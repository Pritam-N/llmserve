# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import httpx  # CHANGED: used to stream from vLLM PD proxy/decode

from ..engines.vllm_prefil import PrefillEngine
from ..engines.vllm_decode import DecodeEngine
from ..scheduler.fairshare import FairShareScheduler
from ..util.prefix_awarness import PrefixHeuristic
from ..util.ratelimit import RateLimiter, RateLimitError
from ..rpc.client import RPCClient

# CHANGED: import existing + new metrics
from ..metrics.prometheus import (
    METRIC_TTFT,
    METRIC_TPS,
    Q_PREFILL,
    Q_DECODE,
    SPEC_ACCEPT,
    SPEC_SPEEDUP,
    RATE_LIMIT_REJECTS,
    RATE_LIMIT_RETRY,
)
from ..metrics.prometheus import (  # NEW
    BACKEND_TTFT,
    REQUESTS_ACCEPTED,
    STREAM_EVENTS,
    STREAM_BYTES,
    STREAM_TOKENS,
    STREAM_ERRORS,
    VLLM_PD_STREAM_ERRORS,
    VLLM_PD_TOKENS_STREAMED,
)

log = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough router-side estimator to avoid tokenizer deps."""
    return max(1, len(text) // 4)


# =========================
# Local callbacks (monolith)
# =========================
class _RouterCallbacks:
    """
    Local/monolith mode: use in-process DecodeEngine directly.
    """
    def __init__(self, decode_pool: list[DecodeEngine]):
        self.decode_pool = decode_pool
        self._rr_decode = 0

    def _pick_decode(self) -> DecodeEngine:
        eng = self.decode_pool[self._rr_decode % len(self.decode_pool)]
        self._rr_decode += 1
        return eng

    async def prefill_chunk(self, ctx, start_token: int, n_tokens: int):
        # Placeholder for a local prefill stage (unused in monolith).
        await asyncio.sleep(0)

    async def decode_stream(self, ctx):
        dec = self._pick_decode()
        async for delta in dec.stream_text(ctx.prompt, **(ctx.opts or {})):
            yield delta


# =========================
# Custom gRPC disaggregation
# =========================
class _RemoteCallbacks:
    """
    Our custom gRPC PD:
      router -> Prefill RPC (optional, chunked)
      router -> Decode RPC (stream)
    """
    def __init__(self, spec):
        self.spec = spec
        self.rpc = RPCClient(spec)

    async def prefill_chunk(self, ctx, start_token: int, n_tokens: int):
        try:
            await self.rpc.prefill_chunk(
                ctx.req_id, ctx.prompt, start_token, n_tokens, ctx.tenant
            )
        except Exception as e:
            log.warning("prefill_chunk gRPC failed: %s", e)
            # Let scheduler proceed; decode can still run with fewer cached pages.
            await asyncio.sleep(0)

    async def decode_stream(self, ctx):
        try:
            stream = await self.rpc.decode_stream(
                ctx.req_id, ctx.prompt, ctx.tenant, ctx.opts or {}
            )
            async for msg in stream:
                # msg is DecodeChunk(delta=str)
                yield msg.delta
        except Exception as e:
            log.warning("decode_stream gRPC failed: %s", e)
            return


# =========================
# vLLM PD disaggregation
# =========================
class _VLLMDisaggCallbacks:
    """
    vLLM's native disaggregated prefill/decode:
      - We do NOT call a separate prefill RPC. vLLM coordinates prefill->decode.
      - We forward requests to a vLLM endpoint (proxy or decode) and stream SSE.
    """
    def __init__(self, spec):
        self.spec = spec
        base = getattr(getattr(spec, "disagg", object()), "vllm", None)
        if not base:
            raise RuntimeError("disagg.provider=vllm but spec.disagg.vllm is missing")

        url = (base.proxy_url or base.decode_url)
        if not url:
            raise RuntimeError("disagg.provider=vllm requires proxy_url or decode_url")
        self.base = str(url).rstrip("/")
        self.auth_header = getattr(base, "auth_header", None)
        self._client = httpx.AsyncClient(timeout=30.0)

        try:
            self._model_id = self.spec.models["primary"].id
        except Exception:
            self._model_id = None

    async def prefill_chunk(self, ctx, start_token: int, n_tokens: int):
        # No-op: vLLM internal PD performs prefill/KV transfer.
        return

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth_header:
            h["Authorization"] = self.auth_header
        return h

    async def decode_stream(self, ctx):
        """
        Streams tokens from vLLM proxy/decode endpoint using OpenAI-compatible SSE.
        Supports both chat and text-completions style payloads (best-effort).
        """
        url = f"{self.base}/v1/chat/completions"
        payload = {
            "model": self._model_id or (ctx.opts or {}).get("model"),
            "messages": [{"role": "user", "content": ctx.prompt}],
            "stream": True,
        }
        if ctx.opts:
            # Sampling overrides if present
            for k in ("temperature", "top_p", "max_tokens", "stop",
                      "presence_penalty", "frequency_penalty"):
                if k in ctx.opts:
                    payload[k] = ctx.opts[k]

        try:
            async with self._client.stream("POST", url, headers=self._headers(), json=payload) as r:
                r.raise_for_status()
                async for raw in r.aiter_lines():
                    if not raw:
                        continue
                    line = raw
                    if line.startswith("data: "):
                        line = line[6:].strip()
                    if line == "[DONE]":
                        break

                    # Parse OpenAI-style JSON delta; fall back to raw text line
                    try:
                        obj = json.loads(line)
                    except Exception:
                        yield line
                        continue

                    if isinstance(obj, dict) and "choices" in obj and obj["choices"]:
                        ch0 = obj["choices"][0]
                        delta = ch0.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
                            continue
                        text = ch0.get("text")
                        if text:
                            yield text
                            continue
                        # Else: tool-calls/finish-reasons; ignore
                        continue

                    # Fallback non-standard shape
                    if "output" in obj and isinstance(obj["output"], str):
                        yield obj["output"]
        except httpx.HTTPError as e:
            log.warning("vLLM PD HTTP error: %s", e)
            # CHANGED: error metrics (both backend-generic and vLLM-specific)
            VLLM_PD_STREAM_ERRORS.labels(reason="http").inc()
            STREAM_ERRORS.labels(backend="vllm", reason="http").inc()
            return
        except Exception as e:
            log.warning("vLLM PD stream failed: %s", e)
            # CHANGED: error metrics (both backend-generic and vLLM-specific)
            VLLM_PD_STREAM_ERRORS.labels(reason="other").inc()
            STREAM_ERRORS.labels(backend="vllm", reason="other").inc()
            return


# =========================
# Router
# =========================
class Router:
    def __init__(
        self,
        spec,
        prefill_pool: list[PrefillEngine],
        decode_pool: list[DecodeEngine],
    ):
        self.spec = spec
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool

        self.prefix = PrefixHeuristic()
        self.rate = RateLimiter(spec)
        self.scheduler = FairShareScheduler(spec, self.rate)

        # CHANGED: choose backend + set a label for metrics
        disagg = bool(getattr(getattr(spec, "deployment", object()), "disaggregated", False))
        provider = getattr(getattr(spec, "disagg", object()), "provider", "custom") if disagg else None

        if disagg and provider == "vllm":
            self._cb = _VLLMDisaggCallbacks(spec)
            self._backend_label = "vllm"
            log.info("Router using vLLM PD backend (base=%s)", self._cb.base)
        elif disagg:
            self._cb = _RemoteCallbacks(spec)  # our custom gRPC PD
            self._backend_label = "grpc"
            log.info("Router using CUSTOM gRPC PD backend")
        else:
            self._cb = _RouterCallbacks(decode_pool)  # local
            self._backend_label = "local"
            log.info("Router using LOCAL backend (monolith)")

        self._task: asyncio.Task | None = None

    async def start(self):
        # Scheduler drives prefill/decode via callbacks
        self._task = asyncio.create_task(self.scheduler.run(self._cb))

    async def stop(self):
        self.scheduler.stop()
        if self._task:
            await asyncio.wait([self._task], timeout=2)

    async def submit_and_stream(
        self, prompt: str, tenant: str | None = None, opts: dict | None = None
    ) -> AsyncGenerator[str, None]:
        tenant = tenant or "default"
        est_tokens = max(1, len(prompt) // 4)
        p_hit = self.prefix.observe(prompt)

        # Rate-limit assessment (penalize or reject-at-submit depending on policy)
        assess = self.rate.assess(tenant, est_tokens)
        if assess.policy == "reject" and assess.tokens_deficit > 0:
            RATE_LIMIT_REJECTS.labels(tenant=tenant, reason="tokens@submit").inc()
            raise RateLimitError(tenant, "tokens")

        # CHANGED: record an accepted request (after passing RL gate)
        REQUESTS_ACCEPTED.labels(backend=self._backend_label, tenant=tenant).inc()

        # Submit to scheduler (it will call callbacks and push chunks into ctx.out_q)
        t_submit = time.monotonic()
        first_seen = False

        # For streaming aggregates
        total_tokens = 0
        total_bytes = 0

        ctx = await self.scheduler.submit(
            tenant,
            prompt,
            est_tokens,
            p_hit,
            penalty_mult=assess.penalty_multiplier,
            penalty_expire=assess.penalty_expires_at,
            opts=opts,
        )

        try:
            while True:
                if ctx.done.is_set() and ctx.out_q.empty():
                    break
                delta = await ctx.out_q.get()
                if not delta:
                    continue

                # CHANGED: first token metrics
                if not first_seen:
                    ttft = max(0.0, time.monotonic() - t_submit)
                    METRIC_TTFT.observe(ttft)                               # existing global TTFT
                    BACKEND_TTFT.labels(backend=self._backend_label).observe(ttft)  # NEW per-backend TTFT
                    first_seen = True

                # CHANGED: streaming counters
                STREAM_EVENTS.labels(backend=self._backend_label).inc()
                b = len(delta.encode("utf-8", errors="ignore"))
                total_bytes += b
                STREAM_BYTES.labels(backend=self._backend_label).inc(b)
                t_est = _estimate_tokens(delta)
                total_tokens += t_est
                STREAM_TOKENS.labels(backend=self._backend_label).inc(t_est)
                if self._backend_label == "vllm":
                    VLLM_PD_TOKENS_STREAMED.inc(t_est)

                yield delta
        finally:
            # Drain any outstanding chunks (still count metrics)
            while not ctx.out_q.empty():
                delta = await ctx.out_q.get()
                if not delta:
                    continue
                if not first_seen:
                    ttft = max(0.0, time.monotonic() - t_submit)
                    METRIC_TTFT.observe(ttft)
                    BACKEND_TTFT.labels(backend=self._backend_label).observe(ttft)
                    first_seen = True

                STREAM_EVENTS.labels(backend=self._backend_label).inc()
                b = len(delta.encode("utf-8", errors="ignore"))
                total_bytes += b
                STREAM_BYTES.labels(backend=self._backend_label).inc(b)
                t_est = _estimate_tokens(delta)
                total_tokens += t_est
                STREAM_TOKENS.labels(backend=self._backend_label).inc(t_est)
                if self._backend_label == "vllm":
                    VLLM_PD_TOKENS_STREAMED.inc(t_est)

            log.debug(
                "stream finished backend=%s tenant=%s bytes=%d tokens~%d",
                self._backend_label, tenant, total_bytes, total_tokens
            )

    async def complete(
        self, prompt: str, tenant: str | None = None, opts: dict | None = None
    ) -> str:
        out = []
        async for d in self.submit_and_stream(prompt, tenant=tenant, opts=opts or {}):
            out.append(d)
        return "".join(out)