# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

import httpx  # streaming client for vLLM PD proxy/decode

from ..engines.vllm_prefil import PrefillEngine
from ..engines.vllm_decode import DecodeEngine
from ..scheduler.fairshare import FairShareScheduler
from ..util.prefix_awarness import PrefixHeuristic
from ..util.ratelimit import RateLimiter, RateLimitError
from ..rpc.client import RPCClient

log = logging.getLogger(__name__)


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

        # Cache model id for payloads
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

        # Optional sampling overrides
        if ctx.opts:
            for k in ("temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"):
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

                    # Try to parse OpenAI-style delta
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # As a last resort treat line as raw text
                        yield line
                        continue

                    # Chat delta path
                    if isinstance(obj, dict) and "choices" in obj and obj["choices"]:
                        ch0 = obj["choices"][0]
                        # Newer OpenAI-style streaming: delta.content
                        delta = ch0.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
                            continue
                        # Some servers send 'text' instead
                        text = ch0.get("text")
                        if text:
                            yield text
                            continue
                        # If neither exists, keep looping (tool-calls, etc.)
                        continue

                    # Fallback keys sometimes used by demos
                    if "output" in obj and isinstance(obj["output"], str):
                        yield obj["output"]
        except httpx.HTTPError as e:
            log.warning("vLLM PD HTTP error: %s", e)
            return
        except Exception as e:
            log.warning("vLLM PD stream failed: %s", e)
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

        # Choose callback backend:
        disagg = bool(getattr(getattr(spec, "deployment", object()), "disaggregated", False))
        provider = getattr(getattr(spec, "disagg", object()), "provider", "custom") if disagg else None

        if disagg and provider == "vllm":
            self._cb = _VLLMDisaggCallbacks(spec)
            log.info("Router using vLLM PD backend (base=%s)", self._cb.base)
        elif disagg:
            self._cb = _RemoteCallbacks(spec)  # our custom gRPC PD
            log.info("Router using CUSTOM gRPC PD backend")
        else:
            self._cb = _RouterCallbacks(decode_pool)  # local
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
            from ..metrics.prometheus import RATE_LIMIT_REJECTS

            RATE_LIMIT_REJECTS.labels(tenant=tenant, reason="tokens@submit").inc()
            raise RateLimitError(tenant, "tokens")

        # Hand to scheduler (which will call callbacks and stream deltas into ctx.out_q)
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
                yield delta
        finally:
            # drain any remaining chunks
            while not ctx.out_q.empty():
                yield await ctx.out_q.get()

    async def complete(
        self, prompt: str, tenant: str | None = None, opts: dict | None = None
    ) -> str:
        out = []
        async for d in self.submit_and_stream(prompt, tenant=tenant, opts=opts or {}):
            out.append(d)
        return "".join(out)