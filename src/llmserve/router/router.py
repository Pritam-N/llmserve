# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio, logging
from typing import AsyncGenerator
from ..engines.vllm_prefil import PrefillEngine
from ..engines.vllm_decode import DecodeEngine
from ..scheduler.fairshare import FairShareScheduler
from ..util.prefix_awarness import PrefixHeuristic
from ..util.ratelimit import RateLimiter, RateLimitError, RateLimitRetry
from ..rpc.client import RPCClient

log = logging.getLogger(__name__)

class _RouterCallbacks:
    def __init__(self, decode_pool):
        self.decode_pool = decode_pool
        self._rr_decode = 0

    def _pick_decode(self) -> DecodeEngine:
        eng = self.decode_pool[self._rr_decode % len(self.decode_pool)]
        self._rr_decode += 1
        return eng

    async def prefill_chunk(self, ctx, start_token: int, n_tokens: int):
        await asyncio.sleep(0)  # placeholder for real prefill

    async def decode_stream(self, ctx):
        dec = self._pick_decode()
        async for delta in dec.stream_text(ctx.prompt, **(ctx.opts or {})):
            yield delta

class Router:
    def __init__(self, spec, prefill_pool: list[PrefillEngine], decode_pool: list[DecodeEngine]):
        self.spec = spec
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool
        self.prefix = PrefixHeuristic()
        self.rate = RateLimiter(spec)
        self.scheduler = FairShareScheduler(spec, self.rate)
        self._cb = _RouterCallbacks(decode_pool)
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self.scheduler.run(self._cb))

    async def stop(self):
        self.scheduler.stop()
        if self._task: await asyncio.wait([self._task], timeout=2)

    async def submit_and_stream(self, prompt: str, tenant: str | None = None, opts: dict | None = None) -> AsyncGenerator[str, None]:
        tenant = tenant or "default"
        est_tokens = max(1, len(prompt) // 4)
        p_hit = self.prefix.observe(prompt)

        # RL assessment (no blocking) -> penalty tag or early reject (policy=reject and deficit>0)
        assess = self.rate.assess(tenant, est_tokens)
        if assess.policy == "reject" and assess.tokens_deficit > 0:
            from ..metrics.prometheus import RATE_LIMIT_REJECTS
            RATE_LIMIT_REJECTS.labels(tenant=tenant, reason="tokens@submit").inc()
            raise RateLimitError(tenant, "tokens")

        ctx = await self.scheduler.submit(
            tenant, prompt, est_tokens, p_hit,
            penalty_mult=assess.penalty_multiplier,
            penalty_expire=assess.penalty_expires_at,
            opts=opts,
        )

        try:
            while True:
                if ctx.done.is_set() and ctx.out_q.empty(): break
                delta = await ctx.out_q.get()
                yield delta
        finally:
            while not ctx.out_q.empty():
                yield await ctx.out_q.get()

    async def complete(self, prompt: str, tenant: str | None = None, opts: dict | None = None) -> str:
        out = []
        async for d in self.submit_and_stream(prompt, tenant=tenant, opts=opts or {}): out.append(d)
        return "".join(out)
    
class _RemoteCallbacks:
    def __init__(self, spec):
        self.spec = spec
        self.rpc = RPCClient(spec)

    async def prefill_chunk(self, ctx, start_token: int, n_tokens: int):
        try:
            await self.rpc.prefill_chunk(ctx.req_id, ctx.prompt, start_token, n_tokens, ctx.tenant)
        except Exception as e:
            log.warning("prefill_chunk gRPC failed: %s", e)
            await asyncio.sleep(0)

    async def decode_stream(self, ctx):
        try:
            stream = await self.rpc.decode_stream(ctx.req_id, ctx.prompt, ctx.tenant, ctx.opts or {})
            async for msg in stream:
                # msg is DecodeChunk(delta=str)
                yield msg.delta
        except Exception as e:
            log.warning("decode_stream gRPC failed: %s", e)
            # fail fast with empty stream
            return