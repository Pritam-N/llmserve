# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import time, heapq, asyncio, uuid
from dataclasses import dataclass, field
from typing import Any
from ..metrics.prometheus import Q_PREFILL, Q_DECODE, RATE_LIMIT_RETRY
from ..util.types import monotonic_time

@dataclass(order=True)
class _ScoredItem:
    score: float
    arrival_id: int
    req_id: str = field(compare=False)
    kind: str = field(compare=False)          # "prefill" | "decode"
    payload: Any = field(compare=False)

@dataclass
class RequestCtx:
    req_id: str
    tenant: str
    prompt: str
    est_tokens: int
    prefix_hit_prob: float
    created_ts: float
    penalty_mult: float
    penalty_expires: float
    opts: dict 
    # client pipes
    out_q: asyncio.Queue[str]
    done: asyncio.Event

class FairShareScheduler:
    def __init__(self, spec, rate_limiter):
        self.spec = spec
        self.rate = rate_limiter
        self.arrival = 0
        self.q: list[_ScoredItem] = []
        self.reqs: dict[str, RequestCtx] = {}
        self._stop = asyncio.Event()

    async def submit(self, tenant: str, prompt: str, est_tokens: int, prefix_hit_prob: float,
                     penalty_mult: float, penalty_expire: float, opts: dict) -> RequestCtx:
        rid = str(uuid.uuid4())
        ctx = RequestCtx(
            req_id=rid, tenant=tenant, prompt=prompt, est_tokens=max(1, est_tokens),
            prefix_hit_prob=prefix_hit_prob, created_ts=time.time(),
            penalty_mult=penalty_mult, penalty_expires=penalty_expire,
            opts=opts or {}, 
            out_q=asyncio.Queue(), 
            done=asyncio.Event(),
        )
        self.reqs[rid] = ctx
        self._enqueue_prefill(ctx)
        return ctx

    def _tenant_weight(self, tenant: str) -> float:
        tenants = self.spec.scheduling.fair_share.get("tenants", {})
        return float(tenants.get(tenant, {}).get("weight", 1.0))

    def _score(self, ctx: RequestCtx, kind: str) -> float:
        weight = self._tenant_weight(ctx.tenant)
        srpt = 1.0 / ctx.est_tokens
        score = (1.0 / weight) * srpt
        pol = self.spec.scheduling.policies
        if pol.prefix_awareness and ctx.prefix_hit_prob > 0.5:
            score *= 0.7
        # RL deprioritization window
        if ctx.penalty_mult > 1.0 and (ctx.penalty_expires == 0.0 or monotonic_time() < ctx.penalty_expires):
            score *= ctx.penalty_mult
        # aging
        waited = max(0.0, time.time() - ctx.created_ts)
        if pol.aging_seconds > 0:
            score *= max(0.8, 1.0 - min(1.0, waited / pol.aging_seconds) * 0.2)
        if kind == "decode": score *= 0.9
        return score

    def _enqueue_prefill(self, ctx: RequestCtx, offset: int = 0):
        self.arrival += 1
        it = _ScoredItem(self._score(ctx, "prefill"), self.arrival, ctx.req_id, "prefill", {"offset": offset})
        heapq.heappush(self.q, it)
        Q_PREFILL.set(sum(1 for i in self.q if i.kind == "prefill"))

    def _enqueue_decode(self, ctx: RequestCtx):
        self.arrival += 1
        it = _ScoredItem(self._score(ctx, "decode"), self.arrival, ctx.req_id, "decode", {})
        heapq.heappush(self.q, it)
        Q_DECODE.set(sum(1 for i in self.q if i.kind == "decode"))

    async def run(self, router_callbacks):
        pol = self.spec.scheduling.policies
        chunk = int(pol.prefill_chunk_tokens)
        min_decode_slots = int(pol.min_decode_slots)

        while not self._stop.is_set():
            if not self.q:
                await asyncio.sleep(0.001); continue

            item = heapq.heappop(self.q)
            ctx = self.reqs.get(item.req_id)
            if ctx is None: continue

            if item.kind == "prefill":
                off = item.payload.get("offset", 0)
                await router_callbacks.prefill_chunk(ctx, off, chunk)
                off2 = off + chunk
                if off2 < ctx.est_tokens and pol.chunked_prefill:
                    self._enqueue_prefill(ctx, off2)
                else:
                    self._enqueue_decode(ctx)

            else:  # decode
                try:
                    # decode-time rate limit enforcement (may reschedule)
                    handle = await self.rate.acquire_for_decode(ctx.tenant, ctx.est_tokens)
                except Exception as e:
                    from ..util.ratelimit import RateLimitRetry, RateLimitError
                    if isinstance(e, RateLimitRetry):
                        RATE_LIMIT_RETRY.labels(tenant=ctx.tenant, reason=e.reason).inc()
                        # reschedule decode after a short pause
                        await asyncio.sleep(0.01)
                        self._enqueue_decode(ctx)
                        continue
                    if isinstance(e, RateLimitError):
                        # rejected; complete with empty stream
                        ctx.done.set()
                        continue
                    raise

                try:
                    async for delta in router_callbacks.decode_stream(ctx):
                        await ctx.out_q.put(delta)
                    ctx.done.set()
                finally:
                    handle.release()

                # run more decodes back-to-back if available
                extra = min_decode_slots - 1
                while extra > 0:
                    nxt_i = next((i for i, it in enumerate(self.q) if it.kind == "decode"), None)
                    if nxt_i is None: break
                    it2 = self.q.pop(nxt_i); heapq.heapify(self.q)
                    ctx2 = self.reqs.get(it2.req_id)
                    if ctx2:
                        try:
                            handle2 = await self.rate.acquire_for_decode(ctx2.tenant, ctx2.est_tokens)
                        except Exception:
                            # put it back for later
                            self._enqueue_decode(ctx2)
                            break
                        try:
                            async for delta in router_callbacks.decode_stream(ctx2):
                                await ctx2.out_q.put(delta)
                            ctx2.done.set()
                        finally:
                            handle2.release()
                    extra -= 1

    def stop(self): self._stop.set()