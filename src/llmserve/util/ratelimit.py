# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio, time
from dataclasses import dataclass
from typing import Dict
from .types import monotonic_time

class RateLimitError(Exception):
    def __init__(self, tenant: str, reason: str): super().__init__(reason); self.tenant=tenant; self.reason=reason

class RateLimitRetry(Exception):
    def __init__(self, tenant: str, reason: str): super().__init__(reason); self.tenant=tenant; self.reason=reason

@dataclass
class _Bucket:
    rate: float
    burst: float
    tokens: float
    last_ts: float

@dataclass
class Assessment:
    tenant: str
    penalty_multiplier: float
    penalty_expires_at: float
    tokens_deficit: float
    policy: str  # "deprioritize" | "queue" | "reject"

class _Handle:
    def __init__(self, release_fn): self._rel = release_fn
    def release(self): 
        try: self._rel()
        except Exception: pass

class RateLimiter:
    """
    Two-stage RL:
      (1) assess() at submission -> returns deprioritization tag (no blocking)
      (2) acquire_for_decode() at execution -> enforces concurrency + tokens
          with policies: deprioritize (non-blocking; reschedule on deficit),
                         queue (wait), reject (raise)
    """
    def __init__(self, spec):
        rl = spec.scheduling.rate_limits or {}
        self.cfg = rl
        self.buckets: Dict[str,_Bucket] = {}
        self.sems: Dict[str, asyncio.Semaphore] = {}
        now = monotonic_time()
        for tenant, c in rl.items():
            rate = float(c.tokens_per_sec)
            burst = float(c.burst or (rate*2 if rate>0 else 0.0))
            self.buckets[tenant] = _Bucket(rate, burst, burst, now)
            self.sems[tenant] = asyncio.Semaphore(int(c.max_concurrency) or 1_000_000)
        if "default" not in self.buckets:
            self.buckets["default"] = _Bucket(0.0, 0.0, 0.0, now)
            self.sems["default"] = asyncio.Semaphore(1_000_000)
        self.spec = spec

    def _cfg(self, tenant: str):
        return self.cfg.get(tenant, self.cfg.get("default"))

    def _refill(self, tenant: str):
        b = self.buckets[tenant]
        if b.rate <= 0: return
        now = monotonic_time()
        dt = now - b.last_ts
        if dt > 0:
            b.tokens = min(b.burst, b.tokens + b.rate*dt)
            b.last_ts = now

    def assess(self, tenant: str, est_tokens: int) -> Assessment:
        t = tenant if tenant in self.buckets else "default"
        pol = self._cfg(t).on_exhaustion
        mult = float(self._cfg(t).deprioritize_multiplier)
        win_ms = int(self._cfg(t).penalty_window_ms)
        self._refill(t)
        deficit = max(0.0, est_tokens - self.buckets[t].tokens)
        if pol == "deprioritize" and deficit > 0.0:
            return Assessment(t, mult, monotonic_time() + (win_ms/1000.0), deficit, pol)
        # queue/reject do not mark penalty at submission
        return Assessment(t, 1.0, 0.0, deficit, pol)

    async def acquire_for_decode(self, tenant: str, cost_tokens: int) -> _Handle:
        t = tenant if tenant in self.buckets else "default"
        c = self._cfg(t)
        policy = c.on_exhaustion
        sem = self.sems[t]

        # concurrency gate
        if policy == "reject":
            if sem.locked() or sem._value <= 0:
                from ..metrics.prometheus import RATE_LIMIT_REJECTS
                RATE_LIMIT_REJECTS.labels(tenant=t, reason="concurrency").inc()
                raise RateLimitError(t, "concurrency")
            sem.acquire()
        elif policy == "queue":
            await sem.acquire()
        else:  # deprioritize: non-blocking, reschedule if no slot
            if sem.locked() or sem._value <= 0:
                from ..metrics.prometheus import RATE_LIMIT_RETRY
                RATE_LIMIT_RETRY.labels(tenant=t, reason="concurrency").inc()
                raise RateLimitRetry(t, "concurrency")
            sem.acquire()

        # tokens bucket
        try:
            b = self.buckets[t]
            self._refill(t)
            if b.rate <= 0:
                return _Handle(lambda: sem.release())
            if b.tokens >= cost_tokens:
                b.tokens -= cost_tokens
                return _Handle(lambda: sem.release())
            if policy == "reject":
                from ..metrics.prometheus import RATE_LIMIT_REJECTS
                RATE_LIMIT_REJECTS.labels(tenant=t, reason="tokens").inc()
                sem.release()
                raise RateLimitError(t, "tokens")
            if policy == "queue":
                deficit = cost_tokens - b.tokens
                wait_s = deficit / max(1e-6, b.rate)
                await asyncio.sleep(min(wait_s, 2.0))
                self._refill(t)
                b.tokens = max(0.0, b.tokens - cost_tokens)
                return _Handle(lambda: sem.release())
            # deprioritize: reschedule
            from ..metrics.prometheus import RATE_LIMIT_RETRY
            RATE_LIMIT_RETRY.labels(tenant=t, reason="tokens").inc()
            sem.release()
            raise RateLimitRetry(t, "tokens")
        except:
            sem.release()
            raise