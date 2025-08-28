# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging, time, random
from typing import AsyncGenerator, Protocol
from ..metrics.prometheus import SPEC_ACCEPT, SPEC_SPEEDUP
from ..util.plugin import load_symbol, PluginLoadError
from ._helpers import resolve_tp_pp

log = logging.getLogger(__name__)

try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams

    VLLM_AVAILABLE = True
except Exception as e:
    log.warning("vLLM not available: %s", e)
    EngineArgs = object  # type: ignore
    AsyncLLMEngine = object  # type: ignore
    SamplingParams = object  # type: ignore
    VLLM_AVAILABLE = False


class DecodeStrategy(Protocol):
    async def generate_text(self, prompt: str, **kw) -> str: ...
    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]: ...


def _build_engine_args(kwargs: dict) -> EngineArgs | None:
    if not VLLM_AVAILABLE:
        return None
    while True:
        try:
            return EngineArgs(**kwargs)
        except TypeError as te:
            msg = str(te)
            removed = False
            for k in list(kwargs.keys()):
                if f"'{k}'" in msg or (k + "=") in msg:
                    kwargs.pop(k)
                    removed = True
                    break
            if not removed:
                raise


class _BaseVLLMStrategy:
    def __init__(self, spec):
        self.spec = spec
        self.engine: AsyncLLMEngine | None = None

    async def _startup(self, extra_args: dict | None = None):
        if not VLLM_AVAILABLE:
            log.warning("Decode strategy in STUB mode (no vLLM)")
            return
        m = self.spec.models["primary"]
        tp, pp = resolve_tp_pp(self.spec, self.role)

        args_kwargs: dict = dict(
            model=m.id,
            dtype=m.dtype,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_prefix_caching=self.spec.kv_cache.prefix_caching,
            enforce_eager=self.spec.scheduling.policies.chunked_prefill,
            gpu_memory_utilization=0.92,
        )
        if extra_args:
            args_kwargs.update(extra_args)
        ea = _build_engine_args(args_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(ea) if ea else None

    async def _telemetry_spec(self, outputs, start_ts: float):
        """Best-effort speculative metrics (acceptance, speedup) if available + sampling."""
        tel = self.spec.telemetry
        if not tel.speculative_metrics_enabled:
            return
        if (
            tel.speculative_sample_rate < 1.0
            and random.random() > tel.speculative_sample_rate
        ):
            return
        try:
            o0 = outputs[0]
            metrics = getattr(o0, "metrics", None) or {}
            acc = metrics.get("spec_acceptance") or metrics.get("acceptance_rate")
            if acc is not None:
                try:
                    SPEC_ACCEPT.observe(float(acc))
                except Exception:
                    pass
            # naive speedup estimate: baseline ~ duration; if speculative tokens present, scale
            dur = max(1e-6, time.time() - start_ts)
            base = dur  # without baseline, we can't do better; keep 1.0
            SPEC_SPEEDUP.observe(base / dur if dur > 0 else 1.0)
        except Exception:
            pass

    async def generate_text(self, prompt: str, **kw) -> str:
        if not VLLM_AVAILABLE or self.engine is None:
            return f"(stub) {prompt[:64]} ..."
        sp = SamplingParams(
            max_tokens=int(kw.get("max_tokens", 256)),
            temperature=float(kw.get("temperature", 0.7)),
            top_p=float(kw.get("top_p", 0.95)),
        )
        t0 = time.time()
        outs = await self.engine.generate(prompt, sp)
        await self._telemetry_spec(outs, t0)
        if not outs:
            return ""
        return outs[0].outputs[0].text

    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        text = await self.generate_text(prompt, **kw)
        chunk = int(kw.get("stream_chunk", 64))
        for i in range(0, len(text), chunk):
            yield text[i : i + chunk]


class BaselineStrategy(_BaseVLLMStrategy):
    def __init__(self, spec, role="decode"):
        super().__init__(spec, role)

    async def startup(self):
        await self._startup()


class SpeculativeStrategy(_BaseVLLMStrategy):
    def __init__(self, spec, role="decode"):
        super().__init__(spec, role)

    async def startup(self):
        sd = self.spec.spec_decode
        extras: dict = {}
        if sd.enabled and sd.method == "draft":
            extras["speculative_model"] = sd.draft_model
            extras["num_speculative_tokens"] = int(sd.num_spec_tokens)
        elif sd.enabled and sd.method in {"eagle", "eagle2", "medusa", "arctic"}:
            extras["speculative_algorithm"] = sd.method
            extras["num_speculative_tokens"] = int(sd.num_spec_tokens)
        try:
            await self._startup(extra_args=extras)
        except Exception as e:
            fb = sd.fallback
            log.warning("Speculative startup failed (%s). Fallback=%s", e, fb)
            if fb == "draft" and sd.draft_model:
                await self._startup(
                    extra_args={
                        "speculative_model": sd.draft_model,
                        "num_speculative_tokens": int(sd.num_spec_tokens),
                    }
                )
            elif fb == "baseline":
                await self._startup(extra_args={})
            else:
                await self._startup(extra_args={})  # disable


class LookaheadStrategy(_BaseVLLMStrategy):
    def __init__(self, spec, role="decode"):
        super().__init__(spec, role)
        self.provider = None
        self.active = False

    async def startup(self):
        await self._startup()
        la = self.spec.lookahead
        if not la.enabled:
            log.info("Lookahead disabled; baseline behavior")
            return
        if not la.plugin:
            log.warning(
                "Lookahead enabled but no plugin specified; fallback=%s", la.fallback
            )
            self._fallback()
            return
        try:
            Provider = load_symbol(la.plugin, "LookaheadProvider")
            self.provider = Provider(self.spec)  # plugin-defined ctor
            # attach may receive the low-level engine if needed
            if self.engine is not None and hasattr(self.provider, "attach"):
                self.provider.attach(self.engine)
            self.active = True
            log.info("Lookahead plugin %s loaded", la.plugin)
        except PluginLoadError as e:
            log.warning("Lookahead plugin load failed: %s; fallback=%s", e, la.fallback)
            self._fallback()

    def _fallback(self):
        # nothing else to do here: we keep engine ready; methods fall back to baseline
        self.active = False

    async def generate_text(self, prompt: str, **kw) -> str:
        if self.active and hasattr(self.provider, "generate_text"):
            return await self.provider.generate_text(prompt, **kw)
        return await super().generate_text(prompt, **kw)

    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        if self.active and hasattr(self.provider, "stream_text"):
            async for ch in self.provider.stream_text(prompt, **kw):
                yield ch
            return
        async for ch in super().stream_text(prompt, **kw):
            yield ch


# add alongside existing strategies
class HybridStrategy(_BaseVLLMStrategy):
    """
    Auto-select per-request:
      - default to vLLM baseline/speculative (fast path)
      - only use lookahead provider if: plugin loaded AND policy allows AND request hints
      - otherwise, fallback to vLLM
    """

    def __init__(self, spec, role="decode"):
        super().__init__(spec, role)
        self.vllm_baseline = BaselineStrategy(spec, role)
        self.vllm_spec = SpeculativeStrategy(spec, role)
        self.la = LookaheadStrategy(spec, role)

    async def startup(self):
        # Always prepare vLLM paths first (fast path)
        await self.vllm_baseline.startup()
        await self.vllm_spec.startup()
        # Prepare lookahead plugin too (may end up inactive)
        await self.la.startup()

    def _should_use_lookahead(
        self, tenant: str, max_tokens: int, strategy_hint: str, workload: str
    ) -> bool:
        s = self.spec
        if s.decode_strategy != "hybrid":
            return False
        if not (s.hybrid.enabled and s.lookahead.enabled and self.la.active):
            return False
        # honor explicit hint
        if strategy_hint == "lookahead":
            return True
        if strategy_hint == "baseline" or strategy_hint == "speculative":
            return False
        # policy checks
        if s.hybrid.tenants and tenant not in s.hybrid.tenants:
            return False
        if max_tokens < s.hybrid.min_decode_tokens:
            return False
        # workload heuristic — prefer lookahead for code/math if caller hints it
        if workload in {"code", "math"}:
            return True
        # prefer vLLM unless policy overrides
        return not s.hybrid.prefer_vllm

    async def generate_text(self, prompt: str, **kw) -> str:
        tenant = kw.get("tenant", "default")
        max_tokens = int(kw.get("max_tokens", 256))
        strategy_hint = (kw.get("strategy_hint") or "auto").lower()
        workload = (kw.get("workload") or "general").lower()

        if self._should_use_lookahead(tenant, max_tokens, strategy_hint, workload):
            return await self.la.generate_text(prompt, **kw)

        # choose between baseline vs speculative (based on spec_decode.enabled)
        if self.spec.spec_decode.enabled:
            return await self.vllm_spec.generate_text(prompt, **kw)
        return await self.vllm_baseline.generate_text(prompt, **kw)

    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        tenant = kw.get("tenant", "default")
        max_tokens = int(kw.get("max_tokens", 256))
        strategy_hint = (kw.get("strategy_hint") or "auto").lower()
        workload = (kw.get("workload") or "general").lower()

        if self._should_use_lookahead(tenant, max_tokens, strategy_hint, workload):
            async for c in self.la.stream_text(prompt, **kw):
                yield c
                return

        if self.spec.spec_decode.enabled:
            async for c in self.vllm_spec.stream_text(prompt, **kw):
                yield c
                return
        async for c in self.vllm_baseline.stream_text(prompt, **kw):
            yield c


class DecodeEngine:
    def __init__(self, spec, role: str = "decode"):
        self.spec = spec
        self.role = role
        self.strategy: DecodeStrategy | None = None

    async def startup(self):
        mode = (self.spec.decode_strategy or "baseline").lower()
        if mode == "speculative" or (
            self.spec.spec_decode.enabled and mode == "baseline"
        ):
            self.strategy = SpeculativeStrategy(self.spec, role=self.role)
        elif mode == "lookahead":
            self.strategy = LookaheadStrategy(self.spec, role=self.role)
        elif mode == "hybrid":
            self.strategy = HybridStrategy(self.spec, role=self.role)
        else:
            self.strategy = BaselineStrategy(self.spec, role=self.role)

        await self.strategy.startup()  # type: ignore[attr-defined]

        log.info("DecodeEngine strategy=%s", self.strategy.__class__.__name__)

    async def generate_text(self, prompt: str, **kw) -> str:
        return await self.strategy.generate_text(prompt, **kw)  # type: ignore

    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        async for c in self.strategy.stream_text(prompt, **kw):  # type: ignore
            yield c
