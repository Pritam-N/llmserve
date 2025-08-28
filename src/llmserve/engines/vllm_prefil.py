# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging
from dataclasses import dataclass
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


@dataclass
class PrefillResult:
    prompt_tokens: int
    # Placeholder for future KV export handle
    kv_handle: str | None = None


class PrefillEngine:
    """
    Wrapper for a vLLM engine configured for prefill.
    NOTE: vLLM does not yet expose a stable public API to export KV for
    cross-process decode. We call this first to keep the architecture shape.
    """
    def __init__(self, spec, role: str = "prefill"):
        self.spec = spec
        self.role = role
        self.engine: AsyncLLMEngine | None = None

    async def startup(self):
        if not VLLM_AVAILABLE:
            log.warning("Starting PrefillEngine in STUB mode (no vLLM)")
            return

        m = self.spec.models["primary"]
        tp, pp = resolve_tp_pp(self.spec, self.role)

        args = dict(
            model=m.id,
            dtype=m.dtype,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_prefix_caching=self.spec.kv_cache.prefix_caching,
            enforce_eager=True,  # prefill benefits from eager kernels
            gpu_memory_utilization=0.92,
        )
        
        ea = EngineArgs(**args)
        self.engine = AsyncLLMEngine.from_engine_args(ea)
        log.info("PrefillEngine ready: %s", m.id)

    async def prefill(self, prompt: str) -> PrefillResult:
        """
        Performs a prefill pass. Until KV export is wired, this just tokenizes
        and returns a token count, so the router/scheduler can reason about cost.
        """
        if not VLLM_AVAILABLE or self.engine is None:
            # STUB path: approximate token count
            approx_tokens = max(1, len(prompt) // 4)
            return PrefillResult(prompt_tokens=approx_tokens, kv_handle=None)

        # In future: call engine to perform a zero-decode prefill and capture KV.
        # For now, we estimate using vLLM's tokenizer via dummy generation with max_tokens=0.
        sp = SamplingParams(max_tokens=0, temperature=0.0)
        _ = await self.engine.generate(prompt, sp)  # warms caches/tokenizer
        # We don't get token count directly in this call; estimate by len(tokenizer)
        try:
            tokenizer = self.engine.engine.tokenizer  # type: ignore[attr-defined]
            prompt_tokens = len(tokenizer.encode(prompt))
        except Exception:
            prompt_tokens = max(1, len(prompt) // 4)

        return PrefillResult(prompt_tokens=prompt_tokens, kv_handle=None)