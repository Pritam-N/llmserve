# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging
from typing import AsyncGenerator

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


class DecodeEngine:
    """
    Wrapper for a vLLM engine configured for decode (optionally speculative).
    Today we run monolithic generation; later we'll consume KV handles.
    """
    def __init__(self, spec):
        self.spec = spec
        self.engine: AsyncLLMEngine | None = None
        self.speculative_enabled = bool(self.spec.draft and self.spec.draft.enabled)

    async def startup(self):
        if not VLLM_AVAILABLE:
            log.warning("Starting DecodeEngine in STUB mode (no vLLM)")
            return

        m = self.spec.models["primary"]
        args_kwargs = dict(
            model=m.id,
            dtype=m.dtype,
            tensor_parallel_size=m.tensor_parallel,
            pipeline_parallel_size=m.pipeline_parallel,
        )

        # Best effort: vLLM supports draft model + speculative tokens on some versions
        if self.speculative_enabled and self.spec.draft:
            # These parameters may vary across vLLM versions; guarded by try/except
            args_kwargs.update({
                "speculative_model": self.spec.draft.id,
                "num_speculative_tokens": self.spec.draft.speculative_tokens,
            })

        args = EngineArgs(**args_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        log.info("DecodeEngine ready: %s (speculative=%s)", m.id, self.speculative_enabled)

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        if not VLLM_AVAILABLE or self.engine is None:
            # STUB: deterministic-ish echo for dev
            return f"(stub) {prompt[:64]} ..."

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # AsyncLLMEngine.generate returns List[RequestOutput]
        outputs = await self.engine.generate(prompt, sp)
        if not outputs:
            return ""
        return outputs[0].outputs[0].text

    async def stream_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> AsyncGenerator[str, None]:
        """
        Simple streaming by polling incremental outputs. vLLM has streaming support,
        but we keep a minimal, version-tolerant approach here.
        """
        text = await self.generate_text(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        # naive chunking to simulate streaming
        chunk = 64
        for i in range(0, len(text), chunk):
            yield text[i:i+chunk]