# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging
from typing import AsyncGenerator

from ..engines.vllm_prefill import PrefillEngine
from ..engines.vllm_decode import DecodeEngine

log = logging.getLogger(__name__)


class Router:
    """
    Minimal router that wires prefill → decode. For now:
      - runs prefill to estimate tokens / warm tokenizer
      - runs monolithic vLLM decode
    Later:
      - KV locality, fair-share scheduling, budgets, and KV handle passing
    """
    def __init__(self, spec, prefill_pool: list[PrefillEngine], decode_pool: list[DecodeEngine]):
        self.spec = spec
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool
        self._rr_prefill = 0
        self._rr_decode = 0

    def _pick_prefill(self) -> PrefillEngine:
        if not self.prefill_pool:
            raise RuntimeError("No prefill engines")
        eng = self.prefill_pool[self._rr_prefill % len(self.prefill_pool)]
        self._rr_prefill += 1
        return eng

    def _pick_decode(self) -> DecodeEngine:
        if not self.decode_pool:
            raise RuntimeError("No decode engines")
        eng = self.decode_pool[self._rr_decode % len(self.decode_pool)]
        self._rr_decode += 1
        return eng

    async def complete(self, prompt: str, max_tokens: int = 256) -> str:
        prefill = self._pick_prefill()
        decode = self._pick_decode()

        # Prefill step (placeholder for future KV export)
        _prefill_res = await prefill.prefill(prompt)

        # Decode step (monolithic today)
        text = await decode.generate_text(prompt, max_tokens=max_tokens)
        return text

    async def stream(self, prompt: str, max_tokens: int = 256) -> AsyncGenerator[str, None]:
        prefill = self._pick_prefill()
        decode = self._pick_decode()
        _prefill_res = await prefill.prefill(prompt)
        async for chunk in decode.stream_text(prompt, max_tokens=max_tokens):
            yield chunk