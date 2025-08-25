# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import os, logging, asyncio
from typing import AsyncGenerator
import torch

log = logging.getLogger(__name__)

class LookaheadProvider:
    """
    Plugin adapter for hao-ai-lab/LookaheadDecoding (LADE).
    Uses HF transformers path (not vLLM). Fallback is handled by the DecodeEngine.
    """
    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_flash = False

    def attach(self, _vllm_engine):
        # Not used: LADE runs through HF; we ignore vLLM engine.
        pass

    async def _lazy_init(self):
        if self.model is not None: 
            return

        # --- 1) Import LADE and HF
        try:
            import lade
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        except Exception as e:
            raise RuntimeError(f"LADE/HF import failed: {e}")

        # --- 2) Activate LADE augmentation
        # Repo docs: set USE_LADE=1 (and optionally LOAD_LADE=1) then augment & config.  [oai_citation:3‡GitHub](https://github.com/hao-ai-lab/LookaheadDecoding)
        os.environ.setdefault("USE_LADE", "1")
        # LOAD_LADE can reduce overhead when repeatedly enabling:
        os.environ.setdefault("LOAD_LADE", "1")
        lade.augment_all()

        la = self.spec.lookahead
        level = int(la.ngram)
        guess = int(la.max_parallel)
        window = level + 3  # heuristic; can expose via manifest later

        # --- 3) FlashAttention setup: prefer specialized wheel; else vanilla FA if present
        self.use_flash = False
        try:
            import flash_attn  # noqa: F401
            self.use_flash = True
        except Exception:
            pass

        lade_opts = dict(LEVEL=level, WINDOW_SIZE=window, GUESS_SET_SIZE=guess, DEBUG=0)
        if self.use_flash:
            lade_opts["USE_FLASH"] = True
        lade.config_lade(**lade_opts)

        # --- 4) Load HF model/tokenizer (LLaMA-family recommended)
        m = self.spec.models["primary"]
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}
        torch_dtype = dtype_map.get(m.dtype, torch.float16)

        # Try to detect model type to warn if not LLaMA (LADE README says LLaMA supported).  [oai_citation:4‡GitHub](https://github.com/hao-ai-lab/LookaheadDecoding)
        try:
            from transformers import AutoConfig
            mt = AutoConfig.from_pretrained(m.id).model_type
            if "llama" not in str(mt).lower():
                log.warning("LookaheadDecoding currently targets LLaMA-family; got model_type=%s. Continuing anyway.", mt)
        except Exception:
            pass

        attn_flag = {"attn_implementation": "flash_attention_2"} if self.use_flash else {}
        self.tokenizer = AutoTokenizer.from_pretrained(m.id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            m.id,
            torch_dtype=torch_dtype,
            device_map="auto",
            **attn_flag,
        )

        # make sure model is in eval
        self.model.eval()

        # prime gpu a bit (async-friendly warmup)
        await asyncio.sleep(0)

    async def generate_text(self, prompt: str, **kw) -> str:
        await self._lazy_init()
        max_tokens = int(kw.get("max_tokens", 256))
        temperature = float(kw.get("temperature", 0.7))
        top_p = float(kw.get("top_p", 0.95))

        device = next(self.model.parameters()).device  # type: ignore[attr-defined]
        toks = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = self.model.generate(
                **toks,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the suffix after the prompt to match vLLM behavior
        return text[len(prompt):]

    async def stream_text(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        # Simple chunked streaming over the full output
        text = await self.generate_text(prompt, **kw)
        chunk = int(kw.get("stream_chunk", 64))
        for i in range(0, len(text), chunk):
            yield text[i : i + chunk]