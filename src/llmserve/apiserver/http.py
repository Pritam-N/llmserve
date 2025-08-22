# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import orjson

def build_app(spec, router) -> FastAPI:
    app = FastAPI(title="LLMServe", version="0.1.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt = body.get("prompt")
        max_tokens = int(body.get("max_tokens", 256))
        stream = bool(body.get("stream", False))
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        if not stream:
            text = await router.complete(prompt, max_tokens=max_tokens)
            return JSONResponse(
                {
                    "model": spec.models["primary"].id,
                    "choices": [{"index": 0, "text": text}],
                }
            )

        async def gen():
            async for chunk in router.stream(prompt, max_tokens=max_tokens):
                yield orjson.dumps({"delta": chunk}).decode() + "\n"

        return StreamingResponse(gen(), media_type="application/jsonl")

    return app