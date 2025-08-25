# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
import orjson
from ..util.ratelimit import RateLimitError

def build_app(spec, router) -> FastAPI:
    app = FastAPI(title="LLMServe", version="0.1.0")

    @app.get("/healthz")
    async def healthz(): return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(body: dict, authorization: str | None = Header(default=None)):
        '''
        curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer <your-api-key>" \
        -d '{
            "prompt": "...",
            "max_tokens": 1024,
            "strategy": "lookahead",   # or "baseline" | "speculative" | "auto"
            "workload": "code",        # "code" | "math" | "general"
            "tenant": "premium"
            }               
        '''
        prompt = body.get("prompt")
        stream = bool(body.get("stream", False))
        tenant = body.get("tenant") or "default"
        max_tokens = int(body.get("max_tokens", 256))
        strategy_hint = (body.get("strategy") or body.get("strategy_hint") or "auto").lower()
        workload = (body.get("workload") or "general").lower()
        if not prompt: raise HTTPException(400, "prompt is required")

        try:
            if not stream:
                text = await router.complete(prompt, tenant=tenant,
                                 opts={"max_tokens": max_tokens,
                                       "strategy_hint": strategy_hint,
                                       "workload": workload})
                return JSONResponse({"model": spec.models["primary"].id, "choices": [{"index": 0, "text": text}]})

            async def gen():
                async for chunk in router.submit_and_stream(prompt, tenant=tenant,
                                               opts={"max_tokens": max_tokens,
                                                     "strategy_hint": strategy_hint,
                                                     "workload": workload}):
                    yield orjson.dumps({"delta": chunk}).decode() + "\n"
            return StreamingResponse(gen(), media_type="application/jsonl")

        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=f"rate_limited: {e.tenant}:{e.reason}")

    return app