# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio
import logging
import uvicorn
from prometheus_client import start_http_server
from .apiserver.http import build_app
from .engines.vllm_prefil import PrefillEngine
from .engines.vllm_decode import DecodeEngine
from .router.router import Router

log = logging.getLogger("llmserve")

class Orchestrator:
    def __init__(self, spec):
        self.spec = spec
        self.prefill_pool: list[PrefillEngine] = []
        self.decode_pool: list[DecodeEngine] = []
        self.router: Router | None = None

    async def _start_prefill_pool(self, n: int):
        for _ in range(max(1, n)):
            eng = PrefillEngine(self.spec)
            await eng.startup()
            self.prefill_pool.append(eng)
        log.info("Prefill pool size: %d", len(self.prefill_pool))

    async def _start_decode_pool(self, n: int):
        for _ in range(max(1, n)):
            eng = DecodeEngine(self.spec)
            await eng.startup()
            self.decode_pool.append(eng)
        log.info("Decode pool size: %d", len(self.decode_pool))

    async def run_local(self, host: str = "0.0.0.0", port: int = 8000, metrics_port: int = 9400):
        # metrics
        start_http_server(metrics_port)
        log.info("metrics server on :%d", metrics_port)

        # engines
        prefill_n = self.spec.deployment.replicas.get("prefill", 1)
        decode_n  = self.spec.deployment.replicas.get("decode", 1)
        await asyncio.gather(
            self._start_prefill_pool(prefill_n),
            self._start_decode_pool(decode_n),
        )

        # router
        self.router = Router(self.spec, self.prefill_pool, self.decode_pool)
        await self.router.start()

        # FastAPI
        app = build_app(self.spec, self.router)
        config = uvicorn.Config(app=app, host=host, port=port, workers=1, loop="asyncio", lifespan="on")
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        if self.router:
            await self.router.stop()
        if self.prefill_pool:
            for eng in self.prefill_pool:
                await eng.shutdown()
        if self.decode_pool:
            for eng in self.decode_pool:
                await eng.shutdown()