# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import asyncio, logging, grpc
from typing import AsyncIterator
from ..engines.vllm_prefil import PrefillEngine
from ..engines.vllm_decode import DecodeEngine
from ...llmserve_pb2 import (
    PrefillChunkRequest, PrefillChunkReply,
    DecodeRequest, DecodeChunk,
)
from ...llmserve_pb2_grpc import (
    PrefillServiceServicer, add_PrefillServiceServicer_to_server,
    DecodeServiceServicer, add_DecodeServiceServicer_to_server,
)

log = logging.getLogger(__name__)

class PrefillRPC(PrefillServiceServicer):
    def __init__(self, spec):
        self.spec = spec
        self.engine = PrefillEngine(spec)
        self._started = False

    async def start(self):
        if not self._started:
            await self.engine.startup()
            self._started = True

    async def PrefillChunk(self, request: PrefillChunkRequest, context) -> PrefillChunkReply:  # noqa: N802
        await self.start()
        # For now, we just approximate/work warm tokenizer; kv_handle reserved
        res = await self.engine.prefill(request.prompt)
        return PrefillChunkReply(ok=True, prompt_tokens=res.prompt_tokens, kv_handle="")

class DecodeRPC(DecodeServiceServicer):
    def __init__(self, spec):
        self.spec = spec
        self.engine = DecodeEngine(spec)
        self._started = False

    async def start(self):
        if not self._started:
            await self.engine.startup()
            self._started = True

    async def DecodeStream(self, request: DecodeRequest, context) -> AsyncIterator[DecodeChunk]:  # noqa: N802
        await self.start()
        opts = {
            "max_tokens": int(request.max_tokens or 256),
            "temperature": float(request.temperature or 0.7),
            "top_p": float(request.top_p or 0.95),
            "tenant": request.tenant or "default",
            "strategy_hint": request.strategy_hint or "auto",
            "workload": request.workload or "general",
        }
        async for ch in self.engine.stream_text(request.prompt, **opts):
            yield DecodeChunk(delta=ch)

async def serve_prefill(spec, host: str, port: int):
    server = grpc.aio.server(options=[
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    add_PrefillServiceServicer_to_server(PrefillRPC(spec), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("Prefill gRPC server on %s:%d", host, port)
    await server.wait_for_termination()

async def serve_decode(spec, host: str, port: int):
    server = grpc.aio.server(options=[
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ])
    add_DecodeServiceServicer_to_server(DecodeRPC(spec), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("Decode gRPC server on %s:%d", host, port)
    await server.wait_for_termination()