# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json, grpc
from ...llmserve_pb2_grpc import PrefillServiceStub, DecodeServiceStub
from ...llmserve_pb2 import PrefillChunkRequest, DecodeRequest

def _channel(addr: str, round_robin: bool):
    # enable client-side round_robin (Python gRPC needs service_config)
    opts = [
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.enable_retries", 1),
    ]
    if round_robin:
        service_config = json.dumps({
            "loadBalancingConfig": [{"round_robin": {}}]
        })
        opts.append(("grpc.service_config", service_config))
    return grpc.aio.insecure_channel(addr, options=opts)

class RPCClient:
    def __init__(self, spec):
        self.spec = spec
        self.prefill_addr = spec.rpc.prefill_service
        self.decode_addr = spec.rpc.decode_service
        self._rr = bool(spec.rpc.round_robin)

        self._ch_prefill = _channel(self.prefill_addr, self._rr)
        self._ch_decode  = _channel(self.decode_addr, self._rr)
        self.prefill = PrefillServiceStub(self._ch_prefill)
        self.decode  = DecodeServiceStub(self._ch_decode)

    async def prefill_chunk(self, req_id: str, prompt: str, start_token: int, n_tokens: int, tenant: str):
        req = PrefillChunkRequest(req_id=req_id, prompt=prompt, start_token=start_token, n_tokens=n_tokens, tenant=tenant)
        return await self.prefill.PrefillChunk(req)

    async def decode_stream(self, req_id: str, prompt: str, tenant: str, opts: dict):
        req = DecodeRequest(
            req_id=req_id, prompt=prompt, tenant=tenant,
            max_tokens=int(opts.get("max_tokens", 256)),
            temperature=float(opts.get("temperature", 0.7)),
            top_p=float(opts.get("top_p", 0.95)),
            strategy_hint=(opts.get("strategy_hint") or "auto"),
            workload=(opts.get("workload") or "general"),
        )
        return self.decode.DecodeStream(req)