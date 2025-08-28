# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import Literal
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# =========================
# Models (existing)
# =========================


class ModelCfg(BaseModel):
    id: str
    dtype: str = "bf16"
    max_model_len: int | None = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1


# (legacy; kept for backward compatibility if older manifests still include it)
class DraftCfg(BaseModel):
    id: str
    enabled: bool = False
    speculative_tokens: int = 8


# =========================
# Scheduling & Budgets
# =========================


class SchedulingPolicies(BaseModel):
    chunked_prefill: bool = True
    prefill_chunk_tokens: int = 512
    prefill_long_prompt_tokens: int = 1024
    min_decode_slots: int = 2
    queue_max_len: int = 2000
    prefix_awareness: bool = True
    aging_seconds: float = 2.0


class RateLimitCfg(BaseModel):
    # logical token bucket per tenant
    tokens_per_sec: float = 0.0
    burst: int = 0
    max_concurrency: int = 0
    # what to do when limits hit:
    # - "deprioritize": let requests in but worsen their scheduler score for a window
    # - "reject": 429 immediately
    # - "queue": wait until capacity (blocking)
    on_exhaustion: Literal["deprioritize", "reject", "queue"] = "deprioritize"
    # penalty applied when on_exhaustion=deprioritize
    deprioritize_multiplier: float = 1.25
    penalty_window_ms: int = 2000


class SchedulingCfg(BaseModel):
    fair_share: dict[str, dict] = Field(default_factory=dict)
    policies: SchedulingPolicies = SchedulingPolicies()
    rate_limits: dict[str, RateLimitCfg] = Field(default_factory=dict)


class BudgetsCfg(BaseModel):
    max_tokens_in_flight: int = 250_000
    max_prefill_concurrency: int = 6
    max_decode_concurrency: int = 12
    max_kv_hbm_gb: int = 60
    max_kv_io_gbps: int = 60


# =========================
# KV Cache
# =========================


class EvictionCfg(BaseModel):
    hot_keep_secs: int = 300
    evict_if_inactive_secs: int = 900
    tenant_hotset_gb: dict[str, int] = Field(default_factory=dict)


class KVCfg(BaseModel):
    page_size_kb: int = 512
    prefix_caching: bool = True
    hbm_dtype: str = "fp16"
    disk_dtype: str = "int8"
    eviction: EvictionCfg = EvictionCfg()


class StorageClassCfg(BaseModel):
    name: str
    type: str  # filesystem|s3|nas
    path: str | None = None
    bucket: str | None = None
    region: str | None = None
    gds: bool | None = None
    prefetch_on_prefix: bool = False


# =========================
# Transfers
# =========================


class TransferCfg(BaseModel):
    engine_order: list[str] = Field(default_factory=lambda: ["nixl", "ucx", "nccl"])
    nixl: dict = Field(default_factory=lambda: {"backend_policy": "auto"})
    ucx: dict = Field(default_factory=lambda: {"rdma": True})
    nccl: dict = Field(default_factory=lambda: {"p2p": True})


# =========================
# Deployment & Security
# =========================


class DeployCfg(BaseModel):
    mode: str = "local"  # local|k8s
    disaggregated: bool = False
    replicas: dict[str, int] = Field(
        default_factory=lambda: {"router": 1, "prefill": 1, "decode": 1}
    )
    resources: dict[str, int] = Field(
        default_factory=lambda: {"prefill_gpu": 1, "decode_gpu": 1}
    )
    router_port: int = 8000
    prefill_port: int = 9001
    decode_port: int = 9002


class SecurityCfg(BaseModel):
    api_keys: list[str] = Field(default_factory=list)


# =========================
# Plugins & Strategies
# =========================


class SpecDecodeCfg(BaseModel):
    enabled: bool = False
    method: Literal["draft", "eagle", "eagle2", "medusa", "arctic"] = "draft"
    num_spec_tokens: int = 8
    draft_model: str | None = None
    # fallback if plugin/feature unsupported
    fallback: Literal["baseline", "draft", "disable"] = "baseline"


# =========================
# Lookahead Spec
# =========================


class LookaheadCfg(BaseModel):
    enabled: bool = False
    ngram: int = 4
    max_parallel: int = 8
    plugin: str | None = None  # e.g. "llmserve_ext_lookahead"
    # fallback if plugin missing/unloadable
    fallback: Literal["baseline", "speculative", "disable"] = "baseline"


class HybridCfg(BaseModel):
    enabled: bool = False
    # if request max_tokens >= this & other predicates match -> consider lookahead
    min_decode_tokens: int = 512
    # restrict lookahead to specific tenants (empty => allow all)
    tenants: list[str] = Field(default_factory=list)
    # prefer vLLM when both are possible (strong default)
    prefer_vllm: bool = True
    # fallback if lookahead plugin missing/unloadable
    fallback: Literal["baseline", "speculative", "disable"] = "baseline"


class TelemetryCfg(BaseModel):
    # speculative acceptance/speedup metrics
    speculative_metrics_enabled: bool = True
    # to reduce cost on very high QPS installs
    speculative_sample_rate: float = 1.0  # 0.0..1.0

class RoleParallelismCfg(BaseModel):
    tp: int = 1  # tensor-parallel GPUs per pod
    pp: int = 1  # pipeline-parallel stages per pod
    dp: int = 1  # pods (K8s replicas) for data-parallel scaling

    shards_across_nodes: bool = (
        False  # shard a single model across nodes (not enabled yet)
    )
    init_method: str | None = (
        None  # e.g., "tcp://llmserve-rdzv:29500" for torch.distributed rendezvous
    )

class RolesCfg(BaseModel):
    prefill: RoleParallelismCfg = RoleParallelismCfg()
    decode: RoleParallelismCfg = RoleParallelismCfg()


class RPCCfg(BaseModel):
    scheme: Literal["grpc"] = "grpc"
    prefill_service: str = "llmserve-prefill:9001"  # K8s service DNS:port
    decode_service: str = "llmserve-decode:9002"
    timeout_s: float = 30.0
    round_robin: bool = True  # client-side LB; fine with ClusterIP too

# =========================
# Disaggregated mode
# =========================

class DisaggVLLMCfg(BaseModel):
    # Where our router will forward user requests (vLLM-provided endpoint)
    # Option A: point to a vLLM "disagg proxy" (recommended when you use their PD orchestration)
    # Option B: point directly to the vLLM decode server if that server fronts the API for your version.
    proxy_url: str | None = None      # e.g., "http://vllm-proxy:8000"
    decode_url: str | None = None     # e.g., "http://vllm-decode:8000"
    auth_header: str | None = None    # optional bearer

class DisaggCfg(BaseModel):
    provider: Literal["custom", "vllm"] = "custom"
    vllm: DisaggVLLMCfg = DisaggVLLMCfg()

# =========================
# Top-level Spec
# =========================


class SpecCfg(BaseModel):
    models: dict[str, ModelCfg] = Field(default_factory=dict)

    # legacy: will not be used when spec_decode is present
    draft: DraftCfg | None = None

    # decoding choices
    decode_strategy: Literal["baseline", "speculative", "lookahead", "hybrid"] = (
        "baseline"
    )
    spec_decode: SpecDecodeCfg = SpecDecodeCfg()
    lookahead: LookaheadCfg = LookaheadCfg()
    hybrid: HybridCfg = HybridCfg()

    # execution policies & limits
    scheduling: SchedulingCfg = SchedulingCfg()
    budgets: BudgetsCfg = BudgetsCfg()

    # kv + storage + transfers
    kv_cache: KVCfg = KVCfg()
    storageClasses: list[StorageClassCfg] = Field(default_factory=list)
    transfer: TransferCfg = TransferCfg()

    # deploy & security
    deployment: DeployCfg = DeployCfg()
    security: SecurityCfg = SecurityCfg()

    # telemetry/tuning
    telemetry: TelemetryCfg = TelemetryCfg()

    roles: RolesCfg = RolesCfg()      # <— NEW
    rpc: RPCCfg = RPCCfg()

    disagg: DisaggCfg = DisaggCfg()

    @model_validator(mode="after")
    def _validate_spec(self):
        if self.spec_decode.enabled and self.spec_decode.method == "draft":
            if not self.spec_decode.draft_model:
                raise ValueError(
                    "spec_decode.method=draft requires spec_decode.draft_model"
                )
        # sensible defaults for rate limits
        if "default" not in self.scheduling.rate_limits:
            self.scheduling.rate_limits["default"] = RateLimitCfg()
        return self


class Manifest(BaseModel):
    apiVersion: str
    kind: str
    metadata: dict
    spec: SpecCfg


# =========================
# Loader
# =========================


def load_manifest(path: str | Path) -> SpecCfg:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
        m = Manifest(**raw)
    except ValidationError as ve:
        raise SystemExit(f"Manifest validation error:\n{ve}") from ve
    if m.kind != "LLMServe":
        raise SystemExit(f"Unexpected kind={m.kind}, expected LLMServe")
    if "primary" not in m.spec.models:
        raise SystemExit("spec.models.primary is required")
    return m.spec

