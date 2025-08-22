from __future__ import annotations
from pydantic import BaseModel, Field, ValidationError
import yaml
from pathlib import Path

class ModelCfg(BaseModel):
    id: str
    dtype: str = "bf16"
    max_model_len: int | None = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

class DraftCfg(BaseModel):
    id: str
    enabled: bool = False
    speculative_tokens: int = 8

class SchedulingPolicies(BaseModel):
    chunked_prefill: bool = True
    prefill_long_prompt_tokens: int = 1024
    min_decode_slots: int = 2
    queue_max_len: int = 2000

class SchedulingCfg(BaseModel):
    fair_share: dict[str, dict] = Field(default_factory=dict)
    policies: SchedulingPolicies = SchedulingPolicies()

class BudgetsCfg(BaseModel):
    max_tokens_in_flight: int = 250_000
    max_prefill_concurrency: int = 6
    max_decode_concurrency: int = 12
    max_kv_hbm_gb: int = 60
    max_kv_io_gbps: int = 60

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

class TransferCfg(BaseModel):
    engine_order: list[str] = ["nixl", "ucx", "nccl"]
    nixl: dict = Field(default_factory=lambda: {"backend_policy": "auto"})
    ucx: dict = Field(default_factory=lambda: {"rdma": True})
    nccl: dict = Field(default_factory=lambda: {"p2p": True})

class DeployCfg(BaseModel):
    mode: str = "local"  # local|k8s
    replicas: dict[str, int] = Field(default_factory=lambda: {"prefill": 1, "decode": 1})
    resources: dict[str, int] = Field(default_factory=lambda: {"prefill_gpu": 1, "decode_gpu": 1})

class SecurityCfg(BaseModel):
    api_keys: list[str] = Field(default_factory=list)

class SpecCfg(BaseModel):
    models: dict[str, ModelCfg] = Field(default_factory=dict)
    draft: DraftCfg | None = None
    scheduling: SchedulingCfg = SchedulingCfg()
    budgets: BudgetsCfg = BudgetsCfg()
    kv_cache: KVCfg = KVCfg()
    storageClasses: list[StorageClassCfg] = Field(default_factory=list)
    transfer: TransferCfg = TransferCfg()
    deployment: DeployCfg = DeployCfg()
    security: SecurityCfg = SecurityCfg()

class Manifest(BaseModel):
    apiVersion: str
    kind: str
    metadata: dict
    spec: SpecCfg

def load_manifest(path: str | Path) -> SpecCfg:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
        m = Manifest(**raw)
    except ValidationError as ve:
        # bubble up a readable message
        raise SystemExit(f"Manifest validation error:\n{ve}") from ve
    if m.kind != "LLMServe":
        raise SystemExit(f"Unexpected kind={m.kind}, expected LLMServe")
    if "primary" not in m.spec.models:
        raise SystemExit("spec.models.primary is required")
    return m.spec