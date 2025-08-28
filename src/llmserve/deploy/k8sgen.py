from .monolith_renderer import render_manifests_monolith
from .custom_renderer import render_manifests_disagg
from .vllm_renderer import render_manifests_vllm_disagg
from .helpers import _indent
from pathlib import Path

# ============================================================
# Switch + Writer
# ============================================================

def render_all(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    svc_type: str = "LoadBalancer",
) -> dict[str, str]:
    if getattr(spec.deployment, "disaggregated", False):
        # Provider switch: "custom" (our gRPC PD) vs "vllm" (vLLM's PD stack)
        provider = getattr(getattr(spec, "disagg", object()), "provider", "custom")
        if provider == "vllm":
            return render_manifests_vllm_disagg(spec, namespace, image, svc_type)
        return render_manifests_disagg(spec, namespace, image, svc_type)
    return render_manifests_monolith(spec, namespace=namespace, image=image, svc_type=svc_type)

def write_out(dirpath: str, docs: dict[str, str], manifest_text: str) -> Path:
    outdir = Path(dirpath)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        p = outdir / name
        if name.endswith("configmap-manifest.yaml"):
            indented = _indent(manifest_text if manifest_text.endswith("\n") else manifest_text + "\n", 8)
            text = text + indented
        p.write_text(text, encoding="utf-8")
    return outdir