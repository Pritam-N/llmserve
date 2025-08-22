from __future__ import annotations

def version_payload(spec) -> str:
    m = spec.models["primary"]
    draft = spec.draft
    parts = [
        f"model={m.id}",
        f"dtype={m.dtype}",
        f"tp={m.tensor_parallel}",
        f"pp={m.pipeline_parallel}",
        f"chunked_prefill={spec.scheduling.policies.chunked_prefill}",
    ]
    if draft and draft.enabled:
        parts.append(f"speculative={draft.id}:{draft.speculative_tokens}")
    return " | ".join(parts)