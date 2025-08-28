from __future__ import annotations

def resolve_tp_pp(spec, role: str) -> tuple[int, int]:
    # Prefer role-specific overrides; fall back to model defaults
    if role == "prefill":
        tp = max(1, int(spec.roles.prefill.tp))
        pp = max(1, int(spec.roles.prefill.pp))
    else:
        tp = max(1, int(spec.roles.decode.tp))
        pp = max(1, int(spec.roles.decode.pp))
    return tp, pp