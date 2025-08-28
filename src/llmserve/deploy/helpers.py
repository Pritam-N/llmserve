from __future__ import annotations
from pathlib import Path
import textwrap
import os
from typing import Any, Dict, List, Optional

# ============================================================
# Helpers
# ============================================================

def _yaml(doc: str) -> str:
    return textwrap.dedent(doc).lstrip()

def _q(val: Any) -> str:
    """Quote a scalar for YAML if it contains special chars; else as-is."""
    s = str(val)
    if any(c in s for c in [":", "{", "}", "[", "]", ",", "#", "&", "*", "!", "|", ">", "@", "%", '"', "'"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s

def _indent(lines: str, spaces: int) -> str:
    pad = " " * spaces
    return "".join(pad + ln if ln.strip() else ln for ln in lines.splitlines(True))

def _to_yaml_kv(d: Dict[str, Any], indent_spaces: int) -> str:
    if not d:
        return ""
    out = ""
    pad = " " * indent_spaces
    for k, v in d.items():
        out += f"{pad}{k}: {_q(v)}\n"
    return out

def _to_yaml_list(name: str, items: List[Dict[str, Any]], indent_spaces: int) -> str:
    if not items:
        return ""
    pad = " " * indent_spaces
    out = f"{pad}{name}:\n"
    for it in items:
        out += f"{pad}- "
        first = True
        for k, v in it.items():
            if first:
                out += f"{k}: {_q(v)}\n"
                first = False
            else:
                out += f"{pad}  {k}: {_q(v)}\n"
    return out

# ============================================================
# Config extractors (safe defaults)
# ============================================================

def _svc_type(spec) -> str:
    return getattr(getattr(spec, "network", object()), "service_type", None) or "LoadBalancer"

def _priority_class(spec) -> Optional[str]:
    return getattr(getattr(spec, "network", object()), "priority_class", None)

def _host_network(spec) -> bool:
    return bool(getattr(getattr(spec, "network", object()), "host_network", False))

def _iface_name(spec) -> Optional[str]:
    # Prefer transfer.ucx.net_devices; else network.iface.name; else None
    nd = getattr(getattr(spec, "transfer", object()), "ucx", {}).get("net_devices", None)
    if nd:
        return nd
    iface = getattr(getattr(spec, "network", object()), "iface", {})
    return iface.get("name") if isinstance(iface, dict) else None

def _rdma_cfg(spec) -> Dict[str, Any]:
    rdma = getattr(getattr(spec, "network", object()), "rdma", None) or {}
    return {
        "enabled": bool(rdma.get("enabled", False)),
        "resource": rdma.get("resource_name", "rdma/hca"),
        "count": int(rdma.get("count", 1)),
    }

def _topology_spread_cfg(spec) -> Dict[str, Any]:
    ts = getattr(getattr(spec, "network", object()), "topology_spread", None) or {}
    return {
        "enabled": bool(ts.get("enabled", False)),
        "max_skew": int(ts.get("max_skew", 1)),
        "topology_key": ts.get("topology_key", "kubernetes.io/hostname"),
        "when_unsatisfiable": ts.get("when_unsatisfiable", "ScheduleAnyway"),
    }

def _selectors_for_role(spec, role: str) -> Dict[str, str]:
    ns = getattr(getattr(spec, "network", object()), "node_selector", None) or {}
    sel = ns.get(role, {}) if isinstance(ns, dict) else {}
    return sel

def _tolerations(spec) -> List[Dict[str, Any]]:
    tol = getattr(getattr(spec, "network", object()), "tolerations", None)
    if isinstance(tol, list) and tol:
        return tol
    # default GPU toleration
    return [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]

def _extra_annotations(spec, role: str) -> Dict[str, str]:
    anns = getattr(getattr(spec, "network", object()), "annotations", None) or {}
    if isinstance(anns, dict):
        return anns.get(role, {}) if isinstance(anns.get(role, {}), dict) else {}
    return {}

def _multus_ann(spec, namespace: str) -> Optional[str]:
    multus = getattr(getattr(spec, "network", object()), "multus", None)
    if not isinstance(multus, dict):
        return None
    if not multus.get("enabled", False):
        return None
    # Allow either raw annotation string or list of network dicts
    if "annotation" in multus and multus["annotation"]:
        return str(multus["annotation"])
    nets = multus.get("networks", [])
    if not isinstance(nets, list) or not nets:
        return None
    entries = []
    for n in nets:
        if not isinstance(n, dict) or "name" not in n:
            continue
        ns = n.get("namespace", namespace)
        iface = n.get("interface")
        e = f"{ns}/{n['name']}"
        if iface:
            e += f"@{iface}"
        entries.append(e)
    return ", ".join(entries) if entries else None

def _env_defaults(spec, role: str) -> List[Dict[str, str]]:
    iface = _iface_name(spec) or "eth0"
    ucx_tls = ",".join(getattr(getattr(spec, "transfer", object()), "ucx", {}).get(
        "tls", ["rc_x","ud_x","cuda_copy","cuda_ipc","mm","self"]
    ))
    envs = [
        {"name": "NCCL_DEBUG", "value": "WARN"},
        {"name": "NCCL_P2P_LEVEL", "value": "SYS"},
        {"name": "NCCL_SOCKET_IFNAME", "value": iface if ":" not in iface else "eth0"},
        {"name": "UCX_TLS", "value": ucx_tls},
        {"name": "UCX_NET_DEVICES", "value": iface},
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
    ]
    # Optional UCX tuning
    ucx = getattr(getattr(spec, "transfer", object()), "ucx", {}) or {}
    if ucx.get("rndv_scheme"):
        envs.append({"name": "UCX_RNDV_SCHEME", "value": str(ucx["rndv_scheme"])})
    if ucx.get("rndv_thresh_bytes") is not None:
        envs.append({"name": "UCX_RNDV_THRESH", "value": str(int(ucx["rndv_thresh_bytes"]))})
    if ucx.get("max_rndv_rails") is not None:
        envs.append({"name": "UCX_MAX_RNDV_RAILS", "value": str(int(ucx["max_rndv_rails"]))})
    # Extra env from spec.network.extra_env[role] if provided
    extra = getattr(getattr(spec, "network", object()), "extra_env", None) or {}
    role_env = extra.get(role, [])
    for kv in role_env if isinstance(role_env, list) else []:
        if isinstance(kv, dict) and "name" in kv and "value" in kv:
            envs.append({"name": str(kv["name"]), "value": str(kv["value"])})
    return envs

def _env_block(role_env: List[Dict[str, str]]) -> str:
    if not role_env:
        return ""
    lines = "    env:\n"
    for kv in role_env:
        lines += f"      - name: {kv['name']}\n"
        lines += f"        value: {_q(kv['value'])}\n"
    return lines

def _resources_block(gpus: int, rdma: Dict[str, Any]) -> str:
    req = {"cpu": "2000m", "memory": "12Gi", "nvidia.com/gpu": str(gpus)}
    lim = {"cpu": "4000m", "memory": "24Gi", "nvidia.com/gpu": str(gpus)}
    if rdma.get("enabled"):
        res = rdma.get("resource", "rdma/hca")
        cnt = str(int(rdma.get("count", 1)))
        req[res] = cnt
        lim[res] = cnt
    return _yaml(f"""
              resources:
                requests:
                  cpu: {_q(req['cpu'])}
                  memory: {_q(req['memory'])}
                  "nvidia.com/gpu": "{req['nvidia.com/gpu']}"
{("                  " + res + ': "' + cnt + '"\n') if rdma.get("enabled") else ""}\
                limits:
                  cpu: {_q(lim['cpu'])}
                  memory: {_q(lim['memory'])}
                  "nvidia.com/gpu": "{lim['nvidia.com/gpu']}"
{("                  " + res + ': "' + cnt + '"\n') if rdma.get("enabled") else ""}\
    """)

def _tolerations_yaml(spec) -> str:
    return _to_yaml_list("tolerations", _tolerations(spec), 10)

def _node_selector_yaml(sel: Dict[str, str]) -> str:
    if not sel:
        return ""
    return "          nodeSelector:\n" + _to_yaml_kv(sel, 12)

def _topology_spread_yaml(spec, role: str) -> str:
    cfg = _topology_spread_cfg(spec)
    if not cfg["enabled"]:
        return ""
    # Spread across hostname by matching app/component labels
    return _yaml(f"""
          topologySpreadConstraints:
            - maxSkew: {cfg['max_skew']}
              topologyKey: {cfg['topology_key']}
              whenUnsatisfiable: {cfg['when_unsatisfiable']}
              labelSelector:
                matchLabels:
                  app.kubernetes.io/name: llmserve
                  app.kubernetes.io/component: {role}
    """)

def _priority_class_yaml(spec) -> str:
    pc = _priority_class(spec)
    if not pc:
        return ""
    return f"          priorityClassName: {pc}\n"

def _host_network_yaml(spec) -> str:
    if not _host_network(spec):
        return ""
    return _yaml("""
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
    """)

def _annotations_yaml(spec, role: str, metrics_port: int, namespace: str) -> str:
    base = {
        "prometheus.io/scrape": "true",
        "prometheus.io/port": str(metrics_port),
        "prometheus.io/path": "/metrics",
    }
    multus = _multus_ann(spec, namespace)
    if multus:
        base["k8s.v1.cni.cncf.io/networks"] = multus
    extra = _extra_annotations(spec, role)
    base.update(extra)
    return "          annotations:\n" + _to_yaml_kv(base, 12)

def _add_cfg_volume_mounts(with_kv: bool) -> str:
    mounts = _yaml("""
              volumeMounts:
                - name: cfg
                  mountPath: /app/llmserve.yaml
                  subPath: llmserve.yaml
    """)
    if with_kv:
        mounts += _yaml("""
                - name: kvpages
                  mountPath: /var/lib/kvpages
        """)
    return mounts

def _add_pod_volumes(with_kv: bool) -> str:
    vols = _yaml("""
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
    """)
    if with_kv:
        vols += _yaml("""
            - name: kvpages
              persistentVolumeClaim:
                claimName: kvpages-pvc
        """)
    return vols

def _ns(namespace: str) -> str:
    return _yaml(f"""
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/part-of: llmserve
    """)

def _cm_header(namespace: str) -> str:
    return _yaml(f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: llmserve-manifest
      namespace: {namespace}
    data:
      llmserve.yaml: |
    """)

def _render_router_only(spec, namespace: str, image: str, svc_type: str, router_port: int, metrics_port: int) -> tuple[str, str]:
    dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmserve-router
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: router
    spec:
      replicas: {max(1, int(spec.deployment.replicas.get("router", 1)))}
      revisionHistoryLimit: 2
      selector:
        matchLabels:
          app.kubernetes.io/name: llmserve
          app.kubernetes.io/component: router
      template:
        metadata:
          labels:
            app.kubernetes.io/name: llmserve
            app.kubernetes.io/component: router
{_annotations_yaml(spec, "router", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "router")).rstrip()}
{_topology_spread_yaml(spec, "router").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: router
              image: {image}
              imagePullPolicy: IfNotPresent
              env:
                - name: ROLE
                  value: "router"
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: http
                  containerPort: {router_port}
                - name: metrics
                  containerPort: {metrics_port}
{_add_cfg_volume_mounts(with_kv=False).rstrip()}
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
    """)
    svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: llmserve-router
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: router
    spec:
      type: {_svc_type(spec)}
      selector:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: router
      ports:
        - name: http
          port: {router_port}
          targetPort: http
        - name: metrics
          port: {metrics_port}
          targetPort: metrics
    """)
    return dep, svc

def _render_nads(spec, namespace: str) -> Dict[str, str]:
    """Render NetworkAttachmentDefinitions if provided in spec.network.multus.definitions[]."""
    docs: Dict[str, str] = {}
    multus = getattr(getattr(spec, "network", object()), "multus", None)
    if not isinstance(multus, dict):
        return docs
    defs = multus.get("definitions", [])
    if not isinstance(defs, list):
        return docs
    idx = 0
    for nad in defs:
        if not isinstance(nad, dict) or "name" not in nad or "config" not in nad:
            continue
        name = str(nad["name"])
        ns = str(nad.get("namespace", namespace))
        cfg = str(nad["config"]).rstrip()
        idx += 1
        docs[f"05-nad-{idx:02d}-{name}.yaml"] = _yaml(f"""
        apiVersion: k8s.cni.cncf.io/v1
        kind: NetworkAttachmentDefinition
        metadata:
          name: {name}
          namespace: {ns}
        spec:
          config: |
{_indent(cfg + ("\n" if not cfg.endswith("\n") else ""), 12)}
        """)
    return docs