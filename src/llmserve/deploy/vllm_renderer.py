# ============================================================
# vLLM disaggregated renderer (router + vLLM prefill/decode/proxy)
# ============================================================

from .helpers import (
    _yaml, 
    _ns, 
    _cm_header, 
    _rdma_cfg, 
    _env_defaults, 
    _env_block,
    _resources_block,
    _host_network_yaml,
    _tolerations_yaml,
    _node_selector_yaml,
    _topology_spread_yaml,
    _priority_class_yaml,
    _svc_type,
    _selectors_for_role,
    _annotations_yaml,
    _render_nads,
    _render_router_only
)
import os

def render_manifests_vllm_disagg(
    spec,
    namespace: str,
    image: str | None,
    svc_type: str,
) -> dict[str, str]:
    image = image or os.environ.get("LLMSERVE_IMAGE", "ghcr.io/yourorg/llmserve:0.2.0")
    vllm_img = os.environ.get("VLLM_IMAGE", "vllm/vllm:latest")

    router_port = int(getattr(spec.deployment, "router_port", 8000))
    metrics_port = 9400

    prefill_gpus = max(1, int(spec.roles.prefill.tp) * int(spec.roles.prefill.pp))
    decode_gpus  = max(1, int(spec.roles.decode.tp)  * int(spec.roles.decode.pp))
    prefill_repl = max(1, int(spec.roles.prefill.dp_replicas))
    decode_repl  = max(1, int(spec.roles.decode.dp_replicas))

    ns = _ns(namespace)
    cm_header = _cm_header(namespace)
    rdma = _rdma_cfg(spec)

    # Router (ours)
    router_dep, router_svc = _render_router_only(spec, namespace, image, _svc_type(spec), router_port, metrics_port)

    # vLLM Prefill
    prefill_env = _env_defaults(spec, role="prefill")
    vllm_prefill_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-prefill
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: prefill
    spec:
      replicas: {prefill_repl}
      selector: {{ matchLabels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: prefill }} }}
      template:
        metadata:
          labels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: prefill }}
{_annotations_yaml(spec, "prefill", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "prefill")).rstrip()}
{_topology_spread_yaml(spec, "prefill").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: prefill
              image: {vllm_img}
              imagePullPolicy: IfNotPresent
{_env_block(prefill_env).rstrip()}
              command: ["bash","-lc"]
              args:
                - |
                  echo "[vLLM] starting prefill..."
                  # Replace with exact flags for your vLLM version:
                  vllm serve "$MODEL_ID" \\
                    --tensor-parallel-size={spec.roles.prefill.tp} \\
                    --pipeline-parallel-size={spec.roles.prefill.pp} \\
                    --port 8001
              env:
                - name: MODEL_ID
                  value: "{spec.models['primary'].id}"
              ports:
                - name: http
                  containerPort: 8001
{_resources_block(gpus=prefill_gpus, rdma=rdma).rstrip()}
    """)
    vllm_prefill_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-prefill
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: prefill
    spec:
      selector:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: prefill
      ports:
        - name: http
          port: 8001
          targetPort: http
    """)

    # vLLM Decode
    decode_env = _env_defaults(spec, role="decode")
    vllm_decode_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-decode
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: decode
    spec:
      replicas: {decode_repl}
      selector: {{ matchLabels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: decode }} }}
      template:
        metadata:
          labels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: decode }}
{_annotations_yaml(spec, "decode", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "decode")).rstrip()}
{_topology_spread_yaml(spec, "decode").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: decode
              image: {vllm_img}
              imagePullPolicy: IfNotPresent
{_env_block(decode_env).rstrip()}
              command: ["bash","-lc"]
              args:
                - |
                  echo "[vLLM] starting decode..."
                  vllm serve "$MODEL_ID" \\
                    --tensor-parallel-size={spec.roles.decode.tp} \\
                    --pipeline-parallel-size={spec.roles.decode.pp} \\
                    --port 8000
              env:
                - name: MODEL_ID
                  value: "{spec.models['primary'].id}"
              ports:
                - name: http
                  containerPort: 8000
{_resources_block(gpus=decode_gpus, rdma=rdma).rstrip()}
    """)
    vllm_decode_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-decode
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: decode
    spec:
      selector:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: decode
      ports:
        - name: http
          port: 8000
          targetPort: http
    """)

    # Optional proxy demo (points router to a single endpoint)
    vllm_proxy_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-proxy
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: proxy
    spec:
      replicas: 1
      selector: {{ matchLabels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: proxy }} }}
      template:
        metadata:
          labels: {{ app.kubernetes.io/name: vllm, app.kubernetes.io/component: proxy }}
{_annotations_yaml(spec, "router", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: proxy
              image: {vllm_img}
              imagePullPolicy: IfNotPresent
              command: ["bash","-lc"]
              args:
                - |
                  echo "[vLLM] starting disagg proxy..."
                  python -m vllm.examples.online_serving.disagg_proxy_demo \\
                    --prefill-url http://vllm-prefill:8001 \\
                    --decode-url  http://vllm-decode:8000 \\
                    --port 8008
              ports:
                - name: http
                  containerPort: 8008
    """)
    vllm_proxy_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-proxy
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: proxy
    spec:
      selector:
        app.kubernetes.io/name: vllm
        app.kubernetes.io/component: proxy
      ports:
        - name: http
          port: 8008
          targetPort: http
    """)

    docs = {
        "00-namespace.yaml": ns,
        "01-configmap-manifest.yaml": cm_header,
        "10-router-deploy.yaml": router_dep,
        "11-router-svc.yaml": router_svc,
        "20-vllm-prefill-deploy.yaml": vllm_prefill_dep,
        "21-vllm-prefill-svc.yaml": vllm_prefill_svc,
        "30-vllm-decode-deploy.yaml": vllm_decode_dep,
        "31-vllm-decode-svc.yaml": vllm_decode_svc,
        "40-vllm-proxy-deploy.yaml": vllm_proxy_dep,
        "41-vllm-proxy-svc.yaml": vllm_proxy_svc,
    }
    docs.update(_render_nads(spec, namespace))
    return docs