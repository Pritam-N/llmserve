# ============================================================
# Disaggregated renderer (custom gRPC): router/prefill/decode
# ============================================================


from .helpers import (
    _yaml, 
    _ns, 
    _cm_header, 
    _rdma_cfg, 
    _env_defaults, 
    _env_block,
    _add_cfg_volume_mounts,
    _resources_block,
    _host_network_yaml,
    _tolerations_yaml,
    _node_selector_yaml,
    _topology_spread_yaml,
    _priority_class_yaml,
    _svc_type,
    _selectors_for_role,
    _annotations_yaml,
    _add_pod_volumes,
    _render_nads,
    _render_router_only
)
import os

def render_manifests_disagg(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    svc_type: str = "LoadBalancer",
) -> dict[str, str]:
    image = image or os.environ.get("LLMSERVE_IMAGE", "ghcr.io/yourorg/llmserve:0.1.0")

    router_port = int(getattr(spec.deployment, "router_port", 8000))
    prefill_port = int(getattr(spec.deployment, "prefill_port", 9001))
    decode_port  = int(getattr(spec.deployment, "decode_port", 9002))
    metrics_port = 9400

    prefill_gpus = max(1, int(spec.roles.prefill.tp) * int(spec.roles.prefill.pp))
    decode_gpus  = max(1, int(spec.roles.decode.tp)  * int(spec.roles.decode.pp))
    prefill_repl = max(1, int(spec.roles.prefill.dp_replicas))
    decode_repl  = max(1, int(spec.roles.decode.dp_replicas))
    router_repl  = max(1, int(spec.deployment.replicas.get("router", 1)))

    ns = _ns(namespace)
    cm_header = _cm_header(namespace)
    rdma = _rdma_cfg(spec)

    # Router
    router_dep, router_svc = _render_router_only(spec, namespace, image, _svc_type(spec), router_port, metrics_port)

    # Prefill
    prefill_env = _env_defaults(spec, role="prefill")
    prefill_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmserve-prefill
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: prefill
    spec:
      replicas: {prefill_repl}
      revisionHistoryLimit: 2
      selector:
        matchLabels:
          app.kubernetes.io/name: llmserve
          app.kubernetes.io/component: prefill
      template:
        metadata:
          labels:
            app.kubernetes.io/name: llmserve
            app.kubernetes.io/component: prefill
{_annotations_yaml(spec, "prefill", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "prefill")).rstrip()}
{_topology_spread_yaml(spec, "prefill").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: prefill
              image: {image}
              imagePullPolicy: IfNotPresent
              env:
                - name: ROLE
                  value: "prefill"
{_env_block(prefill_env).rstrip()}
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: grpc
                  containerPort: {prefill_port}
                - name: metrics
                  containerPort: {metrics_port}
{_add_cfg_volume_mounts(with_kv=True).rstrip()}
{_resources_block(gpus=prefill_gpus, rdma=rdma).rstrip()}
{_add_pod_volumes(with_kv=True).rstrip()}
    """)

    prefill_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: llmserve-prefill
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: prefill
    spec:
      type: ClusterIP
      selector:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: prefill
      ports:
        - name: grpc
          port: {prefill_port}
          targetPort: grpc
    """)

    # Decode
    decode_env = _env_defaults(spec, role="decode")
    decode_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmserve-decode
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: decode
    spec:
      replicas: {decode_repl}
      revisionHistoryLimit: 2
      selector:
        matchLabels:
          app.kubernetes.io/name: llmserve
          app.kubernetes.io/component: decode
      template:
        metadata:
          labels:
            app.kubernetes.io/name: llmserve
            app.kubernetes.io/component: decode
{_annotations_yaml(spec, "decode", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "decode")).rstrip()}
{_topology_spread_yaml(spec, "decode").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: decode
              image: {image}
              imagePullPolicy: IfNotPresent
              env:
                - name: ROLE
                  value: "decode"
{_env_block(decode_env).rstrip()}
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: grpc
                  containerPort: {decode_port}
                - name: metrics
                  containerPort: {metrics_port}
{_add_cfg_volume_mounts(with_kv=True).rstrip()}
{_resources_block(gpus=decode_gpus, rdma=rdma).rstrip()}
{_add_pod_volumes(with_kv=True).rstrip()}
    """)

    decode_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: llmserve-decode
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: decode
    spec:
      type: ClusterIP
      selector:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: decode
      ports:
        - name: grpc
          port: {decode_port}
          targetPort: grpc
    """)

    pvc = _yaml(f"""
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: kvpages-pvc
      namespace: {namespace}
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Gi
      storageClassName: fast-nvme
    """)

    sc = _yaml(f"""
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: fast-nvme
    provisioner: kubernetes.io/no-provisioner
    volumeBindingMode: WaitForFirstConsumer
    reclaimPolicy: Delete
    """)

    docs = {
        "00-namespace.yaml": ns,
        "01-configmap-manifest.yaml": cm_header,
        "10-router-deploy.yaml": router_dep,
        "11-router-svc.yaml": router_svc,
        "20-prefill-deploy.yaml": prefill_dep,
        "21-prefill-svc.yaml": prefill_svc,
        "30-decode-deploy.yaml": decode_dep,
        "31-decode-svc.yaml": decode_svc,
        "50-kvpages-pvc.yaml": pvc,
        "51-storageclass-fast-nvme.yaml": sc,
    }
    docs.update(_render_nads(spec, namespace))
    return docs