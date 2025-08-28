# ============================================================
# Monolith renderer
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
    _render_nads
)
import os

def render_manifests_monolith(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    svc_type: str = "LoadBalancer",
) -> dict[str, str]:
    image = image or os.environ.get("LLMSERVE_IMAGE", "ghcr.io/yourorg/llmserve:0.1.0")
    router_port = int(getattr(spec.deployment, "router_port", 8000))
    metrics_port = 9400
    replicas = int(spec.deployment.replicas.get("decode", 1))

    ns = _ns(namespace)
    cm_header = _cm_header(namespace)
    rdma = _rdma_cfg(spec)
    envs = _env_defaults(spec, role="api")

    dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmserve
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: api
    spec:
      replicas: {replicas}
      revisionHistoryLimit: 2
      strategy:
        type: RollingUpdate
        rollingUpdate:
          maxUnavailable: 1
          maxSurge: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: llmserve
          app.kubernetes.io/component: api
      template:
        metadata:
          labels:
            app.kubernetes.io/name: llmserve
            app.kubernetes.io/component: api
{_annotations_yaml(spec, "api", metrics_port, namespace).rstrip()}
        spec:
{_host_network_yaml(spec).rstrip()}
{_tolerations_yaml(spec).rstrip()}
{_node_selector_yaml(_selectors_for_role(spec, "api")).rstrip()}
{_topology_spread_yaml(spec, "api").rstrip()}
{_priority_class_yaml(spec).rstrip()}
          containers:
            - name: llmserve
              image: {image}
              imagePullPolicy: IfNotPresent
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: http
                  containerPort: {router_port}
                - name: metrics
                  containerPort: {metrics_port}
{_env_block(envs).rstrip()}
{_add_cfg_volume_mounts(with_kv=True).rstrip()}
{_resources_block(gpus=1, rdma=rdma).rstrip()}
              readinessProbe:
                httpGet:
                  path: /healthz
                  port: http
                initialDelaySeconds: 10
                periodSeconds: 5
              livenessProbe:
                httpGet:
                  path: /healthz
                  port: http
                initialDelaySeconds: 30
                periodSeconds: 10
{_add_pod_volumes(with_kv=True).rstrip()}
    """)

    svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: llmserve
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
    spec:
      type: {_svc_type(spec)}
      selector:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: api
      ports:
        - name: http
          port: {router_port}
          targetPort: http
        - name: metrics
          port: {metrics_port}
          targetPort: metrics
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
        "10-deployment.yaml": dep,
        "20-service.yaml": svc,
        "50-kvpages-pvc.yaml": pvc,
        "51-storageclass-fast-nvme.yaml": sc,
    }
    # Optional NADs
    docs.update(_render_nads(spec, namespace))
    return docs