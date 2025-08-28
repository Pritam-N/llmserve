# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from pathlib import Path
import textwrap
import os

def _yaml(doc: str) -> str:
    return textwrap.dedent(doc).lstrip()

def _env_block() -> str:
    # Common performance/env defaults; adjust per-fabric if needed
    return _yaml(f"""
    env:
      - name: NCCL_DEBUG
        value: "WARN"
      - name: NCCL_P2P_LEVEL
        value: "SYS"
      - name: NCCL_SOCKET_IFNAME
        value: "eth0"
      - name: UCX_TLS
        value: "rc,ud,mm,self"
      - name: UCX_NET_DEVICES
        value: "eth0"
      - name: NVIDIA_VISIBLE_DEVICES
        value: "all"
      - name: NVIDIA_DRIVER_CAPABILITIES
        value: "compute,utility"
    """)

# ------------------------------
# Monolith renderer (legacy path)
# ------------------------------
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

    ns = _yaml(f"""
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/part-of: llmserve
    """)

    cm_header = f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: llmserve-manifest
      namespace: {namespace}
    data:
      llmserve.yaml: |
    """

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
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "{metrics_port}"
            prometheus.io/path: "/metrics"
        spec:
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
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
{_env_block().rstrip()}
              volumeMounts:
                - name: cfg
                  mountPath: /app/llmserve.yaml
                  subPath: llmserve.yaml
                - name: kvpages
                  mountPath: /var/lib/kvpages
              resources:
                requests:
                  cpu: "2000m"
                  memory: "12Gi"
                  "nvidia.com/gpu": "1"
                limits:
                  cpu: "4000m"
                  memory: "24Gi"
                  "nvidia.com/gpu": "1"
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
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
            - name: kvpages
              persistentVolumeClaim:
                claimName: kvpages-pvc
          nodeSelector:
            kubernetes.io/arch: amd64
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
      type: {svc_type}
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

    return {
        "00-namespace.yaml": ns,
        "01-configmap-manifest.yaml": cm_header,
        "10-deployment.yaml": dep,
        "20-service.yaml": svc,
        "50-kvpages-pvc.yaml": pvc,
        "51-storageclass-fast-nvme.yaml": sc,
    }

# ------------------------------------------
# Disaggregated renderer (router/prefill/decode)
# GPU sizing: per-pod GPUs = TP × PP ; replicas = DP
# ------------------------------------------
def render_manifests_disagg(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    svc_type: str = "LoadBalancer",
) -> dict[str, str]:
    image = image or os.environ.get("LLMSERVE_IMAGE", "ghcr.io/yourorg/llmserve:0.1.0")

    # Ports
    router_port = int(getattr(spec.deployment, "router_port", 8000))
    prefill_port = int(getattr(spec.deployment, "prefill_port", 9001))
    decode_port  = int(getattr(spec.deployment, "decode_port", 9002))
    metrics_port = 9400

    # GPU math from roles: GPUs per pod = TP × PP
    prefill_gpus = max(1, int(spec.roles.prefill.tp) * int(spec.roles.prefill.pp))
    decode_gpus  = max(1, int(spec.roles.decode.tp)  * int(spec.roles.decode.pp))

    # DP via replicas (pods)
    prefill_repl = max(1, int(spec.roles.prefill.dp_replicas))
    decode_repl  = max(1, int(spec.roles.decode.dp_replicas))
    router_repl  = max(1, int(spec.deployment.replicas.get("router", 1)))

    ns = _yaml(f"""
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/part-of: llmserve
    """)

    cm_header = f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: llmserve-manifest
      namespace: {namespace}
    data:
      llmserve.yaml: |
    """

    # Router (HTTP API)
    router_dep = _yaml(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmserve-router
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: router
    spec:
      replicas: {router_repl}
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
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "{metrics_port}"
            prometheus.io/path: "/metrics"
        spec:
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
              volumeMounts:
                - name: cfg
                  mountPath: /app/llmserve.yaml
                  subPath: llmserve.yaml
              resources:
                requests:
                  cpu: "1000m"
                  memory: "2Gi"
                limits:
                  cpu: "2000m"
                  memory: "4Gi"
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
    """)

    router_svc = _yaml(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: llmserve-router
      namespace: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/component: router
    spec:
      type: {svc_type}
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

    # Prefill worker (gRPC)
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
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "{metrics_port}"
            prometheus.io/path: "/metrics"
        spec:
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          # Optional: steer prefill to specific nodes
          # nodeSelector:
          #   role: prefill
          containers:
            - name: prefill
              image: {image}
              imagePullPolicy: IfNotPresent
              env:
                - name: ROLE
                  value: "prefill"
{_env_block().rstrip()}
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: grpc
                  containerPort: {prefill_port}
                - name: metrics
                  containerPort: {metrics_port}
              volumeMounts:
                - name: cfg
                  mountPath: /app/llmserve.yaml
                  subPath: llmserve.yaml
                - name: kvpages
                  mountPath: /var/lib/kvpages
              resources:
                requests:
                  cpu: "2000m"
                  memory: "12Gi"
                  "nvidia.com/gpu": "{prefill_gpus}"
                limits:
                  cpu: "4000m"
                  memory: "24Gi"
                  "nvidia.com/gpu": "{prefill_gpus}"
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
            - name: kvpages
              persistentVolumeClaim:
                claimName: kvpages-pvc
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

    # Decode worker (gRPC)
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
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "{metrics_port}"
            prometheus.io/path: "/metrics"
        spec:
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          # Optional: steer decode to nodes with fast NVMe
          # nodeSelector:
          #   role: decode
          containers:
            - name: decode
              image: {image}
              imagePullPolicy: IfNotPresent
              env:
                - name: ROLE
                  value: "decode"
{_env_block().rstrip()}
              args: ["llmserve","up","-f","/app/llmserve.yaml"]
              ports:
                - name: grpc
                  containerPort: {decode_port}
                - name: metrics
                  containerPort: {metrics_port}
              volumeMounts:
                - name: cfg
                  mountPath: /app/llmserve.yaml
                  subPath: llmserve.yaml
                - name: kvpages
                  mountPath: /var/lib/kvpages
              resources:
                requests:
                  cpu: "2000m"
                  memory: "12Gi"
                  "nvidia.com/gpu": "{decode_gpus}"
                limits:
                  cpu: "4000m"
                  memory: "24Gi"
                  "nvidia.com/gpu": "{decode_gpus}"
          volumes:
            - name: cfg
              configMap:
                name: llmserve-manifest
            - name: kvpages
              persistentVolumeClaim:
                claimName: kvpages-pvc
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

    return {
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

# ------------------------------
# Switch: pick renderer by spec
# ------------------------------
def render_all(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    svc_type: str = "LoadBalancer",
) -> dict[str, str]:
    if getattr(spec.deployment, "disaggregated", False):
        return render_manifests_disagg(spec, namespace=namespace, image=image, svc_type=svc_type)
    return render_manifests_monolith(spec, namespace=namespace, image=image, svc_type=svc_type)

# ------------------------------
# Writer: dump YAMLs to dir and inject manifest content
# ------------------------------
def write_out(dirpath: str, docs: dict[str, str], manifest_text: str) -> Path:
    outdir = Path(dirpath)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        p = outdir / name
        if name.endswith("configmap-manifest.yaml"):
            indented = "".join(f"        {line}" for line in manifest_text.splitlines(True))
            text = text + indented
        p.write_text(text, encoding="utf-8")
    return outdir