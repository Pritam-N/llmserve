# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from pathlib import Path
import textwrap
import os


def _yaml(doc: str) -> str:
    return textwrap.dedent(doc).lstrip()


def render_manifests(
    spec,
    namespace: str = "llmserve",
    image: str | None = None,
    replicas: int | None = None,
    svc_type: str = "LoadBalancer",
    router_port: int = 8000,
    metrics_port: int = 9400,
):
    """
    Render production-lean manifests for the current monolith process.
    Later we can split into router/prefill/decode using the same patterns.
    """
    image = image or os.environ.get("LLMSERVE_IMAGE", "ghcr.io/yourorg/llmserve:0.1.0")
    replicas = replicas or spec.deployment.replicas.get(
        "decode", 1
    )  # use decode count as overall pool size

    # Namespace
    ns = _yaml(
        f"""
    apiVersion: v1
    kind: Namespace
    metadata:
      name: {namespace}
      labels:
        app.kubernetes.io/name: llmserve
        app.kubernetes.io/part-of: llmserve
    """
    )

    # ConfigMap with the entire user manifest (so pods mount it read-only)
    # NOTE: ensure your CLI path passes the validated yaml content down here.
    # The CLI will read the file and insert as literal block.
    cm_header = f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: llmserve-manifest
      namespace: {namespace}
    data:
      llmserve.yaml: |
    """
    # We fill llmserve.yaml content at write-time (CLI) to preserve formatting.
    # For now keep just the header; CLI will append the block.

    # Deployment (monolith)
    # - Requests 1 GPU
    # - Readiness/Liveness on /healthz
    # - Mounts the manifest ConfigMap at /app/llmserve.yaml
    # - Prometheus annotations for metrics on :9400/metrics
    dep = _yaml(
        f"""
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
          # Uncomment on clusters with GPU runtime class
          # runtimeClassName: nvidia
          # Tolerate GPU nodes if tainted
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
              env:
                # Optional: pin NCCL/UCX for multi-node performance (adjust to your fabric)
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
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "4000m"
                  memory: "24Gi"
                  nvidia.com/gpu: "1"
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
    """
    )

    # Service
    svc = _yaml(
        f"""
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
    """
    )

    # PDB
    pdb = _yaml(
        f"""
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: llmserve-pdb
      namespace: {namespace}
    spec:
      minAvailable: 80%
      selector:
        matchLabels:
          app.kubernetes.io/name: llmserve
          app.kubernetes.io/component: api
    """
    )

    # NetworkPolicy (allow ingress from cluster + Prometheus; restrict egress to DNS + S3 if used)
    netpol = _yaml(
        f"""
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: llmserve-default
      namespace: {namespace}
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/name: llmserve
      policyTypes: ["Ingress","Egress"]
      ingress:
        - {{}}
      egress:
        - to:
            - namespaceSelector: {{}}
          ports:
            - protocol: UDP
              port: 53
            - protocol: TCP
              port: 53
    """
    )

    # Fast NVMe PVC + StorageClass example (adjust to your CSI)
    pvc = _yaml(
        f"""
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
    """
    )

    sc = _yaml(
        f"""
    # Example: local NVMe storage class (replace with your cloud CSI)
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: fast-nvme
    provisioner: kubernetes.io/no-provisioner
    volumeBindingMode: WaitForFirstConsumer
    reclaimPolicy: Delete
    """
    )

    return {
        "00-namespace.yaml": ns,
        "01-configmap-manifest.yaml": cm_header,  # content appended by CLI with actual YAML
        "10-deployment.yaml": dep,
        "20-service.yaml": svc,
        "30-pdb.yaml": pdb,
        "40-networkpolicy.yaml": netpol,
        "50-kvpages-pvc.yaml": pvc,
        "50-storageclass-fast-nvme.yaml": sc,
    }


def write_out(dirpath: str, docs: dict[str, str], manifest_text: str):
    outdir = Path(dirpath)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        p = outdir / name
        if name == "01-configmap-manifest.yaml":
            # append literal manifest payload indented by 6 spaces under data.llmserve.yaml
            indented = "".join(
                f"        {line}" for line in manifest_text.splitlines(True)
            )
            text = text + indented
        p.write_text(text, encoding="utf-8")
    return outdir
