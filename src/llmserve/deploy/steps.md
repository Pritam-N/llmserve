```
# Build & push your image (example)
docker build -t ghcr.io/yourorg/llmserve:0.1.0 .
docker push ghcr.io/yourorg/llmserve:0.1.0

# Render manifests
llmserve up -f llmserve.yaml --mode k8s --namespace llmserve --image ghcr.io/yourorg/llmserve:0.1.0

# Apply (if you want the CLI to do it)
llmserve up -f llmserve.yaml --mode k8s --namespace llmserve --image ghcr.io/yourorg/llmserve:0.1.0 --apply

# Or via kubectl manually
kubectl apply -f deploy_out/k8s/
```

Keep the generator, add these files:
	•	router-deployment.yaml (no GPU), router-service.yaml (LB)
	•	prefill-deployment.yaml (GPU, NVMe optional), decode-deployment.yaml (GPU, NVMe strongly recommended)
	•	Headless Services for worker pools (prefill and decode) + ServiceEndpoints (or gRPC + client-side discovery) so Router can pick workers using KV locality.
	•	A kv-manager StatefulSet (GPU optional) with a Stateful PVC for tiered cache and a StorageClass that supports fast local NVMe; plus S3 creds Secret for the cold tier.
	•	Optional Node labels:
	•	role=prefill (compute-heavy GPUs, fewer NVMe)
	•	role=decode (bandwidth-heavy GPUs, local NVMe)
	•	Add nodeSelector/affinity in templates accordingly.

We’ll extend k8sgen.py to render these once the worker RPC is in place.

⸻

6) Cluster prerequisites (quick checklist)
	•	NVIDIA Device Plugin + CUDA drivers on GPU nodes.
	•	(Optional but recommended) GPU Feature Discovery labels for scheduling.
	•	(Optional) NVIDIA GPU Operator if you want managed drivers/runtime.
	•	StorageClass that maps to local NVMe or high-IOPS SSD for the KV page dir.
	•	(If using GPUDirect Storage / NIXL) ensure kernel modules + libraries on nodes; set envs accordingly (we already template a few).

⸻

7) Tuning knobs you’ll likely touch

In llmserve.yaml:
	•	deployment.replicas.decode → controls number of pods rendered.
	•	budgets.max_tokens_in_flight, scheduling.policies.prefill_chunk_tokens, spec_decode.* → latency/throughput tradeoffs.
	•	kv_cache.page_size_kb, kv_cache.eviction.* → HBM pressure vs IO.

In k8sgen.py (or via flags):
	•	--service-type → ClusterIP for internal ingress, LoadBalancer for public.
	•	Resource requests/limits (CPU/mem/GPU).


