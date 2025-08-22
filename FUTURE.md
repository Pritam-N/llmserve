Great question. Here’s a crisp map of what you can optimize in the prefill vs. decode split, what’s already practical to ship now, and what’s promising but not wired yet.

⸻

What you can do today (ship-ready)

A. Disaggregate prefill & decode
	•	Run prefill and decode in separate engine pools so you can size GPUs differently (compute-heavy vs. bandwidth-heavy) and route by KV-locality. vLLM has an experimental disaggregated prefill mode you can take cues from.  ￼

B. Prefill-phase optimizations
	•	Chunked prefill: split long prompts into chunks and interleave them with ongoing decodes to avoid head-of-line blocking; supported in vLLM and documented as a scheduler policy. (Note: CUDA Graphs benefits are limited when chunked prefill is on.)  ￼ ￼
	•	Continuous batching: always keep the SMs busy by coalescing arriving work; vLLM’s design + Anyscale’s write-ups show large throughput gains.  ￼ ￼
	•	Faster prefill kernels: use FlashAttention/-3 kernels (Hopper: warp specialization, TMA, FP8) to accelerate the heavy context pass.  ￼
	•	Prompt compression (pre-RAG or pre-prefill): LLMLingua-2, CPC/EHPC, etc., shrink the token count (often 2–5×) before prefill, cutting cost and TTFT.  ￼ ￼

C. Decode-phase optimizations
	•	Flash-Decoding for long contexts: specialized kernels can deliver big gains (reports up to multi-× on very long sequence decode).  ￼
	•	Speculative decoding (single- or two-model): Medusa-style, EAGLE, ReDrafter, and production guides in TensorRT-LLM. Use adaptive speculation length to squeeze extra ~10%+ speedups.  ￼ ￼ ￼ ￼
	•	GQA/MQA models: if your model supports Grouped-Query or Multi-Query attention, decode gets much cheaper (far fewer KV heads to fetch) with minor quality loss; you can also uptrain MHA → GQA.  ￼ ￼
	•	Prefix caching: automatically reuse shared prompt prefixes so repeated system/user headers don’t re-compute in prefill. vLLM has Automatic Prefix Caching.  ￼

D. KV-cache, memory, and multi-node
	•	PagedAttention: baseline for high-throughput KV management (non-contiguous paging) used by vLLM.  ￼
	•	Tiered KV offload across HBM→DRAM→NVMe→Object, with KV-aware routing and accelerated transfer via NIXL (RDMA / GPUDirect / storage). See NVIDIA Dynamo architecture & blog.  ￼ ￼
	•	Quantized KV for decode: quantize KV pages (e.g., INT8) for the decode phase—note most studies keep prefill unquantized and apply quantization on decode use to avoid quality loss.  ￼

⸻

What’s possible but not (fully) wired yet (your “next up” backlog)

1) True cross-process KV handoff between prefill and decode
	•	Export/import KV pages (with page tables + RoPE offsets) so prefill pool and decode pool can be separate pods/nodes without re-computing context. vLLM’s disaggregated prefill doc sketches the shape, but the stable KV handoff API is still evolving. Pair with KV-locality routing to pick the best decode worker.  ￼

2) Lookahead decoding (exact, parallel)
	•	Break part of the sequential dependency by verifying n-grams in parallel—no draft model required; results show up to ~1.8× speedups in some tasks and better scalability across GPUs. Integrates well with FlashAttention.  ￼ ￼

3) Smarter speculation
	•	DISCO picks the speculation length dynamically per step instead of a static setting; pairs nicely with EAGLE/ReDrafter.  ￼

4) New decode-efficient attentions
	•	GLA/GTA (2025) aim to increase compute per byte of KV fetched (hardware-efficient), showing up to 2× gains in online serving benchmarks and smaller KV footprints compared to GQA. These need model support or uptraining.  ￼

5) Prefill-only engines for “one-token” tasks
	•	If your workload is “retrieve one token then stop” (classification, routing, tool selection), PrefillOnly shows how to keep only the last-layer KV, cutting memory and latency. Useful for hybrid pipelines.  ￼

6) Scheduler upgrades around chunked prefill
	•	Systems work (OSDI’24 Sarathi-Serve) shows that Decode + Chunked-Prefill dominates “Decode + Full Prefill”, improving tail latency; there’s more room to bake this into production schedulers (SJF/SRPT variants, prefix-aware admission). vLLM discussions also explore prefix-caching-aware scheduling.  ￼ ￼

7) Kernel & precision frontier
	•	FlashAttention-3 (Hopper FP8/TMA) and next-gen fused kernels can further reduce both prefill and decode time; tighten integration paths in vLLM/TensorRT-LLM.  ￼

8) Production-grade tiered KV with NIXL
	•	End-to-end KV offload/orchestration (Dynamo) with NIXL data paths and KV-aware routing across nodes/storage. This is ready conceptually but needs wiring in open stacks (policy, failure handling, encryption, prefetch).  ￼ ￼

9) Prompt-side acceleration as a first-class knob
	•	Make prompt compression a policy in the router (budget-aware): compress when the prefill queue or TTFT SLO is at risk. Track compression ratio vs. answer quality via eval gates.  ￼

⸻

Quick prioritization (what I’d implement next)
	1.	KV handoff & locality routing: finish the prefill→decode KV export/import API; prefer same-node decoders; fall back with NIXL transfers. (Biggest end-to-end win for disaggregation.)  ￼ ￼
	2.	Speculation w/ adaptive lookahead: wire a standard two-model path first, then add DISCO tuning.  ￼ ￼
	3.	Chunked prefill scheduler (Decode-first + small prefill chunks), plus prefix-aware admission; keep CUDA Graphs off when chunking.  ￼ ￼
	4.	Decode kernels: enable Flash-Decoding / FA-3 where available (Hopper).  ￼ ￼
	5.	Tiered KV: start with NVMe (local) → DRAM → HBM; add INT8 KV for decode pages; later wire object store + prefetch.  ￼

⸻

What we’ve not done in our code yet (explicit)
	•	Cross-process KV export/import & KV-locality router (currently stubbed).
	•	Speculative decoding in the decode engine (we created the hooks, not the full path).
	•	Lookahead decoding integration.
	•	NIXL/UCX/NCCL transfer engines (interfaces exist, full data-path TODO).
	•	Tiered KV manager with quantized on-disk pages and hotset policy.
	•	Scheduler with chunked-prefill + prefix-aware/aging fairness.

If you want, I can start by wiring speculative decoding in the decode engine (draft model + adaptive token budget) and sketch the KV handoff proto so we can begin testing prefill→decode separation across processes.