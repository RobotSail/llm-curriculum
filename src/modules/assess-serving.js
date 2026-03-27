// Assessment module for C.3: Serving Infrastructure
// Split from assess-branch-cd.js — per-section test (10 questions)

export const servingAssessment = {
  id: "C.3-assess",
  sectionId: "C.3",
  title: "Assessment: Serving Infrastructure",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The prefill phase (processing the input prompt) and the decode phase (generating output tokens) have fundamentally different computational profiles:",
      options: ["Both phases are memory-bandwidth-bound, since loading model weights from HBM dominates the latency regardless of the number of input tokens", "Prefill is memory-bandwidth-bound while decode is compute-bound, because generating each new token requires full recomputation of the attention matrix", "Prefill is **compute-bound** (large matrix-matrix multiplications over all prompt tokens in parallel) while decode is **memory-bandwidth-bound** (matrix-vector products generating one token at a time)", "Both phases are compute-bound but decode uses more FLOPs per token because it must attend to the entire accumulated KV-cache at each generation step"],
      correct: 2,
      explanation: "Prefill processes all $N$ prompt tokens simultaneously: the main operations are matrix-matrix multiplies $Y = XW$ where $X \\in \\mathbb{R}^{N \\times d}$. This has high arithmetic intensity ($\\sim 2N$ FLOPs/byte) and saturates GPU compute. Decode generates one token at a time: matrix-vector multiplies $y = Wx$ with arithmetic intensity $\\sim 2$ FLOPs/byte, bottlenecked by weight-loading bandwidth. This asymmetry is the key insight driving disaggregated inference architectures."
    },
    {
      type: "mc",
      question: "Disaggregated inference (as in systems like Splitwise or DistServe) separates prefill and decode into different GPU pools because:",
      options: ["Prefill and decode require different model architectures, with prefill using an encoder and decode using a separate autoregressive decoder", "Co-locating them causes **interference** — long prefills block decode steps (increasing time-to-first-token), and decode's low utilization wastes compute-optimized GPUs; separating them allows hardware-specific optimization for each phase", "Decode requires significantly more GPU memory than prefill due to the growing KV-cache, so it needs dedicated high-memory GPUs while prefill can use cheaper ones", "It simplifies the codebase by isolating the two code paths, making each independently testable and deployable without cross-phase interactions"],
      correct: 1,
      explanation: "When prefill and decode share GPUs, a long prefill (e.g., 100K context) stalls all concurrent decode iterations, creating latency spikes. Disaggregation assigns prefill to compute-optimized GPUs (maximizing FLOPs) and decode to bandwidth-optimized or cheaper GPUs (where memory bandwidth is the bottleneck). The KV-cache is transferred between pools after prefill. This can reduce P99 time-to-first-token by 2-5x while improving overall throughput."
    },
    {
      type: "mc",
      question: "When serving multiple LoRA adapters from a single base model, the primary challenge is:",
      options: [
        "Requests using different adapters cannot be efficiently batched together because each requires a **different low-rank weight delta** $\\Delta W = BA$ — naive batching requires separate GEMM calls per adapter, destroying throughput",
        "LoRA adapters are too large to fit in GPU memory alongside the base model, since each adapter's low-rank matrices add substantial overhead when serving many adapters",
        "LoRA adapters change the model's vocabulary by adding task-specific tokens, creating incompatible tokenizations across requests using different adapters in the same batch",
        "Each adapter requires a separate KV-cache because the modified attention weights produce different key-value representations that cannot be shared across adapter variants"
      ],
      correct: 0,
      explanation: "The base model computation $y = Wx$ is shared, but each LoRA adapter adds a different $\\Delta W_i x = B_i A_i x$. In a batch with $K$ different adapters, naive implementation requires $K$ separate GEMM operations for the LoRA components. Solutions include: S-LoRA's custom CUDA kernels for batched irregular GEMMs, Punica's SGMV (segmented gather matrix-vector) kernel, and padding/grouping strategies that batch requests by adapter. The KV-cache is shared since the base attention structure is identical."
    },
    {
      type: "mc",
      question: "The NVIDIA H100 SXM offers ~3.35 TB/s HBM3 bandwidth vs the A100 SXM's ~2.0 TB/s HBM2e. For **decode-phase** serving (memory-bandwidth-bound), switching from A100 to H100 provides approximately:",
      options: ["No improvement — the decode bottleneck is the same on both GPUs since the model weights and KV-cache sizes are hardware-independent", "~4x speedup due to the H100's higher FP16 FLOPS (990 TFLOPS vs 312 TFLOPS), which directly translates to faster matrix-vector products", "~1.67x speedup — decode is bandwidth-bound, so the improvement tracks the bandwidth ratio $3.35/2.0 \\approx 1.67$, not the compute ratio", "~10x speedup from the Transformer Engine's specialized hardware blocks that are specifically designed for autoregressive token generation"],
      correct: 2,
      explanation: "Since decode is bottlenecked by memory bandwidth (loading model weights for matrix-vector products), the speedup is determined by the HBM bandwidth ratio, not the FLOPS ratio. The H100's ~3.35 TB/s vs A100's ~2.0 TB/s gives ~1.67x decode throughput. The H100's massive compute advantage (3.2x in FP16, even more with FP8) primarily benefits the compute-bound prefill phase. This is why hardware selection for inference depends critically on the workload mix."
    },
    {
      type: "mc",
      question: "vLLM, TensorRT-LLM, and SGLang represent three major open-source serving frameworks. A distinguishing feature of SGLang is:",
      options: ["**RadixAttention** — a radix tree-based KV-cache sharing mechanism that enables efficient prefix caching across requests with arbitrary shared prefixes, plus a frontend language for programming complex LLM pipelines", "It only supports NVIDIA GPUs but achieves higher utilization than competing frameworks through custom CUDA kernel optimizations for each GPU generation", "It was the first to implement continuous batching, which all other frameworks later adopted as a standard feature for efficient request scheduling", "It uses a proprietary model format that enables cross-framework weight sharing and allows seamless migration between different serving backends"],
      correct: 0,
      explanation: "SGLang introduced RadixAttention, which organizes the KV-cache as a radix tree indexed by token sequences. This allows automatic, fine-grained cache sharing: if requests share any prefix (not just a pre-defined system prompt), the common KV-cache entries are reused. Combined with SGLang's frontend DSL for expressing multi-call LLM programs (e.g., tree-of-thought, multi-turn), this enables cache-aware scheduling that can reduce redundant prefill computation by 5-10x in structured generation workloads."
    },
    {
      type: "mc",
      question: "A model with 70B float16 parameters requires 140 GB of weight storage. To serve this on hardware with 80 GB GPUs, the minimum number of GPUs needed using tensor parallelism is:",
      options: ["1 GPU — with offloading to CPU memory for the weights that exceed the 80 GB GPU capacity, swapping layers in and out during inference", "8 GPUs — tensor parallelism requires 8x overhead due to the all-reduce communication costs and activation replication across all devices", "4 GPUs — you need significant headroom beyond the raw weight size to accommodate KV-cache growth, activation tensors, and framework overhead", "2 GPUs — the weights are split across GPUs, with each GPU holding approximately 70 GB of weights plus KV-cache and activation memory"],
      correct: 3,
      explanation: "140 GB of weights requires at least 2 GPUs with 80 GB each (160 GB total). Each GPU holds ~70 GB of weights, leaving ~10 GB for KV-cache, activations, and framework overhead. In practice, 2 GPUs may be tight for long contexts (KV-cache grows with sequence length), so production deployments often use 4 GPUs for the headroom. Tensor parallelism splits each layer's matrices across GPUs with all-reduce communication between them — communication overhead is typically 5-15% of total latency."
    },
    {
      type: "mc",
      question: "Time-to-first-token (TTFT) and time-per-output-token (TPOT) are the two primary latency metrics for LLM serving. Increasing the batch size typically:",
      options: [
        "Increases TTFT (longer queue wait and prefill contention) while TPOT stays roughly constant until compute saturation — throughput improves at the cost of latency",
        "Decreases both TTFT and TPOT — larger batches amortize fixed overhead across more requests, reducing latency for both the initial response and subsequent tokens",
        "Has no effect on either metric — batch size only affects total power consumption without changing the per-request latency characteristics of the serving system",
        "Decreases TTFT but increases TPOT — batching accelerates the compute-bound prefill phase through better GPU utilization but slows the bandwidth-bound decode phase"
      ],
      correct: 0,
      explanation: "Larger batches improve throughput (tokens/second across all sequences) by better utilizing GPU compute. However, TTFT increases because: (1) requests may wait in a queue, and (2) ongoing prefills for other requests in the batch cause contention. TPOT remains relatively stable because decode is bandwidth-bound and adding sequences is nearly free until compute saturation. The SLA-optimal batch size balances the throughput-latency trade-off for the specific workload's latency requirements."
    },
    {
      type: "mc",
      question: "FP8 inference (available on H100 and later GPUs) provides a benefit over FP16 primarily by:",
      options: ["Reducing model accuracy in a controlled trade-off that saves power by lowering the voltage required for each floating-point operation, without changing throughput or memory usage", "**Doubling the effective memory bandwidth** (8-bit values are half the size of 16-bit, so 2x more values loaded per second) and doubling the compute throughput (2x more FP8 ops per cycle), benefiting both prefill and decode phases", "Enabling larger vocabulary sizes by compressing the embedding table to fit more token representations within the same GPU memory footprint, doubling the maximum vocabulary", "Reducing the number of required transformer layers by increasing per-layer expressiveness, since FP8's quantization noise acts as implicit regularization that compresses representations"],
      correct: 1,
      explanation: "FP8 halves the bytes per parameter: a 70B model occupies ~70 GB instead of ~140 GB, and the effective bandwidth for weight loading doubles. The H100's Transformer Engine provides 1979 TFLOPS in FP8 vs 990 in FP16. For decode (bandwidth-bound), FP8 nearly doubles throughput. For prefill (compute-bound), FP8 doubles peak FLOPS. Combined, FP8 can provide 1.5-1.9x end-to-end throughput improvement with minimal quality loss when properly calibrated."
    },
    {
      type: "mc",
      question: "Chunked prefill is an optimization that breaks the prefill phase of a long prompt into smaller chunks. The main benefit is:",
      options: ["Reducing the total compute for prefill by processing overlapping chunks that share intermediate attention computations, lowering the overall FLOP count for long prompts", "Preventing a single long prefill from **monopolizing the GPU** for hundreds of milliseconds — by interleaving prefill chunks with decode steps from other requests, it reduces latency spikes", "Enabling longer context lengths than the model's maximum position embedding supports, by processing chunks that each fit within the position limit independently", "Reducing KV-cache memory usage by discarding intermediate chunk states after processing, so only the final chunk's key-value pairs are retained in the cache"],
      correct: 1,
      explanation: "A 100K-token prefill on a 70B model can take several seconds, during which no decode steps execute for other requests. Chunked prefill splits this into e.g., 512-token chunks, interleaving decode iterations between chunks. This caps the maximum time any decode step is delayed to the time for one prefill chunk. The total prefill time increases slightly (due to repeated kernel launches), but the P99 TPOT for concurrent requests improves dramatically — essential for meeting SLA targets."
    },
    {
      type: "mc",
      question: "When evaluating serving frameworks, the metric \"requests per second per dollar\" accounts for:",
      options: ["Only the GPU hardware cost — the metric normalizes raw throughput by the purchase price of the GPUs, ignoring utilization, software efficiency, and operational overhead", "The accuracy of the model's outputs — higher-quality responses are weighted more heavily in the throughput calculation, penalizing frameworks that sacrifice output quality for speed", "The number of parameters in the model — the metric normalizes by model size so that serving smaller models doesn't appear artificially cheaper than serving larger ones", "The **total cost efficiency** — including GPU cost, utilization, batching efficiency, and framework overhead; a framework achieving 100 req/s on a \\$2/hr GPU beats one achieving 150 req/s on a \\$6/hr GPU"],
      correct: 3,
      explanation: "Raw throughput (req/s) is insufficient for cost comparison. Framework A serving 100 req/s on an A100 (\\$2/hr) achieves 50 req/s/\\$ vs. Framework B serving 150 req/s on 2x H100s (\\$8/hr) at 18.75 req/s/\\$. Real cost analysis also considers: GPU utilization under varying load, the latency-throughput trade-off at SLA boundaries, CPU/memory/networking costs, and operational complexity. This is why serving benchmarks should report cost-normalized throughput at a specified latency target."
    }
  ]
};
