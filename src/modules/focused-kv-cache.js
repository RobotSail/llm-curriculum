// Focused module: KV-Cache and Memory-Bound Decoding
// Covers: why autoregressive decoding needs caching, KV-cache structure,
// memory growth, bandwidth bottleneck, and batching as the solution.
// Section C.2: Efficient Decoding

export const kvCacheLearning = {
  id: "C.2-kv-cache-learning-medium",
  sectionId: "C.2",
  title: "KV-Cache and Memory-Bound Decoding",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Why Autoregressive Decoding Is Expensive",
      content: "Autoregressive generation produces one token at a time. Each new token requires a **full forward pass** through the model to compute the next-token probabilities.\n\nConsider generating a 500-token response with a 70B model. Naively, each token requires attending to all previous tokens, meaning:\n- Token 1: attend to 1 position\n- Token 2: attend to 2 positions\n- Token 500: attend to 500 positions\n\nThe attention computation for token $t$ recomputes the key and value projections for **all** previous tokens $1, \\ldots, t-1$, then computes attention scores between the new query and all previous keys. This is wasteful: the keys and values for tokens $1$ through $t-1$ haven't changed since they were first computed.\n\nThe **KV-cache** eliminates this redundancy by storing the key and value vectors from all previous tokens, so each decode step only needs to compute the new token's query, key, and value — not recompute the entire history."
    },
    {
      type: "mc",
      question: "Without a KV-cache, generating token $t$ in an autoregressive model requires recomputing the key and value projections for all $t-1$ previous tokens. The total compute to generate a sequence of length $T$ scales as:",
      options: [
        "$O(T)$ — linear in sequence length because each token reuses the previous step's computation through weight sharing",
        "$O(T \\log T)$ — each step uses a divide-and-conquer attention pattern that processes tokens in $\\log T$ groups",
        "$O(T^2)$ — quadratic because step $t$ does $O(t)$ work and summing $1 + 2 + \\cdots + T = O(T^2)$",
        "$O(T^3)$ — cubic because attention itself is quadratic and we repeat it $T$ times during generation"
      ],
      correct: 2,
      explanation: "Without caching, step $t$ recomputes all $t$ key-value projections and runs attention over $t$ positions, costing $O(t)$ work. The total across all $T$ steps is $\\sum_{t=1}^{T} t = T(T+1)/2 = O(T^2)$. With the KV-cache, each step only computes the new token's projections and attends to the cached keys, reducing per-step work to $O(T)$ (reading the cache) and total work to $O(T^2)$ as well — but the constant factor is dramatically smaller because we skip the redundant projections."
    },
    {
      type: "info",
      title: "What the KV-Cache Stores",
      content: "At each transformer layer, the attention mechanism computes:\n\n$$Q = xW_Q, \\quad K = xW_K, \\quad V = xW_V$$\n\nDuring decoding, only the **new token's** $Q$, $K$, $V$ need to be computed. The $K$ and $V$ vectors from all previous tokens are appended to the cache.\n\nFor a model with:\n- $L$ layers\n- $H$ attention heads per layer\n- Head dimension $d_h$ (so $H \\times d_h$ = hidden size per KV projection)\n\nThe cache stores, per token:\n- One $K$ vector of size $H \\times d_h$ at each of $L$ layers\n- One $V$ vector of size $H \\times d_h$ at each of $L$ layers\n\n**Total per token** (in float16, 2 bytes per element):\n$$\\text{bytes/token} = 2 \\times L \\times H \\times d_h \\times 2$$\n\nThe factor of $2$ at the start accounts for K+V. For a 70B model (80 layers, 64 heads, $d_h = 128$), this is $2 \\times 80 \\times 64 \\times 128 \\times 2 = 2{,}621{,}440$ bytes ≈ **2.5 MB per token**."
    },
    {
      type: "mc",
      question: "A 7B model has 32 layers, 32 attention heads, and head dimension 128. Using float16 storage, how much KV-cache memory does a single 4096-token sequence require?",
      options: [
        "~2 GB — per token: $2 \\times 32 \\times 4096 \\times 2 = 524{,}288$ bytes; for 4096 tokens: $524{,}288 \\times 4096 \\approx 2.15$ GB, since KV pairs across all layers scale linearly with sequence length",
        "~14 GB — approximately equal to the model weight size, since the cache effectively duplicates all model parameters for each cached token position in the attention layers",
        "~128 MB — computed using only the hidden dimension without the layer multiplier, mistakenly treating the cache as a single-layer structure rather than spanning all 32 layers",
        "~32 MB — computed as $32 \\times 4096 \\times 4096 \\times 2$ bytes but missing the factor of 2 for storing both keys and values at each layer position"
      ],
      correct: 0,
      explanation: "Per token: $2 \\text{ (K+V)} \\times 32 \\text{ layers} \\times (32 \\times 128) \\text{ hidden} \\times 2 \\text{ bytes} = 2 \\times 32 \\times 4096 \\times 2 = 524{,}288$ bytes ≈ 0.5 MB. For 4096 tokens: $524{,}288 \\times 4096 = 2{,}147{,}483{,}648$ bytes ≈ 2 GB. This is roughly 1/7th of the 14 GB model weight size — but at 32K context it would be ~16 GB, exceeding the model weights. The KV-cache grows linearly with context while model weights stay fixed."
    },
    {
      type: "info",
      title: "KV-Cache Memory Growth",
      content: "The KV-cache grows **linearly** with sequence length. This linear growth creates a critical constraint for long-context models:\n\n| Model | Context | KV-Cache (float16) | Model Weights |\n|-------|---------|------|--------|\n| 7B (32L, 4096 hidden) | 4K | ~0.5 GB | ~14 GB |\n| 7B | 32K | ~4 GB | ~14 GB |\n| 7B | 128K | ~16 GB | ~14 GB |\n| 70B (80L, 8192 hidden) | 4K | ~1.3 GB | ~140 GB |\n| 70B | 128K | ~42 GB | ~140 GB |\n\nAt 128K context, the KV-cache for a 7B model **exceeds the model weights themselves**. For a 70B model at 128K, it consumes 42 GB — nearly a third of an H100's 80 GB.\n\nThis is why serving a few long-context requests can exhaust GPU memory while hundreds of short-context requests fit comfortably. The KV-cache, not the model weights, is the binding constraint for long-context inference."
    },
    {
      type: "mc",
      question: "A serving system has 80 GB of GPU memory. The model weights occupy 14 GB (7B model, float16). How many concurrent 32K-context requests can the system serve if each request's KV-cache requires ~4 GB?",
      options: [
        "20 requests — dividing total memory by per-request cache size without accounting for model weights",
        "16 requests — using $(80 - 14) / 4 = 16.5$, rounded down, since model weights are loaded once and shared across all requests",
        "5 requests — because each request needs both the 14 GB model weights and 4 GB cache loaded independently into separate memory regions",
        "80 requests — because the KV-cache can be offloaded entirely to CPU memory, leaving all GPU memory available for batching"
      ],
      correct: 1,
      explanation: "Model weights are loaded once and shared across all concurrent requests — they don't need to be duplicated. Available memory for KV-caches is $80 - 14 = 66$ GB. At ~4 GB per request, this fits $\\lfloor 66/4 \\rfloor = 16$ concurrent requests. In practice, the number is lower due to memory fragmentation, activation memory, and CUDA overhead. This calculation shows why KV-cache compression (quantization, MQA/GQA) directly translates to higher serving throughput."
    },
    {
      type: "info",
      title: "The Bandwidth Bottleneck",
      content: "Each decode step generates one token by running a forward pass through the model. The dominant operations are matrix-vector products $y = Wx$ where $W$ is a weight matrix and $x$ is the single token's hidden state.\n\nFor a matrix $W \\in \\mathbb{R}^{m \\times n}$:\n- **Data loaded from memory**: $m \\times n$ parameters × 2 bytes = $2mn$ bytes\n- **Computation performed**: $2mn$ FLOPs (multiply-accumulate)\n- **Arithmetic intensity**: $\\frac{2mn \\text{ FLOPs}}{2mn \\text{ bytes}} = 1$ FLOP/byte\n\nAn H100 GPU has ~3.35 PFLOPS of FP16 compute but only ~3.35 TB/s of HBM bandwidth. The **compute-to-bandwidth ratio** is:\n$$\\frac{3.35 \\times 10^{15}}{3.35 \\times 10^{12}} \\approx 1000 \\text{ FLOPs/byte}$$\n\nWith arithmetic intensity of ~1 FLOP/byte, the GPU is **1000x underutilized** during single-token decode. It spends almost all its time waiting for weight data to arrive from HBM, not computing.\n\nThis is why decode latency is dominated by **memory bandwidth**, not compute capacity. The GPU is an expensive space heater during single-sequence decoding."
    },
    {
      type: "mc",
      question: "A GPU has 2 TB/s of HBM bandwidth. A 7B model (14 GB in float16) generates one token per forward pass by reading all weights once. What is the theoretical maximum token rate for a single sequence?",
      options: [
        "~143 tokens/sec — computed as $2000 / 14 \\approx 143$, since each token requires reading all 14 GB of weights through the bandwidth pipe",
        "~500 tokens/sec — because the GPU caches frequently used weight matrices in L2, reducing the effective bandwidth requirement by ~3.5x",
        "~1000 tokens/sec — because modern GPUs overlap computation and memory access, effectively doubling the apparent bandwidth",
        "~14 tokens/sec — because each weight parameter requires multiple memory reads due to cache conflicts and non-sequential access patterns"
      ],
      correct: 0,
      explanation: "At minimum, each token reads the full 14 GB weight matrix through the memory bus. At 2 TB/s, this takes $14/2000 \\approx 7$ ms per token, giving ~143 tokens/sec. In practice, it's somewhat lower due to KV-cache reads, activation memory, and overhead. This is the fundamental bandwidth limit — no amount of additional compute can speed up single-sequence decoding beyond what the memory bus can deliver. Only reducing bytes read (quantization, pruning) or increasing arithmetic intensity (batching) helps."
    },
    {
      type: "info",
      title: "Batching: Converting Bandwidth-Bound to Compute-Bound",
      content: "The solution to the bandwidth bottleneck is **batching** — processing multiple sequences simultaneously.\n\nWith a batch of $B$ sequences, the matrix-vector product $y = Wx$ becomes a matrix-matrix product $Y = WX$ where $X \\in \\mathbb{R}^{n \\times B}$:\n- **Data loaded**: still $2mn$ bytes (weights loaded once, shared across batch)\n- **Computation**: $2mnB$ FLOPs\n- **Arithmetic intensity**: $\\frac{2mnB}{2mn} = B$ FLOPs/byte\n\nAt batch size $B$, the arithmetic intensity is $B$. To saturate the GPU's compute-to-bandwidth ratio of ~1000, you need $B \\approx 1000$. In practice, useful saturation begins much earlier:\n\n- $B = 1$: ~0.1% compute utilization (bandwidth-bound)\n- $B = 32$: ~3% compute utilization (still bandwidth-bound, but 32x throughput)\n- $B = 256$: ~25% compute utilization (transitioning to compute-bound)\n- $B = 1024$: ~100% compute utilization (fully compute-bound)\n\nThe key insight: **adding sequences to a batch is nearly free** until you hit compute saturation. Each additional sequence adds negligible latency but one more token of throughput. This is why high-throughput serving systems maximize batch size — they fill the gap between what the memory bus delivers and what the compute units can process."
    },
    {
      type: "mc",
      question: "A serving system currently runs at batch size 16, generating 2,000 tokens/sec total across all sequences. If they increase to batch size 64 (still in the bandwidth-bound regime), what total throughput should they expect?",
      options: [
        "~2,000 tokens/sec — throughput is fixed by bandwidth and does not increase with batch size regardless of the operating regime",
        "~8,000 tokens/sec — approximately 4x improvement since batch size increased 4x and the system remains bandwidth-bound with near-linear throughput scaling",
        "~32,000 tokens/sec — quadratic improvement because each new sequence multiplies the throughput of every other sequence in the batch",
        "~500 tokens/sec — throughput decreases because the larger batch creates memory contention that reduces effective bandwidth per sequence"
      ],
      correct: 1,
      explanation: "In the bandwidth-bound regime, the weight matrix is read once regardless of batch size. Going from $B=16$ to $B=64$ performs 4x more computation with (approximately) the same memory reads. Total throughput scales nearly linearly: ~$2000 \\times 4 = 8000$ tokens/sec. Per-sequence latency stays roughly constant because the extra compute is \"free\" — the GPU was idle during memory reads anyway. This near-linear scaling holds until the system transitions to compute-bound, typically around $B = 256$-$1024$ depending on model size and GPU."
    },
    {
      type: "info",
      title: "Prefill vs. Decode: Two Different Regimes",
      content: "LLM inference has two distinct phases with very different computational profiles:\n\n**Prefill (prompt processing)**: The entire input prompt is processed in one forward pass. This is a **matrix-matrix** operation ($Y = WX$ with $X$ being all prompt tokens), so it is **compute-bound**. A 4K-token prompt has arithmetic intensity ~4000 — the GPU runs at near peak FLOPs.\n\n**Decode (token generation)**: Each new token is generated one at a time. This is a **matrix-vector** operation, **bandwidth-bound** as we just discussed.\n\nThis creates a fundamental tension in serving:\n- Prefill wants **high compute** — it benefits from being scheduled on compute-rich hardware\n- Decode wants **high bandwidth** — it benefits from memory-optimized configurations\n\nModern serving systems must balance these two phases. A long prompt with a short response is prefill-dominated (compute-bound). A short prompt with a long response is decode-dominated (bandwidth-bound). The optimal serving strategy depends on the workload mix.\n\nSome systems (e.g., Sarathi, DistServ) physically separate prefill and decode onto different GPUs, optimizing each phase independently."
    },
    {
      type: "mc",
      question: "A chatbot API receives two types of requests: (A) summarization tasks with 8K-token inputs and 200-token outputs, and (B) creative writing tasks with 100-token prompts and 2K-token outputs. Which statement about their serving characteristics is correct?",
      options: [
        "Both request types have identical serving profiles because total token count (input + output) is roughly similar at ~8-10K tokens per request",
        "Request A is decode-dominated because long inputs generate more KV-cache entries, making each decode step slower due to larger attention computations",
        "Request B is more expensive overall because creative writing requires higher-quality model outputs, which demands more FLOPs per token during generation",
        "Request A is prefill-dominated and compute-bound (8K tokens processed in parallel), while request B is decode-dominated and bandwidth-bound (2K sequential decode steps)"
      ],
      correct: 3,
      explanation: "Request A spends most of its time in prefill: 8K tokens processed in one compute-bound pass, then only 200 bandwidth-bound decode steps. Request B does minimal prefill (100 tokens) but runs 2K decode steps, each bandwidth-bound. For request A, GPU compute utilization is high. For request B, the GPU is mostly waiting on memory reads. A serving system receiving a mix of both should schedule them carefully — prefill-heavy requests fill compute, while decode-heavy requests fill bandwidth, and interleaving them can improve overall utilization."
    },
    {
      type: "mc",
      question: "GQA (Grouped-Query Attention) reduces the number of KV heads from $H$ to $G$ (where $G < H$), with each KV head shared across $H/G$ query heads. For a 70B model with 64 query heads, switching from full MHA to GQA with 8 KV groups affects inference by:",
      options: [
        "Reducing model quality by 8x because 87.5% of the attention information is discarded when collapsing 64 independent KV projections down to 8 shared groups",
        "Having no effect on inference speed or memory since GQA only changes the training procedure and the stored model uses the same number of parameters at serving time",
        "Increasing decode latency by 8x because the shared KV heads create a serialization bottleneck where query heads must wait to access shared key-value pairs",
        "Reducing KV-cache by 8x (from 64 to 8 KV heads), enabling 8x more concurrent requests or 8x longer contexts within the same memory budget"
      ],
      correct: 3,
      explanation: "GQA with 8 groups stores 8 K and 8 V heads per layer instead of 64, reducing KV-cache by $64/8 = 8\\times$. For the 70B model at 128K context, this shrinks the cache from ~42 GB to ~5.25 GB — a massive reduction. The quality impact is minimal: experiments show GQA-8 matches MHA quality within 0.5% on standard benchmarks after appropriate training. The key insight is that many query heads learn to attend to similar patterns, so sharing KV projections across groups loses little information while saving enormous memory."
    }
  ]
};
