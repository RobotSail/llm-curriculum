// Assessment module for C.2: Efficient Decoding
// Split from assess-branch-cd.js — per-section test (10 questions)

export const decodingAssessment = {
  id: "C.2-assess",
  sectionId: "C.2",
  title: "Assessment: Efficient Decoding",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The KV-cache stores key and value tensors from previous tokens to avoid recomputation during autoregressive decoding. For a model with $L$ layers, $H$ attention heads, head dimension $d$, and sequence length $S$, the KV-cache size (in float16) is:",
      options: [
        "$2 \\times L \\times H \\times d \\times 2$ bytes — stores one K and one V vector per layer but only for the current token, independent of sequence length",
        "$2 \\times L \\times H \\times d \\times S \\times 2$ bytes — it stores both K and V across all layers and grows linearly with sequence length",
        "$L \\times S \\times 2$ bytes — stores one scalar per layer per token, since attention reduces each head's contribution to a single aggregated value",
        "$H \\times d \\times S \\times 2$ bytes — caches only a single layer's K and V projections, recomputing the remaining layers at each decode step"
      ],
      correct: 1,
      explanation: "Per token, the cache stores a K and V vector of dimension $H \\times d$ at each of $L$ layers. Factor of 2 for K+V, factor of 2 bytes for float16. Total: $2 \\times L \\times H \\times d \\times S \\times 2$ bytes. The linear growth with $S$ is why long-context models are so memory-hungry during inference — this cache, not the model weights, often dominates GPU memory for long sequences."
    },
    {
      type: "mc",
      question: "For a 70B parameter model (80 layers, 64 heads, $d_h = 128$) serving a single request at 128K context in float16, the KV-cache alone requires approximately:",
      options: ["~1.3 GB — computed as $80 \\times 128 \\times 128000 \\times 2$ bytes, caching only the keys without the values", "~5 GB — computed as $2 \\times 80 \\times 128 \\times 128000 \\times 2$ bytes, using only $d_h$ rather than $H \\times d_h$", "~20 GB — computed as $2 \\times 80 \\times 64 \\times 128 \\times 128000 \\times 2$ bytes $\\approx 20.97$ GB", "~80 GB — computed as $2 \\times 80 \\times 64 \\times 128 \\times 128000 \\times 8$ bytes, using float32 per entry"],
      correct: 2,
      explanation: "KV-cache $= 2 \\times 80 \\times 64 \\times 128 \\times 128000 \\times 2 = 2 \\times 80 \\times 8192 \\times 128000 \\times 2$ bytes. Breaking it down: $80 \\times 8192 = 655360$ (per-layer KV dim), $\\times 128000 \\times 2 \\times 2 = 335,544,320,000$ bytes $\\approx 20.97$ GB. This means a single 128K-context request on a 70B model needs ~21 GB just for the KV-cache, in addition to the ~140 GB for model weights. This is why KV-cache optimization is critical."
    },
    {
      type: "mc",
      question: "Paged Attention, introduced by vLLM, addresses KV-cache memory fragmentation by:",
      options: [
        "Compressing the KV-cache entries to 4-bit precision via per-head quantization, reducing the memory footprint without changing the allocation strategy",
        "Pre-allocating the maximum possible KV-cache for every request at launch time, using a memory pool that reserves contiguous blocks upfront",
        "Managing KV-cache in non-contiguous **pages** (like OS virtual memory), allocating blocks on demand and eliminating wasted memory from pre-allocation",
        "Storing the KV-cache on CPU memory and streaming entries to the GPU on demand, trading PCIe transfer latency for reduced GPU memory pressure"
      ],
      correct: 2,
      explanation: "Without paged attention, each request pre-allocates a contiguous KV-cache for the maximum sequence length, wasting memory when actual sequences are shorter. Paged Attention divides the KV-cache into fixed-size blocks (pages) allocated on demand. A block table maps logical positions to physical blocks, enabling non-contiguous storage. This reduces memory waste from ~60-80% to near zero, dramatically increasing the number of concurrent requests a single GPU can serve."
    },
    {
      type: "mc",
      question: "Speculative decoding uses a small **draft model** to generate $K$ candidate tokens, then the large **target model** verifies them in a single forward pass. A key theoretical guarantee is:",
      options: ["The output distribution is **mathematically identical** to sampling from the target model alone — the draft model only affects speed, not the distribution of generated text", "The draft model must share the same vocabulary as the target model but the target can have additional tokens beyond the shared set", "The method only works with greedy decoding because the acceptance criterion requires deterministic token selection at each step", "The output quality matches the draft model since accepted tokens come from the draft model's distribution rather than the target's"],
      correct: 0,
      explanation: "Speculative decoding uses a rejection sampling scheme: each draft token is accepted with probability $\\min(1, \\frac{p_{\\text{target}}(x)}{p_{\\text{draft}}(x)})$. If rejected, a corrected token is sampled from a modified distribution. This guarantees the final output follows the exact target model distribution. The speedup comes from the fact that the target model can verify $K$ tokens in parallel (one forward pass), while generating them autoregressively would require $K$ passes. Typical speedups are 2-3x."
    },
    {
      type: "mc",
      question: "In speculative decoding, the acceptance rate depends on the alignment between draft and target distributions. If the draft model proposes tokens with probability $q(x)$ and the target assigns $p(x)$, the expected acceptance rate is:",
      options: [
        "$1 - \\text{KL}(p \\| q)$ — one minus the KL divergence from the target to the draft distribution",
        "$\\exp(-\\text{JS}(p, q))$ — the exponentiated negative Jensen-Shannon divergence between the two",
        "$\\frac{p(x)}{q(x)}$ averaged over $q$ — the expected likelihood ratio under the draft distribution",
        "$\\sum_x \\min(p(x), q(x))$ — the total variation overlap between the two distributions"
      ],
      correct: 3,
      explanation: "Each token is accepted with probability $\\min(1, p(x)/q(x))$, and the overall acceptance rate (averaged over $x \\sim q$) is $\\sum_x q(x) \\min(1, p(x)/q(x)) = \\sum_x \\min(q(x), p(x))$. This is the total variation overlap. When $p \\approx q$, acceptance is near 100% and speculative decoding achieves maximum speedup. When they diverge, more tokens are rejected and the benefit decreases. This is why the draft model should approximate the target well."
    },
    {
      type: "mc",
      question: "Continuous batching (also called iteration-level or in-flight batching) differs from static batching by:",
      options: ["Allowing new requests to **join the batch as soon as any request finishes**, rather than waiting for all requests in the batch to complete — this eliminates idle GPU cycles from padding shorter sequences", "Using larger batch sizes by accumulating more requests before starting any processing, trading latency for higher total throughput", "Processing requests one at a time for lower per-request latency, eliminating the overhead of batch coordination and padding", "Batching only the prefill phase while processing the decode phase sequentially, since prefill benefits more from parallelism"],
      correct: 0,
      explanation: "In static batching, all requests in a batch must complete before new ones are admitted. Since requests have variable output lengths, short-output requests finish early and their GPU resources sit idle. Continuous batching inserts new requests at each iteration, maintaining high GPU utilization. This can improve throughput by 2-5x over static batching. vLLM, TensorRT-LLM, and SGLang all implement continuous batching as a core feature."
    },
    {
      type: "mc",
      question: "Prefix caching (also called prompt caching) accelerates serving by:",
      options: ["Storing the **computed KV-cache for shared prompt prefixes** (e.g., system prompts) so that multiple requests sharing the same prefix skip the redundant prefill computation", "Caching the model weights in faster memory tiers for rapid loading, reducing the cold-start latency when serving the first request", "Caching the final output tokens for common prompts so that repeated queries return the same precomputed response without any model execution", "Pre-computing all possible outputs for short prompts by enumerating the most likely continuations and storing them in a lookup table"],
      correct: 0,
      explanation: "Many serving scenarios involve repeated prefixes: system prompts in chat APIs, few-shot examples, or shared document contexts. Prefix caching computes the KV-cache for the shared prefix once and reuses it across requests. For a 2K system prompt with a 70B model, this saves ~2K tokens of prefill computation per request. RadixAttention (SGLang) extends this with a radix tree to efficiently share arbitrary prefix subtrees across concurrent requests."
    },
    {
      type: "mc",
      question: "Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce the KV-cache size by:",
      options: ["Using fewer transformer layers in the model architecture, which proportionally reduces the number of KV entries that must be stored per token", "Reducing the number of distinct K and V heads — MQA uses a **single** KV head shared across all query heads, while GQA uses $G$ KV head groups (where $1 < G < H$), reducing cache by a factor of $H/G$", "Quantizing the KV-cache to 4-bit precision, reducing each stored entry's footprint by 4x compared to the standard float16 representation", "Only caching every other layer's KV pairs and recomputing the skipped layers on the fly during each decode step to save memory"],
      correct: 1,
      explanation: "Standard multi-head attention has $H$ distinct K,V projections. MQA collapses these to 1 shared K,V head (cache reduced by $H\\times$). GQA uses $G$ groups, each serving $H/G$ query heads (cache reduced by $H/G\\times$). For example, LLaMA-2 70B uses GQA with 8 KV heads for 64 query heads, reducing KV-cache by $8\\times$. This is critical for long-context serving: the 128K KV-cache drops from ~21 GB to ~2.6 GB."
    },
    {
      type: "mc",
      question: "During autoregressive decoding, each token generation step is **memory-bandwidth bound** rather than compute-bound because:",
      options: [
        "The model weights must be loaded from CPU RAM into GPU memory at each decode step, and this PCIe transfer dominates the latency over the actual matrix computation",
        "Each step processes only **one new token** (batch size 1 per sequence), so the arithmetic intensity — ratio of FLOPs to bytes loaded — is extremely low; the GPU spends most time loading weights from HBM rather than computing",
        "The softmax normalization requires a global reduction across the full vocabulary, creating a sequential memory-access bottleneck that prevents effective parallelization",
        "The KV-cache must be read and rewritten to HBM at every step, and the growing cache size makes these read-write cycles dominate the total step latency"
      ],
      correct: 1,
      explanation: "For a single token, the dominant operation is matrix-vector products $y = Wx$ (not matrix-matrix). With $W \\in \\mathbb{R}^{m \\times n}$, this requires loading $mn$ parameters but performs only $2mn$ FLOPs — an arithmetic intensity of $\\sim 2$ FLOPs/byte. Modern GPUs have compute-to-bandwidth ratios of 100-300 FLOPs/byte, so the hardware is vastly underutilized. This is why batching multiple sequences together is essential: it converts vector operations into matrix operations with higher arithmetic intensity."
    },
    {
      type: "mc",
      question: "A serving system processes a batch of 32 sequences, each generating one token per step. Compared to serving a single sequence, the token generation throughput (tokens/second across all sequences) approximately:",
      options: ["Decreases due to memory contention — the 32 KV-caches compete for HBM bandwidth, reducing the effective memory throughput available for each sequence's decode step", "Stays the same — the GPU processes sequences strictly sequentially, so batching only affects queue wait time without increasing the token generation rate", "Increases by exactly 32x but with 32x higher per-sequence latency, since the GPU must cycle through all 32 sequences in round-robin before returning to each one", "Increases by ~32x because matrix-vector products become matrix-matrix products, better utilizing compute; per-sequence latency stays roughly constant until compute saturation"],
      correct: 3,
      explanation: "Batching converts $Wx$ (matrix-vector, bandwidth-bound) into $WX$ (matrix-matrix, compute-bound), increasing arithmetic intensity from ~2 to $\\sim 2B$ FLOPs/byte where $B$ is batch size. Since decode steps are bandwidth-bound, adding sequences is nearly free until the GPU's compute becomes the bottleneck. A batch of 32 thus achieves ~32x throughput with minimal per-sequence latency increase. The crossover to compute-bound depends on model size and GPU specs — typically around batch size 64-256."
    }
  ]
};
