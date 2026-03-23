// Assessment modules for curriculum branches C and D
// C.1: Quantization, C.2: Efficient Decoding, C.3: Serving Infrastructure, C.4: Compression & Distillation
// D.1: Chain-of-Thought & Reasoning, D.2: Test-Time Compute, D.3: Tool Use & Function Calling, D.4: Agentic Systems
// Pure assessment — no info steps

// ============================================================================
// C.1: Quantization
// ============================================================================
export const quantizationAssessment = {
  id: "C.1-assess",
  sectionId: "C.1",
  title: "Assessment: Quantization",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In post-training quantization (PTQ), weights are quantized **after** training is complete. Quantization-aware training (QAT) instead:",
      options: ["Trains a separate smaller model from scratch", "Quantizes only the optimizer states, not the model weights", "Simulates quantization effects during training via straight-through estimators so the model learns to be robust to reduced precision", "Uses lower learning rates to compensate for precision loss"],
      correct: 2,
      explanation: "QAT inserts fake-quantization nodes during training that round weights/activations to the target precision in the forward pass but pass gradients through unmodified (straight-through estimator). The model thus learns weight configurations that are robust to quantization noise. QAT typically recovers 0.5-1.0 perplexity points over PTQ but requires a full training run, making it far more expensive."
    },
    {
      type: "mc",
      question: "Activation quantization is generally harder than weight quantization because:",
      options: ["Activations use more memory than weights", "The backward pass requires full-precision activations", "Activations are always stored in float64", "Activations exhibit **outlier features** — a small number of hidden dimensions have magnitudes 10-100x larger than the rest, making uniform quantization waste most of its range on a few extreme values"],
      correct: 3,
      explanation: "Research (e.g., LLM.int8(), SmoothQuant) showed that transformer activations contain persistent outlier dimensions with magnitudes far exceeding the typical range. A uniform INT8 grid spanning $[-100, 100]$ to accommodate outliers wastes precision for the majority of values clustered near $[-1, 1]$. SmoothQuant addresses this by mathematically migrating the quantization difficulty from activations to weights via per-channel scaling: $Y = (X \\cdot \\text{diag}(s)^{-1}) \\cdot (\\text{diag}(s) \\cdot W)$."
    },
    {
      type: "mc",
      question: "GPTQ is a popular weight quantization method. Its core strategy is:",
      options: ["Training from scratch with quantized weights", "Quantizing weights **column by column**, using the inverse Hessian to optimally adjust remaining weights to compensate for each column's quantization error", "Quantizing all weights simultaneously with k-means clustering", "Pruning weights below a threshold, then quantizing the rest"],
      correct: 1,
      explanation: "GPTQ extends the Optimal Brain Quantization framework. It processes weight columns sequentially: after quantizing one column, it uses the inverse Hessian $H^{-1}$ of the layer's reconstruction loss to compute the optimal update to all not-yet-quantized columns, minimizing $\\|WX - \\hat{W}X\\|^2$. This Hessian-guided error compensation is what makes GPTQ achieve much better quality than naive round-to-nearest at 4-bit and below."
    },
    {
      type: "mc",
      question: "AWQ (Activation-Aware Weight Quantization) differs from GPTQ by focusing on:",
      options: ["Identifying the small fraction of **salient weight channels** (those corresponding to large activation magnitudes) and protecting them with per-channel scaling before quantization, rather than using Hessian-based error compensation", "Quantizing activations instead of weights", "Using 8-bit instead of 4-bit quantization", "Training a separate quantization network"],
      correct: 0,
      explanation: "AWQ observes that only ~1% of weight channels are critical — those connected to activation outlier features. Rather than expensive Hessian computation, AWQ finds per-channel scaling factors $s$ that protect salient channels: it scales weights by $s$ and inversely scales activations, shifting the quantization difficulty away from important channels. This is simpler than GPTQ and often matches or exceeds its quality with faster quantization time."
    },
    {
      type: "mc",
      question: "A model uses mixed-precision quantization: some layers at 4-bit, others at 8-bit. The decision of which layers get higher precision is typically based on:",
      options: ["Layer index — earlier layers always need more precision", "Parameter count — larger layers get lower precision", "**Per-layer sensitivity analysis** — layers where quantization causes larger increases in output error or perplexity are assigned higher precision, often measured via Hessian trace, Fisher information, or direct calibration loss", "Random assignment with a fixed ratio"],
      correct: 2,
      explanation: "Mixed-precision strategies measure each layer's sensitivity to quantization error, typically by quantizing one layer at a time and measuring the impact on calibration loss. Layers with high Hessian trace ($\\text{tr}(H)$) or large Fisher information are more sensitive. Empirically, attention projection layers and the first/last layers tend to be more sensitive. This yields a constrained optimization: minimize total quality loss subject to a target average bit-width."
    },
    {
      type: "mc",
      question: "SqueezeLLM achieves high-quality ultra-low-bit quantization by combining:",
      options: ["Knowledge distillation with pruning", "Dynamic quantization at inference time", "Layer fusion and operator merging", "Dense-and-sparse decomposition: a low-bit dense representation for the bulk of weights plus a **sparse matrix** storing outlier weights at full precision, keeping the sensitive values exact"],
      correct: 3,
      explanation: "SqueezeLLM decomposes each weight matrix into a dense low-bit component plus a sparse full-precision component for outlier weights. The key insight is that weight sensitivity follows a heavy-tailed distribution — a small number of weights disproportionately affect output quality. By storing these in a sparse matrix (which adds minimal memory overhead due to sparsity), the dense component can be aggressively quantized to 3 or even 2 bits with minimal degradation."
    },
    {
      type: "mc",
      question: "BitNet b1.58 uses ternary weights $\\{-1, 0, +1\\}$, meaning each weight requires $\\log_2(3) \\approx 1.58$ bits. Compared to standard float16 models of the same size, BitNet claims:",
      options: [
        "Identical accuracy with 10x faster inference, due to replacing multiplications with additions/subtractions",
        "Matching perplexity at the same parameter count starting from ~3B parameters, with matrix multiplications reduced to additions since $w \\in \\{-1, 0, 1\\}$ eliminates the need for floating-point multiply hardware",
        "Better accuracy because ternary weights act as regularization",
        "Worse accuracy but 100x memory savings"
      ],
      correct: 1,
      explanation: "With ternary weights, the matrix-vector product $y = Wx$ becomes pure additions and subtractions (multiply by 1, -1, or skip for 0). This eliminates the most expensive operation in inference — floating-point multiplication — and enables dramatically simpler hardware. BitNet b1.58 reports matching LLaMA-equivalent perplexity starting around 3B parameters, suggesting that extreme quantization is viable if applied from the start of training (QAT-style) rather than post-hoc."
    },
    {
      type: "mc",
      question: "A 7B parameter model with float16 weights occupies 14 GB. After GPTQ 4-bit quantization with group size 128, the model size is approximately:",
      options: ["~3.9 GB — each group of 128 weights shares a float16 scale and zero-point, adding overhead: $14 \\times \\frac{4}{16} + \\frac{7 \\times 10^9}{128} \\times 4\\text{ bytes} \\approx 3.5 + 0.22$ GB", "1.75 GB — exactly $14 \\times (4/16)$", "7 GB — quantization only affects compute, not storage", "14 GB — the quantized model includes a full-precision copy"],
      correct: 0,
      explanation: "4-bit quantization reduces the weight payload to $14 \\times (4/16) = 3.5$ GB. However, each group of 128 weights requires a float16 scale and zero-point (4 bytes per group). With $\\frac{7 \\times 10^9}{128} \\approx 54.7\\text{M}$ groups, that adds $\\sim$219 MB of overhead. Total $\\approx 3.7$-$3.9$ GB depending on metadata. Smaller group sizes improve quality but increase overhead; group size 128 is the standard trade-off."
    },
    {
      type: "mc",
      question: "When quantizing a weight value $w$ to a $b$-bit unsigned integer grid $[0, 2^b - 1]$, the standard affine quantization formula is:",
      options: ["$q = \\text{round}(w \\times 2^b)$", "$q = \\text{sign}(w) \\times \\lfloor |w| \\rfloor$", "$q = \\text{clamp}\\left(\\text{round}\\left(\\frac{w - z}{s}\\right), 0, 2^b - 1\\right)$ where $s = \\frac{w_{\\max} - w_{\\min}}{2^b - 1}$ is the scale and $z = w_{\\min}$ is the zero-point", "$q = \\text{round}(w)$ clipped to $b$ bits"],
      correct: 2,
      explanation: "Affine (asymmetric) quantization maps the floating-point range $[w_{\\min}, w_{\\max}]$ to the integer range $[0, 2^b - 1]$. The scale $s = \\frac{w_{\\max} - w_{\\min}}{2^b - 1}$ determines the step size, and the zero-point $z$ handles asymmetric distributions. Dequantization recovers $\\hat{w} = s \\cdot q + z$. The quantization error per weight is bounded by $s/2$, so fewer bits means larger steps and more error."
    },
    {
      type: "mc",
      question: "A researcher quantizes a 70B model to 2-bit weights and observes catastrophic perplexity degradation. Which approach is LEAST likely to help recover quality?",
      options: [
        "Using a larger calibration dataset for GPTQ",
        "Switching to mixed-precision with sensitive layers at 4-bit",
        "Adding LoRA adapters trained on a small dataset after quantization (QLoRA-style)",
        "Increasing the batch size during inference"
      ],
      correct: 3,
      explanation: "Increasing batch size affects throughput but does not change the model's weights or predictions — it cannot recover quality lost to quantization. The other three approaches directly address quantization error: better calibration data improves Hessian estimates in GPTQ, mixed-precision protects sensitive layers, and QLoRA fine-tunes low-rank adapters in float16 on top of quantized weights to compensate for quantization-induced errors. At 2-bit, combining multiple recovery strategies is typically necessary."
    }
  ]
};

// ============================================================================
// C.2: Efficient Decoding
// ============================================================================
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
        "$2 \\times L \\times H \\times d \\times 2$ bytes (independent of sequence length)",
        "$2 \\times L \\times H \\times d \\times S \\times 2$ bytes — it stores both K and V across all layers and grows linearly with sequence length",
        "$L \\times S \\times 2$ bytes",
        "$H \\times d \\times S \\times 2$ bytes (only one layer cached)"
      ],
      correct: 1,
      explanation: "Per token, the cache stores a K and V vector of dimension $H \\times d$ at each of $L$ layers. Factor of 2 for K+V, factor of 2 bytes for float16. Total: $2 \\times L \\times H \\times d \\times S \\times 2$ bytes. The linear growth with $S$ is why long-context models are so memory-hungry during inference — this cache, not the model weights, often dominates GPU memory for long sequences."
    },
    {
      type: "mc",
      question: "For a 70B parameter model (80 layers, 64 heads, $d_h = 128$) serving a single request at 128K context in float16, the KV-cache alone requires approximately:",
      options: ["~20 GB — computed as $2 \\times 80 \\times 64 \\times 128 \\times 128000 \\times 2$ bytes $\\approx 20.97$ GB", "~5 GB", "~1.3 GB", "~80 GB"],
      correct: 0,
      explanation: "KV-cache $= 2 \\times 80 \\times 64 \\times 128 \\times 128000 \\times 2 = 2 \\times 80 \\times 8192 \\times 128000 \\times 2$ bytes. Breaking it down: $80 \\times 8192 = 655360$ (per-layer KV dim), $\\times 128000 \\times 2 \\times 2 = 335,544,320,000$ bytes $\\approx 20.97$ GB. This means a single 128K-context request on a 70B model needs ~21 GB just for the KV-cache, in addition to the ~140 GB for model weights. This is why KV-cache optimization is critical."
    },
    {
      type: "mc",
      question: "Paged Attention, introduced by vLLM, addresses KV-cache memory fragmentation by:",
      options: [
        "Compressing the KV-cache with quantization",
        "Pre-allocating the maximum possible KV-cache for every request",
        "Managing KV-cache in non-contiguous **pages** (like OS virtual memory), allocating blocks on demand and eliminating wasted memory from pre-allocation and fragmentation",
        "Storing the KV-cache on CPU memory instead of GPU"
      ],
      correct: 2,
      explanation: "Without paged attention, each request pre-allocates a contiguous KV-cache for the maximum sequence length, wasting memory when actual sequences are shorter. Paged Attention divides the KV-cache into fixed-size blocks (pages) allocated on demand. A block table maps logical positions to physical blocks, enabling non-contiguous storage. This reduces memory waste from ~60-80% to near zero, dramatically increasing the number of concurrent requests a single GPU can serve."
    },
    {
      type: "mc",
      question: "Speculative decoding uses a small **draft model** to generate $K$ candidate tokens, then the large **target model** verifies them in a single forward pass. A key theoretical guarantee is:",
      options: ["The output quality matches the draft model", "The draft model must share the same vocabulary as the target model but not vice versa", "The method only works with greedy decoding", "The output distribution is **mathematically identical** to sampling from the target model alone — the draft model only affects speed, not the distribution of generated text"],
      correct: 3,
      explanation: "Speculative decoding uses a rejection sampling scheme: each draft token is accepted with probability $\\min(1, \\frac{p_{\\text{target}}(x)}{p_{\\text{draft}}(x)})$. If rejected, a corrected token is sampled from a modified distribution. This guarantees the final output follows the exact target model distribution. The speedup comes from the fact that the target model can verify $K$ tokens in parallel (one forward pass), while generating them autoregressively would require $K$ passes. Typical speedups are 2-3x."
    },
    {
      type: "mc",
      question: "In speculative decoding, the acceptance rate depends on the alignment between draft and target distributions. If the draft model proposes tokens with probability $q(x)$ and the target assigns $p(x)$, the expected acceptance rate is:",
      options: [
        "$1 - \\text{KL}(p \\| q)$",
        "$\\sum_x \\min(p(x), q(x))$ — the total variation overlap between the two distributions",
        "$\\frac{p(x)}{q(x)}$ averaged over $q$",
        "Always exactly 50%"
      ],
      correct: 1,
      explanation: "Each token is accepted with probability $\\min(1, p(x)/q(x))$, and the overall acceptance rate (averaged over $x \\sim q$) is $\\sum_x q(x) \\min(1, p(x)/q(x)) = \\sum_x \\min(q(x), p(x))$. This is the total variation overlap. When $p \\approx q$, acceptance is near 100% and speculative decoding achieves maximum speedup. When they diverge, more tokens are rejected and the benefit decreases. This is why the draft model should approximate the target well."
    },
    {
      type: "mc",
      question: "Continuous batching (also called iteration-level or in-flight batching) differs from static batching by:",
      options: ["Allowing new requests to **join the batch as soon as any request finishes**, rather than waiting for all requests in the batch to complete — this eliminates idle GPU cycles from padding shorter sequences", "Using larger batch sizes", "Processing requests one at a time for lower latency", "Batching only the prefill phase, not decoding"],
      correct: 0,
      explanation: "In static batching, all requests in a batch must complete before new ones are admitted. Since requests have variable output lengths, short-output requests finish early and their GPU resources sit idle. Continuous batching inserts new requests at each iteration, maintaining high GPU utilization. This can improve throughput by 2-5x over static batching. vLLM, TensorRT-LLM, and SGLang all implement continuous batching as a core feature."
    },
    {
      type: "mc",
      question: "Prefix caching (also called prompt caching) accelerates serving by:",
      options: ["Caching the final output tokens for common prompts", "Caching the model weights for faster loading", "Storing the **computed KV-cache for shared prompt prefixes** (e.g., system prompts) so that multiple requests sharing the same prefix skip the redundant prefill computation", "Pre-computing all possible outputs for short prompts"],
      correct: 2,
      explanation: "Many serving scenarios involve repeated prefixes: system prompts in chat APIs, few-shot examples, or shared document contexts. Prefix caching computes the KV-cache for the shared prefix once and reuses it across requests. For a 2K system prompt with a 70B model, this saves ~2K tokens of prefill computation per request. RadixAttention (SGLang) extends this with a radix tree to efficiently share arbitrary prefix subtrees across concurrent requests."
    },
    {
      type: "mc",
      question: "Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce the KV-cache size by:",
      options: ["Using fewer transformer layers", "Only caching every other layer", "Quantizing the KV-cache to 4-bit", "Reducing the number of distinct K and V heads — MQA uses a **single** KV head shared across all query heads, while GQA uses $G$ KV head groups (where $1 < G < H$), reducing cache by a factor of $H/G$"],
      correct: 3,
      explanation: "Standard multi-head attention has $H$ distinct K,V projections. MQA collapses these to 1 shared K,V head (cache reduced by $H\\times$). GQA uses $G$ groups, each serving $H/G$ query heads (cache reduced by $H/G\\times$). For example, LLaMA-2 70B uses GQA with 8 KV heads for 64 query heads, reducing KV-cache by $8\\times$. This is critical for long-context serving: the 128K KV-cache drops from ~21 GB to ~2.6 GB."
    },
    {
      type: "mc",
      question: "During autoregressive decoding, each token generation step is **memory-bandwidth bound** rather than compute-bound because:",
      options: [
        "The model weights are too large to fit in GPU memory",
        "Each step processes only **one new token** (batch size 1 per sequence), so the arithmetic intensity — ratio of FLOPs to bytes loaded — is extremely low; the GPU spends most time loading weights from HBM rather than computing",
        "The softmax operation is inherently memory-bound",
        "GPUs cannot parallelize matrix operations"
      ],
      correct: 1,
      explanation: "For a single token, the dominant operation is matrix-vector products $y = Wx$ (not matrix-matrix). With $W \\in \\mathbb{R}^{m \\times n}$, this requires loading $mn$ parameters but performs only $2mn$ FLOPs — an arithmetic intensity of $\\sim 2$ FLOPs/byte. Modern GPUs have compute-to-bandwidth ratios of 100-300 FLOPs/byte, so the hardware is vastly underutilized. This is why batching multiple sequences together is essential: it converts vector operations into matrix operations with higher arithmetic intensity."
    },
    {
      type: "mc",
      question: "A serving system processes a batch of 32 sequences, each generating one token per step. Compared to serving a single sequence, the token generation throughput (tokens/second across all sequences) approximately:",
      options: ["Increases by ~32x because matrix-vector products become matrix-matrix products, better utilizing compute; per-sequence latency stays roughly constant until compute saturation", "Stays the same — the GPU can only generate one token at a time", "Increases by exactly 32x with 32x higher latency", "Decreases due to memory contention"],
      correct: 0,
      explanation: "Batching converts $Wx$ (matrix-vector, bandwidth-bound) into $WX$ (matrix-matrix, compute-bound), increasing arithmetic intensity from ~2 to $\\sim 2B$ FLOPs/byte where $B$ is batch size. Since decode steps are bandwidth-bound, adding sequences is nearly free until the GPU's compute becomes the bottleneck. A batch of 32 thus achieves ~32x throughput with minimal per-sequence latency increase. The crossover to compute-bound depends on model size and GPU specs — typically around batch size 64-256."
    }
  ]
};

// ============================================================================
// C.3: Serving Infrastructure
// ============================================================================
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
      options: ["Both phases are memory-bandwidth-bound", "Prefill is memory-bound while decode is compute-bound", "Prefill is **compute-bound** (large matrix-matrix multiplications over all prompt tokens in parallel) while decode is **memory-bandwidth-bound** (matrix-vector products generating one token at a time)", "Both phases are compute-bound but decode uses more FLOPs"],
      correct: 2,
      explanation: "Prefill processes all $N$ prompt tokens simultaneously: the main operations are matrix-matrix multiplies $Y = XW$ where $X \\in \\mathbb{R}^{N \\times d}$. This has high arithmetic intensity ($\\sim 2N$ FLOPs/byte) and saturates GPU compute. Decode generates one token at a time: matrix-vector multiplies $y = Wx$ with arithmetic intensity $\\sim 2$ FLOPs/byte, bottlenecked by weight-loading bandwidth. This asymmetry is the key insight driving disaggregated inference architectures."
    },
    {
      type: "mc",
      question: "Disaggregated inference (as in systems like Splitwise or DistServe) separates prefill and decode into different GPU pools because:",
      options: ["Prefill and decode require different model architectures", "It simplifies the codebase", "Decode requires more GPU memory than prefill", "Co-locating them causes **interference** — long prefills block decode steps (increasing time-to-first-token), and decode's low utilization wastes compute-optimized GPUs; separating them allows hardware-specific optimization for each phase"],
      correct: 3,
      explanation: "When prefill and decode share GPUs, a long prefill (e.g., 100K context) stalls all concurrent decode iterations, creating latency spikes. Disaggregation assigns prefill to compute-optimized GPUs (maximizing FLOPs) and decode to bandwidth-optimized or cheaper GPUs (where memory bandwidth is the bottleneck). The KV-cache is transferred between pools after prefill. This can reduce P99 time-to-first-token by 2-5x while improving overall throughput."
    },
    {
      type: "mc",
      question: "When serving multiple LoRA adapters from a single base model, the primary challenge is:",
      options: [
        "LoRA adapters are too large to fit in GPU memory",
        "Requests using different adapters cannot be efficiently batched together because each requires a **different low-rank weight delta** $\\Delta W = BA$ — naive batching requires separate GEMM calls per adapter, destroying throughput",
        "LoRA adapters change the model's vocabulary",
        "Each adapter requires a separate KV-cache"
      ],
      correct: 1,
      explanation: "The base model computation $y = Wx$ is shared, but each LoRA adapter adds a different $\\Delta W_i x = B_i A_i x$. In a batch with $K$ different adapters, naive implementation requires $K$ separate GEMM operations for the LoRA components. Solutions include: S-LoRA's custom CUDA kernels for batched irregular GEMMs, Punica's SGMV (segmented gather matrix-vector) kernel, and padding/grouping strategies that batch requests by adapter. The KV-cache is shared since the base attention structure is identical."
    },
    {
      type: "mc",
      question: "The NVIDIA H100 SXM offers ~3.35 TB/s HBM3 bandwidth vs the A100 SXM's ~2.0 TB/s HBM2e. For **decode-phase** serving (memory-bandwidth-bound), switching from A100 to H100 provides approximately:",
      options: ["~1.67x speedup — decode is bandwidth-bound, so the improvement tracks the bandwidth ratio $3.35/2.0 \\approx 1.67$, not the compute ratio", "~4x speedup due to the H100's higher FP16 FLOPS (990 TFLOPS vs 312 TFLOPS)", "No improvement — the models are identical", "~10x speedup from the Transformer Engine"],
      correct: 0,
      explanation: "Since decode is bottlenecked by memory bandwidth (loading model weights for matrix-vector products), the speedup is determined by the HBM bandwidth ratio, not the FLOPS ratio. The H100's ~3.35 TB/s vs A100's ~2.0 TB/s gives ~1.67x decode throughput. The H100's massive compute advantage (3.2x in FP16, even more with FP8) primarily benefits the compute-bound prefill phase. This is why hardware selection for inference depends critically on the workload mix."
    },
    {
      type: "mc",
      question: "vLLM, TensorRT-LLM, and SGLang represent three major open-source serving frameworks. A distinguishing feature of SGLang is:",
      options: ["It was the first to implement continuous batching", "It only supports NVIDIA GPUs", "**RadixAttention** — a radix tree-based KV-cache sharing mechanism that enables efficient prefix caching across requests with arbitrary shared prefixes, plus a frontend language for programming complex LLM pipelines", "It uses a proprietary model format"],
      correct: 2,
      explanation: "SGLang introduced RadixAttention, which organizes the KV-cache as a radix tree indexed by token sequences. This allows automatic, fine-grained cache sharing: if requests share any prefix (not just a pre-defined system prompt), the common KV-cache entries are reused. Combined with SGLang's frontend DSL for expressing multi-call LLM programs (e.g., tree-of-thought, multi-turn), this enables cache-aware scheduling that can reduce redundant prefill computation by 5-10x in structured generation workloads."
    },
    {
      type: "mc",
      question: "A model with 70B float16 parameters requires 140 GB of weight storage. To serve this on hardware with 80 GB GPUs, the minimum number of GPUs needed using tensor parallelism is:",
      options: ["1 GPU — with offloading to CPU memory", "8 GPUs — tensor parallelism has 8x overhead", "4 GPUs — you need significant headroom beyond the raw weight size", "2 GPUs — the weights are split across GPUs, with each GPU holding approximately 70 GB of weights plus KV-cache and activation memory"],
      correct: 3,
      explanation: "140 GB of weights requires at least 2 GPUs with 80 GB each (160 GB total). Each GPU holds ~70 GB of weights, leaving ~10 GB for KV-cache, activations, and framework overhead. In practice, 2 GPUs may be tight for long contexts (KV-cache grows with sequence length), so production deployments often use 4 GPUs for the headroom. Tensor parallelism splits each layer's matrices across GPUs with all-reduce communication between them — communication overhead is typically 5-15% of total latency."
    },
    {
      type: "mc",
      question: "Time-to-first-token (TTFT) and time-per-output-token (TPOT) are the two primary latency metrics for LLM serving. Increasing the batch size typically:",
      options: [
        "Decreases both TTFT and TPOT",
        "Increases TTFT (longer queue wait and prefill contention) while TPOT stays roughly constant until compute saturation — throughput improves at the cost of latency",
        "Has no effect on either metric",
        "Decreases TTFT but increases TPOT"
      ],
      correct: 1,
      explanation: "Larger batches improve throughput (tokens/second across all sequences) by better utilizing GPU compute. However, TTFT increases because: (1) requests may wait in a queue, and (2) ongoing prefills for other requests in the batch cause contention. TPOT remains relatively stable because decode is bandwidth-bound and adding sequences is nearly free until compute saturation. The SLA-optimal batch size balances the throughput-latency trade-off for the specific workload's latency requirements."
    },
    {
      type: "mc",
      question: "FP8 inference (available on H100 and later GPUs) provides a benefit over FP16 primarily by:",
      options: ["**Doubling the effective memory bandwidth** (8-bit values are half the size of 16-bit, so 2x more values loaded per second) and doubling the compute throughput (2x more FP8 ops per cycle), benefiting both prefill and decode phases", "Reducing model accuracy to save power", "Enabling larger vocabulary sizes", "Reducing the number of required transformer layers"],
      correct: 0,
      explanation: "FP8 halves the bytes per parameter: a 70B model occupies ~70 GB instead of ~140 GB, and the effective bandwidth for weight loading doubles. The H100's Transformer Engine provides 1979 TFLOPS in FP8 vs 990 in FP16. For decode (bandwidth-bound), FP8 nearly doubles throughput. For prefill (compute-bound), FP8 doubles peak FLOPS. Combined, FP8 can provide 1.5-1.9x end-to-end throughput improvement with minimal quality loss when properly calibrated."
    },
    {
      type: "mc",
      question: "Chunked prefill is an optimization that breaks the prefill phase of a long prompt into smaller chunks. The main benefit is:",
      options: ["Reducing the total compute for prefill", "Enabling longer context lengths than the model supports", "Preventing a single long prefill from **monopolizing the GPU** for hundreds of milliseconds — by interleaving prefill chunks with decode steps from other requests, it reduces latency spikes and improves TPOT consistency for concurrent requests", "Reducing KV-cache memory usage"],
      correct: 2,
      explanation: "A 100K-token prefill on a 70B model can take several seconds, during which no decode steps execute for other requests. Chunked prefill splits this into e.g., 512-token chunks, interleaving decode iterations between chunks. This caps the maximum time any decode step is delayed to the time for one prefill chunk. The total prefill time increases slightly (due to repeated kernel launches), but the P99 TPOT for concurrent requests improves dramatically — essential for meeting SLA targets."
    },
    {
      type: "mc",
      question: "When evaluating serving frameworks, the metric \"requests per second per dollar\" accounts for:",
      options: ["Only the GPU cost", "The accuracy of the model's outputs", "The number of parameters in the model", "The **total cost efficiency** — including GPU cost, utilization, batching efficiency, and framework overhead; a framework achieving 100 req/s on a \\$2/hr GPU ($180K/yr$) beats one achieving 150 req/s on a \\$6/hr GPU, despite lower raw throughput"],
      correct: 3,
      explanation: "Raw throughput (req/s) is insufficient for cost comparison. Framework A serving 100 req/s on an A100 (\\$2/hr) achieves 50 req/s/\\$ vs. Framework B serving 150 req/s on 2x H100s (\\$8/hr) at 18.75 req/s/\\$. Real cost analysis also considers: GPU utilization under varying load, the latency-throughput trade-off at SLA boundaries, CPU/memory/networking costs, and operational complexity. This is why serving benchmarks should report cost-normalized throughput at a specified latency target."
    }
  ]
};

// ============================================================================
// C.4: Compression & Distillation
// ============================================================================
export const compressionAssessment = {
  id: "C.4-assess",
  sectionId: "C.4",
  title: "Assessment: Compression & Distillation",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In knowledge distillation, a student model is trained to match the teacher's **soft targets** (softmax outputs with temperature $\\tau > 1$) rather than just the hard labels. The key insight behind using soft targets is:",
      options: [
        "Soft targets are easier to compute",
        "The teacher's probability distribution over **incorrect classes** encodes rich similarity structure (\"dark knowledge\") — e.g., a teacher assigning 0.05 to 'cat' and 0.001 to 'car' for a dog image reveals that dogs are more like cats than cars",
        "Soft targets prevent the student from overfitting to the training data",
        "Temperature scaling makes the loss function convex"
      ],
      correct: 1,
      explanation: "Hinton et al. (2015) showed that the teacher's full probability vector, especially the relative probabilities of incorrect classes, contains far more information per training example than a one-hot label. With temperature $\\tau$, the softmax becomes $p_i = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}$, which smooths the distribution to expose these inter-class relationships. The student loss combines the soft target KL divergence (weighted by $\\tau^2$) with the hard label cross-entropy."
    },
    {
      type: "mc",
      question: "Logit matching distillation minimizes $\\text{KL}(p_{\\text{teacher}} \\| p_{\\text{student}})$ over the output distribution. Feature matching distillation instead:",
      options: ["Aligns **intermediate layer representations** between teacher and student — e.g., minimizing $\\|f_{\\text{teacher}}^{(l)} - g(f_{\\text{student}}^{(k)})\\|^2$ where $g$ is a learned projection to handle dimension mismatches", "Matches only the final prediction", "Uses reinforcement learning to train the student", "Matches the gradient norms of teacher and student"],
      correct: 0,
      explanation: "Feature matching (FitNets, PKD) adds losses that align intermediate representations. Since teacher and student may have different hidden dimensions, a learned linear projection $g$ maps student features to teacher feature space. This provides richer supervision than output-only matching: the student learns not just what to predict but how to represent. For LLM distillation, this can include matching attention patterns, hidden states at specific layers, or the output of feed-forward blocks."
    },
    {
      type: "mc",
      question: "On-policy distillation (used in models like Gemma and some LLaMA variants) differs from standard offline distillation by:",
      options: ["Using a smaller teacher model", "Training without any teacher signal", "Having the **student generate its own outputs**, then using the teacher to score/correct them — this avoids the train-test distribution mismatch where the student is trained on teacher-generated text but must generate its own text at inference", "Using only hard labels from the teacher"],
      correct: 2,
      explanation: "In offline distillation, the student trains on teacher-generated sequences. At inference, the student generates from its own distribution, creating exposure bias — errors compound because the student never learned to recover from its own mistakes. On-policy distillation lets the student generate sequences, then uses the teacher's per-token probabilities as training signal. This is analogous to DAgger in imitation learning. The GKD (Generalized Knowledge Distillation) framework formalizes the spectrum between on-policy and off-policy distillation."
    },
    {
      type: "mc",
      question: "Structured pruning removes entire **structures** (attention heads, neurons, layers), while unstructured pruning removes individual weights. The practical advantage of structured pruning is:",
      options: ["It achieves higher sparsity levels", "It can be applied during training at no cost", "It preserves more model quality at the same sparsity", "The resulting model has **regular, dense tensor shapes** that run efficiently on standard hardware (GPUs/TPUs) without specialized sparse kernels — unstructured pruning creates irregular sparsity patterns that standard hardware cannot accelerate"],
      correct: 3,
      explanation: "Removing 50% of individual weights (unstructured) leaves an irregular sparse matrix requiring specialized sparse GEMM kernels to achieve speedup — and these kernels often underperform dense GEMMs until >90% sparsity on GPUs. Removing 50% of neurons (structured) simply halves the matrix dimension, yielding dense smaller matrices that run at full hardware efficiency. The trade-off: unstructured pruning preserves more quality at the same compression ratio, but structured pruning gives predictable, hardware-friendly speedups."
    },
    {
      type: "mc",
      question: "When merging two LoRA adapters trained on different tasks, the simplest approach is weight-space averaging: $\\Delta W_{\\text{merged}} = \\alpha \\Delta W_A + (1 - \\alpha) \\Delta W_B$. The fundamental limitation of this approach is:",
      options: [
        "The merged weights are always larger than the originals",
        "Weight-space interpolation assumes a **linear loss landscape** between the two solutions — if the loss landscape has barriers between the adapter basins, the merged point may perform poorly on both tasks despite each adapter being individually excellent",
        "LoRA rank prevents merging",
        "The merged adapter has a different rank than the originals"
      ],
      correct: 1,
      explanation: "Linear interpolation in weight space only works well when the two solutions lie in the same loss basin (connected by a low-loss path). If training dynamics led the adapters to different basins, the midpoint can sit on a high-loss ridge. This connects to mode connectivity research: models trained from the same pre-trained checkpoint tend to be linearly connected in the loss landscape, but fine-tuning on very different tasks can break this. Techniques like TIES-Merging and DARE address this by resolving sign conflicts and pruning redundant parameters before merging."
    },
    {
      type: "mc",
      question: "A teacher model has 70B parameters and a student has 7B. After distillation, the student achieves 95% of the teacher's accuracy on benchmarks. The student's inference cost is approximately:",
      options: ["~10% of the teacher's cost — inference cost scales roughly linearly with parameter count (dominated by weight-loading for decode), so a 10x smaller model is ~10x cheaper regardless of how well it was trained", "95% of the teacher's cost", "50% of the teacher's cost due to shared architecture", "The same as the teacher's cost because it was trained by the teacher"],
      correct: 0,
      explanation: "Inference compute and memory scale with parameter count, not training method. The 7B student needs ~10x fewer FLOPs per forward pass, ~10x less memory for weights, and ~10x less KV-cache memory (assuming proportionally smaller hidden dimensions). Distillation improves the student's quality for its size class but doesn't change its computational cost. This is precisely the value proposition of distillation: getting a model that punches above its weight class computationally."
    },
    {
      type: "mc",
      question: "Neural Architecture Search (NAS) for LLMs differs from NAS for vision models primarily because:",
      options: ["LLMs don't have hyperparameters to search", "LLM architectures are already optimal", "The **training cost of each candidate architecture** is enormous (millions of dollars for a full pre-training run), making exhaustive search infeasible — NAS for LLMs typically relies on proxy metrics, scaling law predictions, or searching only within constrained subspaces like depth/width ratios", "NAS requires supervised labels which LLMs don't use"],
      correct: 2,
      explanation: "Vision NAS can evaluate a candidate architecture by training it to convergence in hours on a single GPU. LLM NAS cannot — training a 7B model costs ~\\$100K, making brute-force search over architecture variants prohibitive. Practical LLM NAS uses: (1) scaling law extrapolation from small proxy models, (2) zero-shot proxies based on gradient statistics, (3) constrained search over specific dimensions (depth, width, FFN ratio, head count) with other choices fixed. Results like the Chinchilla scaling laws are a form of two-variable NAS over model size and data quantity."
    },
    {
      type: "mc",
      question: "Layer pruning (removing entire transformer layers) from a 32-layer model shows an interesting pattern: removing layers from the **middle** of the network causes less degradation than removing early or late layers. This suggests:",
      options: ["Middle layers are redundant and should always be removed", "Layer ordering doesn't matter in transformers", "The model was trained incorrectly", "Middle layers exhibit more **representational redundancy** — adjacent middle layers compute similar transformations (high cosine similarity between layer inputs and outputs), while early layers build critical low-level features and late layers perform task-specific computation that is hard to compensate for"],
      correct: 3,
      explanation: "Empirical studies (e.g., ShortGPT, LaCo) show that the cosine similarity between a middle layer's input and output is often >0.99, meaning the layer makes only a small residual update. Early layers show lower similarity (larger transformations building representations), and final layers show specialized computation for the output distribution. This pattern motivates depth pruning strategies that remove middle layers and connects to the broader observation that over-parameterized networks have redundant capacity concentrated in particular regions."
    },
    {
      type: "mc",
      question: "When distilling an LLM for a specific task (e.g., code generation), which data strategy typically yields the best student?",
      options: [
        "Training on the same pre-training corpus as the teacher",
        "Using the teacher to generate **synthetic task-specific data** — having the teacher produce many high-quality examples with chain-of-thought reasoning, then training the student on this curated dataset, which is both task-relevant and teacher-distribution-aligned",
        "Using only human-labeled examples",
        "Random data from the internet"
      ],
      correct: 1,
      explanation: "Synthetic data generation from the teacher provides several advantages: (1) unlimited data at the cost of teacher inference, (2) data that is distributionally aligned with the teacher's capabilities, (3) ability to include reasoning traces that teach the student the process, not just the answer. This is the approach behind Phi-1 (textbook-quality synthetic data), Orca (explanation-augmented distillation), and WizardLM (evolved instructions). The key insight is that the teacher's ability to generate informative training data can be more valuable than its raw predictions."
    },
    {
      type: "mc",
      question: "Pruning at initialization (before training) versus pruning after training represents a fundamental debate. The **lottery ticket hypothesis** states:",
      options: ["Dense randomly-initialized networks contain **sparse subnetworks** (winning tickets) that, when trained in isolation from their original initialization, match the full dense network's accuracy — suggesting that most parameters exist to help find these subnetworks during training", "All neural network architectures are equally expressive", "Pruned networks always outperform dense networks", "Random pruning is as effective as informed pruning"],
      correct: 0,
      explanation: "Frankle & Carlin (2019) showed that within a randomly initialized dense network, there exist sparse subnetworks that can be trained from their original initialization to match the dense network's performance. Finding these \"winning tickets\" requires train-prune-reset cycles. For LLMs, directly finding winning tickets is computationally prohibitive, but the hypothesis motivates the intuition behind successful post-training pruning: well-trained networks contain many low-importance parameters whose removal minimally impacts function. Subsequent work (especially on supermasks) has refined these findings significantly."
    }
  ]
};

// ============================================================================
// D.1: Chain-of-Thought & Reasoning
// ============================================================================
export const cotAssessment = {
  id: "D.1-assess",
  sectionId: "D.1",
  title: "Assessment: Chain-of-Thought & Reasoning",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The **scratchpad hypothesis** for why chain-of-thought (CoT) improves LLM reasoning posits that:",
      options: ["CoT activates special reasoning circuits in the transformer", "CoT prompts contain the answer implicitly", "The intermediate tokens serve as **external working memory** — they allow the model to decompose multi-step problems into sequential single-step computations, each conditioned on the previous result, effectively extending the model's bounded computational depth per token", "Writing more tokens gives the model more time to think in wall-clock time"],
      correct: 2,
      explanation: "A transformer's forward pass has fixed computational depth (number of layers). For problems requiring more sequential computation steps than layers, the model cannot solve them in a single pass. CoT externalizes intermediate computations into generated tokens, which are then fed back as input. Each token generation step adds another forward pass worth of computation, conditioned on previous results. This effectively gives the model $O(T \\times L)$ sequential computation for $T$ output tokens and $L$ layers, rather than just $O(L)$."
    },
    {
      type: "mc",
      question: "Self-consistency (Wang et al., 2022) improves CoT reasoning by:",
      options: ["Training the model to be internally consistent", "Using a verifier model to score reasoning chains", "Asking the model to check its own work", "Sampling **multiple independent reasoning chains** (with temperature > 0), extracting the final answer from each, and returning the answer with the highest frequency — a majority voting scheme that marginalizes over diverse reasoning paths"],
      correct: 3,
      explanation: "Self-consistency samples $N$ reasoning chains (e.g., $N = 40$) with temperature sampling, then takes a majority vote on the final answers. The intuition: correct answers tend to be reachable via multiple valid reasoning paths, while incorrect answers typically result from specific errors that vary across samples. Formally, it approximates $\\arg\\max_a P(a \\mid \\text{question})$ by marginalizing over reasoning paths. This is a simple but powerful technique — it improved GSM8K accuracy from 56% (single CoT) to 74% on PaLM 540B."
    },
    {
      type: "mc",
      question: "Tree-of-Thought (ToT) extends chain-of-thought by:",
      options: [
        "Using tree-structured attention patterns",
        "Exploring a **tree of possible reasoning steps**, where the model evaluates and selects among multiple candidate next steps at each node — using search algorithms like BFS or DFS with self-evaluation to prune unpromising branches",
        "Generating answers in a tree-structured format",
        "Training on tree-structured data"
      ],
      correct: 1,
      explanation: "ToT structures reasoning as a search problem over a tree: each node represents a partial solution, each edge is a reasoning step, and the model both generates candidate steps and evaluates their promise. This enables backtracking (abandoning bad reasoning paths) and look-ahead (evaluating partial solutions before committing). For problems like creative writing or planning, ToT significantly outperforms linear CoT because it avoids the irrecoverable commitment problem — a bad step in linear CoT permanently derails the solution."
    },
    {
      type: "mc",
      question: "Process reward models (PRMs) provide feedback on each **step** of a reasoning chain, as opposed to outcome reward models (ORMs) that only score the final answer. The advantage of PRMs for math reasoning is:",
      options: ["PRMs provide **dense, step-level credit assignment** — they can identify exactly where a reasoning chain went wrong, enabling more efficient search over reasoning paths and better training signal for the policy model, rather than assigning a single reward to an entire multi-step derivation", "PRMs are cheaper to train", "PRMs always produce more accurate final answers", "PRMs don't require human annotations"],
      correct: 0,
      explanation: "ORMs suffer from the sparse reward problem: a 10-step derivation gets a single correct/incorrect label, giving no signal about which step caused a failure. PRMs score each step (e.g., step 3 introduced an algebraic error), enabling: (1) best-first search that prunes reasoning chains at the first erroneous step, (2) more efficient training since each step provides a supervision signal, and (3) interpretable failure analysis. Let's Verify Step by Step (Lightman et al., 2023) showed PRMs substantially outperform ORMs for math reasoning on MATH benchmark."
    },
    {
      type: "mc",
      question: "When does chain-of-thought reasoning **hurt** performance compared to direct answering?",
      options: ["Never — CoT always helps", "Only on mathematical problems", "On tasks requiring **fast pattern matching or factual recall** — CoT can introduce errors by overthinking simple retrievals, and the reasoning steps can lead the model astray on questions where the direct answer is already highly confident. CoT overhead also hurts latency-sensitive applications", "When the model is very large"],
      correct: 2,
      explanation: "CoT helps most on multi-step reasoning tasks (math, logic, multi-hop QA) where the problem genuinely requires sequential computation. It hurts on: (1) simple factual recall (\"What is the capital of France?\") where reasoning steps are unnecessary noise, (2) tasks where the model's first instinct is correct but overthinking introduces doubt, (3) pattern-matching tasks like sentiment analysis where reasoning can rationalize wrong answers. Empirically, CoT provides little benefit or even degrades performance for models below ~100B parameters on many tasks."
    },
    {
      type: "mc",
      question: "The debate over whether LLMs are \"truly reasoning\" versus \"pattern matching\" centers on:",
      options: ["Whether LLMs use symbolic or subsymbolic representations", "Whether LLMs were trained on reasoning data", "Whether LLMs use more parameters than the human brain", "Whether LLMs exhibit **systematic generalization** — solving novel problem compositions they haven't seen in training — or merely interpolate between training examples. Key evidence includes failure on problems with superficially similar structure but different solutions, and sensitivity to irrelevant surface features"],
      correct: 3,
      explanation: "The critical test is out-of-distribution generalization: can models solve problems requiring novel combinations of learned operations? Evidence for pattern matching: performance degrades on math problems with unusual number ranges, models are fooled by irrelevant information that shouldn't affect logical reasoning, and models struggle with problems structurally identical to training examples but with different surface forms. Evidence for reasoning: models can solve some novel compositions, show consistent performance on well-structured problems, and benefit from CoT in ways consistent with genuine computation. The truth likely involves both capabilities in different proportions."
    },
    {
      type: "mc",
      question: "A model is asked to solve $23 \\times 47$ using chain-of-thought. It writes: \"$23 \\times 47 = 23 \\times 40 + 23 \\times 7 = 920 + 161 = 1081$.\" This is a correct decomposition. The model likely learned this strategy because:",
      options: [
        "It has a built-in calculator module",
        "Its training data contains many examples of **distributive property decomposition** for multiplication — the model learned the pattern $a \\times (b + c) = ab + ac$ from seeing similar worked examples, but may fail on numbers requiring different decomposition strategies or carrying patterns it hasn't seen",
        "Transformers can natively perform multiplication",
        "The chain-of-thought prompt forced the correct algorithm"
      ],
      correct: 1,
      explanation: "The model applies a learned strategy (distributive property) that it has seen in training data. It can execute $23 \\times 40 = 920$ and $23 \\times 7 = 161$ as separate, simpler retrievals/computations. This illustrates both the power and limitation of CoT: it enables multi-step computation by chaining learned operations, but the model's \"arithmetic\" is fundamentally pattern-matching on trained examples. It may fail on numbers that trigger different carrying patterns or require strategies not well-represented in training data."
    },
    {
      type: "mc",
      question: "Few-shot CoT prompting provides exemplar reasoning chains in the prompt. Zero-shot CoT instead uses a trigger phrase like \"Let's think step by step.\" The fact that zero-shot CoT works at all suggests:",
      options: ["Instruction-tuned models have **internalized a general reasoning mode** that can be activated by trigger phrases — this mode was learned from the many step-by-step explanations in training data and RLHF, making the model predisposed to produce structured reasoning when prompted appropriately", "The model doesn't actually need examples", "Zero-shot CoT works only on easy problems", "The trigger phrase is mathematically optimal"],
      correct: 0,
      explanation: "Zero-shot CoT (Kojima et al., 2022) showed that \"Let's think step by step\" improves accuracy across diverse reasoning tasks without any exemplars. This works because: (1) pre-training on web data includes millions of step-by-step explanations, (2) instruction tuning explicitly rewards detailed reasoning, (3) RLHF preferences favor thorough explanations. The trigger phrase shifts the model's generation distribution toward the \"explanation\" mode it learned during training. This is evidence that reasoning capabilities are latent in the model and can be elicited by appropriate prompting."
    },
    {
      type: "mc",
      question: "When using self-consistency with $N = 40$ samples, the computational cost is 40x a single generation. Which statement about the cost-accuracy trade-off is correct?",
      options: ["Accuracy scales linearly with $N$ — 80 samples would be twice as good as 40", "All $N$ samples must use the same temperature", "Accuracy improvements follow **diminishing returns** — the marginal gain from each additional sample decreases logarithmically; gains from $N=1$ to $N=5$ are much larger than from $N=20$ to $N=40$, and the optimal $N$ depends on the difficulty distribution of the task", "Self-consistency with $N=2$ is always worse than single CoT"],
      correct: 2,
      explanation: "Majority voting accuracy follows a law of diminishing returns. For an easy problem where the model gets the right answer 80% of the time, even $N=5$ gives >95% accuracy (binomial probability). For a hard problem where the correct rate is 30%, even $N=100$ won't produce a majority-correct result. The marginal improvement from each additional sample decreases because the majority vote converges. The compute-optimal strategy is to allocate more samples to harder problems (adaptive compute) rather than using a fixed $N$ across all inputs."
    },
    {
      type: "mc",
      question: "A model answers a math problem correctly with CoT, but when the same problem is rephrased with a misleading \"common sense\" cue (e.g., adding irrelevant context suggesting a different answer), the model changes its answer. This demonstrates:",
      options: ["The model needs more training data", "CoT doesn't work for math problems", "The rephrased problem was actually harder", "**Reasoning fragility** — the model's chain-of-thought is influenced by surface-level features and social priors rather than following a purely logical computation; genuine reasoning should be invariant to irrelevant perturbations, but the model's reasoning is entangled with pattern-matching heuristics"],
      correct: 3,
      explanation: "This is a key piece of evidence in the reasoning-vs-pattern-matching debate. Studies show that adding irrelevant information (e.g., \"Alice has 5 apples\" in a problem that doesn't involve Alice) or framing a problem to suggest a common-but-wrong answer can cause models to produce incorrect reasoning chains that rationalize the wrong answer. True compositional reasoning should be invariant to such perturbations. This suggests the model's \"reasoning\" is partly a post-hoc rationalization process guided by distributional cues rather than a faithful logical computation."
    }
  ]
};

// ============================================================================
// D.2: Test-Time Compute
// ============================================================================
export const testTimeComputeAssessment = {
  id: "D.2-assess",
  sectionId: "D.2",
  title: "Assessment: Test-Time Compute",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The o1/o3 family of models use \"extended thinking\" at inference time. The core mechanism is:",
      options: [
        "Running the model on multiple GPUs simultaneously",
        "Generating a **long internal chain-of-thought** (potentially thousands of tokens) before producing the final answer — the model has been trained via RL to use this extended reasoning trace to explore, backtrack, and verify, trading compute time for accuracy on hard problems",
        "Fine-tuning the model on each new question",
        "Searching over a database of pre-computed answers"
      ],
      correct: 1,
      explanation: "o1-style models generate an extended hidden reasoning trace before answering. The model was trained with reinforcement learning to use this trace productively: exploring multiple approaches, catching and correcting errors, and verifying intermediate results. This trades inference compute (generating many tokens) for accuracy. The reasoning trace can be thousands of tokens long for hard math/coding problems, but the model learns when to think briefly vs. extensively based on problem difficulty."
    },
    {
      type: "mc",
      question: "Best-of-N sampling generates $N$ complete responses and selects the best one using a reward model or verifier. Compared to self-consistency (majority voting), best-of-N:",
      options: ["Can select based on **response quality rather than just answer frequency** — it can prefer a well-reasoned minority answer over a frequently-repeated shallow answer, but its effectiveness is bounded by the verifier's accuracy and the coverage of $N$ samples", "Always produces better results because it uses a reward model", "Uses less compute since it only generates one response", "Is identical to self-consistency"],
      correct: 0,
      explanation: "Self-consistency selects by answer frequency, implicitly assuming that correct reasoning paths outnumber incorrect ones. Best-of-N uses a learned verifier to evaluate response quality, which can capture reasoning quality beyond just the final answer. However, best-of-N is limited by: (1) verifier accuracy — a miscalibrated verifier may prefer confident-sounding wrong answers, (2) sample coverage — $N$ samples may not include a correct response for very hard problems, and (3) reward hacking if the verifier has exploitable biases."
    },
    {
      type: "mc",
      question: "Applying Monte Carlo Tree Search (MCTS) to LLM reasoning faces a fundamental challenge that doesn't exist in game-playing (e.g., AlphaGo):",
      options: ["LLMs can't play games", "MCTS requires a board representation", "The **branching factor is enormous** — at each token (or reasoning step), there are thousands of possible continuations, making exhaustive tree search intractable; games like Go have ~250 legal moves per position, while an LLM's vocabulary is 32K-128K tokens, and even at the reasoning-step level, the space of possible next thoughts is vast", "LLMs are too slow for real-time search"],
      correct: 2,
      explanation: "MCTS succeeds in Go because: (1) the branching factor (~250) is manageable, (2) a value network provides reliable position evaluations, and (3) the game has a clear terminal reward. For LLM reasoning: (1) the branching factor at the token level is ~32K-128K, requiring either very aggressive pruning or operating at the \"reasoning step\" level, (2) evaluating partial reasoning chains is much harder than evaluating board positions, and (3) defining what constitutes a \"move\" in reasoning is ambiguous. Approaches like LATS and RAP address these by chunking reasoning into coarse steps and using the LLM itself as a value function."
    },
    {
      type: "mc",
      question: "Compute-optimal inference allocation suggests that the optimal amount of test-time compute should depend on:",
      options: ["Only the model size — larger models should always think longer", "The time of day", "The user's subscription tier", "The **difficulty of the specific input** — easy questions should use minimal test-time compute (direct answering), while hard questions benefit from extended reasoning, search, or multiple samples; spending equal compute on all inputs is wasteful"],
      correct: 3,
      explanation: "Inference scaling research shows that the value of additional test-time compute follows a difficulty-dependent curve. For easy problems, the model's first answer is usually correct — additional compute is wasted. For hard problems, extended thinking, multiple samples, or search can dramatically improve accuracy. The optimal strategy adaptively allocates compute: a difficulty classifier or confidence estimator decides how much inference-time computation to invest per query. This mirrors the human intuition of \"thinking harder\" about harder problems."
    },
    {
      type: "mc",
      question: "Inference scaling laws describe how performance improves with test-time compute. Compared to training scaling laws (Chinchilla), inference scaling:",
      options: [
        "Shows the same power-law exponent",
        "Can be **more cost-efficient for hard problems** — a smaller model with 100x inference compute can sometimes match a 10x larger model with 1x inference compute, especially on reasoning tasks where search and verification compound; but for easy tasks, the larger model with direct answering is more efficient",
        "Never matches training-time scaling",
        "Follows an exponential rather than power law"
      ],
      correct: 1,
      explanation: "Research from DeepMind and OpenAI shows that for certain problem types (particularly verifiable reasoning), investing in test-time compute can substitute for model size. A 7B model with best-of-256 sampling can match a 70B model's accuracy on math problems. However, the trade-off depends heavily on task type: for knowledge-intensive tasks, the information must be in the model's parameters, and no amount of inference compute can compensate. The optimal compute allocation between training and inference depends on deployment volume — high-volume tasks favor larger trained models, while rare hard queries favor inference-time scaling."
    },
    {
      type: "mc",
      question: "A system uses best-of-N with a reward model to solve math problems. With $N = 1$, accuracy is 40%. With $N = 100$, accuracy reaches 75%. Doubling to $N = 200$ would most likely yield:",
      options: ["~78-80% — improvements follow **logarithmic scaling** with $N$; each doubling of $N$ provides diminishing marginal gains because the probability of covering the correct solution approaches a ceiling set by the model's coverage distribution", "~100% accuracy — double the samples, double the improvement", "~75% — no improvement beyond 100 samples", "~50% — more samples confuse the reward model"],
      correct: 0,
      explanation: "Best-of-N accuracy scales as $1 - (1 - p)^N$ where $p$ is the per-sample probability of a correct-and-top-ranked response. For large $N$, this saturates. If the model has a 1% chance of producing the correct answer per sample, $N=100$ gives $\\sim 63\\%$ coverage, and $N=200$ gives $\\sim 87\\%$. But with a reward model selector, the binding constraint is often the reward model's ability to distinguish correct from incorrect solutions, not just coverage. Empirically, accuracy scales roughly as $a + b \\log(N)$ in the relevant range."
    },
    {
      type: "mc",
      question: "Process reward models (PRMs) can be used for step-level beam search during test-time reasoning. In this approach, the search:",
      options: ["Generates all tokens simultaneously", "Only evaluates the final answer", "Maintains a beam of $B$ partial reasoning chains, scoring each at every reasoning step using the PRM and pruning low-scoring branches — this focuses compute on the most promising reasoning paths rather than committing to a single chain or running independent samples", "Uses the PRM to generate tokens directly"],
      correct: 2,
      explanation: "Step-level beam search with PRMs is a structured test-time compute strategy: at each reasoning step, $B$ candidate continuations are generated, the PRM scores each partial chain, and only the top-$B$ survive. This is more efficient than best-of-N (which runs $N$ independent chains to completion) because compute is concentrated on promising paths early. The PRM acts as a value function estimating the probability of reaching a correct answer from each partial state. This is the approach advocated by Let's Verify Step by Step and subsequent work on verifier-guided search."
    },
    {
      type: "mc",
      question: "A key distinction between o1-style extended thinking and standard chain-of-thought prompting is:",
      options: ["Extended thinking uses a different model architecture", "Extended thinking only works in English", "Standard CoT produces longer reasoning traces", "Extended thinking models are trained with **RL to optimize the reasoning process itself** — the model learns when to explore alternatives, backtrack from dead ends, and verify results, rather than producing a single linear chain; standard CoT relies on the model's pre-trained generation distribution without explicit optimization of the reasoning strategy"],
      correct: 3,
      explanation: "Standard CoT relies on the model's pre-trained distribution over explanation-like text — it produces plausible-looking reasoning but has no explicit incentive to reason correctly or to self-correct. o1-style models are trained with RL where the reward signal is answer correctness, so the model learns to use its reasoning trace strategically: trying multiple approaches, identifying and recovering from errors, and allocating reasoning effort proportional to problem difficulty. The RL training fundamentally changes the model's relationship to its own reasoning output from \"generating plausible text\" to \"using text as a computational medium.\""
    },
    {
      type: "mc",
      question: "Parallel test-time compute (generating $N$ independent samples) vs sequential test-time compute (one long reasoning chain with $N$ times the tokens) differ in that:",
      options: [
        "They always produce identical results",
        "Parallel sampling provides **diversity** (exploring independent starting points) but no depth, while sequential reasoning provides **depth** (building on previous reasoning) but risks compounding errors — the optimal balance depends on whether the task requires exploration of the solution space or deep sequential computation",
        "Sequential is always better because it generates more tokens",
        "Parallel is always better because it uses more GPUs"
      ],
      correct: 1,
      explanation: "This is a fundamental axis in test-time compute allocation. Parallel sampling (best-of-N, self-consistency) explores breadth: independent samples cover more of the solution space but each sample is limited in depth. Sequential reasoning (extended CoT, iterative refinement) explores depth: building elaborate reasoning chains but committed to a single trajectory. For problems requiring a creative insight (exploration), parallel sampling is better. For problems requiring long derivations (depth), sequential is better. Optimal strategies often combine both: parallel samples of extended reasoning chains."
    },
    {
      type: "mc",
      question: "A company deploys a reasoning model and observes that 80% of user queries are simple (answered correctly with 1 inference step) while 20% are hard (requiring 50 inference steps). If they allocate the same 50-step budget to all queries, the wasted compute fraction is approximately:",
      options: ["About 72% — the 80% of easy queries waste 49 of their 50 allocated steps, so $\\frac{0.8 \\times 49}{0.8 \\times 50 + 0.2 \\times 50} = \\frac{39.2}{50} \\approx 78\\%$ of total compute is wasted on unnecessary reasoning for easy queries", "0% — more compute never hurts", "About 20% — only the hard queries waste compute", "About 50%"],
      correct: 0,
      explanation: "With uniform 50-step allocation: easy queries use 1 useful step + 49 wasted = 0.8 * 49 = 39.2 wasted step-equivalents. Hard queries use all 50 steps productively. Total compute = 50 steps * 100% of queries = 50 units. Wasted = 39.2 units. Waste fraction = 39.2/50 = 78.4%. Adaptive allocation (1 step for easy, 50 for hard) uses 0.8*1 + 0.2*50 = 10.8 units — a 4.6x compute reduction. This illustrates why difficulty-adaptive inference allocation is crucial for cost-effective deployment of reasoning models."
    }
  ]
};

// ============================================================================
// D.3: Tool Use & Function Calling
// ============================================================================
export const toolUseAssessment = {
  id: "D.3-assess",
  sectionId: "D.3",
  title: "Assessment: Tool Use & Function Calling",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Function calling in LLMs is typically trained by:",
      options: ["Hard-coding specific API calls into the model's architecture", "Giving the model direct access to execute code during training", "Fine-tuning on datasets of (user query, function call, function result, final response) sequences — the model learns to emit **structured function call tokens** in a special format, which are intercepted by the serving system, executed, and the results injected back into the context", "Training a separate classifier that maps queries to functions"],
      correct: 2,
      explanation: "Function calling training involves: (1) curating datasets where the correct response involves calling specific functions with appropriate arguments, (2) defining a structured output format (often JSON) that the model generates inline, (3) teaching the model when to call functions vs. answer directly. The serving system parses these structured outputs, executes the function, and appends the result to the context for the model to incorporate. This is how models like GPT-4 and Claude handle tool use — the function call is part of the model's text generation, not a separate system."
    },
    {
      type: "mc",
      question: "In a RAG (Retrieval-Augmented Generation) system, the chunking strategy — how documents are split into retrievable units — critically affects quality. Which statement is correct?",
      options: ["Larger chunks are always better because they provide more context", "All documents should be stored as single chunks", "Chunking doesn't matter if the embedding model is good enough", "Chunk size involves a **precision-recall trade-off**: smaller chunks (128-256 tokens) yield precise retrieval but may miss surrounding context, while larger chunks (512-1024 tokens) capture more context but may dilute relevance with irrelevant text; overlapping chunks and hierarchical strategies can mitigate this"],
      correct: 3,
      explanation: "Too-small chunks: high retrieval precision (the chunk matches the query well) but insufficient context for the LLM to synthesize an answer. Too-large chunks: the relevant passage is buried in irrelevant text, diluting the embedding representation and the LLM's ability to extract the answer. Strategies to mitigate: (1) overlapping windows (e.g., 512 tokens with 128 overlap), (2) hierarchical chunking (retrieve small chunks but expand to parent chunks for context), (3) semantic chunking (split at topic boundaries rather than fixed token counts). The optimal strategy is domain-dependent."
    },
    {
      type: "mc",
      question: "A RAG pipeline uses embedding similarity for initial retrieval (top-100), then a cross-encoder reranker to select the final top-5 passages. The reranker improves results because:",
      options: [
        "It uses a larger embedding dimension",
        "Cross-encoders process the **query and passage jointly** through full attention, capturing fine-grained semantic interactions — unlike bi-encoders that independently embed query and passage into separate vectors and compare with cosine similarity, missing token-level cross-attention between query and document",
        "It has access to more training data",
        "It runs faster than the initial retriever"
      ],
      correct: 1,
      explanation: "Bi-encoders (used for initial retrieval) embed query $q$ and passage $p$ independently: $\\text{sim}(e(q), e(p))$. This enables fast ANN search over millions of passages but misses cross-attention between query and passage tokens. Cross-encoder rerankers process $[q; p]$ jointly through a transformer, enabling rich token-level interactions: the model can attend from query tokens to passage tokens and vice versa. This is dramatically more expressive but expensive ($O(N)$ forward passes for $N$ candidates), hence the two-stage retrieve-then-rerank pipeline."
    },
    {
      type: "mc",
      question: "Multi-step retrieval (also called iterative or multi-hop RAG) is needed when:",
      options: ["The answer requires **synthesizing information from multiple documents** or requires a chain of lookups — e.g., \"What is the GDP of the country where the inventor of the transformer architecture was born?\" requires first retrieving who invented transformers, then their birthplace, then that country's GDP", "The document collection is very large", "The embedding model has limited context length", "The user asks multiple questions in one turn"],
      correct: 0,
      explanation: "Single-shot retrieval fails on multi-hop questions because the initial query doesn't contain the intermediate information needed to formulate the final retrieval. Multi-step RAG decomposes the question: (1) retrieve \"transformer architecture inventor\" -> Vaswani et al., (2) retrieve birthplace information -> identify country, (3) retrieve GDP data. Each retrieval step uses information from prior steps to formulate the next query. Architectures like IRCoT (Interleaving Retrieval with CoT) and Self-RAG automate this decomposition."
    },
    {
      type: "mc",
      question: "When an LLM generates a function call that fails (e.g., API returns an error), the ideal error recovery behavior is:",
      options: ["Immediately returning the error message to the user", "Ignoring the error and generating an answer without the tool result", "Analyzing the error, **reformulating the function call** (e.g., correcting parameter types, trying alternative APIs, or decomposing into simpler sub-calls), and retrying — this error recovery loop should be bounded to prevent infinite retries, with graceful degradation when recovery fails", "Calling the same function again with identical parameters"],
      correct: 2,
      explanation: "Robust tool use requires error handling as a core capability, not an afterthought. The model should: (1) parse the error message to diagnose the failure (auth error? malformed input? rate limit?), (2) determine if the call can be retried with corrections or if an alternative approach is needed, (3) attempt recovery with a bounded retry count, (4) gracefully inform the user if recovery fails, explaining what was attempted. Training for error recovery involves including error scenarios in the fine-tuning data — models that only see successful tool calls perform poorly when tools fail."
    },
    {
      type: "mc",
      question: "Code generation can be viewed as a form of tool use where the \"tool\" is a code interpreter. The key advantage of code execution over pure text reasoning for tasks like data analysis is:",
      options: ["Code is always shorter than natural language", "Code generation doesn't require pre-training", "Code runs faster than the LLM can think", "Code execution provides **exact computation and verification** — the model generates a hypothesis as code, the interpreter executes it on real data and returns exact results, eliminating hallucinated computations; natural language arithmetic and data manipulation are unreliable at scale"],
      correct: 3,
      explanation: "When asked \"What is the mean of these 1000 data points?\", an LLM reasoning in natural language will hallucinate a plausible-sounding number. Code generation + execution computes the exact answer. This is the core insight behind tools like Code Interpreter and open-source equivalents: the LLM's strength is understanding the user's intent and translating it to code; the interpreter's strength is exact, verifiable computation. The model can also inspect intermediate results, detect errors, and iterate — forming a generate-execute-debug loop."
    },
    {
      type: "mc",
      question: "Embedding models used for RAG retrieval are typically trained with **contrastive learning**. The training objective is:",
      options: [
        "Predicting the next word in a document",
        "Learning representations where **semantically similar query-passage pairs have high cosine similarity** while dissimilar pairs have low similarity — using in-batch negatives and hard negatives to push apart embeddings of superficially similar but semantically different texts",
        "Minimizing the reconstruction error of documents",
        "Classifying documents into predefined categories"
      ],
      correct: 1,
      explanation: "Contrastive training (e.g., DPR, E5, GTE) uses pairs $(q_i, p_i^+)$ of queries and relevant passages. The loss pulls together $e(q_i)$ and $e(p_i^+)$ while pushing apart $e(q_i)$ and $e(p_j^+)$ for $j \\neq i$ (in-batch negatives). Hard negatives — passages that are superficially similar to the query but not actually relevant (e.g., same topic but different answer) — are critical for learning fine-grained distinctions. Modern embedding models (GTE, E5-Mistral) build on decoder LLMs fine-tuned with contrastive objectives, achieving strong retrieval by leveraging the LLM's pre-trained language understanding."
    },
    {
      type: "mc",
      question: "A naive RAG system retrieves the top-5 most similar passages and concatenates them into the LLM's context. A more sophisticated approach uses the **lost-in-the-middle** finding to:",
      options: ["Place the most relevant passages at the **beginning and end** of the context rather than ranked order — LLMs attend more strongly to the start and end of long contexts, so relevant information in the middle of concatenated passages is more likely to be ignored", "Only retrieve 1 passage to avoid confusion", "Randomize the passage order", "Retrieve from the middle of documents only"],
      correct: 0,
      explanation: "Liu et al. (2023) showed that LLMs' ability to use information from retrieved passages follows a U-shaped curve: performance is highest when the relevant passage is at the beginning or end of the context, and lowest when it's in the middle. This means naive \"rank by relevance\" ordering (most relevant first, then decreasing) places important passages in the problematic middle positions. Solutions include: reordering passages to place the most relevant at the boundaries, using reciprocal rank fusion, or citing specific passages by reference to force the model's attention."
    },
    {
      type: "mc",
      question: "When deciding whether to call a tool or answer directly, the model must assess its own **epistemic uncertainty**. Which approach to this calibration challenge is most effective?",
      options: ["Always calling tools to be safe", "Using a hardcoded keyword list to trigger tool calls", "Training on data that explicitly includes both tool-calling and direct-answering examples for similar queries, with the decision boundary based on **whether the model's knowledge is sufficient and current** — the model learns to recognize queries where its parametric knowledge may be outdated, incomplete, or unreliable", "Letting the user decide when to use tools"],
      correct: 2,
      explanation: "Effective tool-use routing requires the model to know what it doesn't know. Training strategies include: (1) self-knowledge probing — including examples where the model should say \"I need to look this up\" for recent events or precise numbers, (2) confidence calibration — training the model to express appropriate uncertainty, (3) curriculum with both tool-assisted and direct answers for similar queries, so the model learns the decision boundary. Models like Toolformer automate this by learning to insert API calls only when they improve prediction quality, using a self-supervised filtering criterion."
    },
    {
      type: "mc",
      question: "A RAG system's retrieval component returns passages with relevance scores. At what point does adding more retrieved passages to the LLM context typically **hurt** performance?",
      options: ["Never — more context is always better", "Only when the context window is exceeded", "After exactly 3 passages", "When additional passages have **low relevance scores** — they introduce noise and irrelevant information that can distract the LLM, cause hallucinated synthesis of contradictory sources, or push relevant information into the less-attended middle of the context window. Optimal passage count depends on query type and model capability"],
      correct: 3,
      explanation: "Adding passages with relevance scores below a task-specific threshold introduces noise. The LLM may: (1) hallucinate a synthesis of contradictory information from relevant and irrelevant passages, (2) get distracted by plausible-sounding but incorrect passages (especially problematic for factual QA), (3) suffer from the lost-in-the-middle effect as the context grows. Empirically, accuracy often peaks at 3-5 highly relevant passages and degrades with more. Adaptive retrieval strategies set a relevance threshold or use the LLM's own confidence to decide when enough context has been retrieved."
    }
  ]
};

// ============================================================================
// D.4: Agentic Systems
// ============================================================================
export const agenticAssessment = {
  id: "D.4-assess",
  sectionId: "D.4",
  title: "Assessment: Agentic Systems",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The ReAct (Reasoning + Acting) framework structures agent behavior as an interleaved sequence of:",
      options: [
        "Training and inference steps",
        "**Thought → Action → Observation** cycles — the agent reasons about what to do (Thought), executes a tool call or action (Action), observes the result (Observation), then reasons again based on the new information, continuing until the task is complete or a stopping criterion is met",
        "Encoding and decoding steps",
        "Forward and backward passes"
      ],
      correct: 1,
      explanation: "ReAct (Yao et al., 2022) combines chain-of-thought reasoning with tool use in a structured loop. The Thought step allows the agent to plan, interpret observations, and decide what to do next. The Action step interfaces with external tools (search, calculator, API calls). The Observation step grounds the agent in real-world feedback. This structure outperforms both pure reasoning (no grounding) and pure acting (no planning) because reasoning guides action selection while observations correct reasoning errors."
    },
    {
      type: "mc",
      question: "If an agent makes correct decisions with 95% accuracy at each step, its probability of completing a 10-step task correctly (assuming independent steps) is approximately:",
      options: ["~60% — computed as $0.95^{10} \\approx 0.599$; this **compounding error problem** means even highly reliable per-step performance degrades rapidly over multi-step tasks", "95% — the per-step accuracy", "~90% — nearly as good as per-step accuracy", "50% — it's essentially random over 10 steps"],
      correct: 0,
      explanation: "$0.95^{10} = 0.5987$. This is the compounding error problem: each step's small error probability multiplies. At 20 steps: $0.95^{20} = 0.358$. At 50 steps: $0.95^{50} = 0.077$. This has profound implications for agent design: (1) error recovery mechanisms are essential — agents must detect and recover from mistakes, not just avoid them, (2) reducing task horizon (fewer steps) is as valuable as improving per-step accuracy, (3) verification at intermediate checkpoints can reset the error accumulation by catching and correcting mistakes before they compound."
    },
    {
      type: "mc",
      question: "To achieve 90% task success rate on a 10-step task, the required per-step accuracy is:",
      options: ["90%", "95%", "~99% — since $x^{10} = 0.9$ gives $x = 0.9^{1/10} \\approx 0.9895$; this illustrates why agentic systems need near-perfect per-step reliability for multi-step tasks", "99.9%"],
      correct: 2,
      explanation: "Solving $x^{10} = 0.9$: $x = 0.9^{0.1} = e^{0.1 \\ln(0.9)} = e^{-0.01053} \\approx 0.9895$. So you need ~98.95% per-step accuracy for 90% task completion. For a 20-step task: $x = 0.9^{0.05} \\approx 0.9947$ (99.47% per step). This quantifies why long-horizon agentic tasks are so challenging: the reliability requirement per step grows exponentially with task length. It also explains why current agents work best on short, well-defined tasks and struggle with open-ended, multi-step workflows."
    },
    {
      type: "mc",
      question: "Agent memory architectures typically distinguish between short-term and long-term memory. In LLM-based agents, these correspond to:",
      options: ["GPU memory vs CPU memory", "Attention keys vs attention values", "Training data vs inference data", "The **context window** (short-term: recent observations, current plan, working state) and **external storage** (long-term: vector databases of past experiences, summarized episode histories, learned procedures) — the context window is limited and ephemeral, while external storage persists across sessions"],
      correct: 3,
      explanation: "Short-term memory is the context window: it contains the current task description, recent Thought-Action-Observation steps, and immediate working state. It's limited by the model's context length and is lost between sessions. Long-term memory uses external storage: (1) episodic memory — past task executions stored as retrievable summaries, (2) semantic memory — facts and procedures in a vector database, (3) procedural memory — learned tool-use patterns. The retrieval system selectively loads relevant long-term memories into the context window, analogous to human memory retrieval."
    },
    {
      type: "mc",
      question: "Multi-agent debate systems use multiple LLM instances that argue different positions and critique each other's reasoning. The primary benefit over a single model is:",
      options: [
        "Multi-agent systems are cheaper to run",
        "Adversarial interaction can surface **errors and unstated assumptions** that a single model would not catch — each agent's critique forces others to justify their reasoning, improving the quality of the final consensus through dialectical refinement",
        "Multiple models always agree on the correct answer",
        "Each agent can use a different programming language"
      ],
      correct: 1,
      explanation: "A single model generating a response has no external check on its reasoning. In multi-agent debate: (1) a critic agent identifies logical gaps, unsupported claims, or errors, (2) the original agent must defend or revise its reasoning, (3) this iterative process often converges to higher-quality outputs. However, limitations exist: LLM agents may be too deferential (agreeing with critiques even when the original reasoning was correct), or they may share systematic biases (multiple instances of the same model make the same mistakes). Diversity of model or prompting strategy helps mitigate this."
    },
    {
      type: "mc",
      question: "Evaluating agentic systems is fundamentally harder than evaluating standard LLM outputs because:",
      options: ["Agent evaluation requires assessing **multi-step trajectories** in environments with stochastic outcomes — the same agent may take different valid paths to the same goal, intermediate states are hard to evaluate, success criteria can be ambiguous, and the environment may change between evaluation runs", "Agents produce longer outputs", "Agents only work in production, not in benchmarks", "Agent outputs cannot be compared to ground truth"],
      correct: 0,
      explanation: "Standard LLM evaluation compares a single output to a reference. Agent evaluation must handle: (1) multiple valid solution paths (did the agent take a suboptimal but correct path?), (2) partial credit for partially completed tasks, (3) environment stochasticity (web pages change, APIs have different latencies), (4) difficulty of attributing failure to specific steps vs. the overall strategy, (5) cost of evaluation (each agent run may take minutes and cost dollars in API calls). Benchmarks like SWE-bench, WebArena, and GAIA attempt standardized agent evaluation but each captures only a narrow slice of real-world agent capabilities."
    },
    {
      type: "mc",
      question: "An agent attempts to book a flight: it searches for flights (step 1), selects one (step 2), fills in passenger details (step 3), but enters the wrong date (step 4), and proceeds to payment (step 5). The most robust agent architecture would:",
      options: ["Complete the booking and hope the date is correct", "Start over from step 1 after every step", "Include a **verification step before irreversible actions** — after filling in details (step 3-4), verify the entered information against the original request before proceeding to payment; irreversible actions like payment should always be preceded by explicit confirmation steps that check for accumulated errors", "Use a separate model for each step"],
      correct: 2,
      explanation: "Robust agent design distinguishes between reversible and irreversible actions. Browsing and form-filling are reversible (can go back and correct). Payment is irreversible. The agent should: (1) maintain an explicit plan with checkpoints, (2) verify accumulated state against the original goal before irreversible actions, (3) use self-reflection (\"Does this match what the user asked for?\") at critical junctions. This verification overhead reduces throughput but dramatically reduces costly errors. The principle mirrors software engineering: validate inputs before committing transactions."
    },
    {
      type: "mc",
      question: "The \"inner monologue\" approach to agent reasoning, where the agent maintains an explicit running commentary of its state and plans, helps primarily by:",
      options: ["Making the agent's responses more verbose for users", "Reducing the number of API calls", "Improving the agent's language fluency", "Keeping the agent's **goals, current state, and plan in the active context window** — without explicit tracking, the agent can lose track of what it's doing over long trajectories as earlier context scrolls out of the attention window; the inner monologue acts as a self-maintained working memory"],
      correct: 3,
      explanation: "Over a 50-step trajectory, the early steps (including the original task description) may be hundreds of tokens back in the context, receiving diminished attention. Inner monologue explicitly restates: \"My goal is X. I have completed A, B, C. Current state is Y. Next I need to do Z.\" This keeps critical information in the recent context window where the model attends most strongly. It also serves as a form of self-verification — stating the current plan explicitly can reveal inconsistencies. The cost is additional token generation, but this is usually worthwhile for complex tasks."
    },
    {
      type: "mc",
      question: "When deploying an LLM agent with access to real-world tools (email, file system, web browsing), the primary safety concern is:",
      options: [
        "The agent might generate offensive text",
        "**Unintended or adversarial actions with real-world consequences** — the agent might misinterpret instructions and delete files, send unintended emails, or be manipulated by adversarial content in retrieved web pages (indirect prompt injection) to take actions the user never intended",
        "The agent will use too much compute",
        "The agent's actions are too slow"
      ],
      correct: 1,
      explanation: "Unlike chat-only LLMs where the worst case is bad text, tool-using agents can cause real harm: deleting data, sending emails as the user, making purchases, or modifying code in production. Key risks: (1) misinterpreted instructions amplified by tool capabilities, (2) indirect prompt injection — malicious content in web pages or documents instructing the agent to take harmful actions, (3) compounding errors leading to unintended state changes. Mitigations include: sandboxing, requiring human approval for irreversible actions, input/output filtering, and principle of least privilege for tool access."
    },
    {
      type: "mc",
      question: "An agent system decomposes a complex task into subtasks handled by specialized sub-agents (e.g., a \"researcher\" agent, a \"coder\" agent, a \"reviewer\" agent). The orchestration challenge is:",
      options: ["Ensuring correct **information flow and task dependency management** — the orchestrator must track which sub-tasks are complete, pass the right context to each sub-agent (avoiding both information overload and critical omissions), handle sub-agent failures gracefully, and merge potentially conflicting outputs from parallel sub-tasks", "Sub-agents are more expensive than a single agent", "Sub-agents can't communicate with each other", "Each sub-agent needs a separate GPU"],
      correct: 0,
      explanation: "Multi-agent orchestration is essentially a workflow management problem with LLM-specific challenges: (1) context management — each sub-agent needs enough context to do its job but not so much that it's distracted, (2) dependency tracking — the coder agent can't start until the researcher agent provides specifications, (3) failure handling — if the researcher returns low-quality results, the orchestrator must detect this and retry or adjust, (4) output merging — when the reviewer critiques the coder's output, the orchestrator must route the feedback and manage the revision loop. Frameworks like LangGraph and AutoGen provide abstractions for this but the fundamental complexity remains."
    }
  ]
};
