// Assessment: Distributed Training Infrastructure (Section 1.6)
// 10 MC questions, no info steps. Pure assessment module.

export const distributedTrainingAssessment = {
  id: "1.6-assess",
  sectionId: "1.6",
  title: "Assessment: Distributed Training Infrastructure",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In **Distributed Data Parallel (DDP)** training, each GPU holds a full copy of the model. After the backward pass, gradients are synchronized across GPUs using:",
      options: [
        "A parameter server architecture that collects gradients from every GPU, computes the global average on a dedicated node, and then redistributes the averaged result back to each GPU",
        "An **all-reduce** operation that efficiently computes the sum (or average) of gradients across all GPUs so every replica ends up with identical gradients — typically implemented as a ring all-reduce to minimize communication overhead",
        "Each GPU sends its full gradient tensor to GPU 0, which performs the averaging computation and then broadcasts the result back to all other GPUs in a hub-and-spoke pattern",
        "Gradients are not synchronized at all — each GPU trains on its own data shard independently, and the parameter divergence is reconciled only at periodic checkpoint intervals"
      ],
      correct: 1,
      explanation: "DDP uses all-reduce (typically ring all-reduce or tree all-reduce via NCCL) to synchronize gradients. In ring all-reduce, each GPU sends a chunk of its gradient to its neighbor, and after $2(N-1)$ steps (N = number of GPUs), all GPUs have the complete averaged gradient. The communication volume per GPU is $2 \\cdot (N-1)/N \\cdot |\\text{params}|$, which approaches $2|\\text{params}|$ as $N$ grows — nearly independent of GPU count. This is far more efficient than the naive reduce-broadcast approach via a parameter server, which creates a bottleneck at the central node."
    },
    {
      type: "mc",
      question: "**Tensor parallelism** and **pipeline parallelism** split the model across GPUs in fundamentally different ways. Tensor parallelism:",
      options: ["Splits individual layers (e.g., partitioning weight matrices column-wise or row-wise) across GPUs so each GPU computes a portion of every layer's output, requiring intra-layer communication at each forward and backward step", "Assigns different training examples to different GPUs while each GPU holds a full model replica, synchronizing gradients via all-reduce after each backward pass", "Assigns entire layers to different GPUs in sequence so each GPU runs a contiguous subset of the network's depth, passing activations between stages", "Replicates the model on every GPU and uses an asynchronous parameter server to aggregate updates, trading strict consistency for throughput"],
      correct: 0,
      explanation: "Tensor parallelism (Megatron-LM style) partitions weight matrices within a layer. For example, a linear layer $Y = XW$ can be split column-wise: $W = [W_1 | W_2]$, with each GPU computing $XW_i$. This requires an all-reduce after each layer to combine partial results. Pipeline parallelism, by contrast, assigns whole layers to different GPUs — GPU 0 runs layers 1-10, GPU 1 runs layers 11-20, etc. Tensor parallelism has higher communication frequency (every layer) but lower latency per communication; pipeline parallelism has lower communication frequency but suffers from the bubble problem."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 1** shards the **optimizer states** across GPUs while each GPU still holds a full copy of parameters and gradients. For a model with $\\Psi$ parameters using Adam in mixed precision, Stage 1 reduces per-GPU optimizer memory from $12\\Psi$ bytes to approximately:",
      options: ["$12\\Psi$ bytes — no savings, because Stage 1 only partitions gradients, not optimizer states, so each GPU still holds all of Adam's $m$, $v$, and FP32 master weights", "$4\\Psi$ bytes — only the FP16 parameters remain, because Stage 1 offloads all optimizer state and gradients to CPU RAM via asynchronous memory transfers", "$12\\Psi / N$ bytes, where $N$ is the number of GPUs — each GPU stores only $1/N$ of Adam's first moment ($m$), second moment ($v$), and FP32 master weights", "$2\\Psi$ bytes — only the FP16 gradients, because Stage 1 quantizes all optimizer states to INT8, reducing the $12\\Psi$ overhead to a negligible rounding buffer"],
      correct: 2,
      explanation: "Adam requires per-parameter state: FP32 master weights (4 bytes), FP32 first moment $m$ (4 bytes), and FP32 second moment $v$ (4 bytes) = 12 bytes per parameter. ZeRO Stage 1 partitions these 12$\\Psi$ bytes across $N$ GPUs, so each GPU stores $12\\Psi/N$ bytes of optimizer state. The FP16 parameters ($2\\Psi$) and FP16 gradients ($2\\Psi$) remain fully replicated. For a 7B model on 8 GPUs: optimizer memory drops from 84 GB to ~10.5 GB per GPU, while parameter and gradient memory remain at 14 GB + 14 GB."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 3** (or equivalently, **FSDP** — Fully Sharded Data Parallel) shards parameters, gradients, AND optimizer states. The key runtime overhead compared to DDP is:",
      options: ["No additional overhead compared to DDP — it is strictly better in all respects because the sharded communication patterns have the same total bandwidth cost as the replicated all-reduce", "It cannot overlap communication with computation because each layer's full parameters must be reconstructed and verified before any forward-pass arithmetic can begin on that layer's inputs", "It requires twice as many GPUs as DDP to achieve the same training throughput, because half the GPUs are dedicated to managing the sharded parameter storage and communication scheduling", "All-gather operations to reconstruct full parameter tensors before each forward/backward computation, and reduce-scatter operations to distribute gradients — trading communication volume for memory savings"],
      correct: 3,
      explanation: "In ZeRO-3/FSDP, each GPU stores only a $1/N$ shard of every parameter tensor. Before computing a layer's forward pass, an all-gather reconstructs the full parameters from all shards. After the backward pass, a reduce-scatter distributes gradient shards. The total communication volume per step is $3 \\times 2\\Psi$ (vs. $2\\Psi$ for DDP), a 3x increase. However, this communication can be overlapped with computation by prefetching the next layer's parameters during the current layer's computation. The memory savings are dramatic: total per-GPU memory approaches $(12\\Psi + 2\\Psi + 2\\Psi) / N = 16\\Psi/N$."
    },
    {
      type: "mc",
      question: "The **pipeline bubble problem** in pipeline parallelism arises because:",
      options: [
        "Data cannot be split into micro-batches when using pipeline parallelism because the sequential layer dependencies prevent any form of batch-level decomposition across the pipeline stages",
        "At the start and end of each training step, some pipeline stages are idle waiting for activations from upstream or gradients from downstream — with $p$ pipeline stages and $m$ micro-batches, the bubble fraction is $(p - 1) / m$, wasting compute proportional to the number of stages",
        "GPUs cannot communicate activation tensors across pipeline stages fast enough over PCIe or NVLink, creating a communication bottleneck that serializes the entire forward pass regardless of the pipeline schedule",
        "The model's loss function becomes non-differentiable across pipeline boundaries because the activation tensors are quantized to reduce inter-stage communication, introducing discontinuities that prevent gradient flow"
      ],
      correct: 1,
      explanation: "With naive scheduling, GPU $k$ must wait for GPUs $0, \\dots, k-1$ to complete before starting, creating a \"bubble\" of idle time. Splitting the batch into $m$ micro-batches and interleaving them reduces the bubble fraction to $(p-1)/m$. For example, with 8 pipeline stages and 32 micro-batches, the bubble is $7/32 \\approx 22\\%$ — meaning 22% of compute is wasted. The **1F1B** (one-forward-one-backward) schedule further optimizes memory by limiting the number of in-flight micro-batches, reducing peak activation memory from $O(m)$ to $O(p)$."
    },
    {
      type: "mc",
      question: "The **1F1B** (one-forward-one-backward) pipeline schedule works by:",
      options: ["After an initial warmup phase, alternating between one forward micro-batch and one backward micro-batch on each pipeline stage — this limits the number of in-flight micro-batches per stage to at most $p$ (the pipeline depth), bounding peak activation memory", "Running all $m$ forward passes first across the full pipeline before starting any backward passes, maximizing GPU utilization by keeping all stages active during the forward phase and then active again during the backward phase", "Running the forward and backward passes simultaneously on the same micro-batch by pipelining the gradient computation within each layer so that early layers begin their backward pass while later layers are still completing the forward pass", "Eliminating the pipeline bubble entirely by dynamically reassigning idle pipeline stages to data-parallel replicas, converting wasted bubble time into useful gradient computation on additional data samples"],
      correct: 0,
      explanation: "In 1F1B, each stage goes through a warmup phase (receiving and forwarding micro-batches), then enters a steady state where it performs one forward pass followed by one backward pass in alternation. This means each stage holds activations for at most $p$ micro-batches at any time (rather than all $m$ micro-batches in the naive all-forward-then-all-backward schedule). The bubble fraction remains $(p-1)/m$, but peak memory is dramatically reduced. Interleaved scheduling (where virtual pipeline stages are assigned cyclically) can further reduce the bubble to $(p-1)/(m \\cdot v)$ where $v$ is the number of virtual stages."
    },
    {
      type: "mc",
      question: "**BF16** (bfloat16) is preferred over **FP16** for LLM training because:",
      options: ["BF16 has higher precision for small numbers due to its 10-bit mantissa, making it more accurate than FP16 for the small gradient values that are critical during the early stages of fine-tuning and the final convergence phase", "BF16 uses less memory than FP16 by storing values in 12 bits instead of 16, achieving a 25% reduction in memory footprint per tensor while maintaining sufficient precision for most training workloads through adaptive rounding", "BF16 uses the same 8-bit exponent as FP32 (range $\\pm 3.4 \\times 10^{38}$), avoiding the overflow/underflow issues that plague FP16 (5-bit exponent, range $\\pm 65504$) — this eliminates the need for loss scaling even though BF16 has less mantissa precision (7 bits vs FP16's 10 bits)", "BF16 is the only reduced-precision format supported by modern GPU tensor cores for training workloads — FP16 tensor core support was removed starting with the A100 architecture to simplify the hardware and encourage BF16 adoption"],
      correct: 2,
      explanation: "FP16 has 5 exponent bits (range $\\sim 6 \\times 10^{-8}$ to $6.5 \\times 10^4$) and 10 mantissa bits. BF16 has 8 exponent bits (same range as FP32: $\\sim 10^{-38}$ to $\\sim 10^{38}$) and 7 mantissa bits. In LLM training, gradients and activations span a wide dynamic range — FP16's limited range causes underflow (small gradients become zero) or overflow (large activations become inf), requiring careful loss scaling. BF16's FP32-matching range avoids these issues entirely at the cost of slightly reduced precision. Both use 16 bits (2 bytes per value). The practical result: BF16 training is nearly as stable as FP32 with half the memory."
    },
    {
      type: "mc",
      question: "**Activation checkpointing** (gradient checkpointing) trades compute for memory by:",
      options: ["Compressing activations using quantization from FP16 to INT8 during the forward pass, halving activation memory at the cost of small numerical errors that are corrected during the backward pass via stochastic dequantization", "Reducing the number of active layers in the model by dynamically skipping layers whose gradient contribution falls below a learned threshold, trading model capacity for memory savings during the backward pass", "Storing all intermediate activations on CPU system RAM instead of GPU HBM during the forward pass, then transferring them back to the GPU on demand during the backward pass via PCIe or NVLink-to-host transfers", "Discarding intermediate activations during the forward pass and recomputing them from saved checkpoints during the backward pass — this reduces activation memory from $O(L)$ to $O(\\sqrt{L})$ (with optimal checkpoint placement) at the cost of one additional forward pass, roughly 33% more compute"],
      correct: 3,
      explanation: "During the forward pass, only activations at checkpoint boundaries are saved; intermediate activations are discarded. During the backward pass, when intermediate activations are needed for gradient computation, the forward pass is re-run from the nearest checkpoint. With checkpoints every $\\sqrt{L}$ layers (for $L$ total layers), memory is $O(\\sqrt{L})$ and compute increases by ~33% (one extra forward pass). This is often the single most impactful memory optimization: for a 70B model, it can reduce activation memory from hundreds of GB to a manageable level. The tradeoff is almost always worthwhile — memory is the binding constraint, not compute."
    },
    {
      type: "mc",
      question: "A **70B parameter model** trained with Adam in mixed precision requires approximately how much **optimizer state memory** (across all GPUs combined)?",
      options: ["140 GB (2 bytes per parameter for the FP16 working weights that are used in the forward and backward passes)", "840 GB (12 bytes per parameter: FP32 master weights + FP32 first moment + FP32 second moment)", "280 GB (4 bytes per parameter for the FP32 master copy only, which is the dominant optimizer cost)", "70 GB (1 byte per parameter when using 8-bit Adam with quantized optimizer states)"],
      correct: 1,
      explanation: "Adam maintains three FP32 buffers per parameter: (1) master copy of weights — 4 bytes, (2) first moment estimate $m$ — 4 bytes, (3) second moment estimate $v$ — 4 bytes. Total: $12 \\times 70 \\times 10^9 = 840 \\times 10^9$ bytes $= 840$ GB. This is the dominant memory cost and the primary motivation for ZeRO/FSDP. On 8 GPUs with ZeRO Stage 1, this drops to ~105 GB/GPU. Adding the FP16 model parameters (140 GB) and FP16 gradients (140 GB), total memory is ~1120 GB, or ~140 GB/GPU with 8-way ZeRO-1 for optimizer states alone (parameters and gradients still replicated at Stage 1)."
    },
    {
      type: "mc",
      question: "**Sequence parallelism** addresses a specific limitation of tensor parallelism. In standard Megatron-style tensor parallelism, operations like LayerNorm and dropout are **replicated** on every GPU. Sequence parallelism fixes this by:",
      options: ["Partitioning the sequence dimension across GPUs for these replicated operations (LayerNorm, dropout, activation functions), so each GPU processes a portion of the sequence — then transitioning back to tensor-parallel partitioning for the attention and MLP computations", "Splitting the vocabulary across GPUs so that each GPU computes the embedding lookup and final softmax for a subset of tokens, reducing the per-GPU memory footprint of these large vocabulary-dependent layers", "Using a longer context window by distributing the extended sequence across GPUs, with each GPU responsible for a contiguous chunk of the full context and cross-GPU attention computed via ring communication", "Replacing LayerNorm with a parallelizable alternative such as RMSNorm that decomposes into independent per-GPU computations without requiring the cross-GPU all-reduce needed for computing global mean and variance statistics"],
      correct: 0,
      explanation: "In tensor parallelism, matrix multiplications (attention projections, MLP layers) are split across GPUs, but LayerNorm, dropout, and activation functions operate on the full hidden dimension and are redundantly computed on every GPU. Sequence parallelism (Korthikanti et al., 2022) partitions these operations along the sequence dimension instead: each GPU handles $\\text{seq\\_len}/N$ tokens for LayerNorm/dropout, then the layout transitions to tensor-parallel for the split matrix multiplications. This eliminates the redundant computation and memory for these operations, saving ~30-40% of activation memory that would otherwise be wasted on replicated non-tensor-parallel regions."
    }
  ]
};
