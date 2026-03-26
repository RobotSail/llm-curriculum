// Focused learning module: Tensor Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: why model parallelism is needed, column/row parallelism splits,
// Megatron-LM attention TP, sequence parallelism, communication analysis,
// and practical considerations (3D parallelism, GQA interaction).
// Single-concept module: tensor parallelism for distributing individual layers across GPUs.

export const tensorParallelismLearning = {
  id: "1.6-tensor-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Tensor Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — When Data Parallelism Isn't Enough
    {
      type: "info",
      title: "When Data Parallelism Isn't Enough",
      content: "Data parallelism replicates the entire model on every GPU. For a 70B parameter model trained with Adam, the **model state** per GPU is:\n\n$$\\text{Model state} = (2 + 2 + 12) \\times 70\\text{B} = 1{,}120 \\text{ GB}$$\n\nThe breakdown: 2 bytes for fp16 parameters, 2 bytes for fp16 gradients, and 12 bytes for Adam state (fp32 parameter copy + fp32 first moment + fp32 second moment, each 4 bytes). Even a single replica requires 1.12 TB — far beyond any single GPU's memory.\n\nZeRO Stage 3 shards all three components across GPUs, but each GPU must still **reconstruct full layers** during forward and backward passes. For the largest models, a single transformer layer's parameters plus activations can exceed GPU memory.\n\n**Model parallelism** takes a different approach: split the model itself so each GPU holds and computes only a fraction of each operation. There are two main strategies:\n\n- **Tensor parallelism (TP)**: splits individual layers across GPUs — each GPU computes part of a matrix multiplication within a single layer\n- **Pipeline parallelism (PP)**: assigns different layers to different GPUs — each GPU handles a contiguous block of the network\n\nThis module focuses on tensor parallelism: how to partition weight matrices so that multiple GPUs collaborate on a single layer's computation."
    },
    // Step 2: MC — Why pure DP fails
    {
      type: "mc",
      question: "A team has 512 GPUs, each with 80 GB of memory. They want to train a 175B parameter model using pure data parallelism (no ZeRO, no model parallelism). Why does this fail?",
      options: [
        "The all-reduce gradient synchronization across 512 GPUs introduces so much communication latency that training throughput drops to near zero",
        "Each GPU must hold the full model state, which is $(2 + 2 + 12) \\times 175 \\approx 2{,}800$ GB — far exceeding any single GPU's 80 GB memory",
        "Pure data parallelism requires a global batch size equal to the GPU count, and 512 samples per batch causes severe convergence issues",
        "Data parallelism cannot scale beyond 8 GPUs because the ring all-reduce topology breaks down at higher GPU counts"
      ],
      correct: 1,
      explanation: "Pure data parallelism replicates the full model state on every GPU. With 175B parameters and Adam, each GPU needs $(2 + 2 + 12) \\times 175 \\approx 2{,}800$ GB — over 35x the 80 GB available. Adding more GPUs doesn't help because each one independently needs the full copy. The fundamental bottleneck is **memory per GPU**, not communication bandwidth or batch size constraints. This is the core motivation for model parallelism: distributing the model itself across devices."
    },
    // Step 3: Info — Splitting a Matrix Multiply
    {
      type: "info",
      title: "Splitting a Matrix Multiply",
      content: "The core computation in a transformer is the linear projection $Y = XW$, where $X \\in \\mathbb{R}^{s \\times d}$ (sequence length $\\times$ hidden dim) and $W \\in \\mathbb{R}^{d \\times h}$ (hidden dim $\\times$ output dim). Tensor parallelism splits $W$ across $N$ GPUs so each computes a fraction of $Y$.\n\n**Column parallelism** partitions $W$ along columns. GPU $i$ holds $W_i \\in \\mathbb{R}^{d \\times h/N}$ and computes $Y_i = XW_i$. Each GPU receives the **full input** $X$ and produces $1/N$ of the output columns. The results $[Y_1, Y_2, \\ldots, Y_N]$ are concatenated to form $Y$.\n\n**Row parallelism** partitions $W$ along rows. GPU $i$ holds $W_i \\in \\mathbb{R}^{d/N \\times h}$ and receives the corresponding input slice $X_i \\in \\mathbb{R}^{s \\times d/N}$. Each GPU computes a **partial sum** $Y_i = X_i W_i \\in \\mathbb{R}^{s \\times h}$. The full output is obtained via all-reduce: $Y = \\sum_i Y_i$.\n\nThe key insight from **Megatron-LM** (Shoeybi et al., 2019): in a transformer MLP block with two linear layers ($d \\to 4d \\to d$), use column parallelism for the first layer and row parallelism for the second. The column-parallel output is already sharded along the right dimension to serve as input to the row-parallel layer — **no communication is needed between the two layers**. Only the row-parallel layer's partial sums require an all-reduce. Result: **1 all-reduce per MLP block** in the forward pass."
    },
    // Step 4: MC — All-reduces per MLP block
    {
      type: "mc",
      question: "In Megatron-style tensor parallelism, the MLP block uses column parallelism for the first linear layer ($d \\to 4d$) and row parallelism for the second ($4d \\to d$). How many all-reduce operations occur per MLP block in the **forward pass**?",
      options: [
        "One — only the row-parallel second layer requires an all-reduce to sum the partial results, while the column-parallel first layer's sharded output feeds directly into the row-parallel input",
        "Zero — the column-row pairing eliminates all communication because each GPU independently computes its portion of the output",
        "Two — one all-reduce after each linear layer to recombine the sharded outputs",
        "Four — each linear layer needs an all-gather before it and a reduce-scatter after it"
      ],
      correct: 0,
      explanation: "Column parallelism gives each GPU the full input $X$ and produces a shard of the intermediate activation — no communication needed. This shard is exactly the input slice that row parallelism expects, so the transition is free. The row-parallel second layer computes partial sums $Y_i = X_i W_i$ that must be all-reduced to obtain the final output $Y = \\sum_i Y_i$. Total: **1 all-reduce per MLP block** forward. This elegant pairing is Megatron-LM's key contribution — it minimizes the communication-to-computation ratio."
    },
    // Step 5: Info — Tensor Parallelism in Attention Layers
    {
      type: "info",
      title: "Tensor Parallelism in Attention Layers",
      content: "Self-attention computes $Q = XW_Q$, $K = XW_K$, $V = XW_V$, applies scaled dot-product attention, then projects with $W_O$. Tensor parallelism exploits the fact that **attention heads are independent computations**.\n\nWith $N$ GPUs and $H$ total attention heads (where $N$ divides $H$), each GPU handles $H/N$ heads:\n- The $W_Q$, $W_K$, $W_V$ projections are split by column — each GPU holds weights for its $H/N$ heads (column parallelism)\n- Each GPU computes attention independently for its local heads — no cross-GPU communication needed during the attention computation itself\n- The output projection $W_O$ is split by rows — GPU $i$ holds $W_O^{(i)} \\in \\mathbb{R}^{(d/N) \\times d}$ (row parallelism)\n- The row-parallel output produces partial sums that require one all-reduce\n\nJust like the MLP block: **1 all-reduce per attention block** in the forward pass.\n\nCombining MLP and attention, each transformer layer has 2 all-reduces in the forward pass and 2 in the backward pass, totaling **4 all-reduce operations per layer**. For a model with $L$ layers, that is $4L$ all-reduces per training step. This high communication frequency is why tensor parallelism must use **NVLink** (600+ GB/s) within a single node, not InfiniBand (~50 GB/s) across nodes."
    },
    // Step 6: MC — Why TP GPUs must be within a node
    {
      type: "mc",
      question: "A training cluster has 8 GPUs per node connected by NVLink (600 GB/s) and nodes connected by InfiniBand (50 GB/s). Why must tensor-parallel GPUs be co-located within the same node?",
      options: [
        "Tensor parallelism requires GPUs to share a unified memory address space, which is only possible with NVLink's memory-mapping capability",
        "InfiniBand only supports point-to-point communication and cannot execute the collective all-reduce operations that tensor parallelism requires",
        "NVLink's lower latency allows GPUs to synchronize their random number generators, which is essential for reproducible dropout in TP",
        "Each transformer layer triggers multiple all-reduce operations transferring activation-sized tensors — the 12x bandwidth gap between NVLink and InfiniBand would make TP communication the dominant cost"
      ],
      correct: 3,
      explanation: "Tensor parallelism requires 4 all-reduce operations per transformer layer (2 forward, 2 backward), each moving tensors of size $O(s \\times d)$. For hidden dim $d = 8192$ and sequence length $s = 4096$, each all-reduce transfers tens of megabytes, repeated across every layer and every micro-batch. NVLink at 600 GB/s handles this with minimal overhead; InfiniBand at 50 GB/s (a 12x reduction) would turn these frequent all-reduces into a severe bottleneck, collapsing GPU utilization. The issue is bandwidth for high-frequency communication, not memory sharing or collective operation support."
    },
    // Step 7: Info — Sequence Parallelism
    {
      type: "info",
      title: "Sequence Parallelism",
      content: "Tensor parallelism splits weight matrices across GPUs, but some operations — **LayerNorm** and **dropout** — act on the full hidden dimension and are **replicated identically** across all TP GPUs. Each GPU independently computes the same LayerNorm on the same activations, wasting both compute and memory.\n\n**Sequence parallelism** (SP) fixes this by splitting these non-TP regions along the **sequence dimension**. Between tensor-parallel regions, activations of shape $(s, d)$ are partitioned so each GPU holds $(s/N, d)$ — a different chunk of tokens but the full hidden dimension. Each GPU applies LayerNorm and dropout only to its $s/N$ tokens.\n\nThe transitions between TP and SP regions use familiar collectives:\n- **Before a column-parallel layer**: all-gather along the sequence dimension (each GPU needs the full sequence as input to the column-parallel matrix multiply)\n- **After a row-parallel layer**: reduce-scatter combines the partial sums and distributes the result along the sequence dimension\n\nCritically, the all-gather + reduce-scatter pair has the **same total communication volume** as the all-reduce it replaces — the math is identical, just reorganized. The benefit is purely on the **memory** side: activation memory for LayerNorm and dropout regions is reduced by a factor of $N$ because each GPU stores only $s/N$ tokens' worth of activations in those regions."
    },
    // Step 8: MC — Benefit of sequence parallelism
    {
      type: "mc",
      question: "A model uses 8-way tensor parallelism. Adding sequence parallelism replaces all-reduce operations with all-gather and reduce-scatter pairs. What is the **primary** benefit of this change?",
      options: [
        "Communication volume is cut by a factor of 8 because reduce-scatter only sends $1/N$ of the data compared to all-reduce",
        "Training throughput doubles because the all-gather and reduce-scatter can overlap with computation, unlike all-reduce which blocks the pipeline",
        "Activation memory for LayerNorm and dropout regions is reduced by $8\\times$ because each GPU stores only $s/8$ tokens in those regions, while total communication volume remains the same",
        "Numerical precision improves because reduce-scatter avoids the floating-point accumulation errors inherent in all-reduce summation"
      ],
      correct: 2,
      explanation: "Sequence parallelism replaces all-reduce with all-gather + reduce-scatter. These have **identical total communication volume** — the data is simply reorganized, not reduced. The key win is **activation memory**: in the non-TP regions (LayerNorm, dropout), each GPU now stores activations for only $s/N$ tokens instead of the full $s$ tokens. With $N = 8$, this is an $8\\times$ memory reduction for those regions. This freed memory can be used for larger batch sizes or longer sequences. Communication cost is unchanged."
    },
    // Step 9: Info — Communication Analysis and Scaling
    {
      type: "info",
      title: "Communication Analysis and Scaling",
      content: "Let's analyze when tensor parallelism is efficient versus when it breaks down.\n\nFor a single all-reduce of a tensor with $M$ elements across $N$ GPUs, the data transferred per GPU is approximately $2M \\cdot \\frac{N-1}{N} \\approx 2M$ (for the ring all-reduce algorithm). Each all-reduce in TP moves activation tensors of size $O(s \\times d)$ where $s$ is the sequence length and $d$ is the hidden dimension.\n\nPer transformer layer:\n- **Communication**: 4 all-reduces (2 forward, 2 backward), each moving $O(s \\times d)$ data. Total: $O(s \\times d)$ per layer.\n- **Computation**: the dominant cost is the matrix multiplies. Each MLP block does $O(s \\times d \\times 4d / N) = O(s \\times d^2 / N)$ FLOPs per GPU (the work is split $N$ ways).\n\nThe **compute-to-communication ratio** per GPU scales as:\n\n$$\\frac{\\text{Compute}}{\\text{Communication}} \\propto \\frac{s \\cdot d^2 / N}{s \\cdot d} = \\frac{d}{N}$$\n\nThis reveals two critical insights:\n1. **Larger hidden dimension $d$ improves efficiency** — bigger models extract more compute per byte transferred, making TP more worthwhile.\n2. **Increasing TP degree $N$ hurts efficiency** — each doubling of $N$ halves the ratio. Beyond $N = 8$ (a single NVLink node), the ratio typically becomes unfavorable even on fast interconnects.\n\nThis is why practical TP degrees are almost always 2, 4, or 8 — matching the GPUs within a single NVLink-connected node."
    },
    // Step 10: MC — Scaling behavior with hidden dim
    {
      type: "mc",
      question: "A model uses 8-way tensor parallelism. The hidden dimension is doubled from 4096 to 8192 while keeping the TP degree and sequence length the same. What happens to the compute-to-communication ratio per GPU?",
      options: [
        "The ratio stays the same because both compute and communication scale linearly with hidden dimension",
        "The ratio halves because the larger weight matrices require twice as much all-reduce communication, while the compute per GPU is unchanged",
        "The ratio decreases because more parameters means more frequent synchronization between GPUs",
        "The ratio doubles because compute scales as $d^2$ (matrix multiply) while communication scales as $d$ (activation size), so larger models are more efficient with TP"
      ],
      correct: 3,
      explanation: "The compute-to-communication ratio scales as $d/N$. With fixed $N = 8$: doubling $d$ from 4096 to 8192 **doubles** the ratio. Why? Compute per GPU scales as $O(s \\cdot d^2 / N)$ — the matrix multiply has $d^2$ in its cost. Communication scales as $O(s \\cdot d)$ — all-reduces transfer activation-sized tensors. The extra factor of $d$ in the numerator means **larger models extract more compute per byte of communication**, making tensor parallelism increasingly efficient. This is one reason why scaling up model size is relatively cheap in terms of parallelism overhead."
    },
    // Step 11: Info — Practical Considerations
    {
      type: "info",
      title: "Practical Considerations",
      content: "Tensor parallelism in real training systems involves several practical concerns beyond the core algorithm.\n\n**3D Parallelism**: Large-scale training combines TP, pipeline parallelism (PP), and data parallelism (DP). The standard layout is TP within a node (highest bandwidth needed), PP across nearby nodes (moderate bandwidth), and DP across all remaining GPUs (one gradient sync per step). For example, 512 GPUs = TP$\\,8 \\times$ PP$\\,8 \\times$ DP$\\,8$.\n\n**Uneven head counts**: If the number of attention heads $H$ is not divisible by the TP degree $N$, some GPUs get more heads than others, causing load imbalance. Practitioners choose $N$ to divide $H$ evenly.\n\n**Grouped-query attention (GQA)**: GQA uses fewer key-value heads than query heads — e.g., 32 query heads but only 8 KV heads. With 8-way TP, each GPU gets $32/8 = 4$ query heads and $8/8 = 1$ KV head. GQA naturally aligns with TP because the KV heads are already designed to be shared across groups of query heads.\n\n**Activation checkpointing interaction**: TP reduces per-GPU parameter and gradient memory by $N\\times$, but activations are only partially reduced (the TP-region activations are $1/N$ each, but non-TP regions like LayerNorm are full-sized unless sequence parallelism is used). Activation checkpointing remains important even with TP.\n\n**Expert parallelism in MoE**: Mixture-of-experts models add another parallelism dimension — experts are distributed across GPUs, with all-to-all communication routing tokens to the correct expert. This interacts with TP by adding communication at MoE layers."
    },
    // Step 12: MC — GQA with tensor parallelism
    {
      type: "mc",
      question: "A model has 32 query heads and uses grouped-query attention (GQA) with 8 KV heads. The model is trained with 8-way tensor parallelism. How are the attention heads distributed across GPUs?",
      options: [
        "Each GPU gets 4 query heads and 1 KV head — GQA's shared KV heads divide evenly across the TP degree, with each KV head serving the 4 local query heads on its GPU",
        "Each GPU gets 4 query heads but all 8 KV heads are replicated on every GPU, since KV heads must be accessible to all query head groups",
        "Each GPU gets 4 query heads and 4 KV heads — the KV heads are duplicated $4\\times$ so that each GPU has a local copy for every query head it owns",
        "The 8 KV heads are placed on the first GPU only, and the other 7 GPUs send their query attention scores to GPU 0 for KV lookup via point-to-point transfers"
      ],
      correct: 0,
      explanation: "With 8-way TP: 32 query heads $/ \\, 8$ GPUs $= 4$ query heads per GPU, and 8 KV heads $/ \\, 8$ GPUs $= 1$ KV head per GPU. Each GPU's single KV head serves all 4 of its local query heads — this is exactly the GQA sharing pattern (each KV head shared across a group of query heads). No replication or cross-GPU KV transfer is needed because the query-to-KV grouping aligns perfectly with the TP partition. This natural alignment is one reason GQA (popularized by Llama 2) is popular in large-scale training: it reduces KV cache size **and** plays well with tensor parallelism."
    }
  ]
};
