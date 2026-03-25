// Focused learning module: Data Parallelism and ZeRO
// Section 1.6: Distributed Training Infrastructure
// Covers: why distribution is needed, data parallelism, gradient synchronization,
// all-reduce, ZeRO stages, and FSDP.
// Single-concept module: data-parallel training and its memory-efficient extensions.

export const dataParallelismLearning = {
  id: "1.6-data-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Data Parallelism and ZeRO",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why distribute?
    {
      type: "info",
      title: "The Memory Wall: Why One GPU Isn't Enough",
      content: "Training a large language model requires storing several components in GPU memory simultaneously:\n\n1. **Model parameters**: For a model with $\\Psi$ parameters in FP16, this is $2\\Psi$ bytes\n2. **Gradients**: Same size as parameters — $2\\Psi$ bytes in FP16\n3. **Optimizer states**: Adam stores FP32 master weights ($4\\Psi$), first moment $m$ ($4\\Psi$), and second moment $v$ ($4\\Psi$) = $12\\Psi$ bytes\n4. **Activations**: Intermediate values saved for backpropagation, scaling with batch size and sequence length\n\nFor a 7B parameter model: parameters = 14 GB, gradients = 14 GB, optimizer states = 84 GB. **Total: 112 GB just for the model** — before any activations. A single A100 has 80 GB of HBM.\n\nFor a 70B model: the optimizer states alone require **840 GB**. Even the most powerful single GPU can't hold this.\n\n**Distributed training** spreads this memory burden across multiple GPUs while ensuring they collectively compute the same result as a single (impossibly large) GPU would."
    },
    // Step 2: MC — memory calculation
    {
      type: "mc",
      question: "A 13B parameter model is trained with Adam in mixed precision. What is the minimum number of 80 GB A100 GPUs needed just to hold the model state (parameters + gradients + optimizer states), ignoring activation memory entirely?",
      options: [
        "1 GPU — 13B parameters at 2 bytes each is 26 GB, which fits on a single 80 GB GPU with room to spare for gradients and optimizer states",
        "2 GPUs — the model state totals $16 \\times 13B = 208$ GB when including optimizer states, so two 80 GB GPUs (160 GB total) aren't quite enough, but with some offloading it works",
        "3 GPUs — the total model state is $(2 + 2 + 12) \\times 13 = 208$ GB, requiring at least $\\lceil 208/80 \\rceil = 3$ GPUs",
        "8 GPUs — practical training requires 8-way parallelism as a minimum due to communication overhead, regardless of the memory arithmetic"
      ],
      correct: 2,
      explanation: "Total model state: FP16 params ($2 \\times 13 = 26$ GB) + FP16 gradients ($26$ GB) + Adam optimizer states ($12 \\times 13 = 156$ GB) = **208 GB**. With 80 GB per GPU, you need at least $\\lceil 208/80 \\rceil = 3$ GPUs. In practice you'd use more GPUs to also fit activations and to increase throughput. This calculation shows why optimizer states are the dominant memory cost — they account for $156/208 = 75\\%$ of the total model state."
    },
    // Step 3: Data parallelism basics
    {
      type: "info",
      title: "Data Parallelism: The Simplest Distribution Strategy",
      content: "**Data parallelism** (DP) is the most straightforward distributed training approach:\n\n1. **Replicate** the entire model on every GPU\n2. **Split** each batch of data across GPUs — if the global batch has $B$ samples and there are $N$ GPUs, each GPU processes $B/N$ samples\n3. **Forward + backward** each GPU computes gradients on its local data shard independently\n4. **Synchronize** average the gradients across all GPUs\n5. **Update** each GPU applies the same averaged gradient to its local model copy\n\nAfter step 5, all GPUs have identical model parameters (they started identical and applied the same update). The process is mathematically equivalent to training on the full batch $B$ on a single device:\n\n$$g_{\\text{sync}} = \\frac{1}{N}\\sum_{i=1}^{N} g_i = \\frac{1}{N}\\sum_{i=1}^{N} \\frac{1}{B/N}\\sum_{j \\in \\mathcal{D}_i} \\nabla_\\theta \\ell_j = \\frac{1}{B}\\sum_{j=1}^{B} \\nabla_\\theta \\ell_j$$\n\nData parallelism scales **throughput** linearly with GPUs (ideally $N\\times$ samples per second), but does not reduce **per-GPU memory** — every GPU still holds the full model, full gradients, and full optimizer states."
    },
    // Step 4: MC — DP understanding
    {
      type: "mc",
      question: "A team trains a model with data parallelism across 64 GPUs. They use a global batch size of 1M tokens. Which statement is correct about the gradient computation?",
      options: [
        "Each GPU computes gradients on $1M/64 \\approx 15.6K$ tokens, and the averaged gradient is mathematically identical to computing the gradient on all 1M tokens on a single GPU",
        "Each GPU computes gradients on all 1M tokens independently, and the averaging step removes noise by exploiting the redundancy of 64 independent gradient estimates",
        "The 64 GPUs collectively compute a gradient that is 64x more accurate than a single-GPU gradient, enabling the use of a 64x larger learning rate",
        "Each GPU computes gradients on $1M/64$ tokens, but the averaged result is only an approximation of the true gradient due to information loss during the all-reduce communication"
      ],
      correct: 0,
      explanation: "Data parallelism is exact, not approximate. The gradient of a sum equals the sum of gradients: $\\nabla_\\theta \\frac{1}{B}\\sum_{j=1}^{B} \\ell_j = \\frac{1}{B}\\sum_{j=1}^{B} \\nabla_\\theta \\ell_j$. Splitting the sum across GPUs and averaging is algebraically identical. The all-reduce operation introduces no information loss (in exact arithmetic — floating-point reduction order can cause minor differences). Each GPU processes $\\sim$15.6K tokens, making each GPU's workload 64x smaller, which is why data parallelism provides near-linear throughput scaling."
    },
    // Step 5: All-reduce
    {
      type: "info",
      title: "All-Reduce: The Communication Backbone",
      content: "The gradient synchronization step uses an **all-reduce** operation: given $N$ GPUs each holding a tensor $g_i$, compute $\\sum_i g_i / N$ and distribute the result to all GPUs.\n\nThe naive approach — send all gradients to one GPU, average, broadcast back — creates a bottleneck at the central GPU and doesn't scale.\n\n**Ring all-reduce** is the standard solution. Arrange $N$ GPUs in a logical ring. Each GPU sends a chunk of its gradient to its neighbor. After $2(N-1)$ communication steps:\n- Phase 1 (reduce-scatter): Each GPU has a $1/N$-sized chunk containing the partial sum from all GPUs\n- Phase 2 (all-gather): Each GPU has the complete averaged gradient\n\nThe key property: **each GPU sends and receives exactly $2 \\cdot \\frac{N-1}{N} \\cdot |g|$ bytes total**, regardless of $N$. As $N \\to \\infty$, this approaches $2|g|$ bytes — the communication cost per GPU is nearly constant.\n\nFor a 7B model: $|g| = 14$ GB (FP16 gradients). Each GPU transfers about 28 GB of data. With NVLink at 600 GB/s between adjacent GPUs, this takes about 50 ms — during which time the GPUs are otherwise idle unless communication is overlapped with computation."
    },
    // Step 6: MC — all-reduce
    {
      type: "mc",
      question: "Ring all-reduce synchronizes gradients across $N$ GPUs with each GPU transferring approximately $2|g|$ bytes (for gradient tensor size $|g|$). A team doubles their GPU count from 32 to 64 while keeping the same model. What happens to the per-GPU communication volume?",
      options: [
        "It doubles — each GPU must now communicate with twice as many peers, sending twice as much data through the ring",
        "It stays approximately the same — the ring all-reduce formula is $2 \\cdot \\frac{N-1}{N} \\cdot |g|$, which is nearly $2|g|$ for both $N=32$ and $N=64$",
        "It halves — with more GPUs, each GPU is responsible for a smaller chunk of the gradient, reducing the per-GPU data transfer proportionally",
        "It increases by $\\sqrt{2}$ — the communication cost scales as $O(\\sqrt{N})$ per GPU in ring all-reduce due to the increasing number of ring hops"
      ],
      correct: 1,
      explanation: "Per-GPU communication is $2 \\cdot \\frac{N-1}{N} \\cdot |g|$. For $N=32$: $2 \\times 31/32 \\times |g| = 1.94|g|$. For $N=64$: $2 \\times 63/64 \\times |g| = 1.97|g|$. The difference is negligible — this is why ring all-reduce scales so well. The total network bandwidth grows linearly with $N$, but the per-GPU cost stays flat. However, the **latency** does increase: the $2(N-1)$ sequential steps mean more round-trip delays. For very large $N$, hierarchical approaches (tree all-reduce, or 2D ring) are used to manage latency."
    },
    // Step 7: ZeRO motivation
    {
      type: "info",
      title: "The Redundancy Problem: Why ZeRO Was Invented",
      content: "Standard data parallelism has an obvious waste: **every GPU stores identical copies** of the model parameters, gradients, and optimizer states. With 64 GPUs, you have 64 copies of everything — 64x redundancy.\n\nFor a 7B model across 64 GPUs:\n- Optimizer states: $84$ GB $\\times$ 64 = **5,376 GB** (total system memory for optimizer states)\n- But only 84 GB of unique data!\n\n**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminates this redundancy by **partitioning** (sharding) the model state across GPUs instead of replicating it. The insight: each GPU only needs the full parameters during the forward/backward pass — it doesn't need full optimizer states or gradients at all times.\n\nZeRO is designed in three stages, each eliminating more redundancy:\n- **Stage 1**: Partition optimizer states\n- **Stage 2**: Partition optimizer states + gradients\n- **Stage 3**: Partition optimizer states + gradients + parameters\n\nEach stage trades more communication for more memory savings."
    },
    // Step 8: MC — ZeRO motivation
    {
      type: "mc",
      question: "A 30B parameter model is trained on 8 GPUs using standard DDP (full replication). The total memory for optimizer states across all GPUs is $12 \\times 30B \\times 8 = 2{,}880$ GB. With ZeRO Stage 1 (partitioned optimizer states), the total system-wide optimizer memory is:",
      options: [
        "$2{,}880$ GB — ZeRO Stage 1 doesn't reduce total memory, it only redistributes it more evenly across GPUs to prevent any single GPU from running out",
        "$360$ GB — each GPU stores $1/8$ of the optimizer states, and the total is $12 \\times 30B = 360$ GB, the same as a single GPU would need",
        "$2{,}880 / 8 = 360$ GB total, which is also $360/8 = 45$ GB per GPU — a 64x total reduction compared to full replication",
        "$12 \\times 30B / 8 = 45$ GB per GPU, totaling $45 \\times 8 = 360$ GB system-wide — the redundancy is eliminated, so the total equals what one GPU would store"
      ],
      correct: 3,
      explanation: "With ZeRO Stage 1, the $12 \\times 30B = 360$ GB of optimizer states is partitioned into 8 equal shards of 45 GB each. The total system-wide memory is 360 GB (down from 2,880 GB with full replication). Each GPU stores only its 45 GB shard. The key insight: the redundancy factor dropped from 8x to 1x. Parameters ($2 \\times 30 = 60$ GB) and gradients (60 GB) remain fully replicated in Stage 1, so each GPU still needs $45 + 60 + 60 = 165$ GB. Stages 2 and 3 address the remaining redundancy."
    },
    // Step 9: ZeRO stages
    {
      type: "info",
      title: "ZeRO Stages: Progressive Memory Savings",
      content: "For a model with $\\Psi$ parameters, Adam in mixed precision, and $N$ GPUs:\n\n| Stage | What's partitioned | Per-GPU memory | Communication vs DDP |\n|---|---|---|---|\n| DDP (baseline) | Nothing | $2\\Psi + 2\\Psi + 12\\Psi = 16\\Psi$ | $2\\Psi$ (all-reduce) |\n| **Stage 1** | Optimizer states | $2\\Psi + 2\\Psi + 12\\Psi/N$ | $2\\Psi$ (same as DDP) |\n| **Stage 2** | + Gradients | $2\\Psi + 2\\Psi/N + 12\\Psi/N$ | $2\\Psi$ (same as DDP) |\n| **Stage 3** | + Parameters | $(2\\Psi + 2\\Psi + 12\\Psi)/N = 16\\Psi/N$ | $3 \\times 2\\Psi$ (1.5x DDP) |\n\n**Stage 1** is nearly free — the optimizer update is local (each GPU updates only its shard), and the all-reduce for gradients is unchanged. The only added cost is an all-gather to broadcast updated parameters, which can overlap with computation.\n\n**Stage 2** replaces the gradient all-reduce with a reduce-scatter (each GPU gets its gradient shard), saving gradient memory. Communication volume is unchanged.\n\n**Stage 3** (equivalent to **FSDP** — Fully Sharded Data Parallel) is the most aggressive. Parameters are sharded; before each layer's computation, an all-gather reconstructs full parameters, and after backward, reduce-scatter distributes gradient shards. Communication increases by 1.5x, but memory drops to $16\\Psi/N$ — linear scaling with GPUs."
    },
    // Step 10: MC — ZeRO stage selection
    {
      type: "mc",
      question: "A team has 8 A100 (80 GB) GPUs and wants to train a 13B model with Adam. Per-GPU model state with DDP: $16 \\times 13 = 208$ GB (doesn't fit). Which is the minimum ZeRO stage needed to fit the model state on these GPUs?",
      options: [
        "Stage 1 — partitioning optimizer states gives per-GPU cost of $4 \\times 13 + 12 \\times 13 / 8 = 52 + 19.5 = 71.5$ GB, which fits in 80 GB",
        "Stage 2 — Stage 1 gives $2 \\times 13 + 2 \\times 13 + 12 \\times 13 / 8 = 26 + 26 + 19.5 = 71.5$ GB, but adding activations pushes it over 80 GB, so Stage 2's gradient savings ($26 + 26/8 + 19.5 = 48.75$ GB) provides enough headroom",
        "Stage 3 — only full sharding ($16 \\times 13 / 8 = 26$ GB) provides enough memory for model state plus activations",
        "No ZeRO stage is sufficient — 8 GPUs cannot train a 13B model and the team needs at least 16 GPUs"
      ],
      correct: 0,
      explanation: "Let's compute each stage. Stage 1: params ($2 \\times 13 = 26$ GB) + gradients ($26$ GB) + optimizer shard ($12 \\times 13 / 8 = 19.5$ GB) = **71.5 GB**. This fits in 80 GB with 8.5 GB headroom for activations (tight but feasible with activation checkpointing). Stage 2 would give $26 + 3.25 + 19.5 = 48.75$ GB, more comfortable. Stage 3 gives $26$ GB, very comfortable. The minimum is Stage 1, though Stage 2 is more practical since it provides headroom for activations. This is why Stage 1 is the most commonly used — it provides the biggest savings relative to its zero additional communication cost."
    },
    // Step 11: FSDP and practical usage
    {
      type: "info",
      title: "FSDP: ZeRO Stage 3 in Practice",
      content: "**FSDP** (Fully Sharded Data Parallel), implemented in PyTorch, is the practical realization of ZeRO Stage 3. It has become the standard approach for training large models.\n\nThe execution flow for each layer:\n\n**Forward pass**:\n1. All-gather: Reconstruct full parameters from shards across GPUs\n2. Compute: Run the layer's forward pass with full parameters\n3. Discard: Free the non-local parameter shards (only keep the local shard)\n\n**Backward pass**:\n1. All-gather: Reconstruct full parameters again (needed for gradient computation)\n2. Compute: Run the layer's backward pass\n3. Reduce-scatter: Compute gradient shards — each GPU gets $1/N$ of the gradient\n4. Discard: Free full parameters and non-local gradient data\n\nThe key optimization: **prefetching**. While GPU $i$ computes layer $l$, it starts the all-gather for layer $l+1$ in the background. This overlaps communication with computation, hiding much of the latency.\n\nFSDP also supports **mixed sharding**: shard within a node (e.g., 8 GPUs with fast NVLink) but replicate across nodes (slower interconnect). This balances memory savings with communication efficiency."
    },
    // Step 12: MC — practical integration
    {
      type: "mc",
      question: "A 70B model is trained across 64 A100 GPUs using FSDP (ZeRO Stage 3). The per-GPU model state memory is $16 \\times 70 / 64 \\approx 17.5$ GB. The team notices that training throughput is 40% lower than expected from perfect scaling. Which factor most likely explains the throughput gap?",
      options: [
        "The model parameters are too large to fit in the GPU's L2 cache, causing memory bandwidth bottlenecks during the matrix multiplications that dominate both forward and backward compute",
        "The FSDP all-gather and reduce-scatter communications for parameter reconstruction and gradient distribution cannot be fully overlapped with computation, especially for small layers where compute time is shorter than communication latency",
        "Python's Global Interpreter Lock (GIL) prevents true parallelism across the 64 GPU processes, serializing the CPU-side orchestration of gradient synchronization and parameter updates",
        "The optimizer update step is 64x slower because each GPU must wait for all other GPUs to complete their local optimizer shard updates before any GPU can begin the next forward pass"
      ],
      correct: 1,
      explanation: "FSDP's communication-computation overlap is imperfect. The all-gather for layer $l+1$ must complete before layer $l+1$'s computation begins. For layers with small parameter counts (e.g., LayerNorm, small attention projections), the compute finishes before the next layer's all-gather completes, creating idle time. Additionally, the first layer's all-gather cannot be overlapped (no preceding computation to hide behind), and the last layer's reduce-scatter has the same issue. The 1.5x communication overhead of ZeRO-3 vs DDP contributes to the gap. Techniques like computation-communication co-scheduling and adaptive sharding granularity help reduce this."
    }
  ]
};
