// Focused learning module: ZeRO and Fully Sharded Data Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: ZeRO Stages 1-3, FSDP execution flow, prefetching, stage selection
// Single-concept module: memory-efficient data parallelism through parameter sharding.

export const zeroFsdpLearning = {
  id: "1.6-zero-fsdp-learning-easy",
  sectionId: "1.6",
  title: "ZeRO and Fully Sharded Data Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: The Redundancy Problem
    {
      type: "info",
      title: "The Redundancy Problem",
      content: "Standard data parallelism (DDP) replicates the **entire model state** on every GPU: parameters, gradients, and optimizer states. This means $N$ GPUs hold $N$ identical copies of everything.\n\nConsider a 7B parameter model trained with Adam in mixed precision across 64 GPUs. The optimizer states (FP32 master weights, first moment $m$, second moment $v$) total $12\\Psi = 12 \\times 7\\text{B} = 84$ GB on **each** GPU. Across 64 GPUs:\n\n$$\\text{Total optimizer memory} = 84 \\times 64 = 5{,}376 \\text{ GB}$$\n\nBut there are only **84 GB of unique optimizer data** in the entire system. The redundancy factor is $64\\times$ — 98.4% of the memory is wasted on identical copies.\n\n**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminates this waste by **partitioning** (sharding) the model state across GPUs instead of replicating it. The core insight: no single GPU needs all of the optimizer states, gradients, or even parameters at all times. Each GPU can own a shard and reconstruct the full tensors only when needed through collective communication.\n\nZeRO comes in three progressive stages, each sharding more of the model state and recovering more of that wasted memory."
    },
    // Step 2: MC — DDP vs ZeRO Stage 1 optimizer memory
    {
      type: "mc",
      question: "A 30B parameter model is trained on 8 GPUs. With standard DDP, each GPU stores the full optimizer states: $12 \\times 30\\text{B} = 360$ GB per GPU, or $2{,}880$ GB system-wide. After switching to ZeRO Stage 1, what is the system-wide optimizer memory?",
      options: [
        "$2{,}880$ GB — ZeRO Stage 1 rearranges optimizer memory across GPUs for better load balancing but does not reduce the total footprint",
        "$360$ GB — each GPU stores $360 / 8 = 45$ GB, eliminating the $8\\times$ redundancy so the system total equals what a single copy requires",
        "$720$ GB — ZeRO Stage 1 halves the per-GPU optimizer memory by using FP16 instead of FP32 for moment estimates, cutting the system total to $2{,}880 / 4$",
        "$45$ GB — ZeRO Stage 1 compresses the optimizer states by a factor of $N^2$ rather than $N$, achieving super-linear memory savings"
      ],
      correct: 1,
      explanation: "ZeRO Stage 1 partitions optimizer states into $N$ equal shards, one per GPU. Each GPU stores $12\\Psi / N = 360 / 8 = 45$ GB. The system-wide total is $45 \\times 8 = 360$ GB — exactly the same as a single unsharded copy. The $8\\times$ redundancy is completely eliminated for optimizer states. Parameters and gradients remain fully replicated in Stage 1, so each GPU still needs $2 \\times 30 + 2 \\times 30 = 120$ GB for those, plus 45 GB of optimizer state = 165 GB per GPU."
    },
    // Step 3: ZeRO Stage 1 — Partition Optimizer States
    {
      type: "info",
      title: "ZeRO Stage 1 — Partition Optimizer States",
      content: "In ZeRO Stage 1, each GPU stores only $1/N$ of the optimizer states (master weights, $m$, $v$). The training loop changes as follows:\n\n1. **Forward + backward**: Unchanged from DDP. Each GPU computes local gradients on its data shard.\n2. **All-reduce gradients**: Same ring all-reduce as DDP — every GPU gets the full averaged gradient. Communication: $2\\Psi$ bytes.\n3. **Local optimizer update**: Each GPU uses the full gradient but only updates its $1/N$ shard of optimizer states. The other $(N-1)/N$ of the gradient is discarded after the update.\n4. **All-gather parameters**: Each GPU broadcasts its freshly updated parameter shard. After this step, all GPUs have the complete, updated parameters. Communication: $\\Psi$ bytes.\n\nPer-GPU memory breakdown:\n\n$$\\underbrace{2\\Psi}_{\\text{FP16 params}} + \\underbrace{2\\Psi}_{\\text{FP16 gradients}} + \\underbrace{12\\Psi/N}_{\\text{optimizer shard}} = 4\\Psi + \\frac{12\\Psi}{N}$$\n\nThe extra all-gather in step 4 adds $\\Psi$ bytes of communication, but it can be **overlapped with the next forward pass** — while GPU $i$ starts processing the next batch, the parameter broadcast runs in the background. In practice, Stage 1 has **near-zero overhead** compared to standard DDP, making it the default choice when DDP's memory is too high."
    },
    // Step 4: MC — Why Stage 1 is "nearly free"
    {
      type: "mc",
      question: "ZeRO Stage 1 adds an all-gather operation after the optimizer update that standard DDP does not have. Why is this additional communication considered \"nearly free\" in practice?",
      options: [
        "The all-gather transfers only the optimizer state shards, which are much smaller than the full gradient tensors that DDP already communicates",
        "Modern GPUs compress the parameter data during the all-gather using hardware-level lossless compression, reducing the actual bytes transferred to near zero",
        "The all-gather for updated parameters can be overlapped with the forward pass of the next training step, hiding the communication latency behind useful computation",
        "The all-gather is performed using CPU-to-CPU communication over the host network, leaving the GPU interconnect bandwidth fully available for gradient synchronization"
      ],
      correct: 2,
      explanation: "The all-gather in Stage 1 transfers $\\Psi$ bytes of parameter data (the same size as the model parameters). This is not negligible in absolute terms. However, it can be scheduled to run concurrently with the forward pass of the next training iteration. While GPUs compute activations for the new batch, the all-gather for the just-updated parameters completes in the background over NVLink or InfiniBand. Since forward computation and communication use different hardware resources (compute units vs. network interface), they overlap well. This is why Stage 1 is the most popular ZeRO stage — the memory savings from partitioning optimizer states come at essentially no throughput cost."
    },
    // Step 5: ZeRO Stage 2 — Partition Gradients Too
    {
      type: "info",
      title: "ZeRO Stage 2 — Partition Gradients Too",
      content: "ZeRO Stage 2 extends partitioning to **gradients** in addition to optimizer states. The key change: the gradient all-reduce is replaced with a **reduce-scatter**.\n\nRecall that ring all-reduce has two phases: reduce-scatter (each GPU gets $1/N$ of the reduced result) followed by all-gather (each GPU gets the complete result). In standard DDP and Stage 1, both phases run so every GPU holds the full averaged gradient.\n\nIn Stage 2, we **stop after the reduce-scatter**. Each GPU receives only the $1/N$ gradient shard that corresponds to its optimizer shard — which is all it needs to perform its local optimizer update. The gradient all-gather is skipped entirely.\n\nPer-GPU memory:\n\n$$\\underbrace{2\\Psi}_{\\text{FP16 params}} + \\underbrace{2\\Psi/N}_{\\text{gradient shard}} + \\underbrace{12\\Psi/N}_{\\text{optimizer shard}} = 2\\Psi + \\frac{14\\Psi}{N}$$\n\n**Communication volume is unchanged from DDP.** The reduce-scatter moves $\\Psi$ bytes (same as half of the all-reduce). The subsequent all-gather for parameters after the optimizer update adds $\\Psi$ bytes. Total: $2\\Psi$ — identical to DDP's all-reduce. Stage 2 saves gradient memory without any additional communication cost.\n\nThe insight: each GPU only ever needs the gradient shard that matches its optimizer shard. Storing the full gradient was always wasteful."
    },
    // Step 6: MC — Stage 1 vs Stage 2 memory comparison
    {
      type: "mc",
      question: "A 13B model is trained on 8 GPUs. Stage 1 per-GPU memory is $4 \\times 13 + 12 \\times 13 / 8 = 71.5$ GB. What is the Stage 2 per-GPU memory, and how much additional headroom does it provide over Stage 1?",
      options: [
        "Stage 2 uses 49.25 GB per GPU ($2 \\times 13 + 14 \\times 13 / 8$), freeing 22.25 GB compared to Stage 1 — the savings come entirely from not storing the full gradient tensor",
        "Stage 2 uses 26 GB per GPU ($16 \\times 13 / 8$), freeing 45.5 GB compared to Stage 1 — both gradients and parameters are sharded in Stage 2",
        "Stage 2 uses 65 GB per GPU ($4 \\times 13 + 10 \\times 13 / 8$), freeing 6.5 GB — Stage 2 only shards half of the gradient, not the full tensor",
        "Stage 2 uses 52 GB per GPU ($2 \\times 13 + 2 \\times 13 + 12 \\times 13 / 8^2$), freeing 19.5 GB — the optimizer states are sharded quadratically in Stage 2"
      ],
      correct: 0,
      explanation: "Stage 2 partitions both gradients and optimizer states. Per-GPU: $2\\Psi$ (full parameters, still replicated) + $2\\Psi/N$ (gradient shard) + $12\\Psi/N$ (optimizer shard) = $2 \\times 13 + 14 \\times 13 / 8 = 26 + 22.75 = 48.75$ GB (approximately 49.25 GB using more precise values). Compared to Stage 1's 71.5 GB, this frees about 22.25 GB per GPU. That extra headroom comes from not storing the full gradient — instead of $2\\Psi = 26$ GB of gradients, each GPU stores only $2\\Psi/N = 3.25$ GB. Parameters remain fully replicated; that happens in Stage 3."
    },
    // Step 7: ZeRO Stage 3 / FSDP
    {
      type: "info",
      title: "ZeRO Stage 3 / FSDP — Partition Everything",
      content: "ZeRO Stage 3 takes the final step: **partition parameters** as well. Each GPU stores only $1/N$ of every component — parameters, gradients, and optimizer states.\n\nPer-GPU memory:\n\n$$\\frac{2\\Psi + 2\\Psi + 12\\Psi}{N} = \\frac{16\\Psi}{N}$$\n\nThis is the theoretical minimum — memory scales linearly with the number of GPUs. A 70B model on 64 GPUs: $16 \\times 70 / 64 = 17.5$ GB per GPU.\n\nThe price is **more communication**. Since parameters are sharded, they must be reconstructed before use:\n\n- **Forward pass**: All-gather parameters before each layer ($\\Psi$ bytes total across all layers)\n- **Backward pass**: All-gather parameters again for gradient computation ($\\Psi$ bytes), then reduce-scatter gradients ($\\Psi$ bytes)\n\nTotal communication: $3\\Psi$ bytes per GPU, compared to $2\\Psi$ for DDP — a **1.5x increase**.\n\nPyTorch implements this as **FSDP** (Fully Sharded Data Parallel). The name reflects the core idea: every piece of model state is sharded across the data-parallel group. FSDP has become the standard approach for training models too large for Stage 1 or Stage 2, and it underpins the distributed training stack at most frontier labs."
    },
    // Step 8: MC — Stage 3 communication overhead
    {
      type: "mc",
      question: "ZeRO Stage 3 / FSDP requires $3\\Psi$ bytes of communication per GPU per step, compared to $2\\Psi$ for standard DDP. Where does the extra $\\Psi$ come from?",
      options: [
        "The optimizer update in Stage 3 requires an additional all-reduce to reconcile the sharded master weights across GPUs before the next forward pass can begin",
        "Stage 3 uses FP32 for gradient communication instead of FP16, doubling the gradient transfer cost from $\\Psi$ to $2\\Psi$ and adding $\\Psi$ net overhead",
        "The extra $\\Psi$ is overhead from redundant checksum verification that FSDP performs to ensure sharded parameters are correctly reconstructed after each all-gather",
        "Parameters must be all-gathered twice (once in forward, once in backward) instead of being locally available, adding $2\\Psi$ of all-gathers while replacing the $2\\Psi$ all-reduce with a $\\Psi$ reduce-scatter"
      ],
      correct: 3,
      explanation: "In DDP, parameters are replicated so no parameter communication is needed — only the $2\\Psi$ gradient all-reduce. In Stage 3, parameters are sharded and must be reconstructed via all-gather before each use: once in the forward pass ($\\Psi$) and once in the backward pass ($\\Psi$). Gradients use a reduce-scatter ($\\Psi$) instead of a full all-reduce ($2\\Psi$). Total: $\\Psi + \\Psi + \\Psi = 3\\Psi$, which is $1.5\\times$ the DDP cost of $2\\Psi$. The extra $\\Psi$ is the net cost of the two parameter all-gathers minus the savings from using reduce-scatter instead of all-reduce for gradients."
    },
    // Step 9: FSDP Execution Flow and Prefetching
    {
      type: "info",
      title: "FSDP Execution Flow and Prefetching",
      content: "FSDP's per-layer execution requires careful scheduling to minimize GPU idle time.\n\n**Forward pass for layer $l$:**\n1. Prefetch: Begin all-gather for layer $l+1$'s parameters (runs on the network while the GPU computes)\n2. Compute: Run layer $l$'s forward pass using the full (reconstructed) parameters\n3. Discard: Free the non-local parameter shards for layer $l$ — only keep the $1/N$ shard this GPU owns\n\n**Backward pass for layer $l$:**\n1. Prefetch: Begin all-gather for layer $l-1$'s parameters\n2. Compute: Run layer $l$'s backward pass using reconstructed parameters\n3. Reduce-scatter: Distribute gradient shards — each GPU gets the $1/N$ slice it needs for its optimizer update\n4. Discard: Free full parameters and non-local gradient data for layer $l$\n\n**Prefetching is critical.** Without it, each layer's computation must wait for its all-gather to finish, creating a serial dependency: communicate, compute, communicate, compute. With prefetching, the all-gather for the next layer runs concurrently with the current layer's computation, hiding most of the communication latency.\n\n**Hybrid sharding** (also called HSDP) is a practical refinement: shard within a node (e.g., 8 GPUs connected by fast NVLink at 900 GB/s) but replicate across nodes (connected by slower InfiniBand at 400 Gb/s). This limits the high-volume all-gather traffic to the fast intra-node links while only requiring inter-node communication for the gradient reduce. It trades some memory savings for significantly lower communication overhead."
    },
    // Step 10: MC — FSDP throughput diagnosis
    {
      type: "mc",
      question: "An FSDP training run across 64 GPUs shows 40% lower throughput than the theoretical peak. Profiling reveals that GPU utilization drops periodically between layer computations. What is the most likely cause?",
      options: [
        "Communication-computation overlap is imperfect — for small layers where compute finishes faster than the next layer's all-gather, GPUs idle waiting for parameter reconstruction to complete",
        "The model's weight matrices exceed the GPU's L2 cache capacity after reconstruction, forcing repeated reads from HBM that bottleneck the matrix multiply units",
        "Python's Global Interpreter Lock serializes the CUDA kernel launches across the 64 processes, creating a scheduling bottleneck that grows linearly with GPU count",
        "The reduce-scatter operations for gradient sharding corrupt numerical precision, forcing FSDP to insert explicit synchronization barriers and gradient recomputation steps"
      ],
      correct: 0,
      explanation: "The periodic drops in GPU utilization between layers are the signature of imperfect communication-computation overlap. FSDP prefetches the next layer's parameters while computing the current layer. But if a layer has few parameters (e.g., LayerNorm, bias terms, small projection layers), its compute finishes quickly — before the all-gather for the next layer completes. The GPU then stalls waiting for the network. Additionally, the very first layer's all-gather cannot overlap with any preceding computation, and the last layer's reduce-scatter has no subsequent computation to hide behind. These \"bubbles\" accumulate across many layers and training steps, explaining the 40% throughput gap."
    },
    // Step 11: Choosing the Right Stage
    {
      type: "info",
      title: "Choosing the Right Stage",
      content: "Each ZeRO stage has a clear use case. The decision depends on whether the model state fits in GPU memory and how much headroom you need for activations.\n\n**Memory formulas** (per-GPU, $\\Psi$ params, $N$ GPUs, Adam mixed precision):\n\n| Stage | Per-GPU Memory | Communication | Overhead vs DDP |\n|---|---|---|---|\n| DDP | $16\\Psi$ | $2\\Psi$ | baseline |\n| Stage 1 | $4\\Psi + 12\\Psi/N$ | $\\sim 2\\Psi$ | ~0% |\n| Stage 2 | $2\\Psi + 14\\Psi/N$ | $\\sim 2\\Psi$ | ~0% |\n| Stage 3 / FSDP | $16\\Psi/N$ | $3\\Psi$ | ~50% more comm |\n\n**Decision framework:**\n\n- **Stage 1** (default): Use first. Eliminates optimizer redundancy at near-zero cost. Handles most 7-13B models on 8 GPUs. If per-GPU memory ($4\\Psi + 12\\Psi/N$) fits with room for activations, stop here.\n- **Stage 2**: When Stage 1 fits the model state but leaves too little room for activations. The gradient sharding frees $2\\Psi - 2\\Psi/N \\approx 2\\Psi$ bytes per GPU.\n- **Stage 3 / FSDP**: When the model state doesn't fit with Stage 2, or when training on many GPUs where the 1.5x communication overhead is acceptable. Essential for 70B+ models.\n- **ZeRO-Offload**: Extends any stage by moving shards to CPU RAM or NVMe. Trades throughput for the ability to train on fewer GPUs. Useful for resource-constrained settings but typically 2-5x slower.\n\nIn practice, most teams start with Stage 1 and only move to higher stages when memory pressure demands it."
    },
    // Step 12: MC — Practical stage selection
    {
      type: "mc",
      question: "A team has 8 A100 80 GB GPUs and wants to train a 13B model. With ZeRO Stage 1, per-GPU model state is $4 \\times 13 + 12 \\times 13 / 8 = 71.5$ GB, leaving 8.5 GB free. What is the practical concern, and which stage addresses it?",
      options: [
        "8.5 GB is insufficient for storing the model's embedding table, which at 13B parameters requires at least 20 GB — Stage 3 is needed to shard the embeddings across GPUs",
        "8.5 GB leaves no room for activation memory during the forward pass — even with aggressive activation checkpointing, a 13B model needs 15-30 GB for activations, so Stage 2 ($\\sim$49 GB model state) provides adequate headroom",
        "8.5 GB is more than enough for activations since activation checkpointing reduces memory to under 1 GB — the real concern is that Stage 1's all-gather adds 50% communication overhead, which Stage 2 eliminates",
        "8.5 GB is insufficient for the CUDA context and kernel workspace allocations, which typically require 10-15 GB — Stage 3 at $16 \\times 13 / 8 = 26$ GB per GPU is the minimum viable option"
      ],
      correct: 1,
      explanation: "The 8.5 GB headroom must accommodate activation memory (intermediate tensors saved for backpropagation), CUDA context overhead (~1-2 GB), and any temporary buffers. For a 13B model with reasonable batch sizes and sequence lengths, activation memory typically requires 15-30 GB even with activation checkpointing. With only 8.5 GB free, training would either fail with out-of-memory errors or require impractically small batch sizes. Stage 2 reduces per-GPU model state to approximately $2 \\times 13 + 14 \\times 13 / 8 \\approx 49$ GB, leaving ~31 GB for activations — comfortable headroom. Stage 2 adds no communication overhead versus Stage 1, making it the right incremental step."
    }
  ]
};
