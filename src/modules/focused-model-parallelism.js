// Focused learning module: Tensor and Pipeline Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: why data parallelism alone isn't enough, tensor parallelism (column/row splits),
// pipeline parallelism (layer sharding), micro-batching, and the bubble problem.
// Single-concept module: model parallelism strategies for training models too large for one GPU.

export const modelParallelismLearning = {
  id: "1.6-model-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Tensor and Pipeline Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why model parallelism?
    {
      type: "info",
      title: "When Data Parallelism Isn't Enough",
      content: "Data parallelism replicates the entire model on every GPU. For a 70B parameter model with Adam, the model state alone is $(2 + 2 + 12) \\times 70 = 1{,}120$ GB. Even ZeRO Stage 3 (which shards everything) requires each GPU to **reconstruct full layers** during forward and backward passes — and a single transformer layer of a 70B model can exceed the memory of one GPU when activations are included.\n\n**Model parallelism** takes a fundamentally different approach: instead of replicating the model everywhere, it **splits the model itself** across GPUs. Each GPU holds only a portion of the model's parameters, computes only part of each operation, and communicates intermediate results with other GPUs.\n\nThere are two main strategies:\n- **Tensor parallelism (TP)**: splits individual layers across GPUs — each GPU computes part of a matrix multiplication\n- **Pipeline parallelism (PP)**: assigns different layers to different GPUs — each GPU handles a contiguous block of the network\n\nModern large-scale training (GPT-4, Llama, etc.) combines both with data parallelism in a strategy called **3D parallelism**: DP across nodes, TP within a node (where inter-GPU bandwidth is high), and PP across nodes."
    },
    // Step 2: MC — why DP isn't sufficient
    {
      type: "mc",
      question: "A team wants to train a 175B parameter model. Each GPU has 80 GB of memory. Why can't they use pure data parallelism (without ZeRO), even with 1,000 GPUs?",
      options: [
        "The communication overhead of all-reduce across 1,000 GPUs makes gradient synchronization prohibitively slow",
        "Data parallelism replicates the full model on every GPU, and the 175B model's state (~2.8 TB with Adam) doesn't fit on any single 80 GB GPU",
        "Data parallelism requires the global batch size to equal the number of GPUs, and 1,000 samples per batch leads to poor convergence",
        "Pure data parallelism is limited to 8 GPUs because the all-reduce ring topology can't scale beyond a single node"
      ],
      correct: 1,
      explanation: "Pure data parallelism (without ZeRO) replicates the entire model state on every GPU. With 175B parameters and Adam, the model state is roughly $(2 + 2 + 12) \\times 175 \\approx 2{,}800$ GB = 2.8 TB. This must fit on each individual GPU, but each GPU only has 80 GB. No matter how many GPUs you add, each one still needs to hold the full model. This is the fundamental limitation that model parallelism addresses — it distributes the model itself across GPUs rather than replicating it."
    },
    // Step 3: Tensor parallelism — the idea
    {
      type: "info",
      title: "Tensor Parallelism: Splitting a Matrix Multiply",
      content: "The core computation in a transformer is the linear projection $Y = XW + b$, where $X \\in \\mathbb{R}^{s \\times d}$ (sequence length × hidden dim) and $W \\in \\mathbb{R}^{d \\times h}$ (hidden dim × output dim). **Tensor parallelism** splits $W$ across GPUs so each GPU computes part of $Y$.\n\nThe two fundamental splits:\n\n**Column parallelism**: partition $W$ along columns. With $N$ GPUs, GPU $i$ holds $W_i \\in \\mathbb{R}^{d \\times h/N}$ and computes $Y_i = XW_i$. Each GPU receives the **full input** $X$ and produces $1/N$ of the output columns. The partial results $[Y_1, Y_2, \\ldots, Y_N]$ are concatenated to form $Y$.\n\n**Row parallelism**: partition $W$ along rows. GPU $i$ holds $W_i \\in \\mathbb{R}^{d/N \\times h}$ and receives the corresponding slice $X_i$ of the input. Each GPU computes $Y_i = X_i W_i$, which is a **partial sum** of the full output. The partial results are summed via all-reduce: $Y = \\sum_i Y_i$.\n\nThe key insight of Megatron-LM (Shoeybi et al., 2019) is that by pairing column-parallel with row-parallel layers, you can **minimize communication**. In a transformer MLP block ($h \\to 4h \\to h$), the first linear uses column parallelism and the second uses row parallelism, requiring only one all-reduce in the forward pass and one in the backward pass."
    },
    // Step 4: MC — tensor parallelism mechanics
    {
      type: "mc",
      question: "In Megatron-style tensor parallelism for a transformer MLP block with 2 linear layers, the first layer uses column parallelism and the second uses row parallelism. How many all-reduce operations are needed per MLP block in the forward pass?",
      options: [
        "Zero — column and row parallelism are arranged so that each GPU's local computation feeds directly into the next without any communication",
        "One — the row-parallel second layer produces partial sums that must be all-reduced to get the correct output, but the column-parallel first layer needs no communication because each GPU gets the full input",
        "Two — one all-reduce after each linear layer to synchronize partial results across GPUs",
        "Four — each linear layer requires an all-gather before and an all-reduce after, totaling four collective operations"
      ],
      correct: 1,
      explanation: "Column parallelism gives each GPU the full input and produces a shard of the output — no communication needed. This sharded output feeds directly into the row-parallel second layer (each GPU already has the correct slice). Row parallelism produces partial sums that require one all-reduce to combine. So the total is **one all-reduce per MLP block** in the forward pass. This elegant pairing is the key insight of Megatron-LM — it minimizes the communication-to-computation ratio."
    },
    // Step 5: TP for attention
    {
      type: "info",
      title: "Tensor Parallelism in Attention Layers",
      content: "Self-attention computes $Q = XW_Q$, $K = XW_K$, $V = XW_V$, applies attention, then projects with $W_O$. Tensor parallelism exploits the fact that **attention heads are independent**.\n\nWith $N$ GPUs and $H$ attention heads (where $H$ is divisible by $N$), each GPU handles $H/N$ heads:\n- GPU $i$ holds the $Q$, $K$, $V$ projection weights for its $H/N$ heads\n- Each GPU computes attention for its local heads independently (column parallelism)\n- The output projection $W_O$ is split by rows — each GPU holds $W_O^{(i)} \\in \\mathbb{R}^{(d/N) \\times d}$\n- The row-parallel output produces partial sums that are all-reduced\n\nLike the MLP block, the attention block requires **one all-reduce in the forward pass** and one in the backward pass. The total per transformer layer is thus 2 all-reduces forward + 2 backward = **4 all-reduce operations per layer**.\n\nThis is why tensor parallelism is used **within a node** where GPUs are connected by NVLink (600+ GB/s bandwidth) rather than between nodes connected by InfiniBand (~50 GB/s). The frequent all-reduces demand high bandwidth."
    },
    // Step 6: MC — TP bandwidth requirements
    {
      type: "mc",
      question: "A training cluster has 8 GPUs per node connected by NVLink (600 GB/s) and nodes connected by InfiniBand (50 GB/s). A 70B model uses 8-way tensor parallelism. Why is it critical to place all 8 TP GPUs within the same node?",
      options: [
        "NVLink supports special collective operations that InfiniBand cannot execute, so tensor parallelism is technically impossible across nodes",
        "Each transformer layer requires multiple all-reduce operations that transfer activation-sized tensors at every step — the 12x bandwidth gap between NVLink and InfiniBand would make inter-node TP a severe bottleneck",
        "Tensor parallelism requires shared memory between GPUs, which is only available within a node via NVLink's unified memory architecture",
        "InfiniBand adds 100ms of latency per operation, meaning the 4 all-reduces per layer would add 400ms of pure communication overhead per layer"
      ],
      correct: 1,
      explanation: "Tensor parallelism requires all-reduce communication **within every transformer layer** — 4 times per layer (2 forward, 2 backward), each transferring activation tensors of size $O(s \\times d)$. With hidden dim $d = 8192$ and sequence length $s = 4096$, each all-reduce moves ~67 MB. Across all layers and forward/backward passes, this is enormous aggregate bandwidth demand. NVLink at 600 GB/s handles this efficiently; InfiniBand at 50 GB/s would make the all-reduces the dominant cost, destroying GPU utilization. The communication pattern is too frequent and too large for cross-node links."
    },
    // Step 7: Pipeline parallelism — the idea
    {
      type: "info",
      title: "Pipeline Parallelism: Splitting by Layers",
      content: "**Pipeline parallelism** (PP) assigns different layers of the network to different GPUs. A model with $L$ layers distributed across $P$ GPUs gives each GPU a contiguous block of $L/P$ layers (called a **stage**).\n\nThe forward pass flows sequentially: GPU 0 processes its layers and sends the output activations to GPU 1, which processes its layers and sends to GPU 2, and so on. The backward pass flows in reverse.\n\nThe advantage over tensor parallelism: PP only communicates **between stages** — the activation tensors at stage boundaries. This is a single point-to-point send/receive per stage per micro-batch, far less communication than TP's all-reduces within every layer.\n\nThe problem is the **pipeline bubble**. In naive PP, when GPU 0 is computing the forward pass, GPUs 1-3 are idle. When GPU 3 is computing the backward pass, GPUs 0-2 are idle. If each stage takes time $t$, the total time is roughly $2Pt$ (forward + backward), but the useful compute per GPU is only $2t$. The **bubble fraction** is:\n\n$$\\text{bubble} = \\frac{P - 1}{P}$$\n\nWith 4 stages, 75% of GPU time is wasted in the bubble. This is unacceptable — and the solution is **micro-batching**."
    },
    // Step 8: MC — pipeline bubble
    {
      type: "mc",
      question: "A model is split across 8 GPUs using pipeline parallelism with no micro-batching (naive PP). What fraction of total GPU-time is spent idle (the pipeline bubble)?",
      options: [
        "12.5% — only the first and last GPUs have any idle time, and they're each idle for 1/8 of the total time",
        "50% — half the time is forward pass and half is backward, so on average each GPU is idle for half the computation",
        "87.5% — each GPU is active for only 1 forward + 1 backward stage out of 8 forward + 8 backward total phases",
        "0% — all GPUs are always computing because the backward pass on earlier stages overlaps with the forward pass on later stages"
      ],
      correct: 2,
      explanation: "With naive PP and $P = 8$ stages, the bubble fraction is $(P-1)/P = 7/8 = 87.5\\%$. The pipeline processes one batch sequentially: GPU 0 computes forward, sends to GPU 1, waits. Then GPU 1 computes forward, sends to GPU 2, etc. The total timeline has $P$ forward steps + $P$ backward steps = $2P$ time slots, but each GPU is active for only 2 of them (its forward and backward). So utilization is $2/(2P) = 1/P = 12.5\\%$, meaning $87.5\\%$ idle time. This is why naive PP is never used in practice."
    },
    // Step 9: Micro-batching (GPipe and 1F1B)
    {
      type: "info",
      title: "Micro-Batching: Filling the Pipeline Bubble",
      content: "**Micro-batching** splits a mini-batch into $M$ smaller micro-batches and feeds them into the pipeline in sequence. While GPU 0 processes micro-batch 2's forward pass, GPU 1 processes micro-batch 1's forward pass — the pipeline fills up.\n\n**GPipe** (Huang et al., 2019) uses the schedule: all $M$ micro-batch forwards, then all $M$ micro-batch backwards. The bubble shrinks to:\n\n$$\\text{bubble} = \\frac{P - 1}{M + P - 1}$$\n\nWith $P = 8$ stages and $M = 24$ micro-batches: bubble $= 7/31 \\approx 22.6\\%$, compared to $87.5\\%$ without micro-batching.\n\n**1F1B** (one-forward-one-backward) interleaves forwards and backwards more tightly. Once the pipeline is full, each GPU alternates: one forward micro-batch, one backward micro-batch. This achieves the same bubble fraction as GPipe but with a crucial advantage: **lower peak memory**. GPipe must store activations for all $M$ micro-batches simultaneously during the forward phase, while 1F1B only stores activations for at most $P$ micro-batches at any time.\n\nWith enough micro-batches ($M \\gg P$), the bubble becomes negligible and pipeline parallelism achieves near-linear scaling."
    },
    // Step 10: MC — micro-batching tradeoffs
    {
      type: "mc",
      question: "A pipeline-parallel setup uses $P = 4$ stages with GPipe scheduling. The team increases from $M = 4$ to $M = 32$ micro-batches. What happens to the bubble fraction and peak activation memory?",
      options: [
        "Bubble decreases from 43% to 8.6%, and peak activation memory increases 8x because GPipe must store activations for all micro-batches during the all-forward phase",
        "Bubble decreases from 43% to 8.6%, and peak activation memory stays the same because each micro-batch is proportionally smaller",
        "Bubble stays roughly the same because it depends on the number of stages, not micro-batches — but memory increases due to more stored activations",
        "Bubble decreases from 75% to 8.6%, and peak activation memory is unchanged because activation checkpointing eliminates the need to store intermediate activations"
      ],
      correct: 0,
      explanation: "Bubble fraction: $(P-1)/(M+P-1)$. With $M=4$: $3/7 \\approx 43\\%$. With $M=32$: $3/35 \\approx 8.6\\%$. More micro-batches fill the pipeline more efficiently. However, GPipe runs all forward passes before any backward passes, so it must store activations for all $M$ micro-batches. Going from 4 to 32 micro-batches means 8x more stored activations in the forward phase. This is exactly the tradeoff that 1F1B scheduling addresses — it limits peak activation memory to $O(P)$ instead of $O(M)$."
    },
    // Step 11: 3D parallelism
    {
      type: "info",
      title: "3D Parallelism: Combining All Three Strategies",
      content: "Large-scale training combines data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP) — known as **3D parallelism**. Each strategy has different communication characteristics that determine where it's best applied:\n\n**Tensor parallelism**: frequent all-reduces within every layer → needs highest bandwidth → placed **within a node** (NVLink)\n\n**Pipeline parallelism**: point-to-point transfers between stages → moderate bandwidth, tolerates some latency → placed **across nearby nodes** (InfiniBand)\n\n**Data parallelism**: one all-reduce per training step (gradient sync) → lowest frequency, largest message → placed **across all remaining GPUs**\n\nExample: training a 175B model on 512 GPUs (64 nodes × 8 GPUs/node):\n- TP = 8 (within each node)\n- PP = 8 (across 8 nodes per pipeline)\n- DP = 8 (8 independent pipeline replicas)\n- Total: $8 \\times 8 \\times 8 = 512$ GPUs\n\nThe TP dimension handles the per-layer splits, PP handles the cross-layer splits, and DP provides throughput scaling. Each dimension multiplies the total GPU count, and the product must equal the cluster size."
    },
    // Step 12: MC — 3D parallelism configuration
    {
      type: "mc",
      question: "A cluster has 256 GPUs arranged as 32 nodes with 8 GPUs each (NVLink within nodes, InfiniBand between nodes). A training run uses TP=8, PP=4, DP=8. Which configuration correctly places each parallelism dimension?",
      options: [
        "TP across 8 nodes (one GPU per node), PP within 4 GPUs of a single node, DP across the remaining GPUs — this minimizes intra-node communication",
        "TP within each 8-GPU node, PP across 4 nodes forming a pipeline, DP across 8 pipeline replicas — this matches communication frequency to available bandwidth",
        "PP within each node (2 GPUs per stage), TP across 4 nodes (2 GPUs per node), DP across the remaining 8 groups — this balances load evenly",
        "All three dimensions are interchangeable — modern InfiniBand is fast enough that placement doesn't meaningfully affect training throughput"
      ],
      correct: 1,
      explanation: "The placement follows a bandwidth hierarchy. TP requires all-reduces within every transformer layer (highest communication frequency) → must use the highest bandwidth link → NVLink within a node (8 GPUs). PP transfers activations between pipeline stages (moderate frequency) → InfiniBand between nodes. Each pipeline spans 4 nodes. DP synchronizes gradients once per step (lowest frequency) → whatever bandwidth remains. 8 pipeline replicas run in parallel. Total: $8 \\times 4 \\times 8 = 256$ GPUs. Misplacing TP across nodes would bottleneck on InfiniBand's ~12x lower bandwidth."
    },
    // Step 13: Sequence parallelism
    {
      type: "info",
      title: "Sequence Parallelism: Distributing the Non-Tensor-Parallel Regions",
      content: "Tensor parallelism splits the weight matrices, but operations like **LayerNorm** and **dropout** operate on the full hidden dimension and are replicated across all TP GPUs. This is wasteful — each GPU computes the same LayerNorm on the same activations.\n\n**Sequence parallelism** (SP) addresses this by splitting these operations along the **sequence dimension** instead. Between tensor-parallel regions, activations of shape $(s, d)$ are partitioned so each GPU holds $(s/N, d)$ — a different chunk of tokens but the full hidden dimension.\n\nThe transitions work naturally:\n- Before a column-parallel linear layer: gather along sequence dim, scatter along hidden dim → all-gather\n- After a row-parallel linear layer: gather along hidden dim, scatter along sequence dim → reduce-scatter\n\nThese all-gather and reduce-scatter operations have the **same communication volume** as the all-reduces they replace, but sequence parallelism reduces **activation memory** by a factor of $N$ for the LayerNorm and dropout regions. For long sequences, this memory saving is substantial and enables larger batch sizes or longer context lengths."
    },
    // Step 14: MC — putting it all together
    {
      type: "mc",
      question: "A team is designing the parallelism strategy for a 405B parameter model on 2,048 GPUs (256 nodes × 8 GPUs/node). They need TP=8 within each node. Given that the model has 126 layers, which pipeline parallelism degree best balances bubble overhead against memory constraints?",
      options: [
        "PP=1 (no pipeline parallelism) — with TP=8 and ZeRO Stage 3 across the remaining 256-way DP, each GPU's memory share is only $405B \\times 16 / 2048 \\approx 3.2$ GB, which easily fits",
        "PP=126 (one layer per stage) — this minimizes per-GPU memory to a single layer, and with enough micro-batches the bubble can be made arbitrarily small",
        "PP=16 (each stage holds ~8 layers, spanning 16 nodes per pipeline) — this balances per-stage memory (~8 layers worth) against bubble overhead, giving DP=16 pipeline replicas",
        "PP=2 (63 layers per stage) — minimal bubble overhead at $(2-1)/(M+1)$, and the two stages split the 405B parameters roughly in half"
      ],
      correct: 2,
      explanation: "PP=1 fails because even with TP=8, each GPU must hold $405B/8$ parameters' worth of model state — optimizer states alone would be $12 \\times 405/8 \\approx 607$ GB per GPU, far exceeding 80 GB. PP=126 would give 125/($M$+125) bubble — even with 512 micro-batches, that's 20% idle time and impractical coordination. PP=2 requires each stage to hold ~63 layers — with TP=8, that's still $\\sim$50 GB of optimizer states per GPU plus activations, likely too tight. PP=16 gives each stage ~8 layers, making per-GPU memory manageable, and the bubble with $M=64$ micro-batches is only $15/(64+15) \\approx 19\\%$, acceptable for this scale. It yields $2048/(8 \\times 16) = 16$-way DP for throughput."
    }
  ]
};
