// Focused learning module: Tensor and Pipeline Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: why data parallelism isn't enough, tensor parallelism (column/row splitting),
// pipeline parallelism, the pipeline bubble problem, 1F1B scheduling,
// and how parallelism strategies compose.
// Single-concept module: model parallelism strategies for large LLMs.

export const modelParallelismLearning = {
  id: "1.6-model-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Tensor and Pipeline Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why data parallelism isn't enough
    {
      type: "info",
      title: "When Data Parallelism Breaks Down",
      content: "Data parallelism replicates the entire model on every GPU and splits the data batch. ZeRO/FSDP shards optimizer states (and optionally parameters) across GPUs to save memory. But even with ZeRO Stage 3 (sharding everything), each GPU must still **reconstruct the full layer parameters** during the forward and backward pass.\n\nFor a 70B model with $d_{\\text{model}} = 8192$ and an FFN hidden dimension of $28{,}672$: a single FFN weight matrix $W_{\\text{up}} \\in \\mathbb{R}^{8192 \\times 28672}$ has 235M parameters — nearly 500 MB in FP16. The full FFN has ~1.4B parameters per layer. During computation, these parameters must be gathered onto each GPU, requiring the full matrix to fit in memory alongside activations.\n\nAt some model scale, even the parameters for a **single layer** exceed what one GPU can hold during computation. ZeRO-3 can shard parameters across GPUs, but gathering them creates massive communication overhead — every GPU must receive a full copy of every layer's parameters.\n\n**Model parallelism** takes a different approach: instead of replicating the model, **split the model itself** across GPUs so that no single GPU ever needs to hold all parameters."
    },
    // Step 2: MC
    {
      type: "mc",
      question: "With ZeRO Stage 3 across 8 GPUs, each GPU stores $1/8$ of the parameters at rest. During the forward pass for a given layer, what happens?",
      options: [
        "Each GPU computes on its $1/8$ shard independently, and the partial results are combined with an all-reduce to produce the correct output",
        "All 8 GPUs perform an all-gather to reconstruct the full layer parameters on every GPU, then each GPU computes the forward pass on its own data shard using the full parameters",
        "Only the GPU that owns the largest shard computes the forward pass, and the result is broadcast to the other 7 GPUs",
        "The layer computation is skipped for GPUs that don't own the majority of its parameters, and only GPUs holding at least 50% participate"
      ],
      correct: 1,
      explanation: "ZeRO Stage 3 shards parameters for memory savings at rest, but computation still requires the full parameter matrix. Before computing each layer, an all-gather collective reconstructs the full parameters on all GPUs. Each GPU then runs the forward pass on its local data shard using these full parameters. After the layer completes, the gathered parameters are discarded. This means every layer triggers an all-gather communication — the bandwidth cost is the full model size, which becomes prohibitive at extreme scales."
    },
    // Step 3: Tensor parallelism — the idea
    {
      type: "info",
      title: "Tensor Parallelism: Splitting Layers Across GPUs",
      content: "**Tensor parallelism** (TP) splits individual weight matrices across GPUs so that each GPU computes a portion of the layer's output. The partial results are combined to produce the same result as the full computation.\n\nConsider a linear layer $Y = XW$ where $X \\in \\mathbb{R}^{b \\times d}$ and $W \\in \\mathbb{R}^{d \\times h}$.\n\n**Column-parallel split**: Partition $W$ along columns into $[W_1, W_2]$ across 2 GPUs. Each GPU computes $Y_i = X W_i$, producing half the output features. The results are concatenated: $Y = [Y_1, Y_2]$. Each GPU needs only the full input $X$ and half the weight matrix.\n\n**Row-parallel split**: Partition $W$ along rows into $\\begin{bmatrix} W_1 \\\\ W_2 \\end{bmatrix}$. This requires splitting the input too: $X = [X_1, X_2]$. Each GPU computes $Y_i = X_i W_i$, producing a partial sum. The results are added: $Y = Y_1 + Y_2$ (an all-reduce). Each GPU needs half the input features, half the weight matrix, and an all-reduce at the end.\n\nIn transformers, Megatron-LM (Shoeybi et al., 2019) applies tensor parallelism to the two main sub-layers:\n- **Attention**: Split Q, K, V projections column-wise → each GPU handles a subset of attention heads\n- **FFN**: Split the up-projection column-wise and the down-projection row-wise"
    },
    // Step 4: MC
    {
      type: "mc",
      question: "A transformer's FFN has $W_{\\text{up}} \\in \\mathbb{R}^{4096 \\times 11008}$ and $W_{\\text{down}} \\in \\mathbb{R}^{11008 \\times 4096}$. With tensor parallelism degree 4, each GPU stores:",
      options: [
        "Full copies of both matrices — tensor parallelism only splits the computation, not the storage",
        "One quarter of $W_{\\text{up}}$ ($\\in \\mathbb{R}^{1024 \\times 11008}$) split along the input dimension, and the full $W_{\\text{down}}$ since the down-projection cannot be parallelized",
        "Alternating rows of each matrix (rows 0, 4, 8, ... to GPU 0; rows 1, 5, 9, ... to GPU 1) to maximize cache locality during matrix multiplication",
        "One quarter of each matrix: $W_{\\text{up}}$ shard $\\in \\mathbb{R}^{4096 \\times 2752}$ and $W_{\\text{down}}$ shard $\\in \\mathbb{R}^{2752 \\times 4096}$, totaling ~45M parameters per GPU vs ~180M for the full FFN"
      ],
      correct: 3,
      explanation: "Megatron-style TP splits $W_{\\text{up}}$ column-wise (each GPU gets $4096 \\times 2752$ — a quarter of the hidden dimension) and $W_{\\text{down}}$ row-wise (each GPU gets $2752 \\times 4096$). The column split on $W_{\\text{up}}$ means each GPU computes a quarter of the hidden activations. The row split on $W_{\\text{down}}$ means each GPU multiplies its local hidden activations by its weight shard, producing partial output sums that are combined with an all-reduce. Total per GPU: $\\sim 22.5$M parameters per matrix, $\\sim 45$M for the FFN — exactly $1/4$ of the full 180M."
    },
    // Step 5: TP communication costs
    {
      type: "info",
      title: "Tensor Parallelism Communication",
      content: "Tensor parallelism requires **synchronization within each layer**. In Megatron-LM's design, each transformer block needs:\n\n- **2 all-reduce operations in the forward pass**: one after the attention sub-layer, one after the FFN\n- **2 all-reduce operations in the backward pass**: same locations\n\nEach all-reduce communicates a tensor of shape $(b \\times s \\times d_{\\text{model}})$ — the full activation tensor. For a batch size $b = 1$, sequence length $s = 4096$, and $d_{\\text{model}} = 8192$ in BF16:\n\n$$\\text{per all-reduce} = 1 \\times 4096 \\times 8192 \\times 2 \\text{ bytes} \\approx 67 \\text{ MB}$$\n\nWith 4 all-reduces per layer and 80 layers: $4 \\times 80 \\times 67 \\text{ MB} \\approx 21 \\text{ GB}$ of communication per training step. This must happen **within** the forward/backward computation, not overlapping with it.\n\nBecause of this tight coupling, tensor parallelism requires **high-bandwidth interconnects** between GPUs. It works well within a single node (NVLink: 600-900 GB/s) but poorly across nodes (InfiniBand: 50-200 GB/s). In practice, TP degree is limited to the number of GPUs per node (typically 4 or 8)."
    },
    // Step 6: MC
    {
      type: "mc",
      question: "A cluster has 8 GPUs per node connected by NVLink (900 GB/s), and nodes are connected by InfiniBand (200 GB/s). For a model requiring TP degree 16, what is the main challenge?",
      options: [
        "TP degree 16 requires 16 attention heads minimum, which limits the model architecture choices",
        "NVLink's bandwidth is insufficient for the all-reduce volume — even intra-node TP would be too slow at degree 16",
        "The all-reduce operations within each layer must cross the inter-node InfiniBand link (200 GB/s) for 8 of the 16 GPUs, creating a $\\sim 4.5\\times$ slowdown compared to keeping TP within a single 8-GPU node",
        "Pipeline parallelism automatically replaces tensor parallelism beyond degree 8, so TP-16 is not a valid configuration"
      ],
      correct: 2,
      explanation: "With 8 GPUs per node, TP-16 spans two nodes. The all-reduce must include GPUs on both nodes, bounded by the slower inter-node link (200 GB/s vs. NVLink's 900 GB/s). Since all-reduce must complete before computation proceeds (it's on the critical path), this creates a major bottleneck. The standard solution: use TP-8 within a node (fast NVLink) and combine with pipeline parallelism or data parallelism across nodes (where communication can be overlapped with computation)."
    },
    // Step 7: Pipeline parallelism
    {
      type: "info",
      title: "Pipeline Parallelism: Splitting Layers Across GPUs",
      content: "**Pipeline parallelism** (PP) takes a different approach to model splitting: instead of splitting individual layers horizontally (like TP), it splits the model **vertically** by assigning different layers to different GPUs.\n\nA 32-layer model with PP degree 4:\n- GPU 0: layers 1-8\n- GPU 1: layers 9-16\n- GPU 2: layers 17-24\n- GPU 3: layers 25-32\n\nDuring the forward pass, data flows through the pipeline: GPU 0 computes layers 1-8 and sends the output activations to GPU 1, which computes layers 9-16 and passes to GPU 2, and so on.\n\nPP communication is **point-to-point** (one GPU sends to the next), not collective (all-to-all like TP). The data transferred is just the activation tensor at the boundary between stages — the same $(b \\times s \\times d_{\\text{model}})$ tensor as TP, but sent only once per stage, not per layer. This makes PP well-suited for **inter-node** communication where bandwidth is limited.\n\nPP also reduces per-GPU memory: each GPU stores only $L / P$ layers' parameters and activations, where $L$ is total layers and $P$ is the PP degree."
    },
    // Step 8: MC
    {
      type: "mc",
      question: "A 64-layer model uses pipeline parallelism with 4 stages. Each stage handles 16 layers. Compared to tensor parallelism with degree 4, what is the key structural difference in communication?",
      options: [
        "PP communicates only at stage boundaries (3 point-to-point transfers in the forward pass), while TP communicates within every layer (2 all-reduces $\\times$ 64 layers = 128 collectives in the forward pass)",
        "PP requires all-to-all communication at every stage boundary, while TP uses cheaper broadcast operations",
        "Both PP and TP have identical communication volumes — the difference is only in whether parameters or activations are transmitted",
        "PP has zero communication overhead because each GPU operates independently on its own layers without needing any data from other GPUs"
      ],
      correct: 0,
      explanation: "This is PP's main advantage for inter-node deployment. TP requires 2 all-reduce collectives per layer (128 total for 64 layers), all on the critical path. PP requires only point-to-point activation transfers at the 3 boundaries between stages. The total communication volume per step is much lower, and each transfer is a simple send/receive rather than a collective operation. The tradeoff is the pipeline bubble problem — GPUs sit idle while waiting for earlier stages to finish."
    },
    // Step 9: The pipeline bubble
    {
      type: "info",
      title: "The Pipeline Bubble Problem",
      content: "Naive pipeline parallelism has a critical inefficiency: **pipeline bubbles**. When the first micro-batch enters the pipeline:\n\n- Time 0: GPU 0 processes the micro-batch (GPUs 1, 2, 3 are idle)\n- Time 1: GPU 1 processes it (GPU 0 starts micro-batch 2, GPUs 2, 3 idle)\n- Time 2: GPU 2 processes it (GPUs 0, 1 on micro-batch 2-3, GPU 3 idle)\n- Time 3: All GPUs finally active\n\nThe pipeline takes $P - 1$ time steps to fill and $P - 1$ steps to drain, where $P$ is the number of stages. With $M$ micro-batches, the bubble fraction is:\n\n$$\\text{Bubble fraction} = \\frac{P - 1}{M + P - 1}$$\n\nWith $P = 4$ stages and $M = 4$ micro-batches: bubble = $3/7 \\approx 43\\%$ — nearly half the compute is wasted! To make the bubble tolerable ($< 5\\%$), you need $M \\gg P$. With $P = 4$ and $M = 64$: bubble = $3/67 \\approx 4.5\\%$.\n\nThis means pipeline parallelism requires **large effective batch sizes** (many micro-batches) to be efficient, which can conflict with learning rate scaling and convergence properties."
    },
    // Step 10: MC
    {
      type: "mc",
      question: "A training setup uses pipeline parallelism with 8 stages and 16 micro-batches. What fraction of total compute time is wasted in the pipeline bubble?",
      options: [
        "50% — with 8 stages, half the GPUs are always idle regardless of the number of micro-batches",
        "About 30% — the bubble fraction is $(8-1)/(16+8-1) = 7/23 \\approx 30\\%$, meaning nearly a third of compute is wasted on pipeline fill and drain",
        "About 6% — the 16 micro-batches keep the pipeline busy for most of the time, with only brief idle periods at the start and end",
        "0% — modern pipeline scheduling (1F1B) completely eliminates the bubble by interleaving forward and backward passes"
      ],
      correct: 1,
      explanation: "Bubble fraction = $(P-1)/(M+P-1) = 7/23 \\approx 30.4\\%$. With 8 stages and only 16 micro-batches, the pipeline spends a significant fraction of time in fill/drain phases. To reduce this below 5%, you'd need $M > 7/0.05 - 7 = 133$ micro-batches — impractical for most training setups. 1F1B scheduling (interleaved forward and backward passes) helps reduce peak memory but doesn't eliminate the bubble. The practical solution is to keep $P$ small (4-8) and $M$ large."
    },
    // Step 11: 1F1B scheduling
    {
      type: "info",
      title: "1F1B Scheduling: Reducing Memory, Not Bubbles",
      content: "**1F1B (One Forward, One Backward)** is a pipeline scheduling strategy that interleaves forward and backward micro-batches to reduce peak memory.\n\nIn **naive (GPipe) scheduling**: all $M$ forward passes run first, then all $M$ backward passes. This means GPU 0 must store activations for all $M$ micro-batches simultaneously — peak activation memory scales as $O(M)$.\n\nIn **1F1B scheduling**: after the pipeline fills (all stages are active), each GPU alternates between one forward and one backward micro-batch. As soon as a backward pass completes for a micro-batch, its stored activations are freed.\n\nThe memory benefit is dramatic:\n- GPipe: each stage stores activations for $M$ micro-batches\n- 1F1B: each stage stores activations for at most $P$ micro-batches (the pipeline depth)\n\nWith $M = 64$ and $P = 4$: GPipe stores 64$\\times$ the activations; 1F1B stores only 4$\\times$. This is the difference between feasible and impossible for large models with long sequences.\n\nImportant: 1F1B has the **same bubble fraction** as GPipe — $(P-1)/(M+P-1)$. It reduces memory, not compute waste. The bubble is a fundamental property of the pipeline structure, not the scheduling order."
    },
    // Step 12: MC
    {
      type: "mc",
      question: "Switching from GPipe to 1F1B scheduling with pipeline parallelism ($P = 8$ stages, $M = 32$ micro-batches) would:",
      options: [
        "Reduce training throughput because interleaving forward and backward passes prevents efficient batching of matrix multiplications",
        "Eliminate the pipeline bubble entirely by keeping all 8 GPUs busy at all times through the interleaved schedule",
        "Require twice as many micro-batches ($M = 64$) to achieve the same throughput because each stage processes half as many micro-batches at a time",
        "Reduce peak activation memory from $O(M) = O(32)$ per stage to $O(P) = O(8)$ per stage — a 4$\\times$ reduction — while keeping the same pipeline bubble fraction"
      ],
      correct: 3,
      explanation: "1F1B's benefit is purely memory: by interleaving forward and backward passes, activations are freed earlier, reducing peak storage from $M = 32$ to $P = 8$ micro-batches per stage. The bubble fraction remains $(8-1)/(32+8-1) = 7/39 \\approx 18\\%$ in both cases — 1F1B doesn't improve compute utilization. Throughput is identical to GPipe (same amount of useful compute per unit time). The memory savings are critical though: they can mean the difference between fitting in GPU memory or not."
    },
    // Step 13: Composing parallelism strategies
    {
      type: "info",
      title: "3D Parallelism: Composing TP, PP, and DP",
      content: "Large-scale training combines all three parallelism strategies, each operating at a different scale:\n\n**Tensor parallelism (TP)**: Within a node. Splits layers across GPUs connected by NVLink. Typical degree: 4 or 8 (one node). Handles the per-layer memory and compute.\n\n**Pipeline parallelism (PP)**: Across a few nodes. Splits layer groups across stages. Typical degree: 4-8. Point-to-point communication tolerates lower inter-node bandwidth.\n\n**Data parallelism (DP)**: Across many nodes. Replicates the (already TP+PP split) model and distributes data. Gradient all-reduce is overlapped with backward computation.\n\nThe total GPU count is $N = TP \\times PP \\times DP$.\n\n**Example**: Training a 175B model (GPT-3 scale) on 1024 A100 GPUs:\n- TP = 8: each node's 8 GPUs share every layer via tensor parallelism\n- PP = 8: 8 pipeline stages across 8 nodes, each stage holding ~10 layers\n- DP = 16: 16 replicas of the full TP+PP arrangement\n- Total: $8 \\times 8 \\times 16 = 1024$ GPUs\n\nMegatron-LM, DeepSpeed, and similar frameworks automate this decomposition. The key design principle: use TP for intra-node (high bandwidth needed), PP for near-neighbor inter-node (moderate bandwidth), and DP for the remaining GPUs (communication can overlap with compute)."
    },
    // Step 14: MC
    {
      type: "mc",
      question: "A team has 256 GPUs across 32 nodes (8 GPUs/node, NVLink within nodes, InfiniBand between nodes). They want to train a model too large for 8 GPUs. Which parallelism configuration best matches the hardware?",
      options: [
        "TP=8, PP=4, DP=8 — tensor parallelism within each node (NVLink), pipeline parallelism across 4 nearby nodes (moderate bandwidth), data parallelism across the remaining 8 groups",
        "TP=1, PP=32, DP=8 — use a 32-stage pipeline across all nodes and data parallelism within nodes for maximum parameter sharding",
        "TP=8, PP=1, DP=32 — use tensor parallelism within each node and pure data parallelism across nodes, avoiding pipeline bubbles entirely",
        "TP=32, PP=1, DP=8 — maximize tensor parallelism for the highest compute efficiency, using InfiniBand for the TP all-reduce across nodes"
      ],
      correct: 0,
      explanation: "The configuration should match communication patterns to hardware topology. TP=8 uses the fast intra-node NVLink for the frequent all-reduce operations. PP=4 spans 4 nodes — the point-to-point activation transfers tolerate the slower InfiniBand. DP=8 handles the remaining parallelism with gradient all-reduce that can overlap with backward computation. Total: $8 \\times 4 \\times 8 = 256$ GPUs. TP=32 would require cross-node all-reduce for every layer — far too slow. PP=32 would create a 32-stage pipeline with devastating bubble overhead."
    }
  ]
};
