// Focused learning module: Pipeline Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: layer-wise model splitting, the pipeline bubble, GPipe micro-batching,
// 1F1B scheduling, interleaved pipeline schedules, and PP in 3D parallelism.
// Single-concept module: pipeline parallelism for distributing model layers across GPUs.

export const pipelineParallelismLearning = {
  id: "1.6-pipeline-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Pipeline Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Layer-wise model splitting
    {
      type: "info",
      title: "Layer-Wise Model Splitting",
      content: "**Pipeline parallelism** (PP) distributes a model across GPUs by assigning contiguous blocks of layers — called **stages** — to different devices. A model with $L$ layers split across $P$ GPUs gives each GPU a stage of $L/P$ consecutive layers.\n\nThe forward pass flows sequentially through stages: GPU 0 computes its layers and sends the output activation tensor to GPU 1, which computes its layers and sends to GPU 2, and so on. The backward pass flows in reverse: gradients propagate from the last stage back to the first.\n\nThe communication pattern is remarkably simple. At each stage boundary, the sending GPU performs a **point-to-point send** of the activation tensor (or gradient tensor during backward), and the receiving GPU performs a matching **point-to-point receive**. This is fundamentally different from tensor parallelism (TP), which requires **all-reduce** operations within every single layer.\n\nWhy does this matter? An all-reduce across $N$ GPUs transfers $2(N-1)/N$ times the data size in total — and TP triggers this multiple times per layer. Pipeline parallelism's point-to-point transfers happen only at the $P - 1$ stage boundaries, and each transfer involves just one sender and one receiver. This makes PP far more tolerant of lower-bandwidth interconnects, which is why it is typically placed **across nodes** connected by InfiniBand rather than within a node."
    },
    // Step 2: MC — PP vs TP communication
    {
      type: "mc",
      question: "A 96-layer model is split across 8 GPUs using pipeline parallelism (12 layers per stage). During one forward pass of a single micro-batch, how many inter-GPU communication events occur, and what type are they?",
      options: [
        "96 all-reduce operations — one per layer, since each layer's output must be synchronized across all 8 GPUs before the next layer can proceed",
        "8 point-to-point transfers — one per GPU, as each GPU sends its full stage output to every other GPU",
        "7 point-to-point send/receive pairs — one at each of the 7 stage boundaries, each involving only the two adjacent GPUs",
        "16 all-gather operations — two per stage boundary (one for activations, one for auxiliary state), each gathering data from all GPUs"
      ],
      correct: 2,
      explanation: "With $P = 8$ stages, there are $P - 1 = 7$ stage boundaries. At each boundary, one GPU sends an activation tensor to the next GPU via a point-to-point transfer. No collective operations (all-reduce, all-gather) are involved — each communication event involves exactly two GPUs. This is the key advantage of pipeline parallelism over tensor parallelism: TP requires all-reduce operations within every layer (multiple times per layer), while PP only needs simple point-to-point transfers at the boundaries between contiguous blocks of layers."
    },
    // Step 3: The pipeline bubble
    {
      type: "info",
      title: "The Pipeline Bubble",
      content: "Pipeline parallelism has an Achilles' heel: the **pipeline bubble**. In naive PP with $P$ stages and a single batch, the stages execute sequentially. When GPU 0 computes the forward pass for its layers, GPUs 1 through $P - 1$ sit completely idle. When GPU $P - 1$ computes backward, GPUs 0 through $P - 2$ are idle.\n\nThe timeline looks like this: $P$ sequential forward steps followed by $P$ sequential backward steps, for a total wall-clock time proportional to $2P$. But each individual GPU is active for only 2 of those $2P$ steps (one forward, one backward). The **bubble fraction** — the proportion of total GPU-time spent idle — is:\n\n$$\\text{bubble} = \\frac{P - 1}{P}$$\n\nLet's see how devastating this is:\n- $P = 2$ stages: bubble $= 1/2 = 50\\%$\n- $P = 4$ stages: bubble $= 3/4 = 75\\%$\n- $P = 8$ stages: bubble $= 7/8 = 87.5\\%$\n- $P = 16$ stages: bubble $= 15/16 = 93.75\\%$\n\nWith 8 stages, each GPU is productive for only $12.5\\%$ of the total time. You are paying for 8 GPUs but getting the effective throughput of 1. This is catastrophic and makes naive pipeline parallelism completely impractical. The solution, as we will see, is to inject multiple micro-batches into the pipeline so that stages can work on different micro-batches simultaneously."
    },
    // Step 4: MC — bubble fraction calculation
    {
      type: "mc",
      question: "A team deploys naive pipeline parallelism (no micro-batching) with 8 stages. They observe that training is barely faster than running on a single GPU. What bubble fraction explains this, and what formula produces it?",
      options: [
        "Bubble is $87.5\\%$ from $(P-1)/P = 7/8$, meaning each GPU is productive for only $12.5\\%$ of wall-clock time — 8 GPUs yield roughly 1 GPU's worth of throughput",
        "Bubble is $50\\%$ from $(P-1)/(2P) = 7/16$, meaning each GPU is idle for about half the time — this halves throughput but doesn't explain single-GPU-level performance",
        "Bubble is $75\\%$ from $1 - 1/\\sqrt{P} = 1 - 1/\\sqrt{8}$, meaning utilization scales as the square root of stage count rather than linearly",
        "Bubble is $100\\%$ from $(P-1)/(P-1) = 1$, because with 8 stages no two GPUs can ever be active simultaneously in a sequential pipeline"
      ],
      correct: 0,
      explanation: "The bubble fraction for naive PP is $(P-1)/P$. With $P = 8$: $(8-1)/8 = 7/8 = 87.5\\%$. Each GPU spends $87.5\\%$ of the time idle, so the 8-GPU setup achieves only $8 \\times 12.5\\% = 1$ GPU-equivalent of useful compute. This explains why naive PP is nearly as slow as a single GPU — you pay for 8 GPUs but the sequential stage execution means only one GPU is working at any given moment during either the forward or backward pass."
    },
    // Step 5: GPipe — micro-batching
    {
      type: "info",
      title: "GPipe: Micro-Batching to Fill the Bubble",
      content: "**GPipe** (Huang et al., 2019) solves the bubble problem by splitting each mini-batch into $M$ **micro-batches** and injecting them into the pipeline one after another. While GPU 0 processes micro-batch 2, GPU 1 is already processing micro-batch 1 — the pipeline fills up and multiple stages work simultaneously.\n\nGPipe uses a simple two-phase schedule: run all $M$ micro-batch forward passes first, then run all $M$ micro-batch backward passes. Gradients are accumulated across micro-batches and applied in a single optimizer step at the end.\n\nThe bubble shrinks dramatically. With $M$ micro-batches and $P$ stages:\n\n$$\\text{bubble} = \\frac{P - 1}{M + P - 1}$$\n\nExamples with $P = 8$:\n- $M = 8$: bubble $= 7/15 \\approx 46.7\\%$\n- $M = 24$: bubble $= 7/31 \\approx 22.6\\%$\n- $M = 56$: bubble $= 7/63 \\approx 11.1\\%$\n\nAs $M \\to \\infty$, the bubble vanishes. In practice, $M \\gg P$ is needed for good efficiency.\n\nThe tradeoff: during the all-forward phase, GPipe must **store activations for all $M$ micro-batches** simultaneously, because the backward passes haven't started yet. Peak activation memory scales as $O(M)$ — with 56 micro-batches, you store 56 copies of each stage's intermediate activations. This memory cost is GPipe's main limitation."
    },
    // Step 6: MC — GPipe bubble and memory tradeoff
    {
      type: "mc",
      question: "A GPipe setup uses $P = 4$ stages. The team increases from $M = 4$ to $M = 32$ micro-batches. How do the bubble fraction and peak activation memory change?",
      options: [
        "Bubble drops from $75\\%$ to $8.6\\%$; peak activation memory doubles because each additional micro-batch adds only a marginal amount of stored activations",
        "Bubble drops from $42.9\\%$ to $8.6\\%$; peak activation memory increases $8\\times$ because GPipe stores activations for all $M$ micro-batches during the all-forward phase",
        "Bubble stays at $75\\%$ because it depends only on $P$, not $M$; peak activation memory increases $8\\times$ due to storing more micro-batches",
        "Bubble drops from $42.9\\%$ to $8.6\\%$; peak activation memory stays constant because each micro-batch is $8\\times$ smaller, canceling out the $8\\times$ more micro-batches"
      ],
      correct: 1,
      explanation: "Bubble fraction is $(P-1)/(M+P-1)$. With $M=4$: $3/7 \\approx 42.9\\%$. With $M=32$: $3/35 \\approx 8.6\\%$ — a major improvement. However, GPipe runs all forward passes before any backward passes, meaning activations for all $M$ micro-batches are stored simultaneously in the forward phase. Going from $M=4$ to $M=32$ means $8\\times$ more stored activations. The micro-batch size shrinks by $8\\times$ (keeping the total mini-batch fixed), but there are $8\\times$ more of them, so each stage still holds $8\\times$ more activation tensors at peak. This memory scaling is the fundamental limitation that 1F1B scheduling addresses."
    },
    // Step 7: 1F1B scheduling
    {
      type: "info",
      title: "1F1B Scheduling: Interleaving Forward and Backward",
      content: "**1F1B** (one-forward-one-backward) scheduling achieves the same bubble fraction as GPipe but with dramatically lower peak memory. The schedule has three phases:\n\n**Warmup phase**: The first $P$ micro-batches enter the pipeline sequentially, just like GPipe. Each stage completes its forward pass and sends activations downstream. After $P$ forward passes, the pipeline is full — every stage has at least one micro-batch's activations stored.\n\n**Steady state**: Once the pipeline is full, each stage alternates: compute one **forward** micro-batch, then one **backward** micro-batch. The backward pass consumes and releases the activations from an earlier micro-batch. This is the key insight — by interleaving backward passes during the forward phase, activations are freed continuously rather than accumulating.\n\n**Cooldown phase**: After all forward micro-batches are issued, the remaining $P$ backward passes drain the pipeline.\n\nThe bubble fraction remains $(P-1)/(M+P-1)$ — identical to GPipe. But peak activation memory is now $O(P)$ instead of $O(M)$. Why? During steady state, at most $P$ micro-batches' activations are live at any time. Each backward pass releases one set of activations just as a new forward pass creates one. Since typically $M \\gg P$, this is a massive memory saving.\n\nExample: $P = 8$, $M = 64$. GPipe stores 64 micro-batches' activations. 1F1B stores at most 8. That's an $8\\times$ reduction in peak activation memory with no increase in bubble time."
    },
    // Step 8: MC — 1F1B vs GPipe memory
    {
      type: "mc",
      question: "A pipeline with $P = 8$ stages and $M = 64$ micro-batches switches from GPipe to 1F1B scheduling. What changes?",
      options: [
        "Bubble fraction drops from $22.6\\%$ to $11.1\\%$ because 1F1B overlaps forward and backward passes, but peak memory remains the same since the same total activations must be computed",
        "Both bubble fraction and peak memory improve substantially — 1F1B achieves near-zero bubble by fully overlapping forward and backward computation on each GPU",
        "Peak memory drops from $O(M)$ to $O(P)$, but bubble fraction increases from $9.9\\%$ to $43.8\\%$ because interleaving forward and backward passes creates additional synchronization stalls",
        "Peak activation memory drops from $O(M) = O(64)$ to $O(P) = O(8)$ because 1F1B releases activations during steady state, while the bubble fraction remains at $(P-1)/(M+P-1) \\approx 9.9\\%$ for both schedules"
      ],
      correct: 3,
      explanation: "1F1B and GPipe have the **same** bubble fraction: $(P-1)/(M+P-1) = 7/71 \\approx 9.9\\%$. The bubble depends on how many micro-batches fill the pipeline, not on the forward-backward ordering within the schedule. What changes is peak memory: GPipe stores all $M = 64$ micro-batches' activations during its all-forward phase, while 1F1B's steady-state interleaving means at most $P = 8$ micro-batches' activations are live simultaneously. This $8\\times$ memory reduction is 1F1B's key advantage and why it replaced GPipe in practice."
    },
    // Step 9: Interleaved pipeline schedules
    {
      type: "info",
      title: "Interleaved Pipeline Schedules",
      content: "Even with 1F1B, the bubble fraction $(P-1)/(M+P-1)$ can be significant when the number of physical stages $P$ is large relative to $M$. **Interleaved pipeline schedules** (Narayanan et al., 2021) reduce the bubble further by assigning **non-contiguous** layers to each GPU.\n\nThe idea: instead of each GPU holding one contiguous block of layers, each GPU holds $v$ smaller blocks from different parts of the network. These are called **virtual pipeline stages**. With $v$ virtual stages per GPU, the total number of virtual stages is $v \\times P$, and the bubble fraction becomes:\n\n$$\\text{bubble} = \\frac{P - 1}{v \\times M + P - 1}$$\n\nThe denominator grows by a factor of $v$, dramatically shrinking the bubble. For example, with $P = 8$ and $M = 32$:\n- $v = 1$ (standard 1F1B): bubble $= 7/39 \\approx 17.9\\%$\n- $v = 2$: bubble $= 7/71 \\approx 9.9\\%$\n- $v = 4$: bubble $= 7/135 \\approx 5.2\\%$\n\nThe tradeoff: with $v$ virtual stages per GPU, there are $v \\times P$ stage boundaries instead of $P$, so the number of point-to-point communications increases by a factor of $v$. Each micro-batch must now traverse $v$ times more stage boundaries, increasing the communication overhead. Modern systems like Megatron-LM typically use $v = 2$ to $4$ as a sweet spot between reduced bubble and manageable communication cost."
    },
    // Step 10: MC — interleaved schedule calculation
    {
      type: "mc",
      question: "A system uses $P = 8$ physical stages with $v = 4$ virtual stages per GPU and $M = 32$ micro-batches. What is the approximate bubble fraction, and what is the main cost of this approach compared to standard 1F1B ($v = 1$)?",
      options: [
        "Bubble is $\\approx 1.3\\%$ from $(P-1)/(v^2 \\times M + P-1) = 7/519$; the main cost is $16\\times$ more activation memory since virtual stages prevent activation reuse",
        "Bubble is $\\approx 5.2\\%$ from $(P-1)/(v \\times M + P-1) = 7/135$; the main cost is $4\\times$ more point-to-point communications because each micro-batch traverses $v \\times P$ stage boundaries instead of $P$",
        "Bubble is $\\approx 5.2\\%$ from $(P-1)/(v \\times M + P-1) = 7/135$; the main cost is $4\\times$ more peak activation memory since each GPU must store activations for $v$ separate layer groups",
        "Bubble is $\\approx 17.9\\%$ — same as standard 1F1B because virtual stages only reduce memory, not idle time; the main cost is load imbalance across non-contiguous layer groups"
      ],
      correct: 1,
      explanation: "The interleaved bubble fraction is $(P-1)/(v \\times M + P-1) = 7/(4 \\times 32 + 7) = 7/135 \\approx 5.2\\%$, a significant improvement over standard 1F1B's $7/39 \\approx 17.9\\%$. The cost is communication: with $v = 4$ virtual stages per GPU, there are $v \\times P = 32$ virtual stage boundaries instead of 8 physical ones, meaning $4\\times$ more point-to-point transfers per micro-batch. Each transfer moves an activation tensor between GPUs, adding latency and bandwidth consumption. This is why $v$ is kept small (2-4) in practice — beyond that, the communication overhead outweighs the bubble reduction."
    },
    // Step 11: PP in 3D parallelism
    {
      type: "info",
      title: "Pipeline Parallelism in 3D Parallelism",
      content: "In large-scale training, pipeline parallelism is combined with tensor parallelism (TP) and data parallelism (DP) in **3D parallelism**. Each strategy occupies a different dimension of the GPU cluster, matched to the available interconnect bandwidth:\n\n**TP within a node** (NVLink, 600+ GB/s): Tensor parallelism triggers all-reduce operations within every transformer layer — the highest communication frequency. It demands the highest bandwidth and is confined to GPUs within the same node.\n\n**PP across nodes** (InfiniBand, ~50 GB/s): Pipeline parallelism only performs point-to-point activation transfers at stage boundaries. This moderate bandwidth requirement makes it well-suited for the inter-node network.\n\n**DP across pipeline replicas** (InfiniBand): Data parallelism synchronizes gradients once per step via all-reduce. The infrequent but large transfers tolerate lower bandwidth.\n\nA practical challenge with PP is **load imbalance**. In a transformer-based language model, the first pipeline stage typically includes the **embedding layer** and the last stage includes the **language model head** (output projection + softmax). These components have different compute profiles than the middle transformer layers. The embedding lookup is memory-bound and fast; the LM head includes a large matrix multiply over the vocabulary. If the first and last stages are faster or slower than middle stages, they create **additional idle time** beyond the theoretical bubble — the pipeline can only move as fast as the slowest stage.\n\nOther practical considerations include: activation memory at stage boundaries must be budgeted carefully, activation checkpointing interacts with the pipeline schedule (recomputation during backward adds to stage latency), and the number of layers $L$ must be divisible by $P$ for even partitioning."
    },
    // Step 12: MC — load imbalance in practice
    {
      type: "mc",
      question: "A 126-layer model uses $PP = 16$ with 1F1B scheduling and $M = 64$ micro-batches. The first and last pipeline stages include the embedding layer and language model head respectively, making them approximately $30\\%$ faster than the middle stages. What is the primary consequence?",
      options: [
        "The theoretical bubble of $(P-1)/(M+P-1) = 15/79 \\approx 19\\%$ is achieved exactly, because 1F1B scheduling dynamically adjusts micro-batch routing to compensate for heterogeneous stage latencies",
        "Training throughput improves by $30\\%$ overall because the faster first and last stages act as pipeline accelerators, pulling micro-batches through the system more quickly end-to-end",
        "The model diverges because the faster stages apply optimizer updates before slower stages complete their backward passes, introducing stale parameter updates that compound across steps",
        "The faster first and last stages idle while waiting for the slower middle stages, adding bubble time beyond the theoretical $15/79 \\approx 19\\%$ — the pipeline runs at the slowest stage's speed"
      ],
      correct: 3,
      explanation: "The pipeline moves at the speed of its **slowest stage**. When the first and last stages are $30\\%$ faster, they complete each micro-batch's computation sooner and then idle, waiting for middle stages to finish. This adds idle time on top of the theoretical bubble of $(P-1)/(M+P-1) = 15/79 \\approx 19\\%$. The 1F1B schedule cannot compensate — it determines the order of forward and backward passes, not their duration. In practice, teams address load imbalance by assigning fewer layers to stages that also handle embedding/LM head, or by profiling per-stage latencies and rebalancing. This load imbalance is one of the most significant practical challenges in deploying pipeline parallelism at scale."
    }
  ]
};
