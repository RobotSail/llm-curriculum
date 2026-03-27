// G.2: Memory-Efficient Training Assessment
// Pure assessment — no info steps

export const memoryEfficientAssessment = {
  id: "G.2-assess",
  sectionId: "G.2",
  title: "Assessment: Memory-Efficient Training",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Gradient checkpointing trades memory for compute by not storing all intermediate activations. For a network with $L$ sequential layers, what is the optimal memory reduction achievable?",
      options: ["Memory drops from $O(L)$ to $O(\\sqrt{L})$ by placing checkpoints at $\\sqrt{L}$ evenly-spaced layers and recomputing activations within each segment during the backward pass, at the cost of at most one extra forward pass", "Memory drops from $O(L)$ to $O(1)$ by recomputing all activations from the input at every backward step, requiring $L$ extra forward passes per step but eliminating all intermediate activation storage entirely from GPU memory", "Memory drops from $O(L)$ to $O(\\log L)$ by using a recursive binary checkpointing scheme that saves activations at the midpoint of each segment and recursively subdivides during recomputation, requiring $O(\\log L)$ extra forward passes", "Memory is halved from $O(L)$ to $O(L/2)$ by checkpointing every other layer and recomputing only the non-checkpointed layers during the backward pass, keeping half the activations in memory at all times at the cost of roughly half an extra forward pass"],
      correct: 0,
      explanation: "With $\\sqrt{L}$ checkpoints, the network is divided into $\\sqrt{L}$ segments of $\\sqrt{L}$ layers each. During the backward pass, the activations within a segment are recomputed from its checkpoint. Only one segment's activations ($\\sqrt{L}$) plus the checkpoints ($\\sqrt{L}$) are in memory at once, giving $O(\\sqrt{L})$ total. The compute overhead is at most one extra forward pass because each activation is recomputed exactly once. For a 96-layer model, this reduces activation memory from 96 units to about 10 units."
    },
    {
      type: "mc",
      question: "For a model with $N$ parameters trained with the Adam optimizer in mixed-precision, what is the total optimizer state memory?",
      options: [
        "$2N$ bytes: one FP16 copy of the first moment ($N$ bytes) and one FP16 copy of the second moment ($N$ bytes), with no FP32 state needed since moments can accumulate safely in half-precision",
        "$12N$ bytes: a FP32 master copy of the weights ($4N$), FP32 first moment ($4N$), and FP32 second moment ($4N$), in addition to the $2N$ bytes of FP16 working weights",
        "$4N$ bytes: one FP32 copy of the momentum vector only, since Adam's second moment can be computed on the fly from the current gradient without storing it persistently across steps",
        "$8N$ bytes: FP16 copies of both moments ($2N$ each) plus the FP16 gradients ($2N$) and the FP16 working weights ($2N$), with all state kept in half-precision to minimize memory"
      ],
      correct: 1,
      explanation: "Mixed-precision Adam maintains: FP16 weights for forward/backward ($2N$ bytes), FP32 master weights for the optimizer step ($4N$ bytes), FP32 first moment $m_t$ ($4N$ bytes), and FP32 second moment $v_t$ ($4N$ bytes). The FP32 copies are essential because accumulating small updates in FP16 causes underflow. Total: $2N + 4N + 4N + 4N = 14N$ bytes. For a 7B parameter model, this is approximately 98 GB just for optimizer states, dominating the memory budget."
    },
    {
      type: "mc",
      question: "8-bit Adam (as in bitsandbytes) reduces optimizer memory by quantizing the optimizer states. How does it maintain training stability despite the quantization?",
      options: ["It uses stochastic rounding to ensure unbiased quantization in expectation, randomly rounding each moment value up or down with probability proportional to its distance from the nearest quantization level", "It falls back to FP32 Adam for layers whose gradient variance exceeds a learned threshold, maintaining a per-layer variance tracker that dynamically switches between 8-bit and 32-bit precision", "It only quantizes the second moment $v_t$ to INT8, keeping the first moment $m_t$ in full FP32 precision because directional information in the first moment is more sensitive to quantization noise", "It uses dynamic quantization with block-wise scaling: the first and second moments are stored in INT8 with per-block normalization factors, and a dynamic exponent is maintained to track the range, preserving the ratio between large and small values across training"],
      correct: 3,
      explanation: "8-bit Adam stores $m_t$ and $v_t$ in INT8 (1 byte each instead of 4), reducing optimizer state memory from $12N$ to roughly $6N$ bytes. Block-wise quantization divides the state into blocks of 2048 values, computing a separate scaling factor per block. A dynamic exponent tracks the tensor-wide range, allowing the block-wise quantization to adapt as moment values grow or shrink during training. Empirically, this introduces negligible degradation: the quantization error in the moments is small relative to the noise in SGD."
    },
    {
      type: "mc",
      question: "Adafactor reduces memory by factoring the second moment matrix. Instead of storing the full second moment $v_t \\in \\mathbb{R}^{m \\times n}$, what does it store?",
      options: ["A single scalar $\\bar{v}_t$ representing the global mean of $v_t$ across all $mn$ entries, applying a uniform adaptive learning rate to every parameter in the matrix regardless of per-coordinate gradient variance", "A random projection $P^\\top v_t P$ of $v_t$ into a lower-dimensional subspace via a fixed random matrix $P \\in \\mathbb{R}^{mn \\times k}$, compressing the second moment into $k \\ll mn$ values while preserving approximate distances", "A rank-1 factorization: row-wise statistics $r_t \\in \\mathbb{R}^m$ and column-wise statistics $c_t \\in \\mathbb{R}^n$, then reconstructs the second moment as $v_t \\approx r_t c_t^\\top / \\textbf{1}^\\top c_t$, reducing memory from $O(mn)$ to $O(m + n)$", "Only the diagonal entries of $v_t \\in \\mathbb{R}^{m \\times n}$, assuming all off-diagonal interaction terms are zero and that each parameter's second moment is independent of the others, reducing storage to $O(\\min(m,n))$"],
      correct: 2,
      explanation: "For a weight matrix $W \\in \\mathbb{R}^{m \\times n}$, Adam stores $mn$ values for $v_t$. Adafactor instead maintains row factors $r_t \\in \\mathbb{R}^m$ (mean of $v_t$ along columns) and column factors $c_t \\in \\mathbb{R}^n$ (mean along rows), using only $m + n$ values. The approximation $\\hat{v}_t = r_t c_t^\\top / \\textbf{1}^\\top c_t$ preserves the row and column marginals. For a $4096 \\times 4096$ matrix, this reduces second moment storage from $16{,}777{,}216$ to $8{,}192$ values — a 2048x reduction."
    },
    {
      type: "mc",
      question: "GaLore (Gradient Low-Rank Projection) reduces memory by projecting gradients into a low-rank subspace. How does it differ from LoRA?",
      options: ["GaLore projects the gradient $G \\in \\mathbb{R}^{m \\times n}$ to $\\tilde{G} = P^\\top G Q$ where $P, Q$ are obtained from periodic SVD of the gradient, maintains optimizer states only in the low-rank space, then projects back for the weight update — unlike LoRA, the full-rank weight $W$ itself is updated, enabling full-rank training with low-rank memory", "GaLore uses the same low-rank parameterization $\\Delta W = BA$ as LoRA but applies it during pretraining instead of fine-tuning, periodically resetting the low-rank factors to allow the pretrained weights to absorb the accumulated updates", "GaLore quantizes the gradient to 1-bit using sign-based compression and accumulates the residual quantization error in a feedback buffer, reducing optimizer state memory to one bit per parameter while maintaining convergence guarantees", "GaLore freezes a random subset of weights at each step and only computes gradients for the active subset, rotating the frozen set periodically so that all parameters are eventually updated while peak memory usage stays bounded"],
      correct: 0,
      explanation: "The key distinction: LoRA constrains the weight update to a fixed low-rank subspace for the entire training run ($\\Delta W = BA$). GaLore periodically (e.g., every 200 steps) computes the SVD of the full gradient to find the current top-$r$ subspace, projects gradients into it, runs Adam in that compressed space (storing $r(m+n)$ instead of $mn$ for moments), then projects back to apply a full-rank update to $W$. Because the projection subspace is updated, the cumulative weight change can have rank much higher than $r$, enabling full-rank training dynamics with low-rank memory cost."
    },
    {
      type: "mc",
      question: "In mixed-precision training, a loss scaling factor is applied before the backward pass. Why is this necessary?",
      options: [
        "It speeds up convergence by amplifying the effective learning rate, since multiplying the loss by $S$ is equivalent to scaling the step size by $S$, allowing faster traversal of flat regions in the loss landscape without changing the optimizer hyperparameters",
        "Small gradient values underflow to zero in FP16 (which has a minimum positive normal value of $\\sim 6 \\times 10^{-5}$). Loss scaling multiplies the loss by a large factor $S$ before backpropagation so that gradients are computed as $S \\cdot \\nabla_\\theta \\mathcal{L}$, preserving small values in FP16; the scale is then divided out before the optimizer step",
        "It normalizes gradients to unit norm for numerical stability, preventing the gradient magnitudes from growing unboundedly during backpropagation through deep networks and serving as an implicit form of gradient clipping built into the mixed-precision pipeline",
        "It compensates for the reduced precision of FP16 matrix multiplications by adding a correction term that accounts for the accumulated rounding errors in each layer's forward pass, ensuring that the backward pass gradients remain faithful to the FP32 loss surface"
      ],
      correct: 1,
      explanation: "FP16 has limited dynamic range: values smaller than $\\sim 6 \\times 10^{-5}$ flush to zero. Many gradient values fall in this range, especially in early layers. By multiplying the loss by $S$ (e.g., $S = 1024$ or dynamically adjusted), all gradients are scaled up by $S$ during backpropagation, moving them into the representable FP16 range. Before the optimizer step, gradients are divided by $S$ to recover the true values. Dynamic loss scaling starts with a large $S$ and halves it when overflow (INF/NaN) is detected, doubling it periodically when training is stable."
    },
    {
      type: "mc",
      question: "ZeRO (Zero Redundancy Optimizer) has three stages that progressively partition optimizer states across data-parallel workers. What does each stage partition?",
      options: ["Stage 1: activations across the batch; Stage 2: gradients across layers; Stage 3: weight matrices across heads — each stage partitions a different tensor type by a different axis to eliminate redundancy", "Stage 1: attention projection weights; Stage 2: MLP and feedforward weights; Stage 3: embedding tables and output heads — each stage targets a specific sublayer type to balance memory across workers", "Stage 1: layers 1 to $L/3$; Stage 2: layers $L/3$ to $2L/3$; Stage 3: layers $2L/3$ to $L$ — each stage assigns a contiguous block of layers to different workers, similar to pipeline parallelism", "Stage 1: optimizer states ($m_t$, $v_t$, FP32 master weights); Stage 2: optimizer states + gradients; Stage 3: optimizer states + gradients + model parameters. Each worker stores only a $1/N$ shard and communicates via all-gather/reduce-scatter as needed"],
      correct: 3,
      explanation: "With $N$ data-parallel workers, standard data parallelism replicates everything. ZeRO Stage 1 partitions only optimizer states: each worker stores $1/N$ of $m_t$, $v_t$, and the FP32 master copy, reducing optimizer memory by $N$x. Stage 2 additionally partitions gradients (reduce-scatter instead of all-reduce). Stage 3 further partitions the model parameters themselves, requiring an all-gather before each forward/backward computation. Each stage trades more communication for less memory. Stage 3 with 64 GPUs reduces per-GPU memory by up to 64x."
    },
    {
      type: "mc",
      question: "Activation memory during training scales with batch size $B$, sequence length $L$, hidden dimension $d$, and number of layers $N$. For a transformer, which component dominates the activation memory?",
      options: ["The word embedding table of shape $V \\times d$, which must be stored alongside its gradient and is replicated across all layers during backpropagation, consuming $O(Vd)$ memory that scales with vocabulary size", "The final logit output of shape $B \\times L \\times V$, which materializes the full vocabulary distribution for every token in the batch and must persist in memory throughout the backward pass for the cross-entropy gradient computation", "The attention score matrices: each layer stores the $B \\times H \\times L \\times L$ attention scores (before and after softmax), where $H$ is the number of heads. For long sequences, this $O(BHL^2)$ per-layer cost dominates and grows quadratically with sequence length", "The dropout masks of shape $B \\times L \\times d$ per layer, which store a binary keep/drop decision for every activation element and must be saved during the forward pass so the same mask can be reapplied during the backward pass"],
      correct: 2,
      explanation: "For a hidden size $d = 4096$, 32 heads, $L = 4096$ sequence length, and batch size $B = 1$: each layer's attention scores require $B \\times H \\times L^2 = 1 \\times 32 \\times 4096^2 \\approx 537M$ values (stored pre- and post-softmax for the backward pass). The linear layer activations are $O(BLd)$ per layer, which is much smaller when $L \\gg d/H$. This quadratic scaling is why FlashAttention (which avoids materializing the full attention matrix) and gradient checkpointing are critical for long-context training."
    },
    {
      type: "mc",
      question: "Communication-efficient distributed training techniques aim to reduce the bandwidth cost of gradient synchronization. Which of the following correctly describes gradient compression via top-$k$ sparsification with error feedback?",
      options: ["Only the top-$k$ gradient values by magnitude are communicated. The residual error $e_t = g_t - \\text{TopK}(g_t + e_{t-1})$ is accumulated locally and added to the next iteration's gradient before sparsification, ensuring that all gradient information is eventually communicated", "Only the top-$k$ gradient values by magnitude are communicated each step. The residual (uncommunicated values) is permanently discarded, introducing a systematic bias into the training that accumulates over time but is bounded by the sparsification rate", "Gradients are randomly sampled with probability $k/d$ at each coordinate, and unsampled coordinates are set to zero. The sampling mask is synchronized across workers so that all nodes communicate the same subset of gradient dimensions per step", "The gradient is projected to $k$ dimensions via a fixed random matrix, communicated in the compressed space, and projected back to the original dimension. The random matrix is shared across all workers and generated from a common seed"],
      correct: 0,
      explanation: "Top-$k$ sparsification without error feedback causes training divergence because small but persistent gradient signals are permanently lost. Error feedback fixes this: at each step, the accumulated residual $e_{t-1}$ is added to the current gradient $g_t$, and top-$k$ is applied to $g_t + e_{t-1}$. The difference between the input and the sparsified output becomes the new residual. This guarantees that every gradient component eventually accumulates enough magnitude to be communicated. With $k = 0.1\\%$ of parameters, this achieves 1000x compression with minimal accuracy loss."
    },
    {
      type: "mc",
      question: "When training a model that does not fit in a single GPU's memory even with all the above techniques, pipeline parallelism is used. What is the \"bubble\" problem in naive pipeline parallelism, and how does 1F1B (one-forward-one-backward) scheduling address it?",
      options: [
        "The bubble is caused by all-reduce gradient synchronization blocking computation between pipeline stages. Each stage must wait for the global all-reduce of gradients to complete before starting the next microbatch's forward pass. 1F1B overlaps the reduce-scatter communication of one microbatch with the forward computation of the next, hiding the communication latency behind useful compute work on each stage's GPU",
        "In naive pipeline parallelism (GPipe-style), all microbatches complete their forward passes before any backward passes begin, creating idle time (the bubble) proportional to $(p-1)/m$ of the total time, where $p$ is the pipeline depth and $m$ is the number of microbatches. 1F1B interleaves forward and backward passes so that each stage starts its backward pass as soon as possible, reducing peak activation memory from $O(m)$ to $O(p)$ microbatches",
        "The bubble refers to GPU memory fragmentation caused by variable-size activation tensors allocated and freed in non-sequential order across pipeline stages. As stages process microbatches at different rates, the allocator creates holes in the HBM address space. 1F1B defragments memory between forward and backward passes by enforcing a strict allocation order that prevents heap fragmentation",
        "The bubble is wasted compute from padding variable-length sequences to equal length within each microbatch. Since each pipeline stage operates on the same padded tensor shape, shorter sequences waste FLOPs on padding tokens. 1F1B uses dynamic batching to group sequences of similar length together, minimizing padding tokens and ensuring that each pipeline stage processes roughly the same number of real tokens"
      ],
      correct: 1,
      explanation: "In GPipe, all $m$ microbatches run forward through all $p$ stages before backward passes begin. During the warmup and cooldown phases, some stages are idle, creating a bubble of fraction $(p-1)/(m + p - 1)$. 1F1B instead has each stage execute one forward pass then one backward pass in alternation (after an initial warmup). This means each stage holds activations for at most $p$ microbatches simultaneously (instead of $m$), dramatically reducing memory. The bubble fraction remains similar, but the memory reduction is the key benefit, especially for large models."
    }
  ]
};
