// Branch G Assessments: Efficient Training & Inference
// G.1: Parameter-Efficient Fine-Tuning, G.2: Memory-Efficient Training, G.3: Hardware-Aware ML
// Tier 0 Assessments: Foundations
// 0.3: Optimization Theory, 0.4: Systems & Hardware
// Pure assessment — no info steps

// ============================================================================
// G.1: Parameter-Efficient Fine-Tuning
// ============================================================================
export const peftAssessment = {
  id: "G.1-assess",
  sectionId: "G.1",
  title: "Assessment: Parameter-Efficient Fine-Tuning",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In LoRA, the pretrained weight matrix $W_0 \\in \\mathbb{R}^{d \\times k}$ is frozen and the update is parameterized as $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ with rank $r \\ll \\min(d, k)$. How many trainable parameters does this introduce per adapted weight matrix?",
      options: ["$d \\times k$ (same size as the original weight matrix, no parameter savings)", "$r^2$ (the square of the rank, independent of the matrix dimensions $d$ and $k$)", "$r \\times (d + k)$ (the sum of parameters in the two low-rank factors $B$ and $A$)", "$d \\times k \\times r$ (the original matrix size multiplied by the rank factor $r$)"],
      correct: 2,
      explanation: "Matrix $B$ has $d \\times r$ parameters and matrix $A$ has $r \\times k$ parameters, for a total of $r(d + k)$. Since $r \\ll \\min(d, k)$, this is dramatically fewer than the $d \\times k$ parameters in the full matrix. For example, with $d = k = 4096$ and $r = 16$, LoRA uses $2 \\times 4096 \\times 16 = 131{,}072$ parameters vs. $4096^2 = 16{,}777{,}216$ for the full matrix — a 128x reduction."
    },
    {
      type: "mc",
      question: "LoRA applies a scaling factor $\\frac{\\alpha}{r}$ to the low-rank update, so the forward pass computes $h = W_0 x + \\frac{\\alpha}{r} BAx$. What is the purpose of this $\\alpha / r$ scaling?",
      options: ["It allows changing the rank $r$ without retuning the learning rate: the scaling keeps the effective update magnitude roughly constant across different rank choices, so $\\alpha$ can be fixed (e.g., $\\alpha = 16$) while sweeping $r$", "It normalizes the output of the low-rank branch to unit variance regardless of the chosen rank, ensuring consistent activation magnitudes when merging the LoRA update back into the frozen weights during inference", "It prevents gradient explosion in the backward pass by dynamically clipping the update magnitude proportional to $1/r$, acting as an implicit form of gradient scaling that stabilizes training for higher-rank adaptations", "It ensures the low-rank matrices $A$ and $B$ remain approximately orthogonal during training, preserving the geometric structure of the update subspace and preventing rank collapse in the learned adaptation"],
      correct: 0,
      explanation: "When $\\alpha$ is fixed, increasing $r$ increases the capacity of the update but the $\\alpha / r$ factor reduces the per-component contribution, keeping the overall update magnitude stable. This means you can use the same learning rate and $\\alpha$ across experiments with different ranks. In practice, $\\alpha$ is often set equal to the first rank tried (e.g., $\\alpha = r = 16$), and subsequent rank sweeps only change $r$ while keeping $\\alpha$ and the learning rate constant."
    },
    {
      type: "mc",
      question: "In the original LoRA paper, matrix $A$ is initialized from a Gaussian distribution and matrix $B$ is initialized to zero. Why is $B$ initialized to zero rather than both being random?",
      options: [
        "It makes $B$ sparser at initialization, reducing memory usage by allowing compressed storage of the zero matrix and lowering the initial compute cost of the forward pass through the LoRA branch",
        "It ensures $\\Delta W = BA = 0$ at the start of training, so the model begins from the exact pretrained weights and the low-rank adaptation starts as a perturbation from a known-good solution",
        "Random initialization of both matrices causes numerical instability in FP16 due to the product $BA$ amplifying floating-point errors, while zeroing $B$ keeps the initial product within a safe representable range",
        "Zero initialization of $B$ makes the gradient of $A$ exactly zero at the first step, enabling a curriculum-style warmup where $A$ learns feature directions before $B$ begins adapting"
      ],
      correct: 1,
      explanation: "With $B = 0$, the product $BA = 0$ regardless of $A$'s values, so $h = W_0 x + \\frac{\\alpha}{r} \\cdot 0 \\cdot x = W_0 x$ at initialization. This means training starts from the exact pretrained model behavior. The gradients with respect to $B$ are non-zero (they depend on $A$ and the loss), so $B$ immediately begins to update. This is a deliberate design: the pretrained model is already good, and LoRA learns a small perturbation on top."
    },
    {
      type: "mc",
      question: "QLoRA combines 4-bit quantization of the base model with LoRA adapters. Which quantization format does QLoRA introduce, and what is its key property?",
      options: [
        "INT4 uniform quantization, which divides the full weight range $[\\min, \\max]$ into 16 equally spaced bins, assigning each weight to its nearest bin center regardless of the underlying weight distribution",
        "NormalFloat4 (NF4), a data type whose 16 quantization levels are chosen so that each level has equal probability mass under a standard normal distribution, making it information-theoretically optimal for normally-distributed weights",
        "FP4 (4-bit floating point) with 1 sign bit, 2 exponent bits, and 1 mantissa bit, providing a fixed dynamic range that prioritizes values near zero at the cost of reduced precision for larger weight magnitudes",
        "Log4 quantization, which spaces its 16 levels logarithmically to allocate finer resolution near zero and coarser resolution for outliers, optimizing for heavy-tailed weight distributions common in deep networks"
      ],
      correct: 1,
      explanation: "NF4 is based on the empirical observation that pretrained neural network weights are approximately normally distributed. QLoRA computes quantile-based breakpoints: the 16 levels are set at the quantiles $\\{1/32, 3/32, 5/32, \\ldots, 31/32\\}$ of $\\mathcal{N}(0, 1)$, then weights are normalized per-block and mapped to the nearest level. This yields zero quantization error in expectation for truly normal weights. QLoRA also uses double quantization — the per-block quantization constants are themselves quantized to FP8."
    },
    {
      type: "mc",
      question: "QLoRA uses \"double quantization\" to reduce the memory overhead of quantization constants. What does this mean?",
      options: ["The model weights are quantized twice in sequence, first to INT8 then to INT4, with each stage refining the quantization grid to minimize reconstruction error across both passes", "The gradients are quantized to 4 bits during the backward pass using the same NF4 format as the weights, halving the memory needed for gradient storage during LoRA fine-tuning", "Two separate LoRA adapters are applied to each layer and their outputs averaged, providing an ensemble-like effect that compensates for the quantization noise in the frozen base weights", "The quantization constants (scale factors) used for NF4 quantization are themselves quantized to 8-bit, reducing their memory footprint from 32 bits to 8 bits per block"],
      correct: 3,
      explanation: "Block-wise quantization (e.g., blocks of 64 weights) requires storing a scale factor per block. With FP32 scales, this adds $32/64 = 0.5$ bits per parameter. Double quantization quantizes these scale factors themselves — grouping 256 scale factors and quantizing them to FP8, reducing the overhead to roughly $8/64 + 32/(64 \\times 256) \\approx 0.127$ bits per parameter. This is critical: for a 65B parameter model, the savings are several gigabytes."
    },
    {
      type: "mc",
      question: "DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes the weight update differently from standard LoRA. How does DoRA parameterize the adapted weight?",
      options: ["It applies LoRA to both the attention and MLP layers simultaneously with shared low-rank factors $B$ and $A$, tying the adaptation across sublayers to reduce parameters while capturing cross-sublayer interactions", "It doubles the rank of LoRA and applies dropout between the two low-rank matrices $B$ and $A$, using the stochastic regularization to prevent the low-rank subspace from overfitting to the fine-tuning data distribution", "It decomposes the weight into magnitude and direction components: $W' = m \\cdot \\frac{W_0 + BA}{\\|W_0 + BA\\|_c}$, where $m$ is a trainable magnitude vector and the direction is updated via LoRA, inspired by weight normalization", "It uses a dictionary of rank-1 updates $\\{u_i v_i^\\top\\}$ and selects the top-$k$ most relevant updates at each step via a learned gating mechanism, forming a sparse mixture-of-adaptations approach"],
      correct: 2,
      explanation: "DoRA is motivated by analyzing the difference between full fine-tuning and LoRA: full fine-tuning tends to make large directional changes with small magnitude changes, while LoRA couples both. By decomposing $W' = m \\cdot \\frac{V'}{\\|V'\\|_c}$ where $V' = W_0 + BA$ (direction via LoRA) and $m$ is a learnable per-column magnitude vector, DoRA decouples these two aspects. The $\\|\\cdot\\|_c$ denotes column-wise normalization. This consistently improves over LoRA, sometimes matching full fine-tuning performance."
    },
    {
      type: "mc",
      question: "LoRA+ proposes using different learning rates for matrices $A$ and $B$. Specifically, it recommends that $\\eta_B \\gg \\eta_A$. What is the theoretical justification?",
      options: ["Analysis of the update dynamics shows that in standard LoRA, the effective learning rate for $B$ is suboptimal: $A$ acts as a feature extractor that changes slowly while $B$ maps features to outputs and should adapt faster. Setting $\\eta_B / \\eta_A = \\lambda$ with $\\lambda \\approx 16$ improves efficiency", "Matrix $B$ has $d \\times r$ parameters while $A$ has $r \\times k$, and since $d > k$ in most transformer layers, $B$ needs a proportionally larger learning rate to converge in the same number of steps as $A$", "A larger learning rate for $B$ acts as implicit regularization by injecting gradient noise into the output projection, smoothing the loss landscape in a way analogous to dropout applied selectively to the low-rank branch", "It compensates for the zero initialization of $B$: since $B$ starts at the origin while $A$ starts from a random Gaussian, $B$ must traverse a longer distance in parameter space and therefore requires a larger step size to reach the optimum"],
      correct: 0,
      explanation: "LoRA+ analyzes the training dynamics in the infinite-width limit and finds that using a single learning rate is inefficient: the two matrices play asymmetric roles. Matrix $A$ projects the input into the low-rank subspace (feature extraction), and $B$ maps from that subspace to the output. The optimal ratio $\\eta_B / \\eta_A$ depends on the width, but empirically $\\lambda \\approx 16$ works well across model sizes, yielding up to 2x speedup in training convergence with no additional cost."
    },
    {
      type: "mc",
      question: "Under what conditions can LoRA match or closely approximate full fine-tuning performance?",
      options: [
        "Only when the model is small enough that $r = d$ can be used, so the low-rank factorization effectively becomes a full-rank update and LoRA degenerates into standard weight tuning with no rank bottleneck",
        "When the task-specific weight update $\\Delta W^*$ has a low intrinsic rank — empirically observed for many downstream tasks, especially when fine-tuning large models where the adaptation lies in a low-dimensional subspace of the full parameter space",
        "Only when both $A$ and $B$ matrices are initialized from slices of the pretrained weights rather than from random or zero initialization, so the low-rank subspace begins aligned with the model's learned representations",
        "When the base model has been pre-quantized to INT8 before applying LoRA, because quantization noise acts as regularization that confines the gradient updates to a low-rank manifold matching LoRA's capacity"
      ],
      correct: 1,
      explanation: "The Aghajanyan et al. (2020) \"intrinsic dimensionality\" paper showed that fine-tuning updates for many tasks lie in a surprisingly low-dimensional subspace. LoRA exploits this directly: if the true $\\Delta W^*$ has rank $\\leq r$, then LoRA with rank $r$ can represent it exactly. In practice, ranks as small as $r = 4$-$16$ suffice for many NLP tasks on models like GPT-3. However, for complex tasks or smaller models where the update has higher effective rank, LoRA with small $r$ will underperform full fine-tuning."
    },
    {
      type: "mc",
      question: "Adapter methods (Houlsby et al., 2019) insert small bottleneck modules into transformer layers. How does a typical adapter block work, and how does it compare to LoRA?",
      options: ["Adapters replace the attention mechanism entirely with a smaller learned bottleneck version, reducing the per-layer parameter count while preserving the residual connection structure of the original transformer block", "Adapters only modify the embedding layer and final classification head with trainable bottleneck projections, leaving all intermediate transformer layers frozen and unchanged during the fine-tuning process", "Adapters are mathematically identical to LoRA in their low-rank parameterization but differ only in initialization strategy, using Xavier uniform instead of Gaussian for the down-projection and zeros for the up-projection", "An adapter applies $h \\leftarrow h + f(hW_{\\text{down}})W_{\\text{up}}$, inserting a down-projection, nonlinearity $f$, and up-projection as a residual. Unlike LoRA, adapters add sequential computation (extra latency at inference) and cannot be merged into the base weights"],
      correct: 3,
      explanation: "A Houlsby adapter projects from dimension $d$ to a bottleneck $m \\ll d$ via $W_{\\text{down}} \\in \\mathbb{R}^{d \\times m}$, applies a nonlinearity (e.g., ReLU), and projects back via $W_{\\text{up}} \\in \\mathbb{R}^{m \\times d}$, with a skip connection. This adds $2md + m$ trainable parameters per adapter. The key disadvantage vs. LoRA is inference overhead: adapters are sequential modules that cannot be folded into the pretrained weights, adding latency proportional to their depth. LoRA's $\\Delta W = BA$ can be merged into $W_0$ post-training, yielding zero inference overhead."
    },
    {
      type: "mc",
      question: "When applying LoRA to a transformer, which weight matrices are typically adapted, and why?",
      options: [
        "Only the embedding and output layers, because they contain the majority of the model's parameters and adapting them captures the input-output mapping while leaving the internal representations unchanged",
        "All weight matrices uniformly with equal rank, because each layer contributes equally to the adaptation and distributing rank evenly ensures balanced capacity across the entire model's depth",
        "The query and value projection matrices ($W_Q$, $W_V$) in attention are most common, though recent work shows adapting all linear layers (including $W_K$, $W_O$, and MLP projections) with smaller rank per matrix often outperforms adapting fewer matrices with larger rank",
        "Only the LayerNorm parameters ($\\gamma$ and $\\beta$), because they control the distribution of activations flowing through each layer and provide a high-leverage, low-parameter point for steering model behavior"
      ],
      correct: 2,
      explanation: "The original LoRA paper found that adapting $W_Q$ and $W_V$ was sufficient for GPT-3. However, subsequent work (including QLoRA and LLaMA-Adapter) demonstrated that distributing the parameter budget across all linear layers — attention projections ($W_Q, W_K, W_V, W_O$) and MLP layers ($W_{\\text{gate}}, W_{\\text{up}}, W_{\\text{down}}$) — with a proportionally smaller rank per matrix often yields better results than concentrating rank in fewer matrices. This makes intuitive sense: each layer captures different aspects of the adaptation."
    }
  ]
};

// ============================================================================
// G.2: Memory-Efficient Training
// ============================================================================
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
      options: ["The word embedding table, which stores $V \\times d$ parameters", "The final logit output of shape $B \\times L \\times V$", "The attention score matrices: each layer stores the $B \\times H \\times L \\times L$ attention scores (before and after softmax), where $H$ is the number of heads. For long sequences, this $O(BHL^2)$ per-layer cost dominates and grows quadratically with sequence length", "The dropout masks, which require one bit per activation"],
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
        "The bubble is caused by all-reduce gradient synchronization blocking computation between pipeline stages. 1F1B overlaps the reduce-scatter communication of one microbatch with the forward computation of the next, hiding the communication latency behind useful compute work on each stage's GPU",
        "In naive pipeline parallelism (GPipe-style), all microbatches complete their forward passes before any backward passes begin, creating idle time (the bubble) proportional to $(p-1)/m$ of the total time, where $p$ is the pipeline depth and $m$ is the number of microbatches. 1F1B interleaves forward and backward passes so that each stage starts its backward pass as soon as possible, reducing peak activation memory from $O(m)$ to $O(p)$ microbatches",
        "The bubble refers to GPU memory fragmentation caused by variable-size activation tensors allocated and freed in non-sequential order across pipeline stages. 1F1B defragments memory between forward and backward passes by enforcing a strict allocation order that prevents heap fragmentation",
        "The bubble is wasted compute from padding variable-length sequences to equal length within each microbatch. 1F1B uses dynamic batching to group sequences of similar length together, minimizing padding tokens and ensuring that each pipeline stage processes roughly the same number of real tokens"
      ],
      correct: 1,
      explanation: "In GPipe, all $m$ microbatches run forward through all $p$ stages before backward passes begin. During the warmup and cooldown phases, some stages are idle, creating a bubble of fraction $(p-1)/(m + p - 1)$. 1F1B instead has each stage execute one forward pass then one backward pass in alternation (after an initial warmup). This means each stage holds activations for at most $p$ microbatches simultaneously (instead of $m$), dramatically reducing memory. The bubble fraction remains similar, but the memory reduction is the key benefit, especially for large models."
    }
  ]
};

// ============================================================================
// G.3: Hardware-Aware Machine Learning
// ============================================================================
export const hardwareAwareAssessment = {
  id: "G.3-assess",
  sectionId: "G.3",
  title: "Assessment: Hardware-Aware Machine Learning",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "FlashAttention computes exact standard attention but with reduced HBM (High Bandwidth Memory) accesses. It achieves this by tiling the computation into SRAM. What is the key insight that makes this possible?",
      options: ["It approximates the attention matrix with a low-rank factorization $A \\approx UV^\\top$ where $U, V \\in \\mathbb{R}^{N \\times r}$, reducing both compute and memory from $O(N^2 d)$ to $O(Nrd)$ at the cost of bounded approximation error controlled by the rank $r$", "It uses tensor cores to compute the full $N \\times N$ attention matrix faster by reformulating the dot-product attention as a sequence of FP16 matrix multiplications, achieving higher throughput without changing the memory access pattern or total HBM usage", "It replaces softmax with ReLU attention, which can be computed blockwise without tracking normalization constants, enabling each block to be processed independently in SRAM and reducing HBM accesses from $O(N^2)$ to $O(N)$ per attention head", "It tiles $Q$, $K$, $V$ into blocks that fit in SRAM and uses the online softmax trick (tracking running max and sum) to compute exact attention without ever materializing the full $N \\times N$ attention matrix in HBM, reducing HBM accesses from $O(N^2)$ to $O(N^2 d / M)$ where $M$ is the SRAM size"],
      correct: 3,
      explanation: "The core challenge is that softmax requires knowing the maximum and sum across the entire row to normalize. The online softmax algorithm (Milakov & Gimelshein, 2018) maintains running statistics: as each new block of $K$ is processed, the running max $m$ and sum $l$ are updated, and previously computed partial results are rescaled. This means each block of $Q$ can be processed against all blocks of $K$ sequentially, storing only $O(B_r \\times d)$ intermediate results in SRAM at a time. The FLOPs are identical to standard attention ($O(N^2 d)$), but HBM reads/writes drop from $O(N^2 + Nd)$ to $O(N^2 d / M)$."
    },
    {
      type: "mc",
      question: "The roofline model characterizes whether an operation is compute-bound or memory-bound. What determines which regime an operation falls into?",
      options: ["The total number of parameters in the model — larger models have more FLOPs per forward pass and are therefore always compute-bound, while smaller models are memory-bound because their weight matrices fit in cache", "The batch size alone determines the regime — small batches are memory-bound because weight loading dominates, and large batches are compute-bound because arithmetic grows linearly with batch size while memory stays constant", "The arithmetic intensity $I = \\text{FLOPs} / \\text{Bytes}$ of the operation compared to the machine's compute-to-bandwidth ratio $\\pi / \\beta$ (peak FLOP/s divided by peak memory bandwidth). If $I < \\pi / \\beta$, the operation is memory-bound; if $I > \\pi / \\beta$, it is compute-bound", "Whether the GPU uses FP32 or FP16 determines the regime — FP16 operations are always compute-bound because tensor cores double the throughput, while FP32 operations are always memory-bound because they lack tensor core acceleration"],
      correct: 2,
      explanation: "The roofline model plots attainable FLOP/s against arithmetic intensity $I$ (FLOPs per byte transferred). Below the ridge point $I^* = \\pi / \\beta$, performance is limited by memory bandwidth (e.g., elementwise ops have $I \\approx 1$ FLOP/byte). Above it, performance is limited by peak compute (e.g., large matrix multiplications have $I \\approx 2n$ for $n \\times n$ matmul). For an A100 with $\\pi = 312$ TFLOP/s (FP16) and $\\beta = 2$ TB/s, the ridge point is $I^* = 156$ FLOP/byte. Operations must perform at least 156 FLOPs per byte loaded to be compute-bound."
    },
    {
      type: "mc",
      question: "On an NVIDIA A100 GPU, SRAM (shared memory per SM) is approximately 192 KB, while HBM capacity is 80 GB with 2 TB/s bandwidth. Why does FlashAttention's use of SRAM provide such a large speedup despite SRAM being much smaller?",
      options: [
        "SRAM has approximately 19 TB/s aggregate bandwidth across all SMs, roughly 10x more than HBM, so tiling computations to reuse data in SRAM dramatically reduces the memory bottleneck for bandwidth-bound operations like attention",
        "SRAM uses a fundamentally different data format that allows 4x compression",
        "SRAM allows the GPU to skip the softmax computation entirely",
        "SRAM is only faster because it avoids the overhead of virtual memory translation"
      ],
      correct: 0,
      explanation: "A100 has 108 SMs, each with 192 KB of shared memory, totaling about 20 MB of SRAM with aggregate bandwidth around 19 TB/s (each SM can access its SRAM at ~180 GB/s). HBM has 2 TB/s bandwidth but is shared across all operations. By tiling Q, K, V blocks into SRAM, FlashAttention ensures that the expensive dot products and softmax computations read from fast SRAM rather than slow HBM. The key is data reuse: each block of K/V loaded into SRAM is used for all computations against the corresponding Q block before moving to the next tile."
    },
    {
      type: "mc",
      question: "Triton is a Python-based language for writing GPU kernels. What is its primary advantage over writing CUDA kernels directly for ML researchers?",
      options: [
        "Triton kernels run faster than hand-optimized CUDA because Triton uses a superior compilation strategy",
        "Triton operates at the block level rather than the thread level: the programmer specifies operations on blocks (tiles) of data, and the Triton compiler handles thread scheduling, shared memory management, and memory coalescing automatically, dramatically reducing the expertise needed to write efficient GPU code",
        "Triton is a pure Python library that interprets kernels at runtime without compilation",
        "Triton only works on AMD GPUs and is needed because CUDA is NVIDIA-exclusive"
      ],
      correct: 1,
      explanation: "CUDA programming requires managing threads within warps, explicit shared memory allocation and synchronization via `__syncthreads()`, memory coalescing patterns, and occupancy optimization — demanding deep hardware expertise. Triton abstracts this: the programmer writes code that operates on `tl.load()` / `tl.store()` blocks of data with `tl.dot()` for matrix operations, and the compiler automatically handles the thread-level decomposition, SRAM allocation, and memory access optimization. This makes writing fused kernels (e.g., fused attention, fused LayerNorm + activation) accessible to ML researchers."
    },
    {
      type: "mc",
      question: "FP8 training uses two FP8 formats: E4M3 (4 exponent bits, 3 mantissa bits) and E5M2 (5 exponent bits, 2 mantissa bits). How are these typically used in practice?",
      options: ["E4M3 is used for all computations in both forward and backward passes, while E5M2 is used only for storing checkpointed activations to disk, where the larger dynamic range helps preserve values across the wider distribution of saved intermediate states", "E5M2 is used for the forward pass weights and activations because its extra exponent bit provides the wider dynamic range needed to represent the full span of pretrained weight values without overflow during matrix multiplications", "Both formats are used interchangeably within each layer, with the specific format selected randomly per operation to provide a form of stochastic quantization that reduces systematic rounding bias and improves training stability over deterministic format assignment", "E4M3 (higher precision, smaller range) is used for forward pass weights and activations, while E5M2 (lower precision, larger range) is used for gradients in the backward pass, because gradients have a wider dynamic range and benefit more from the extra exponent bit"],
      correct: 3,
      explanation: "E4M3 has range $\\pm 448$ with higher precision (8 significand values per exponent), suitable for weights and activations whose distribution is relatively concentrated. E5M2 has range $\\pm 57344$ with lower precision (4 significand values per exponent), better for gradients which can span many orders of magnitude, especially in early layers or with loss scaling. Per-tensor scaling factors $s$ are maintained: values are stored as $x_{\\text{FP8}} = \\text{cast}(x / s)$ and restored as $x \\approx s \\cdot x_{\\text{FP8}}$. The scaling factors are updated periodically based on the observed tensor statistics."
    },
    {
      type: "mc",
      question: "Per-tensor scaling is essential for FP8 training. Why can't you simply cast FP32/FP16 values to FP8 without scaling?",
      options: ["FP8 casting without scaling would trigger hardware-level type errors on current GPU architectures, because the FP8 ALUs require a scaling metadata header attached to each tensor to configure the internal datapath for the correct exponent bias", "Scaling is only needed for backward compatibility with older GPU architectures that lack native FP8 support — on newer hardware like H100, the tensor cores handle the dynamic range mapping internally and scaling is applied automatically by the hardware", "FP8 has extremely limited dynamic range (e.g., E4M3 max is 448). Without scaling, tensors whose values exceed this range overflow to INF, while tensors with small values (e.g., gradients of $10^{-4}$) lose precision to quantization noise. Per-tensor scaling maps the tensor's range into the FP8 representable range, maximizing the use of available precision", "Without scaling, FP8 matmuls produce outputs in a non-standard intermediate format that cannot be accumulated into FP16 or FP32 accumulators, because the hardware accumulator expects the inputs to be pre-scaled to a canonical range matching its fixed-point summation logic"],
      correct: 2,
      explanation: "E4M3 can represent values from $2^{-9}$ to $448$, roughly 6 orders of magnitude. Transformer weights might range from $-0.5$ to $0.5$, using only a fraction of FP8's range and losing precision. Gradients might range from $10^{-7}$ to $10^{-2}$, falling partially below E4M3's minimum. Per-tensor scaling computes $s = \\max(|x|) / 448$ (for E4M3) and stores $x/s$ in FP8, placing the maximum value at FP8's maximum. Delayed scaling (using statistics from the previous iteration) is used to avoid the synchronization cost of computing the current tensor's max before casting."
    },
    {
      type: "mc",
      question: "Operator fusion combines multiple sequential GPU operations into a single kernel launch. Why does this provide significant speedups for operations like LayerNorm followed by a GeLU activation?",
      options: ["Without fusion, each elementwise operation reads its input from HBM, computes, and writes the output back to HBM. Fusing multiple operations into a single kernel reads data from HBM once, performs all operations in registers/SRAM, and writes the final result once, eliminating intermediate HBM round-trips. Since these operations are memory-bound, the speedup is proportional to the number of fused operations", "Fused operators use a mathematically reformulated combined function that requires fewer total FLOPs than executing the operations separately — for example, fusing LayerNorm and GeLU exploits algebraic cancellations between the normalization and activation computations", "Fusion allows the GPU to dispatch the compound operation to specialized hardware units (similar to tensor cores for matmul) that are designed to execute common operation sequences like LayerNorm + activation in a single fixed-function pipeline stage", "Fusion reduces the numerical precision requirements of the intermediate values, allowing the fused kernel to use FP8 for internal computation instead of FP16, since the intermediate results are never written to HBM where higher precision would be needed"],
      correct: 0,
      explanation: "Consider LayerNorm ($\\mu = \\text{mean}(x)$, $\\sigma = \\text{std}(x)$, $y = (x - \\mu)/\\sigma \\cdot \\gamma + \\beta$) followed by GeLU ($z = y \\cdot \\Phi(y)$). Unfused: read $x$ from HBM, write intermediate $y$ to HBM, read $y$ from HBM, write $z$ to HBM — 4 HBM transfers. Fused: read $x$ from HBM, compute both operations in SRAM/registers, write $z$ to HBM — 2 HBM transfers. Since these are memory-bound (arithmetic intensity $\\approx 10$ FLOP/byte, well below the ridge point), halving the memory traffic nearly halves the wall-clock time. `torch.compile` automates this fusion."
    },
    {
      type: "mc",
      question: "In the GPU memory hierarchy, what are the relative sizes and bandwidths of registers, L1/shared memory (SRAM), L2 cache, and HBM on an NVIDIA A100?",
      options: ["All levels have roughly the same bandwidth, but differ in capacity", "Registers: 256 KB/SM at ~50 TB/s aggregate; L1/SRAM: 192 KB/SM at ~19 TB/s aggregate; L2: 40 MB at ~5 TB/s; HBM: 80 GB at 2 TB/s. Each level trades capacity for bandwidth by roughly an order of magnitude", "HBM is faster than SRAM but smaller, which is why data is cached in HBM", "Registers: 1 MB/SM; L1: 10 MB; L2: 1 GB; HBM: 80 GB, all at uniform 2 TB/s bandwidth"],
      correct: 1,
      explanation: "The memory hierarchy follows a clear pattern: as capacity increases, bandwidth decreases. Registers (256 KB per SM, ~27.6 MB total across 108 SMs) can be accessed in a single cycle at aggregate bandwidths exceeding 50 TB/s. Shared memory/L1 (192 KB per SM, configurable) provides ~19 TB/s aggregate. L2 cache (40 MB, shared across all SMs) offers ~5 TB/s. HBM (80 GB) provides 2 TB/s. Efficient kernel design maximizes data reuse at the fastest level — this is exactly what FlashAttention and operator fusion exploit."
    },
    {
      type: "mc",
      question: "A matrix multiplication $C = AB$ where $A \\in \\mathbb{R}^{M \\times K}$ and $B \\in \\mathbb{R}^{K \\times N}$ has arithmetic intensity $I = \\frac{2MNK}{(MK + KN + MN) \\times \\text{bytes\\_per\\_element}}$. For large square matrices ($M = N = K = n$), what does $I$ simplify to, and what does this imply?",
      options: ["$I \\approx 2/3$ FLOP/byte regardless of $n$, because the data transfer and compute both scale as $n^3$ and their ratio is a constant determined only by the number of bytes per element", "$I = 2$ FLOP/byte regardless of matrix size, because each element is read once and used in exactly two floating-point operations (one multiply and one add), making the intensity a fixed architectural constant", "$I = n^2$ FLOP/byte, growing quadratically with matrix dimension and making even small matmuls compute-bound on modern GPUs since the ridge point is quickly exceeded for any $n > 16$", "$I \\approx \\frac{2n}{3 \\times \\text{bytes\\_per\\_element}}$, which grows linearly with $n$. For FP16 with $n = 4096$, $I \\approx 1365$ FLOP/byte, far above the A100 ridge point of ~156 FLOP/byte, making large matmuls firmly compute-bound"],
      correct: 3,
      explanation: "For square matrices: FLOPs $= 2n^3$, bytes transferred $= 3n^2 \\times \\text{bytes\\_per\\_element}$ (reading $A$, $B$, and writing $C$). So $I = 2n^3 / (3n^2 \\times \\text{bpe}) = 2n/(3 \\times \\text{bpe})$. For FP16 (bpe = 2), $I = n/3$. At $n = 4096$, $I \\approx 1365$, vastly exceeding the A100 ridge point. This is why ML workloads are structured around large matrix multiplications — they efficiently utilize the GPU's compute capacity. Small or skinny matrices (e.g., batch size 1 during inference) have much lower $I$ and become memory-bound."
    },
    {
      type: "mc",
      question: "When profiling a transformer training step with PyTorch Profiler or NVIDIA Nsight Systems, you observe that GPU utilization is only 40%. The trace shows many small gaps between kernel launches. What is the most likely cause and solution?",
      options: ["The model is too large for the GPU's HBM, causing constant swapping between HBM and system RAM as weight tensors are paged in and out — the solution is to reduce the model size or enable ZeRO-Offload to manage the memory hierarchy explicitly", "The GPU's memory bandwidth is saturated by the weight loads, and the compute units are starved waiting for data even though they are architecturally capable of higher throughput — the solution is to switch to a GPU with higher HBM bandwidth such as an H100", "CPU overhead: the Python-side kernel launch overhead and data preprocessing cannot keep the GPU busy. Small, non-fused kernels exacerbate this because each requires a separate CPU-side launch. Solutions include `torch.compile` (which fuses kernels and uses CUDA graphs to batch launches), increasing batch size (to make each kernel launch do more work), and moving data preprocessing to a separate process", "40% utilization is expected and typical for transformer training because attention is inherently memory-bound, and the roofline model predicts that attention kernels can only utilize ~40% of peak FLOP/s on current hardware architectures regardless of optimization"],
      correct: 2,
      explanation: "Each CUDA kernel launch has ~10-20 microseconds of CPU-side overhead. With hundreds of small kernels per training step (each elementwise op, each matmul, each reduction), this adds up. The GPU finishes a small kernel before the CPU can launch the next one. `torch.compile` addresses this by: (1) fusing elementwise ops into fewer kernels, (2) using CUDA graphs to record and replay entire sequences of kernel launches with a single CPU call, eliminating per-kernel launch overhead. CUDA graphs alone can improve throughput by 10-30% for small-batch workloads."
    }
  ]
};

// ============================================================================
// 0.3: Optimization Theory
// ============================================================================
export const optimizationAssessment = {
  id: "0.3-assess",
  sectionId: "0.3",
  title: "Assessment: Optimization Theory",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The KKT (Karush-Kuhn-Tucker) conditions are necessary conditions for optimality in constrained optimization. For the problem $\\min f(x)$ subject to $g_i(x) \\leq 0$ and $h_j(x) = 0$, which of the following correctly states the complementary slackness condition?",
      options: ["For each inequality constraint, $\\mu_i g_i(x^*) = 0$, meaning either the constraint is active ($g_i(x^*) = 0$) or its dual variable is zero ($\\mu_i = 0$). An inactive constraint contributes no force to the optimality conditions", "All inequality constraints must be active (binding) at the optimum: $g_i(x^*) = 0$ for all $i$, because any slack in a constraint represents wasted capacity that could be used to further decrease the objective value", "The gradient of $f$ must be zero at the optimum regardless of the constraints, because the constraints only restrict the feasible set and do not contribute terms to the first-order stationarity condition", "The dual variables $\\mu_i$ must all be strictly positive at the optimum, ensuring that every inequality constraint actively influences the optimal solution and contributes a non-zero force in the Lagrangian"],
      correct: 0,
      explanation: "The KKT conditions are: (1) Stationarity: $\\nabla f + \\sum_i \\mu_i \\nabla g_i + \\sum_j \\lambda_j \\nabla h_j = 0$; (2) Primal feasibility: $g_i(x^*) \\leq 0$, $h_j(x^*) = 0$; (3) Dual feasibility: $\\mu_i \\geq 0$; (4) Complementary slackness: $\\mu_i g_i(x^*) = 0$. The intuition for complementary slackness: if constraint $i$ is not active (i.e., $g_i(x^*) < 0$, we have slack), then $\\mu_i$ must be 0 — the constraint does not influence the solution. Only active constraints can have $\\mu_i > 0$, pushing the solution away from infeasibility."
    },
    {
      type: "mc",
      question: "Strong duality holds when the optimal primal value equals the optimal dual value: $f(x^*) = g(\\mu^*, \\lambda^*)$. Under what conditions does strong duality hold for the problem $\\min f(x)$ subject to $g_i(x) \\leq 0$?",
      options: [
        "Strong duality always holds for any constrained optimization problem",
        "Strong duality holds when Slater's condition is satisfied: the problem is convex (convex $f$, convex $g_i$) and there exists a strictly feasible point $\\hat{x}$ such that $g_i(\\hat{x}) < 0$ for all $i$. For non-convex problems, there is generally a duality gap $f(x^*) - g(\\mu^*, \\lambda^*) > 0$",
        "Strong duality holds only when $f$ is linear and $g_i$ are affine",
        "Strong duality holds whenever the problem has a unique solution"
      ],
      correct: 1,
      explanation: "Weak duality ($g(\\mu^*, \\lambda^*) \\leq f(x^*)$) always holds. Strong duality requires a constraint qualification. Slater's condition is the most commonly used: for convex problems with a strictly feasible point, the duality gap is zero. This is fundamental to SVMs (where the dual problem is solved instead of the primal), convex relaxations in ML, and Lagrangian methods. For non-convex neural network loss landscapes, strong duality generally fails, which is why Lagrangian-based constrained optimization in deep learning requires careful algorithmic design."
    },
    {
      type: "mc",
      question: "In high-dimensional non-convex optimization (such as neural network training), saddle points are much more prevalent than local minima. What is the theoretical basis for this claim?",
      options: ["Neural network loss functions are convex in all but one direction due to the overparameterized regime, so saddle points are rare and most critical points encountered during training are true local minima with a single negative eigenvalue direction", "Local minima are more common than saddle points in high-dimensional spaces, but they are harder for gradient-based methods to find because their basins of attraction are narrow, so optimizers encounter the more accessible saddle points more frequently", "Saddle points only occur in linear networks where the loss is a polynomial in the weights, not in networks with nonlinear activations like ReLU or GELU, because the nonlinearities break the symmetry that creates saddle-point structure", "At a critical point, each eigenvalue of the Hessian is independently positive or negative with roughly equal probability. In $d$ dimensions, a local minimum requires all $d$ eigenvalues to be positive (probability $\\sim 2^{-d}$), making local minima exponentially rare compared to saddle points (which have a mix of signs). This is supported by random matrix theory applied to the loss landscape"],
      correct: 3,
      explanation: "Bray and Dean (2007) showed that for random Gaussian fields in high dimensions, the probability of a critical point being a local minimum decreases exponentially with dimension. Dauphin et al. (2014) connected this to neural networks: at a critical point of a $d$-dimensional loss surface, the Hessian has $d$ eigenvalues. If each sign is roughly random, the probability that all are positive (local min) is $\\sim 2^{-d}$. For $d = 10^9$ parameters, local minima are astronomically rare. Most critical points are saddle points with $O(d)$ negative eigenvalues, which SGD can escape via noise."
    },
    {
      type: "mc",
      question: "SGD with mini-batches uses gradient estimates $\\hat{g} = \\frac{1}{|B|} \\sum_{i \\in B} \\nabla \\ell_i(\\theta)$. How does the gradient noise from mini-batch sampling act as regularization?",
      options: ["The noise causes the optimizer to diverge from all solutions — both sharp and flat minima — requiring early stopping as a compensating regularizer that halts training before the accumulated noise pushes the weights too far from any minimum", "SGD noise has no regularization effect on its own; any observed generalization benefit is entirely due to the learning rate schedule and weight decay, which are the true sources of implicit bias toward flat minima in practical training configurations", "The noise in mini-batch gradients has covariance approximately $\\frac{\\Sigma}{|B|}$ where $\\Sigma$ is the per-sample gradient covariance. This noise biases SGD toward flatter minima: sharp minima (large Hessian eigenvalues) are destabilized because gradient noise causes escape, while flat minima (small eigenvalues) are stable. The noise implicitly penalizes sharpness, acting as a form of regularization controlled by the learning rate-to-batch size ratio $\\eta / |B|$", "The noise adds an implicit L2 penalty to the loss that is mathematically equivalent to weight decay with strength proportional to the gradient variance, making mini-batch SGD identical to full-batch SGD with explicit L2 regularization"],
      correct: 2,
      explanation: "The SGD noise covariance is $C = \\frac{\\Sigma}{|B|}$, and the effective temperature of the stochastic process is governed by $\\eta / |B|$ (learning rate divided by batch size). Near a minimum with Hessian $H$, the noise causes fluctuations of order $\\sqrt{\\eta \\sigma^2 / |B|}$ in each eigendirection. If $\\eta \\lambda_{\\max}(H) \\sigma^2 / |B|$ is too large, the optimizer escapes the minimum. This means sharp minima (large $\\lambda_{\\max}$) are destabilized first. This is why reducing batch size or increasing learning rate (both increase $\\eta / |B|$) can improve generalization."
    },
    {
      type: "mc",
      question: "Adam maintains exponential moving averages of the first moment $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ and second moment $v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$. Why is bias correction ($\\hat{m}_t = m_t / (1 - \\beta_1^t)$ and $\\hat{v}_t = v_t / (1 - \\beta_2^t)$) necessary?",
      options: [
        "Bias correction normalizes the moments to unit variance for numerical stability",
        "Since $m_0 = 0$ and $v_0 = 0$, the exponential moving averages are biased toward zero in early iterations. $\\mathbb{E}[m_t] = (1 - \\beta_1^t) \\mathbb{E}[g_t]$, so dividing by $(1 - \\beta_1^t)$ corrects this initialization bias. Without correction, early updates are too small (especially for $\\beta_2 = 0.999$, where $1 - \\beta_2^t$ is near zero for many steps)",
        "Bias correction prevents the learning rate from decaying to zero",
        "Bias correction is optional and only affects convergence speed, not the final solution"
      ],
      correct: 1,
      explanation: "At step $t$, unrolling the recurrence gives $m_t = (1-\\beta_1) \\sum_{i=1}^{t} \\beta_1^{t-i} g_i$. Taking expectations: $\\mathbb{E}[m_t] = (1-\\beta_1) \\sum_{i=1}^{t} \\beta_1^{t-i} \\mathbb{E}[g_i] = \\mathbb{E}[g] \\cdot (1 - \\beta_1^t)$. The factor $(1 - \\beta_1^t)$ is the bias. For $\\beta_2 = 0.999$, $1 - \\beta_2^{10} \\approx 0.01$, meaning $v_{10}$ underestimates the true second moment by 100x without correction. This would make early updates 10x too large (since Adam divides by $\\sqrt{v}$), potentially destabilizing training."
    },
    {
      type: "mc",
      question: "Why does Adam work particularly well for transformers compared to SGD with momentum?",
      options: ["Transformers have loss landscapes with highly non-uniform curvature: the Hessian eigenvalue spectrum spans many orders of magnitude (from attention logits to embedding weights). Adam's per-parameter adaptive learning rate ($\\eta / \\sqrt{\\hat{v}_t}$) effectively rescales each parameter's update by the inverse of its RMS gradient, providing implicit preconditioning that handles this curvature variation. SGD uses a single learning rate for all parameters", "Adam uses less memory than SGD with momentum because it replaces the explicit momentum buffer with a running average that can be stored in FP16, while SGD with momentum requires a full FP32 copy of the gradient history for numerical stability across training steps", "Adam converges to the global minimum of the non-convex transformer loss landscape while SGD with momentum gets stuck in suboptimal local minima, because Adam's adaptive step sizes allow it to escape saddle points that trap fixed-learning-rate methods", "Adam is only preferred because of its well-tuned default hyperparameters ($\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$) which happen to work across many architectures, not because of any fundamental algorithmic advantage over properly tuned SGD with momentum"],
      correct: 0,
      explanation: "In a transformer, the gradient magnitudes across different parameter groups can differ by orders of magnitude (e.g., attention QK projections vs. output embeddings). Adam's update $\\Delta \\theta_i \\propto \\hat{m}_i / \\sqrt{\\hat{v}_i}$ normalizes each parameter's step size by its gradient RMS, effectively adapting to local curvature. This approximates a diagonal preconditioner: $\\Delta \\theta \\approx \\text{diag}(\\hat{v})^{-1/2} \\hat{m}$, compared to the ideal Newton step $H^{-1} g$. For SGD, a single learning rate must compromise between too large for some parameters and too small for others, making it harder to tune and slower to converge."
    },
    {
      type: "mc",
      question: "The natural gradient uses the Fisher information matrix $F = \\mathbb{E}_{p(x|\\theta)}[\\nabla \\log p(x|\\theta) \\nabla \\log p(x|\\theta)^\\top]$ to precondition the gradient update: $\\Delta \\theta = -F^{-1} \\nabla \\mathcal{L}$. What geometric interpretation does this provide?",
      options: [
        "It computes the gradient in Euclidean space, which is always the most efficient direction",
        "The natural gradient computes the steepest descent direction in the space of distributions rather than in parameter space: it moves $\\theta$ to maximize the loss decrease per unit of KL divergence $D_{\\text{KL}}(p_\\theta \\| p_{\\theta + \\Delta\\theta})$, making the optimization invariant to reparameterization of the model",
        "It averages the gradient over the Fisher distribution to reduce variance",
        "It projects the gradient onto the principal components of the data"
      ],
      correct: 1,
      explanation: "In parameter space, Euclidean distance $\\|\\Delta\\theta\\|_2$ can be misleading: a small change in $\\theta$ might cause a large change in the distribution $p_\\theta$ (or vice versa). The Fisher matrix defines a Riemannian metric on the statistical manifold where the local distance between $p_\\theta$ and $p_{\\theta+\\Delta\\theta}$ is $\\Delta\\theta^\\top F \\Delta\\theta \\approx 2 D_{\\text{KL}}(p_\\theta \\| p_{\\theta+\\Delta\\theta})$. The natural gradient $F^{-1} \\nabla \\mathcal{L}$ gives steepest descent under this KL metric. This is reparameterization-invariant: if you change variables $\\theta \\to \\phi(\\theta)$, the natural gradient transform correctly."
    },
    {
      type: "mc",
      question: "The debate around sharp vs. flat minima and generalization has been ongoing. What is the main criticism of using sharpness (eigenvalues of the Hessian) as a predictor of generalization?",
      options: ["The Hessian is too expensive to compute, making sharpness impractical as a metric", "The Hessian is always positive definite at local minima, so all minima have the same sharpness", "Flat minima always generalize worse than sharp minima in practice", "Dinh et al. (2017) showed that for networks with ReLU activations, sharpness is not reparameterization-invariant: you can reparameterize the network (e.g., scale one layer's weights by $\\alpha$ and the next by $1/\\alpha$) to make any minimum arbitrarily sharp or flat without changing the function the network computes or its generalization"],
      correct: 3,
      explanation: "For a network with ReLU activations and layers $W_1, W_2$, the function $f(x) = W_2 \\cdot \\text{ReLU}(W_1 x)$ is identical to $f(x) = (W_2/\\alpha) \\cdot \\text{ReLU}(\\alpha W_1 x)$ for any $\\alpha > 0$ (by homogeneity of ReLU). But scaling $W_1$ by $\\alpha$ multiplies the Hessian eigenvalues by $\\alpha^2$, making the minimum arbitrarily sharp. This means raw sharpness cannot predict generalization. Recent work defines sharpness measures that are reparameterization-invariant (e.g., SAM's sharpness-aware minimization uses normalized perturbations) to address this critique."
    },
    {
      type: "mc",
      question: "SGD with learning rate $\\eta$ and batch size $B$ converges to a neighborhood of a local minimum, with the size of that neighborhood depending on $\\eta/B$. What is the convergence rate of SGD for a general (non-convex) smooth function to an approximate stationary point ($\\|\\nabla f(\\theta)\\|^2 \\leq \\epsilon$)?",
      options: [
        "$O(1/\\epsilon)$ iterations — linear convergence",
        "$O(1/\\epsilon^2)$ iterations: after $T$ steps, $\\min_{t \\leq T} \\mathbb{E}[\\|\\nabla f(\\theta_t)\\|^2] \\leq O(\\sigma^2 / (\\sqrt{T} \\cdot |B|) + (f(\\theta_0) - f^*) / (\\eta T))$, where $\\sigma^2$ is the gradient noise variance. The $1/\\sqrt{T}$ dependence on noise is unavoidable for stochastic methods",
        "$O(\\log(1/\\epsilon))$ iterations — exponential convergence",
        "SGD does not converge for non-convex functions"
      ],
      correct: 1,
      explanation: "For $L$-smooth non-convex $f$, with learning rate $\\eta \\leq 1/(2L)$ and batch size $B$, after $T$ steps: $\\frac{1}{T} \\sum_{t=0}^{T-1} \\mathbb{E}[\\|\\nabla f(\\theta_t)\\|^2] \\leq \\frac{2(f(\\theta_0) - f^*)}{\\eta T} + \\frac{L \\eta \\sigma^2}{B}$. Setting $\\eta = \\sqrt{B/(T \\sigma^2 L)}$ gives convergence rate $O(\\sigma / \\sqrt{BT})$ to an approximate stationary point. This $O(1/\\sqrt{T})$ rate is tight: it matches the lower bound for stochastic first-order methods. Increasing batch size $B$ helps linearly until communication costs dominate."
    },
    {
      type: "mc",
      question: "Sharpness-Aware Minimization (SAM) modifies the training objective to explicitly seek flat minima. What does SAM optimize?",
      options: ["SAM adds an L2 penalty to encourage small weights, which indirectly leads to flat minima", "SAM computes the Hessian trace and adds it as a regularization term to the loss", "SAM minimizes the worst-case loss in a neighborhood: $\\min_\\theta \\max_{\\|\\epsilon\\| \\leq \\rho} \\mathcal{L}(\\theta + \\epsilon)$. At each step, it first computes the adversarial perturbation $\\hat{\\epsilon} = \\rho \\nabla \\mathcal{L}(\\theta) / \\|\\nabla \\mathcal{L}(\\theta)\\|$ and then takes a gradient step at the perturbed point: $\\theta \\leftarrow \\theta - \\eta \\nabla \\mathcal{L}(\\theta + \\hat{\\epsilon})$", "SAM randomly perturbs the weights each step and averages the resulting losses"],
      correct: 2,
      explanation: "SAM's minimax objective seeks parameters $\\theta$ where not just $\\mathcal{L}(\\theta)$ is low, but the loss remains low for all perturbations $\\epsilon$ within an $\\ell_2$ ball of radius $\\rho$. This explicitly penalizes sharp minima where small perturbations cause large loss increases. The practical algorithm uses a first-order approximation: the inner maximization is approximated by a single gradient ascent step (giving $\\hat{\\epsilon}$), and the outer minimization takes a gradient step at $\\theta + \\hat{\\epsilon}$. This requires two forward-backward passes per step (2x compute cost) but consistently improves generalization across vision and language tasks."
    }
  ]
};

// ============================================================================
// 0.4: Systems & Hardware
// ============================================================================
export const systemsAssessment = {
  id: "0.4-assess",
  sectionId: "0.4",
  title: "Assessment: Systems & Hardware for ML",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "An NVIDIA GPU organizes execution into Streaming Multiprocessors (SMs), each running multiple warps. What is a warp, and why does warp-level execution matter for ML kernel performance?",
      options: ["A warp is a group of 32 threads that execute the same instruction simultaneously (SIMT — Single Instruction, Multiple Threads). If threads within a warp take different branches (warp divergence), both paths are executed serially with threads masked, reducing throughput. For ML kernels, data-parallel operations (like elementwise ops) naturally avoid divergence, while operations with data-dependent branching can suffer significant slowdowns", "A warp is a group of 64 threads that share an instruction cache and execute instructions in a round-robin fashion. Warps are purely a software abstraction with no hardware-level significance — the GPU scheduler treats individual threads as the fundamental unit of execution", "A warp is a dedicated hardware unit within each SM that performs matrix multiplication using tensor cores. Each SM contains multiple warps, and they operate in parallel to tile large matrix operations across the available tensor core compute units", "A warp is a memory controller that manages 32 consecutive bytes in HBM, coalescing memory requests from threads into aligned transactions. Each SM has multiple warp controllers that pipeline memory accesses to hide latency through transaction-level parallelism"],
      correct: 0,
      explanation: "The warp is the fundamental unit of execution on NVIDIA GPUs. All 32 threads in a warp share a program counter and execute in lockstep. When an `if` statement causes some threads to take one branch and others a different branch, the hardware serializes: it executes the `if` path (masking threads that took `else`), then the `else` path (masking the others). For ML, this means kernels should minimize data-dependent branching. Quantization kernels that branch on per-element conditions can suffer severe warp divergence. Structured operations (matmul, reductions, elementwise) naturally keep warps coherent."
    },
    {
      type: "mc",
      question: "PyTorch's autograd system records operations on tensors with `requires_grad=True` to enable automatic differentiation. How does it work internally?",
      options: [
        "PyTorch uses symbolic differentiation: it transforms the Python source code to produce a derivative function before execution",
        "PyTorch builds a dynamic computational graph (the autograd tape) during the forward pass: each operation records a `grad_fn` node containing the backward function and references to input tensors. Calling `.backward()` traverses this graph in reverse topological order, applying the chain rule. The graph is destroyed after each backward pass (unless `retain_graph=True`), allowing Python control flow to change the graph each iteration",
        "PyTorch uses numerical differentiation (finite differences) to estimate gradients",
        "PyTorch precompiles all possible computation graphs at import time and selects the correct one at runtime"
      ],
      correct: 1,
      explanation: "PyTorch's define-by-run paradigm builds the graph dynamically. For `y = torch.relu(x @ W + b)`, the forward pass creates nodes: MatmulBackward -> AddBackward -> ReluBackward, each storing the information needed for the backward pass (e.g., MatmulBackward stores $x$ and $W$ for computing $\\partial L/\\partial W = x^\\top \\partial L/\\partial y$ and $\\partial L/\\partial x = \\partial L/\\partial y \\cdot W^\\top$). This dynamic graph naturally handles Python `if` statements, loops, and recursion — the graph simply reflects whatever operations were executed. The cost is per-operation overhead, which `torch.compile` mitigates."
    },
    {
      type: "mc",
      question: "`torch.compile` transforms PyTorch code for faster execution. What does it actually do under the hood?",
      options: ["It converts the PyTorch model definition to C++ source code and compiles it with GCC, bypassing the Python interpreter entirely to eliminate per-operation overhead and producing a standalone executable that runs the model's forward and backward passes natively", "It replaces all PyTorch operations with pre-optimized cuDNN library calls by matching each operator to the closest cuDNN primitive, leveraging NVIDIA's hand-tuned implementations to achieve peak throughput without requiring any changes to the model code or graph structure", "It enables CUDA graphs for all operations without performing any graph transformation or operator fusion, simply recording the sequence of kernel launches and replaying them to eliminate per-launch CPU overhead while keeping each kernel identical to the eager-mode version", "It uses TorchDynamo to capture the computational graph by intercepting Python bytecode, then passes the graph to a backend compiler (TorchInductor by default) that performs operator fusion, memory planning, and generates optimized Triton kernels or CUDA code. Graph breaks occur when unsupported Python constructs are encountered"],
      correct: 3,
      explanation: "The `torch.compile` pipeline has three stages: (1) TorchDynamo modifies CPython's frame evaluation to trace through Python bytecode, capturing the computation as an FX graph while handling Python control flow via guards and graph breaks. (2) AOTAutograd traces the forward and backward passes ahead-of-time, enabling joint optimization. (3) TorchInductor (the default backend) applies optimizations like operator fusion (combining pointwise ops into single kernels), memory planning (reusing buffers), and code generation (emitting optimized Triton kernels). The result is significantly fewer, larger kernel launches."
    },
    {
      type: "mc",
      question: "In data-parallel distributed training, all-reduce is the standard collective communication operation for synchronizing gradients. What does all-reduce compute, and what is the bandwidth cost of the ring all-reduce algorithm?",
      options: ["All-reduce sends all gradients to a designated central server, which computes the element-wise mean and broadcasts the result back to all workers. The total communication cost is $O(N \\cdot M)$ where $N$ is the number of workers and $M$ is the message size, making it increasingly expensive as the cluster grows", "All-reduce randomly selects one worker's gradient tensor at each synchronization step and copies it to all other workers, providing an unbiased estimate of the true gradient mean while requiring only $M$ bytes of communication per step regardless of the number of workers $N$", "All-reduce computes the sum (or mean) of tensors across all $N$ workers and places the result on every worker. Ring all-reduce achieves this in $2(N-1)/N \\cdot M$ bytes of data transferred per worker (asymptotically $2M$), which is independent of $N$ — making it bandwidth-optimal. It proceeds in two phases: reduce-scatter and all-gather, each taking $N-1$ steps", "All-reduce only works with exactly 2 workers at a time and must be composed recursively via a binary tree structure for larger clusters, resulting in $O(M \\log N)$ total communication per worker and $O(\\log N)$ synchronization rounds, which is suboptimal compared to direct multi-worker algorithms"],
      correct: 2,
      explanation: "Ring all-reduce arranges $N$ workers in a logical ring. Phase 1 (reduce-scatter): over $N-1$ steps, each worker sends a $1/N$-th chunk of its data to the next worker while receiving and accumulating a chunk from the previous worker. After $N-1$ steps, each worker holds the fully reduced version of one chunk. Phase 2 (all-gather): over $N-1$ steps, these reduced chunks are circulated so every worker gets all chunks. Total data per worker: $(N-1)/N \\cdot M$ per phase, or $2(N-1)/N \\cdot M$ total. This is bandwidth-optimal: the lower bound is $2(N-1)/N \\cdot M$. The latency is $O(N)$ steps, which tree-based algorithms reduce to $O(\\log N)$."
    },
    {
      type: "mc",
      question: "A parameter server architecture is an alternative to all-reduce for distributed training. How does it differ, and when is each approach preferred?",
      options: ["In a parameter server architecture, dedicated server nodes store model parameters, and worker nodes push gradients and pull updated weights. This enables asynchronous SGD (workers don't wait for each other) but introduces stale gradients. All-reduce is synchronous and bandwidth-optimal for dense gradients. Parameter servers are preferred for sparse models (e.g., recommendation models with large embedding tables) where each worker only accesses a subset of parameters", "Parameter servers are always faster than all-reduce because asynchronous gradient updates eliminate synchronization barriers entirely, and the server nodes provide dedicated compute for the optimizer step that would otherwise compete with forward/backward computation on the worker GPUs", "Parameter servers and all-reduce are mathematically identical algorithms that always produce the same model updates given the same data, differing only in their implementation topology — the parameter server uses a star topology while all-reduce uses a ring topology for the same communication", "Parameter servers operate exclusively on CPU memory to store and update parameters, while all-reduce operates exclusively on GPU memory for gradient aggregation — the two approaches target different hardware tiers and cannot be combined within a single training framework"],
      correct: 0,
      explanation: "Parameter servers split the model across server nodes. Workers compute gradients on their data, push gradients to servers, and pull updated parameters. This naturally handles sparse updates: if a worker only touches embedding rows $\\{5, 1000, 50000\\}$, it only communicates those rows, not the entire embedding table. Asynchronous PS allows workers to proceed without synchronization barriers, improving throughput but introducing gradient staleness that can hurt convergence. For dense transformer training, synchronous all-reduce is preferred because all parameters are updated every step and ring all-reduce is bandwidth-optimal for this pattern."
    },
    {
      type: "mc",
      question: "NVLink and NVSwitch provide high-bandwidth GPU-to-GPU interconnects within a node, while InfiniBand connects nodes across a cluster. What are the typical bandwidth differences, and why does this matter for model parallelism?",
      options: [
        "NVLink and InfiniBand have the same bandwidth, so it doesn't matter where computation is placed",
        "NVLink (4th gen, A100/H100) provides 600-900 GB/s bidirectional bandwidth between GPUs, while InfiniBand HDR provides ~25 GB/s per port (200 Gbps). NVSwitch enables all-to-all NVLink connectivity within a node. This ~30x bandwidth gap means tensor parallelism (which requires high-bandwidth all-reduce every layer) should be confined within NVLink-connected GPUs, while pipeline and data parallelism (less frequent communication) span across nodes via InfiniBand",
        "InfiniBand is faster than NVLink but has higher latency",
        "NVLink connects GPUs to CPUs, not GPUs to each other"
      ],
      correct: 1,
      explanation: "In an 8-GPU DGX A100 node with NVSwitch, any GPU can communicate with any other at 600 GB/s. Across nodes, InfiniBand HDR gives ~25 GB/s per link (often 4-8 links per node for 100-200 GB/s aggregate). Tensor parallelism splits individual matrix multiplications across GPUs, requiring an all-reduce of the full hidden dimension after every layer — this demands the highest bandwidth and must use NVLink. Pipeline parallelism only communicates activations between pipeline stages (much less data), suitable for the slower InfiniBand. This creates a hierarchical parallelism strategy: TP within a node, PP across nodes, DP across replicas."
    },
    {
      type: "mc",
      question: "When profiling a training run with PyTorch Profiler, you see that a single linear layer (`nn.Linear(4096, 4096)`) takes 0.5ms with batch size 1 but only 0.6ms with batch size 64. What explains this sublinear scaling of time with batch size?",
      options: ["PyTorch automatically reduces the numerical precision from FP16 to INT8 for larger batches to maintain throughput, trading a small amount of accuracy for a 2x reduction in memory bandwidth requirements per element, which keeps the wall-clock time nearly constant as batch size grows", "Batch size 64 triggers cuBLAS to select a fundamentally different CUDA kernel implementation that uses a Strassen-like algorithm with $O(n^{2.81})$ complexity instead of the standard $O(n^3)$ algorithm, yielding a multiplicative speedup that is independent of the memory subsystem", "The GPU dynamically overclocks its compute units when it detects larger batch sizes via a hardware-level occupancy monitor, boosting the clock frequency by up to 30% when the workload is large enough to thermally justify the increased power consumption without risking thermal throttling", "With batch size 1, the matmul is $y = Wx$ where $W \\in \\mathbb{R}^{4096 \\times 4096}$ and $x \\in \\mathbb{R}^{4096}$: 33M FLOPs but 32MB of weight data to load — arithmetic intensity of ~1 FLOP/byte (memory-bound). With batch size 64, $X \\in \\mathbb{R}^{64 \\times 4096}$: 2.1B FLOPs but the same 32MB of weights — arithmetic intensity of ~64 FLOP/byte (compute-bound). The weights are loaded once and reused across all batch elements, so increasing batch size is nearly free until compute saturation"],
      correct: 3,
      explanation: "This is the classic memory-bound to compute-bound transition. For $y = WX^\\top$ with $W \\in \\mathbb{R}^{m \\times k}$, $X \\in \\mathbb{R}^{B \\times k}$: FLOPs $= 2mBk$, bytes $\\approx (mk + Bk) \\times 2$ (FP16). Arithmetic intensity $I \\approx 2mB/(m+B)$ FLOP/byte. For $m = k = 4096$: $B = 1$ gives $I \\approx 2$, $B = 64$ gives $I \\approx 124$, and $B = 256$ gives $I \\approx 470$. On an A100 (ridge point ~156), $B = 64$ is still somewhat memory-bound but much closer to compute-bound, while $B = 256$ is fully compute-bound. This is why batching is critical for inference throughput."
    },
    {
      type: "mc",
      question: "NVIDIA Nsight Systems and Nsight Compute are two profiling tools. What is the key difference between them, and when would you use each?",
      options: ["They are identical tools with different names for different GPU architectures", "Nsight Systems profiles CPU code and Nsight Compute profiles GPU memory", "Nsight Systems provides a system-level timeline view showing CPU activity, GPU kernel launches, memory transfers, and inter-device communication across the entire application. Nsight Compute provides detailed per-kernel analysis: occupancy, memory throughput, compute utilization, warp stall reasons, and roofline analysis. Use Nsight Systems first to identify bottleneck regions, then Nsight Compute to drill into specific slow kernels", "Nsight Compute is the newer version that replaces Nsight Systems entirely"],
      correct: 2,
      explanation: "The profiling workflow is: (1) Run Nsight Systems (`nsys profile python train.py`) to get the full timeline. Look for gaps between kernel launches (CPU overhead), unexpectedly slow kernels, or communication stalls. (2) Once you identify a specific kernel, use Nsight Compute (`ncu --set full python train.py --kernel-name regex:my_kernel`) to analyze it in detail: is it compute-bound or memory-bound? What is the occupancy? Are there bank conflicts in shared memory? What are the warp stall reasons? This two-level approach efficiently narrows down performance issues."
    },
    {
      type: "mc",
      question: "GPU occupancy measures the ratio of active warps to the maximum possible warps per SM. Why is 100% occupancy not always optimal, and what limits occupancy?",
      options: ["Occupancy is limited by three resources per SM: registers per thread, shared memory per block, and the maximum number of thread blocks. A kernel using many registers per thread can only run a few warps concurrently. However, fewer warps with more registers can outperform many warps with fewer registers, because register pressure causes spills to local memory (slow). The optimal occupancy balances latency hiding (more warps to cover memory stalls) against per-thread resource availability", "100% occupancy is always optimal because more warps means more parallelism", "Occupancy is always 100% on modern GPUs due to hardware scheduling improvements", "Occupancy only matters for graphics workloads, not ML training"],
      correct: 0,
      explanation: "An A100 SM has 65,536 registers and supports up to 64 warps (2048 threads). A kernel using 128 registers per thread allows $65536 / (128 \\times 32) = 16$ warps per SM, giving 25% occupancy. Reducing to 64 registers allows 32 warps (50% occupancy) but may require spilling registers to local memory (~100x slower than register access). For compute-bound kernels (like matmul), fewer warps with full register usage often wins. For memory-bound kernels, more warps are needed to hide memory latency through warp switching. Shared memory limits work similarly: more shared memory per block means fewer blocks per SM."
    },
    {
      type: "mc",
      question: "Operator fusion in `torch.compile` or custom Triton kernels eliminates intermediate tensors. For a transformer block with LayerNorm, QKV projection, attention, output projection, another LayerNorm, and an MLP with SiLU gating, which operations are typically fused and which are not?",
      options: [
        "All operations in the transformer block are fused into a single kernel for maximum efficiency",
        "Large matrix multiplications (QKV projection, output projection, MLP up/down projections) are NOT fused with adjacent operations because they are compute-bound and already efficient as standalone cuBLAS calls. However, pointwise operations (LayerNorm, residual add, SiLU activation, dropout) ARE fused with each other and sometimes with the adjacent matmul epilogue, because these operations are memory-bound and fusion eliminates HBM round-trips between them",
        "Only the attention computation (softmax and masking) is fused; everything else runs as separate kernels",
        "Fusion is never applied to training, only to inference"
      ],
      correct: 1,
      explanation: "The optimization strategy follows the roofline model. Matmuls have high arithmetic intensity and run near peak FLOP/s via cuBLAS — fusing a cheap elementwise op into them provides negligible benefit. But between matmuls, there are sequences of memory-bound ops: residual add + LayerNorm (read input, compute stats, normalize, write output), or SiLU activation + elementwise multiply in gated MLPs. Each unfused op reads from and writes to HBM. Fusing N consecutive pointwise ops reduces HBM accesses from 2N to 2 (one read, one write). `torch.compile`'s TorchInductor automatically identifies and fuses these pointwise chains."
    }
  ]
};
