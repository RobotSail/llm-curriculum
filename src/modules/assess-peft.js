// G.1: Parameter-Efficient Fine-Tuning Assessment
// Pure assessment — no info steps

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
