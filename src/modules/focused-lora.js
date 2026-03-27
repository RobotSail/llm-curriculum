// Focused module: Low-Rank Adaptation (LoRA)
// Covers motivation from intrinsic dimensionality, the LoRA mechanism,
// parameter counting, rank selection, alpha scaling, initialization,
// inference merging, and QLoRA.
// Connects to SVD and eigendecomposition from Tier 0 (section 0.1).
// Grounded in Hu et al. (2021), Aghajanyan et al. (2020), Dettmers et al. (2023).

export const loraLearning = {
  id: "G.1-lora-learning-easy",
  sectionId: "G.1",
  title: "Low-Rank Adaptation (LoRA)",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "The Cost of Full Fine-Tuning",
      content: "Full fine-tuning updates **every parameter** in the model. For a 7B model with AdamW, this requires:\n\n- **Parameters**: 7B $\\times$ 2 bytes (BF16) = 14 GB\n- **Optimizer states**: 7B $\\times$ 2 (first + second moment) $\\times$ 4 bytes (FP32) = 56 GB\n- **Gradients**: 7B $\\times$ 2 bytes = 14 GB\n- **Total**: ~84 GB just for model state — before activations\n\nThis means a single 80 GB A100 cannot fine-tune a 7B model with AdamW without memory optimization tricks. A 70B model requires a multi-node setup.\n\nBut there is a deeper question: **do we need to update all 7B parameters?** If the pretrained model already has strong representations, perhaps fine-tuning only adjusts a small subspace of the full parameter space. If so, we are paying for 7B parameters of freedom when the actual adaptation lives in a much smaller space.\n\nThis intuition turns out to be correct, and it is the foundation of parameter-efficient fine-tuning (PEFT)."
    },
    {
      type: "mc",
      question: "A 70B parameter model with AdamW requires approximately how much GPU memory just for parameters + optimizer states (ignoring activations and gradients)?",
      options: [
        "140 GB — parameters in BF16 (2 bytes each) account for all memory since AdamW stores optimizer states in the same format",
        "280 GB — parameters in FP32 (4 bytes each) plus a single momentum buffer of equal size",
        "700 GB — BF16 parameters (140 GB) plus two FP32 optimizer states for AdamW's first and second moments (2 $\\times$ 280 GB = 560 GB)",
        "1.4 TB — each parameter requires 20 bytes total when accounting for master weights, optimizer states, and internal buffers"
      ],
      correct: 2,
      explanation: "Parameters: 70B $\\times$ 2 bytes (BF16) = 140 GB. AdamW maintains first moment ($m$) and second moment ($v$), each in FP32: 70B $\\times$ 4 bytes $\\times$ 2 = 560 GB. Total: 140 + 560 = 700 GB. Each parameter costs 10 bytes (2 for BF16 weight + 4 for first moment + 4 for second moment). This is before activations and gradients — the full training footprint is even larger. This enormous memory cost is what motivates parameter-efficient methods like LoRA."
    },
    {
      type: "info",
      title: "Intrinsic Dimensionality: The Adaptation Lives in a Small Subspace",
      content: "Aghajanyan et al. (2020) measured the **intrinsic dimensionality** of fine-tuning tasks: the minimum number of free parameters needed to reach 90% of full fine-tuning performance.\n\nTheir finding was striking: for a 355M parameter model, most NLP tasks had intrinsic dimensionality around **200-800** — orders of magnitude smaller than the full parameter count. The model has 355 million degrees of freedom, but the adaptation only uses a few hundred.\n\nThis means the weight change $\\Delta W = W_{\\text{fine-tuned}} - W_{\\text{pretrained}}$ is approximately **low-rank**. If you compute the SVD of $\\Delta W$, most singular values are near zero — the update is concentrated in a small number of directions.\n\nThis connects directly to the SVD and eigendecomposition theory from Section 0.1: a matrix with only $r$ significant singular values can be well-approximated by a rank-$r$ factorization, saving storage proportional to $r/\\min(d, k)$.\n\nLoRA exploits this directly: instead of learning a full $\\Delta W \\in \\mathbb{R}^{d \\times k}$, learn a low-rank factorization $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ with $r \\ll \\min(d, k)$."
    },
    {
      type: "mc",
      question: "A 355M parameter model has intrinsic dimensionality of ~500 for a sentiment classification task. This means:",
      options: [
        "Fine-tuning needs at most ~500 free parameters to reach 90% of full fine-tuning performance, because the adaptation lies in a ~500-dimensional subspace",
        "The model activates only ~500 neurons per forward pass on sentiment inputs, with the remaining neurons contributing near-zero activations",
        "The training set must contain at least 500 labeled examples for the fine-tuning loss to converge, since each free parameter needs one example",
        "The model's hidden dimension should be reduced to 500 via pruning before fine-tuning, matching capacity to the task's complexity"
      ],
      correct: 0,
      explanation: "Intrinsic dimensionality measures the number of degrees of freedom actually needed for the task-specific adaptation. A value of ~500 means that a random 500-dimensional projection of the full parameter space captures enough variation to reach 90% of the quality achieved by updating all 355M parameters. This is strong evidence that the weight change $\\Delta W$ is approximately low-rank — it lives in a tiny subspace relative to the full space."
    },
    {
      type: "info",
      title: "The LoRA Mechanism",
      content: "LoRA (Hu et al., 2021) adds a **low-rank update** to each adapted weight matrix. For a pretrained weight $W_0 \\in \\mathbb{R}^{d \\times k}$:\n\n$$h = W_0 x + \\frac{\\alpha}{r} BAx$$\n\nwhere:\n- $W_0$ is **frozen** (not updated during training)\n- $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ are the trainable low-rank matrices\n- $r$ is the **rank** — a hyperparameter controlling capacity\n- $\\alpha$ is a scaling constant (more on this shortly)\n\nThe forward pass computes the original linear transformation $W_0 x$ plus a low-rank correction $BAx$. During training, only $B$ and $A$ receive gradients — $W_0$ stays exactly as pretrained.\n\n**Parameter count**: $B$ has $dr$ parameters, $A$ has $rk$ parameters. Total: $r(d + k)$. For a typical transformer layer with $d = k = 4096$ and $r = 16$, this is $16 \\times 8192 = 131{,}072$ parameters vs. $4096^2 \\approx 16.8$M for the full matrix — a **128x reduction**.\n\nSince only the LoRA parameters have optimizer states, memory drops from $\\sim$84 GB (full fine-tuning a 7B model) to $\\sim$16 GB (frozen model + small LoRA optimizer states)."
    },
    {
      type: "mc",
      question: "A transformer has weight matrices of dimension $4096 \\times 4096$. LoRA with rank $r = 8$ is applied to all 4 attention projections ($W_Q, W_K, W_V, W_O$) and 3 MLP projections ($W_{\\text{gate}}, W_{\\text{up}}, W_{\\text{down}}$) per layer. How many trainable parameters does LoRA add per layer?",
      options: [
        "$7 \\times 4096^2 = 117$M — LoRA doesn't reduce parameters, it just freezes the originals and trains copies",
        "$7 \\times 8 \\times 4096 = 229{,}376$ — each adapted matrix adds $r \\times d$ parameters from one factor only",
        "$7 \\times 8^2 = 448$ — LoRA parameters scale as $r^2$ independent of the weight matrix dimensions",
        "$7 \\times 8 \\times (4096 + 4096) = 458{,}752$ — each adapted matrix adds $r(d + k)$ parameters from both factors $B$ and $A$"
      ],
      correct: 3,
      explanation: "Each LoRA adapter has $B \\in \\mathbb{R}^{4096 \\times 8}$ and $A \\in \\mathbb{R}^{8 \\times 4096}$, contributing $8 \\times (4096 + 4096) = 65{,}536$ parameters. With 7 adapted matrices per layer: $7 \\times 65{,}536 = 458{,}752$. Compare to the full layer parameters: $7 \\times 4096^2 \\approx 117$M. LoRA uses $458{,}752 / 117$M $\\approx 0.4\\%$ of the parameters — a dramatic reduction."
    },
    {
      type: "info",
      title: "Initialization: Starting from the Pretrained Model",
      content: "LoRA uses a specific initialization strategy:\n- **$A$** is initialized from a random Gaussian $\\mathcal{N}(0, \\sigma^2)$\n- **$B$** is initialized to **zero**\n\nThis ensures that $\\Delta W = BA = 0$ at the start of training. The model begins with exactly the pretrained behavior: $h = W_0 x + \\frac{\\alpha}{r} \\cdot \\mathbf{0} \\cdot x = W_0 x$.\n\nWhy this matters: the pretrained model already works well. We want LoRA to learn a **small perturbation** on top of a known-good solution, not to start from a random point and hope it converges to something useful.\n\nDoes $B = 0$ mean training stalls? No. The gradient of the loss with respect to $B$ is:\n$$\\frac{\\partial L}{\\partial B} = \\frac{\\alpha}{r} \\frac{\\partial L}{\\partial h} (Ax)^\\top$$\n\nThis is nonzero as long as $A \\neq 0$ (which it isn't — $A$ has random values) and the loss has a nonzero gradient. So $B$ immediately begins updating from the first step.\n\nThis initialization is analogous to the residual learning idea from ResNets: learn the **residual** $\\Delta W$ starting from zero, rather than learning $W$ from scratch."
    },
    {
      type: "mc",
      question: "What would happen if both $A$ and $B$ were initialized randomly (instead of $B = 0$)?",
      options: [
        "Training would fail entirely because the random $BA$ product creates a rank-deficient Jacobian that blocks gradient flow through the adapted layers permanently",
        "The random $BA \\neq 0$ would corrupt pretrained representations from step 0, forcing the optimizer to waste early training undoing random damage before learning the task",
        "Performance would be identical because AdamW's momentum corrects any initialization-dependent effects within the first few hundred optimization steps regardless",
        "The model would converge faster because random $B$ breaks symmetry between rank-1 components, letting each component specialize toward different adaptation directions immediately"
      ],
      correct: 1,
      explanation: "With random $B$, the product $BA$ would be a random matrix added to $W_0$, corrupting the pretrained representations from step 0. The optimizer would need to spend early training steps undoing this random corruption before making useful task-specific updates. With $B = 0$, training starts from the exact pretrained model and every gradient step moves toward the task — no wasted compute fixing initialization damage. Empirically, $B = 0$ consistently outperforms random initialization."
    },
    {
      type: "info",
      title: "The Alpha/r Scaling Trick",
      content: "The LoRA forward pass includes a scaling factor $\\alpha / r$:\n\n$$h = W_0 x + \\frac{\\alpha}{r} BAx$$\n\nWhy not just $h = W_0 x + BAx$? The scaling serves a practical purpose: **it allows changing rank $r$ without retuning the learning rate**.\n\nConsider what happens when you increase $r$: more rank-1 components contribute to $BA$, so the magnitude of $BAx$ grows. Without scaling, doubling $r$ would roughly double the output magnitude, requiring a proportionally smaller learning rate to maintain training stability.\n\nWith $\\alpha / r$ scaling:\n- Fix $\\alpha$ (e.g., $\\alpha = 16$)\n- Sweep $r \\in \\{4, 8, 16, 32, 64\\}$ without changing the learning rate\n- The scaling automatically compensates for the increased capacity\n\nIn practice, $\\alpha$ is often set equal to the first rank tried (e.g., $\\alpha = r = 16$), giving a scaling factor of 1 for that configuration. This is a convenience, not a fundamental constraint — $\\alpha$ is just another hyperparameter, but one that rarely needs tuning once set.\n\nThe effective learning rate for the LoRA parameters scales as $\\eta_{\\text{eff}} \\propto \\eta \\cdot \\alpha / r$. Increasing $r$ while keeping $\\alpha$ and $\\eta$ fixed reduces the per-component learning rate, maintaining overall update magnitude."
    },
    {
      type: "mc",
      question: "A team trains LoRA with $\\alpha = 16$, $r = 16$, and learning rate $\\eta = 2 \\times 10^{-4}$. They want to try $r = 64$ to see if more capacity helps. Keeping $\\alpha = 16$ and the same $\\eta$, what happens to the effective magnitude of LoRA updates?",
      options: [
        "Updates are 4x larger because 4x more rank-1 components contribute additively to the output",
        "Updates are the same magnitude — the $\\alpha/r$ scaling automatically compensates, reducing the per-component contribution by 4x to offset the 4x more components",
        "Updates are 4x smaller because $\\alpha/r = 16/64 = 0.25$ reduces each component's contribution by 4x, and the number of components doesn't affect total magnitude",
        "Updates are 16x larger because rank scales quadratically with output magnitude through the $BA$ matrix product"
      ],
      correct: 1,
      explanation: "At $r = 16$: scaling is $\\alpha/r = 1$. At $r = 64$: scaling is $\\alpha/r = 0.25$. Each of the 64 rank-1 components contributes 1/4 as much as each of the 16 components did. Since there are 4x more components each contributing 4x less, the total magnitude stays approximately constant. This is the entire point of the $\\alpha/r$ scaling: it makes the overall update magnitude roughly invariant to $r$, so the same learning rate works across rank settings."
    },
    {
      type: "info",
      title: "Merging at Inference: Zero Overhead",
      content: "One of LoRA's key advantages over other PEFT methods (like adapters) is **zero inference overhead**.\n\nDuring training, the forward pass is:\n$$h = W_0 x + \\frac{\\alpha}{r} BAx$$\n\nAfter training, you can **merge** the LoRA weights into the base model:\n$$W' = W_0 + \\frac{\\alpha}{r} BA$$\n\nNow the forward pass is simply $h = W'x$ — a single matrix multiplication, identical in cost to the original model. No additional layers, no branching, no latency increase.\n\nThis is not possible with adapter methods, which insert sequential bottleneck layers (down-project → nonlinearity → up-project). Adapters add inference latency because each token must pass through additional computation that cannot be absorbed into existing weights.\n\nLoRA's merging also enables **multi-task serving**: keep one base model in memory and swap different LoRA weights ($\\Delta W_1, \\Delta W_2, \\ldots$) for different tasks. Switching tasks only requires loading the small LoRA parameters (a few MB), not the full model (many GB). Systems like S-LoRA and Punica exploit this for efficient multi-tenant serving."
    },
    {
      type: "mc",
      question: "A serving system hosts a single 70B base model and needs to support 100 different fine-tuned variants for different customers. With LoRA (rank 16), approximately how much additional memory is needed for all 100 variants?",
      options: [
        "~14 TB — each variant requires a full model copy in BF16 (140 GB), since LoRA weights must be merged before serving and cannot share a base",
        "~14 GB — each variant stores 0.1% of the model parameters at full FP32 precision, giving 140 MB per adapter times 100 variants",
        "~70 GB — each variant stores a copy of the attention projection weights (half the total parameters), since LoRA only adapts attention layers",
        "~700 MB — each rank-16 adapter is ~7 MB (0.005% of model size), so 100 adapters total ~700 MB while all sharing the single 140 GB base"
      ],
      correct: 3,
      explanation: "With rank 16 applied to all linear layers, LoRA parameters are roughly $r(d_{\\text{in}} + d_{\\text{out}})$ per layer summed across all layers. For a 70B model this is typically 0.005-0.01% of total parameters — around 3.5-7M parameters per adapter, or ~7-14 MB in BF16. 100 adapters: ~700 MB-1.4 GB. All variants share the single 140 GB base model. This is the key serving advantage: 100 LoRA variants cost ~1% additional memory vs. 100x for full fine-tuned copies."
    },
    {
      type: "info",
      title: "QLoRA: Fine-Tuning Quantized Models",
      content: "**QLoRA** (Dettmers et al., 2023) combines two ideas to enable fine-tuning on consumer hardware:\n\n1. **Quantize the frozen base model** to 4-bit precision (NF4 format)\n2. **Train LoRA adapters** in BF16 on top of the quantized base\n\nThe NormalFloat4 (NF4) format is specifically designed for neural network weights, which are approximately normally distributed. Its 16 quantization levels are placed at the quantiles of a standard normal distribution, so each level captures equal probability mass. This is information-theoretically optimal for normal weights.\n\nMemory savings are dramatic:\n- **Full fine-tuning** (7B, BF16 + AdamW): ~84 GB\n- **LoRA** (7B, BF16 base + LoRA in BF16): ~16 GB\n- **QLoRA** (7B, NF4 base + LoRA in BF16): ~6 GB\n\nQLoRA also introduces **double quantization**: the per-block scale factors used in NF4 quantization are themselves quantized to FP8, saving an additional ~0.4 bits per parameter.\n\nThe key insight: quantization errors in the frozen base weights act as a fixed perturbation. LoRA can learn to partially compensate for these errors while simultaneously learning the task adaptation. In practice, QLoRA matches full BF16 LoRA performance on most benchmarks."
    },
    {
      type: "mc",
      question: "QLoRA quantizes the base model to NF4 (4-bit) but keeps LoRA parameters in BF16. During the backward pass, gradients must flow through the quantized base weights. How does QLoRA handle this?",
      options: [
        "The base weights are dequantized to BF16 on-the-fly during the backward pass to compute gradients, then discarded — only the NF4 weights are stored persistently, trading compute for memory",
        "Gradients are computed in NF4 as well, using a custom 4-bit backpropagation kernel that maintains the same precision as the forward pass",
        "Gradients are approximated using straight-through estimators that treat quantization as an identity function in the backward pass, ignoring quantization error",
        "QLoRA skips gradient computation through the base weights entirely, only computing gradients for the LoRA parameters via a detached forward pass"
      ],
      correct: 0,
      explanation: "QLoRA dequantizes weights to BF16 in working memory during each forward and backward pass, then discards the BF16 copies. Only the compressed NF4 weights are stored in GPU memory. This trades a small amount of compute (dequantization) for large memory savings. The gradients through the LoRA parameters are exact (computed via the dequantized weights), so there is no gradient approximation — the only approximation is in the forward pass activation values, which use quantized weights."
    }
  ]
};
