// Focused learning module: Mixed Precision Training
// Section 1.6: Distributed Training Infrastructure
// Covers: IEEE floating point formats, FP32/FP16/BF16 tradeoffs,
// loss scaling for FP16, the mixed precision training recipe,
// and practical considerations for LLM training.
// Single-concept module building from first principles.
// Grounded in Goodfellow et al. (2016) Ch. 4 (Numerical Computation)
// and Micikevicius et al. (2018) "Mixed Precision Training".

export const mixedPrecisionLearning = {
  id: "1.6-mixed-precision-learning-easy",
  sectionId: "1.6",
  title: "Mixed Precision Training",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Why precision matters
    {
      type: "info",
      title: "The Precision-Efficiency Tradeoff",
      content: "Every number in a neural network — every parameter, gradient, activation, and optimizer state — is stored as a **floating-point number** that occupies a fixed number of bits in GPU memory.\n\nStandard training uses **FP32** (32-bit single precision): 4 bytes per number. For a 70B parameter model, parameters alone consume $70 \\times 10^9 \\times 4 = 280$ GB. With gradients and Adam optimizer states, the total exceeds **1 TB**.\n\nIf we could use **FP16** (16-bit half precision) instead: 2 bytes per number. Parameters drop to 140 GB. Compute is also faster — modern GPUs (A100, H100) have specialized **tensor cores** that perform FP16 matrix multiplications at 2-4$\\times$ the throughput of FP32.\n\n**Mixed precision training** combines both: use lower precision where possible for speed and memory savings, but keep critical computations in higher precision for numerical stability. The key challenge is figuring out which operations need high precision and which don't."
    },
    // Step 2: MC
    {
      type: "mc",
      question: "A 7B parameter model trained in pure FP32 uses 28 GB for parameters. Switching all parameters and gradients to FP16 would save how much memory on parameters and gradients combined?",
      options: [
        "14 GB — parameters drop from 28 GB to 14 GB, but gradients must remain in FP32 for accurate accumulation",
        "28 GB — both parameters (28 → 14 GB) and gradients (28 → 14 GB) are halved, saving 28 GB total",
        "56 GB — switching to FP16 provides a 4$\\times$ compression because FP16 also eliminates padding bytes used by FP32 alignment",
        "0 GB — the GPU hardware always operates on 32-bit registers regardless of the declared precision, so the memory layout doesn't change"
      ],
      correct: 1,
      explanation: "FP32 uses 4 bytes per number, FP16 uses 2 bytes. Parameters: $7B \\times 4 = 28$ GB → $7B \\times 2 = 14$ GB (save 14 GB). Gradients: same calculation, another 14 GB saved. Total savings: 28 GB. The reduction is exactly 2$\\times$ because we're going from 4 bytes to 2 bytes per value. However, as we'll see, mixed precision training typically keeps a FP32 master copy of parameters for the optimizer, so the real savings are more nuanced."
    },
    // Step 3: Floating point anatomy
    {
      type: "info",
      title: "Floating Point Formats: Exponent vs Mantissa",
      content: "A floating-point number is stored as: $(-1)^{\\text{sign}} \\times 2^{\\text{exponent}} \\times (1 + \\text{mantissa})$\n\nThe bits are split between three fields:\n\n| Format | Total bits | Sign | Exponent | Mantissa | Dynamic range | Precision |\n|---|---|---|---|---|---|---|\n| FP32 | 32 | 1 | 8 | 23 | $\\sim 10^{\\pm 38}$ | ~7 decimal digits |\n| FP16 | 16 | 1 | 5 | 10 | $\\sim 10^{\\pm 5}$ | ~3 decimal digits |\n| BF16 | 16 | 1 | 8 | 7 | $\\sim 10^{\\pm 38}$ | ~2 decimal digits |\n\nThe **exponent** determines the **dynamic range** — how large or small the number can be. The **mantissa** determines the **precision** — how many significant digits are accurate.\n\n**FP16** has only 5 exponent bits, giving a maximum value of ~65,504. Gradient values or loss values larger than this cause **overflow** ($\\to \\infty$). Gradient values smaller than $\\sim 6 \\times 10^{-8}$ **underflow** ($\\to 0$).\n\n**BF16** (Brain Float 16, developed by Google Brain) keeps FP32's 8-bit exponent for full dynamic range, but sacrifices mantissa bits. It can represent values up to $\\sim 3.4 \\times 10^{38}$ — the same as FP32 — at the cost of lower precision."
    },
    // Step 4: MC
    {
      type: "mc",
      question: "During backpropagation, a gradient value of $2.7 \\times 10^{-9}$ is computed. What happens when this gradient is stored in FP16 vs BF16?",
      options: [
        "FP16 underflows this to exactly zero because $2.7 \\times 10^{-9}$ is below FP16's minimum subnormal ($\\sim 6 \\times 10^{-8}$), while BF16 preserves it because BF16's 8-bit exponent supports the same range as FP32",
        "FP16 rounds it to a nearby representable value with 10 bits of mantissa precision, while BF16 rounds it with only 7 bits — both preserve the value but BF16 is less precise",
        "Both FP16 and BF16 represent it accurately — the difference between formats only matters for very large values, not small ones",
        "Both formats underflow to zero because 16-bit formats cannot represent values below $10^{-5}$ regardless of exponent size"
      ],
      correct: 0,
      explanation: "FP16's 5-bit exponent gives a minimum positive subnormal value around $6 \\times 10^{-8}$. The gradient $2.7 \\times 10^{-9}$ is smaller than this, so FP16 rounds it to zero — the gradient is **lost entirely**. BF16's 8-bit exponent supports the same range as FP32 (down to $\\sim 10^{-38}$), so it can represent this small gradient, though with only ~2 decimal digits of precision. This is why FP16 training requires loss scaling (to keep gradients in representable range) while BF16 typically does not."
    },
    // Step 5: The mixed precision recipe
    {
      type: "info",
      title: "The Mixed Precision Training Recipe",
      content: "Micikevicius et al. (2018) established the standard mixed precision training algorithm:\n\n**1. Maintain FP32 master weights.** The optimizer (Adam) stores a full-precision FP32 copy of all parameters. This is the \"source of truth.\"\n\n**2. Forward and backward pass in FP16/BF16.** Before each forward pass, cast the FP32 master weights to FP16/BF16. Run the forward pass and backward pass entirely in reduced precision. Activations and gradients are stored in FP16/BF16.\n\n**3. Gradient accumulation in FP32.** After the backward pass, cast gradients back to FP32 before the optimizer step. The optimizer updates the FP32 master weights using FP32 gradients.\n\nWhy keep FP32 master weights? Consider Adam's update: $\\theta \\leftarrow \\theta - \\alpha \\cdot m / (\\sqrt{v} + \\epsilon)$. If $\\theta \\approx 1.0$ and the update is $\\sim 10^{-5}$, adding them in FP16 ($\\sim$3 decimal digits) gives $1.0 + 0.00001 = 1.0$ — the update is **rounded away**. In FP32 ($\\sim$7 digits), the update is preserved. Small but consistent updates are critical for training convergence.\n\nThe memory cost: we now store parameters three times (FP32 master + FP16 forward copy + FP16 gradients), but the optimizer states were already FP32, so the overhead vs. pure FP32 is modest."
    },
    // Step 6: MC
    {
      type: "mc",
      question: "In mixed precision training, the FP32 master weights are essential because:",
      options: [
        "FP16 cannot represent the weight values themselves — model weights routinely exceed FP16's maximum of 65,504",
        "The GPU hardware requires FP32 inputs for matrix multiplication and silently converts any FP16 weights to FP32 before computation",
        "Small optimizer updates ($\\sim 10^{-5}$) get rounded to zero when added to FP16 weights ($\\sim 3$ digit precision), so accumulated updates would never change the weights",
        "FP32 master weights are only needed for the first few thousand training steps until the weights stabilize, after which training can switch to pure FP16"
      ],
      correct: 2,
      explanation: "The key issue is the **weight update magnitude ratio**. Model weights are typically $O(1)$ in magnitude, while per-step updates are $O(10^{-4})$ to $O(10^{-6})$. FP16 has ~3 decimal digits of precision, so $1.0 + 0.00001 \\approx 1.0$ in FP16 — the update vanishes. Over thousands of steps, this means the model stops learning. FP32's ~7 digits of precision preserves these small updates. This problem persists throughout training, not just early on."
    },
    // Step 7: Loss scaling for FP16
    {
      type: "info",
      title: "Loss Scaling: Rescuing FP16 Gradients",
      content: "FP16's limited dynamic range creates a specific problem: many gradient values during backpropagation fall below FP16's minimum representable value ($\\sim 6 \\times 10^{-8}$) and underflow to zero.\n\nMicikevicius et al. (2018) found that in trained networks, a large fraction of gradient values are small but nonzero — they cluster in the range $[10^{-8}, 10^{-5}]$. In FP16, these all become zero, losing critical training signal.\n\n**Loss scaling** fixes this by multiplying the loss by a large constant $S$ before backpropagation:\n\n$$\\tilde{\\mathcal{L}} = S \\cdot \\mathcal{L}$$\n\nBy the chain rule, all gradients are also scaled by $S$: $\\nabla \\tilde{\\mathcal{L}} = S \\cdot \\nabla \\mathcal{L}$. This shifts the gradient distribution into FP16's representable range. After the backward pass, gradients are **unscaled** by dividing by $S$ (in FP32) before the optimizer step.\n\n**Dynamic loss scaling** (the standard approach) automatically adjusts $S$:\n1. Start with a large $S$ (e.g., $2^{15}$)\n2. If gradients contain `inf` or `NaN` (overflow from too-large $S$): skip the optimizer step and halve $S$\n3. If $N$ consecutive steps succeed without overflow: double $S$\n\nThis finds the largest safe scale factor automatically, maximizing the number of gradient values preserved in FP16."
    },
    // Step 8: MC
    {
      type: "mc",
      question: "During FP16 training with dynamic loss scaling, the scale factor $S = 2^{15}$ causes gradient overflow (`inf` values appear). What should the training loop do?",
      options: [
        "Terminate training — overflow indicates a fundamental numerical instability that cannot be fixed by adjusting the scale factor",
        "Skip this optimizer step (discard the corrupted gradients), reduce $S$ to $2^{14}$, and continue training with the smaller scale factor",
        "Clip the overflowed gradients to FP16's maximum value (65,504) and proceed with the optimizer step using the clipped gradients",
        "Switch the entire training run to FP32 for the remainder, since FP16 has proven inadequate for this model's gradient distribution"
      ],
      correct: 1,
      explanation: "Dynamic loss scaling handles overflow gracefully: the current step's gradients are corrupted, so we skip the optimizer update (don't apply those gradients) and reduce $S$ — typically by halving it. Training continues normally from the next step. This is expected behavior, not an error. The algorithm continuously adapts: if the new $S$ still causes overflow, it halves again; if many steps succeed, it doubles $S$ to recover more small gradients. In practice, after initial tuning, $S$ stabilizes and overflows become rare."
    },
    // Step 9: BF16 — the modern default
    {
      type: "info",
      title: "BF16: Why Loss Scaling Became Optional",
      content: "**BF16** (bfloat16) was designed specifically for deep learning by Google Brain. By keeping FP32's 8-bit exponent, it avoids the dynamic range problem that makes FP16 training fragile.\n\nWith BF16:\n- **No gradient underflow**: The range extends to $\\sim 10^{-38}$, same as FP32. Gradients of $10^{-9}$ are representable (though imprecise).\n- **No loss scaling needed**: Without underflow, the complex loss scaling machinery is unnecessary.\n- **No gradient overflow**: The maximum is $\\sim 3.4 \\times 10^{38}$, same as FP32. Loss values won't overflow.\n\nThe tradeoff: BF16 has only 7 mantissa bits vs. FP16's 10, giving **less precision** ($\\sim$2 decimal digits vs. $\\sim$3). In practice, this matters less than it might seem — the stochastic nature of SGD means that small precision errors in individual gradient values are averaged out across mini-batches.\n\nMost modern LLM training uses BF16 as the default reduced precision format:\n- LLaMA, Gemma, Mistral: all BF16\n- H100 GPUs: native BF16 tensor core support at 2$\\times$ FP32 throughput\n- A100 GPUs: same BF16 throughput as FP16\n\nFP16 with loss scaling is still used when hardware lacks BF16 support (older V100 GPUs) or when the extra mantissa precision is needed for sensitive operations."
    },
    // Step 10: MC
    {
      type: "mc",
      question: "A team is choosing between FP16 and BF16 for training a 13B model on H100 GPUs. Both formats use 16 bits and offer the same compute throughput on this hardware. What is the strongest argument for choosing BF16?",
      options: [
        "BF16 produces more accurate gradient values due to its higher mantissa precision, leading to faster convergence and better final model quality",
        "BF16 uses less memory than FP16 because its smaller mantissa allows more efficient GPU memory alignment and packing",
        "BF16 is backward-compatible with FP32 checkpoints, allowing seamless checkpoint loading without format conversion, while FP16 requires explicit casting",
        "BF16's 8-bit exponent eliminates the need for loss scaling entirely — no gradient underflow, no overflow, and no skipped optimizer steps — simplifying the training pipeline with no practical quality loss"
      ],
      correct: 3,
      explanation: "BF16's decisive advantage is its FP32-matching dynamic range. FP16's 5-bit exponent creates a narrow representable range that requires loss scaling to work around — adding complexity (dynamic scale factor tuning, occasional skipped steps) and potential failure modes. BF16 eliminates all of this. Option A is backwards: FP16 actually has MORE mantissa precision (10 bits vs 7). Option B is false: both are 16 bits and use the same memory. In practice, the tiny precision difference between BF16 and FP16 has negligible effect on model quality, while the loss scaling complexity is a real operational burden."
    },
    // Step 11: Memory accounting
    {
      type: "info",
      title: "Memory Accounting: What Goes Where",
      content: "Mixed precision training stores numbers in different formats at different stages. For a model with $\\Psi$ parameters:\n\n| Component | Format | Size |\n|---|---|---|\n| Master weights | FP32 | $4\\Psi$ bytes |\n| Forward-pass weights | BF16/FP16 | $2\\Psi$ bytes |\n| Gradients | BF16/FP16 | $2\\Psi$ bytes |\n| Adam first moment $m$ | FP32 | $4\\Psi$ bytes |\n| Adam second moment $v$ | FP32 | $4\\Psi$ bytes |\n| **Total model state** | | $\\mathbf{16\\Psi}$ **bytes** |\n\nCompare with pure FP32 training: parameters ($4\\Psi$) + gradients ($4\\Psi$) + Adam $m$ ($4\\Psi$) + Adam $v$ ($4\\Psi$) = $16\\Psi$ bytes.\n\nSurprisingly, the total model state is the **same**! Mixed precision doesn't save memory on the model state because the optimizer dominates and it stays in FP32. The savings come from:\n\n1. **Activations**: Stored in BF16, cutting activation memory in half. For long sequences, this is enormous — activations can exceed model state memory.\n2. **Communication**: Gradient all-reduce in distributed training moves half the data in BF16.\n3. **Compute**: BF16 matrix multiplications run at 2$\\times$ the throughput of FP32 on tensor cores."
    },
    // Step 12: MC
    {
      type: "mc",
      question: "A 70B model is trained with mixed precision (BF16 forward/backward, FP32 optimizer). The total model state memory is $16 \\times 70B = 1.12$ TB. A colleague claims mixed precision saves 50% memory compared to pure FP32. Is this correct?",
      options: [
        "Yes — switching from FP32 to BF16 for parameters and gradients halves those components, saving $4 \\times 70B = 280$ GB from the $1.12$ TB total",
        "No — the model state is $16\\Psi$ bytes in both cases because the FP32 optimizer states dominate; the real savings come from halved activation memory and faster BF16 compute, not model state reduction",
        "Yes — the total drops to $8\\Psi$ bytes because the optimizer states can also use BF16, cutting all components in half",
        "No — mixed precision actually uses MORE total memory because it stores three copies of the weights (FP32 master + BF16 forward + BF16 gradient) instead of two (FP32 weights + FP32 gradient)"
      ],
      correct: 3,
      explanation: "The model state totals $16\\Psi$ in both cases. Pure FP32: $4\\Psi$ (params) + $4\\Psi$ (grad) + $4\\Psi$ (m) + $4\\Psi$ (v) = $16\\Psi$. Mixed precision: $4\\Psi$ (FP32 master) + $2\\Psi$ (BF16 params) + $2\\Psi$ (BF16 grad) + $4\\Psi$ (m) + $4\\Psi$ (v) = $16\\Psi$. The optimizer states ($12\\Psi$ bytes) remain FP32 in both cases and dominate the total. Mixed precision's real benefits are halved activation memory (which can be enormous for long sequences), 2$\\times$ compute throughput, and halved gradient communication in distributed training."
    },
    // Step 13: Practical considerations
    {
      type: "info",
      title: "Practical Considerations for LLM Training",
      content: "Several operations in transformers are sensitive to reduced precision and typically use FP32 even in mixed precision training:\n\n**Softmax**: The attention softmax $\\alpha_{ij} = \\exp(s_{ij}) / \\sum_k \\exp(s_{ik})$ involves exponentiation of potentially large values. In FP16, $\\exp(12) \\approx 162{,}755$ already approaches the maximum ($65{,}504$). Many implementations compute softmax in FP32 and cast the result back.\n\n**Layer normalization**: Computing variance $\\sigma^2 = \\frac{1}{d}\\sum(x_i - \\mu)^2$ involves squaring deviations that can have very different magnitudes. Low precision causes large relative errors in the variance estimate, especially for features with small variance. Typically computed in FP32.\n\n**Loss computation**: Cross-entropy loss involves $\\log(\\text{softmax})$. Accumulating the loss across thousands of tokens in FP16 causes significant rounding errors.\n\n**Gradient accumulation**: When using gradient accumulation (summing gradients across micro-batches before an optimizer step), the accumulated gradients should be in FP32 to avoid rounding errors from repeatedly adding small values.\n\nModern frameworks (PyTorch AMP, DeepSpeed, Megatron-LM) handle these automatically — the programmer specifies the default precision and the framework maintains a list of operations that require FP32."
    },
    // Step 14: MC
    {
      type: "mc",
      question: "A team trains a 7B model in BF16 and notices that validation loss plateaus higher than expected. They suspect numerical issues. Which operation is most likely to benefit from being cast to FP32?",
      options: [
        "Layer normalization — the variance computation $\\frac{1}{d}\\sum(x_i - \\mu)^2$ involves squaring small deviations, where BF16's 2-digit precision can introduce large relative errors that propagate through the network",
        "Embedding table lookups — the discrete index-to-vector mapping loses precision in BF16, corrupting input representations before any computation begins",
        "Linear layers ($XW + b$) — matrix multiplications are the most numerically sensitive operations in transformers and should always run in FP32",
        "The attention mask application — adding $-\\infty$ to masked positions requires FP32 because BF16 cannot represent negative infinity"
      ],
      correct: 0,
      explanation: "Layer normalization is one of the most precision-sensitive operations. Computing the variance involves squaring differences that may be small ($x_i - \\mu \\approx 0.001$), producing values like $10^{-6}$. In BF16 with ~2 decimal digits, these get heavily rounded, and the accumulated variance estimate can be substantially wrong. This causes incorrect normalization, which propagates through every subsequent layer. Linear layers handle BF16 well because tensor core matrix multiplications accumulate partial sums in FP32 internally. Embedding lookups are just table reads — no arithmetic involved."
    }
  ]
};
