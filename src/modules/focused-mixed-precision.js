// Focused learning module: Mixed Precision Training
// Section 1.6: Distributed Training Infrastructure
// Covers: IEEE floating point representation, FP32/FP16/BF16 formats,
// loss scaling, the mixed precision training recipe, and memory savings.
// Single-concept module building from first principles.
// Grounded in Goodfellow et al. (2016) Ch. 8 (optimization) and Micikevicius et al. (2018).

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
      title: "The Memory Wall: Why We Can't Just Use FP32",
      content: "A 70B parameter model stored in 32-bit floating point (FP32) requires $70 \\times 10^9 \\times 4$ bytes $= 280$ GB just for the weights. Add optimizer states (Adam stores two additional copies: first and second moments), and you need $280 \\times 3 = 840$ GB for parameters + optimizer alone — before counting activations, gradients, or data.\n\nThis exceeds the memory of any single GPU (80 GB for an A100, 80-192 GB for H100/H200). Even with data parallelism and ZeRO sharding, memory pressure is the dominant constraint.\n\n**Mixed precision training** reduces memory by storing most quantities in 16-bit formats (2 bytes instead of 4), cutting memory roughly in half. But there's a catch: not all operations are safe to run in reduced precision. The art of mixed precision is knowing which computations need full precision and which can safely use half precision.\n\nThis isn't just about fitting models into memory — 16-bit operations are also **faster**. Modern GPUs have specialized tensor cores that compute 16-bit matrix multiplications at 2-8× the throughput of FP32. Mixed precision training typically provides a **2-3× speedup** alongside the memory savings."
    },
    // Step 2: MC — motivation
    {
      type: "mc",
      question: "A 13B parameter model is trained with Adam in full FP32 precision. What is the total memory for parameters + optimizer states (ignoring activations and gradients)?",
      options: [
        "52 GB — the model weights are 13B × 4 bytes = 52 GB; Adam states use no additional memory because they are computed on-the-fly",
        "104 GB — 52 GB for weights plus 52 GB for Adam's single momentum buffer",
        "156 GB — 52 GB for weights plus 52 GB for Adam's first moment (m) plus 52 GB for Adam's second moment (v)",
        "208 GB — 52 GB each for weights, gradients, first moment, and second moment"
      ],
      correct: 2,
      explanation: "In FP32, each parameter is 4 bytes, so 13B parameters = 52 GB. Adam maintains two optimizer states per parameter: the first moment $m$ (exponential moving average of gradients) and the second moment $v$ (exponential moving average of squared gradients). Each is the same size as the parameters: 52 GB. Total: 52 + 52 + 52 = 156 GB. Gradients are also 52 GB but are typically computed and discarded per-layer, so peak gradient memory is smaller. The key insight: optimizer states alone are 2× the model size."
    },
    // Step 3: Floating point formats
    {
      type: "info",
      title: "Floating Point Formats: FP32, FP16, and BF16",
      content: "A floating point number is stored as: $(-1)^s \\times 2^{e - \\text{bias}} \\times (1 + f)$, where $s$ is the sign bit, $e$ is the exponent, and $f$ is the fraction (mantissa).\n\nThe three formats used in LLM training:\n\n**FP32** (32 bits): 1 sign + 8 exponent + 23 mantissa\n- Range: $\\pm 3.4 \\times 10^{38}$\n- Precision: ~7 decimal digits\n- The \"safe\" format — virtually no risk of overflow or precision loss in neural network training\n\n**FP16** (16 bits): 1 sign + 5 exponent + 10 mantissa\n- Range: $\\pm 65{,}504$\n- Precision: ~3.3 decimal digits\n- Problem: the small range means values above 65,504 **overflow to infinity**. Gradient values can easily exceed this during training.\n\n**BF16** (bfloat16, 16 bits): 1 sign + 8 exponent + 7 mantissa\n- Range: $\\pm 3.4 \\times 10^{38}$ (same as FP32!)\n- Precision: ~2.4 decimal digits\n- Tradeoff: same range as FP32 (no overflow risk) but lower precision. Designed specifically for deep learning by Google Brain.\n\nThe key distinction: **FP16 has a range problem** (overflow), while **BF16 has a precision problem** (rounding). For neural network training, range matters more than precision — a single overflow to infinity propagates through the entire computation and corrupts the model. Small rounding errors average out over many operations."
    },
    // Step 4: MC — formats
    {
      type: "mc",
      question: "During backpropagation, a gradient value of 80,000 is computed. What happens when this value is stored in FP16 vs. BF16?",
      options: [
        "Both formats store it correctly — 80,000 is within the representable range of all 16-bit formats",
        "FP16 overflows to infinity (max representable value is 65,504), corrupting all downstream computations. BF16 stores it with some rounding error but no overflow",
        "FP16 rounds it to 65,504 (the nearest representable value) while BF16 stores it exactly",
        "Both formats underflow to zero because 80,000 requires more mantissa bits than either format provides"
      ],
      correct: 1,
      explanation: "FP16's maximum representable value is 65,504 (limited by its 5-bit exponent). A value of 80,000 exceeds this and overflows to infinity ($+\\infty$). Any computation involving infinity produces infinity or NaN, corrupting the entire training step. BF16 has the same 8-bit exponent as FP32, giving it a range up to $3.4 \\times 10^{38}$. It stores 80,000 with some rounding (only ~2.4 decimal digits of precision) but no overflow. This is why BF16 has largely replaced FP16 for LLM training — overflow is catastrophic, but rounding is tolerable."
    },
    // Step 5: The mixed precision recipe
    {
      type: "info",
      title: "The Mixed Precision Recipe",
      content: "Mixed precision training (Micikevicius et al., 2018) uses a specific protocol:\n\n**1. Master weights in FP32**: Maintain a full-precision copy of all model parameters. This is the \"source of truth\" that accumulates small gradient updates accurately.\n\n**2. Forward and backward pass in FP16/BF16**: Cast the FP32 weights to 16-bit, run the forward pass, compute the loss, and backpropagate — all in reduced precision. This is where the speed and memory gains come from, since matrix multiplications (the dominant cost) run on fast tensor cores.\n\n**3. Update in FP32**: Cast the 16-bit gradients back to FP32 and apply the optimizer update to the FP32 master weights.\n\nWhy keep FP32 master weights? Consider a parameter with value $w = 1.0$ and a gradient update $\\Delta w = 10^{-5}$. In FP32, $1.0 + 10^{-5} = 1.00001$ — the update is captured. In BF16, the precision is only ~2.4 decimal digits, so $1.0 + 10^{-5}$ rounds back to $1.0$. The update is **silently lost**.\n\nOver thousands of steps, these lost updates accumulate. The model appears to train but makes no progress — a subtle failure mode that's harder to diagnose than a loss spike. The FP32 master copy ensures that every small gradient update is faithfully accumulated."
    },
    // Step 6: MC — master weights
    {
      type: "mc",
      question: "A team accidentally stores master weights in BF16 instead of FP32. The learning rate is $3 \\times 10^{-4}$ and typical gradient magnitudes are $\\sim 0.01$. The typical parameter update is $\\sim 3 \\times 10^{-6}$. For a parameter with value $w = 2.0$ in BF16, the smallest representable increment is approximately $2^{-7} \\approx 0.0078$. What happens?",
      options: [
        "Training proceeds normally — BF16's precision is sufficient because Adam's momentum accumulates many small updates into larger ones before applying them",
        "Training appears to proceed (loss decreases initially from large early updates) but eventually stalls, because gradient updates of $3 \\times 10^{-6}$ are below BF16's representable precision at $w = 2.0$ and are silently rounded away",
        "Training immediately diverges because BF16 cannot represent the value 2.0 accurately",
        "Training succeeds but takes exactly 2× longer because each update must be applied twice to overcome the rounding threshold"
      ],
      correct: 1,
      explanation: "BF16 with 7 mantissa bits can represent values near 2.0 with a step size of about $2^{1-7} = 2^{-6} \\approx 0.016$. An update of $3 \\times 10^{-6}$ is ~5,000× smaller than this step size, so it rounds to zero. Early in training, updates are larger (learning rate warmup, initial large gradients) and may be captured. But as training progresses and updates shrink, progress silently stalls. This is a particularly insidious bug because the loss curve looks like normal plateauing rather than an error."
    },
    // Step 7: Loss scaling for FP16
    {
      type: "info",
      title: "Loss Scaling: Rescuing FP16 Gradients",
      content: "When using FP16 (not BF16), there's an additional problem: many gradient values are very small — small enough to **underflow to zero** in FP16. The smallest positive normal FP16 number is $2^{-14} \\approx 6 \\times 10^{-5}$. Gradients below this magnitude become zero, and the model stops learning in those parameters.\n\n**Loss scaling** (Micikevicius et al., 2018) fixes this by multiplying the loss by a large factor $S$ before backpropagation:\n\n$$\\mathcal{L}_{\\text{scaled}} = S \\cdot \\mathcal{L}$$\n\nBy the chain rule, all gradients are also multiplied by $S$, shifting them into FP16's representable range. After backpropagation (but before the optimizer update), we divide the gradients by $S$ to recover the true values:\n\n$$g_{\\text{true}} = g_{\\text{scaled}} / S$$\n\n**Dynamic loss scaling** automates the choice of $S$:\n- Start with a large scale factor (e.g., $S = 2^{16}$)\n- If a forward/backward pass produces NaN or Inf (overflow from over-scaling), halve $S$ and skip this step\n- If $N$ consecutive steps succeed without overflow, double $S$\n- This finds the largest safe scale factor adaptively\n\nWith BF16, loss scaling is typically unnecessary because BF16's range matches FP32 — underflow and overflow are not issues. This is another reason BF16 has become the default for LLM training."
    },
    // Step 8: MC — loss scaling
    {
      type: "mc",
      question: "During FP16 training with dynamic loss scaling, the scale factor is $S = 2^{14} = 16{,}384$. A backward pass produces a gradient of $g_{\\text{scaled}} = 40{,}000$ for some parameter. The true gradient is $g = 40{,}000 / 16{,}384 \\approx 2.44$. What happens next?",
      options: [
        "The scaled gradient of 40,000 underflows in FP16 (max value 65,504), so the scale factor is halved to $2^{13}$",
        "The scaled gradient of 40,000 is within FP16 range (max 65,504), so training proceeds normally: the gradient is unscaled to ~2.44 in FP32 and the optimizer update is applied",
        "The optimizer update is applied using the scaled gradient of 40,000 directly, producing a 16,384× larger update than intended",
        "The scaled gradient is stored as 40,000 in FP16, then cast to FP32 as 40,000.0, then divided by 16,384 to give 2.44, which is then discarded because it exceeds the clipping threshold"
      ],
      correct: 1,
      explanation: "The scaled gradient (40,000) is within FP16's representable range (max 65,504), so no overflow occurs. The procedure is: (1) compute the backward pass in FP16 with scaled loss, producing FP16 gradients, (2) cast these gradients to FP32, (3) divide by $S$ to recover true gradients, (4) apply the optimizer update in FP32. The unscaling step (dividing by $S$) is critical — without it, all updates would be $S$ times too large. Dynamic loss scaling only adjusts $S$ when overflow is detected (NaN/Inf in the gradients)."
    },
    // Step 9: Memory accounting
    {
      type: "info",
      title: "Memory Savings: The Full Picture",
      content: "Let's compare the memory footprint for a model with $N$ parameters:\n\n**Pure FP32 training:**\n- Parameters: $4N$ bytes\n- Gradients: $4N$ bytes\n- Adam states ($m$ and $v$): $8N$ bytes\n- **Total: $16N$ bytes**\n\n**Mixed precision (FP32 master + BF16 compute):**\n- FP32 master weights: $4N$ bytes\n- BF16 working weights: $2N$ bytes\n- BF16 gradients: $2N$ bytes\n- FP32 Adam states ($m$ and $v$): $8N$ bytes\n- **Total: $16N$ bytes** (same!)\n\nWait — where are the savings? The memory for parameters and optimizer states is the same because we still need FP32 master weights and FP32 optimizer states.\n\nThe real savings come from **activations**. During the forward pass, intermediate activations stored for backpropagation are in BF16 (2 bytes) instead of FP32 (4 bytes). For large models with long sequences, activation memory dominates — it can be 10-50× the parameter memory. Halving activation memory is the primary memory benefit.\n\nAdditionally, BF16 matrix multiplications use 2× less memory bandwidth (moving 2 bytes instead of 4 between GPU memory and compute units), which is the main source of the **speed improvement**."
    },
    // Step 10: MC — memory
    {
      type: "mc",
      question: "A 7B parameter model is trained with mixed precision (BF16 compute, FP32 master weights). During the forward pass on a batch of 2048-token sequences, the activation memory is 40 GB in BF16. What would the activation memory be in pure FP32 training?",
      options: [
        "40 GB — activation memory depends only on the model architecture and batch size, not on the numeric precision",
        "20 GB — FP32 uses half the memory of BF16 because FP32 has better compression from higher mantissa precision",
        "80 GB — FP32 activations use 4 bytes per value vs BF16's 2 bytes, doubling the activation memory",
        "160 GB — FP32 activations are 4× larger because both the real and imaginary parts of each activation must be stored in full precision"
      ],
      correct: 2,
      explanation: "Each activation value takes 2 bytes in BF16 and 4 bytes in FP32 — a 2× difference. So 40 GB in BF16 becomes 80 GB in FP32. For a 7B model, parameter + optimizer memory is about 112 GB (16N bytes) in either case. But activation memory going from 80 GB to 40 GB is a significant saving — it could be the difference between fitting the training batch on available GPUs or needing to reduce batch size. This is why activation memory, not parameter memory, is the primary beneficiary of mixed precision."
    },
    // Step 11: BF16 vs FP16 in practice
    {
      type: "info",
      title: "BF16 vs FP16: Why BF16 Won",
      content: "Both FP16 and BF16 are 16-bit formats, but they make different tradeoffs:\n\n**FP16 advantages:**\n- Higher precision (10 mantissa bits vs 7) — less rounding error per operation\n- Supported on all modern GPUs (including older V100s)\n\n**FP16 disadvantages:**\n- Limited range ($\\pm 65{,}504$) — requires loss scaling to avoid overflow/underflow\n- Loss scaling adds complexity and failure modes (scale too high → overflow → skipped steps; scale too low → underflow → lost gradients)\n\n**BF16 advantages:**\n- Same range as FP32 ($\\pm 3.4 \\times 10^{38}$) — no loss scaling needed\n- Simpler training recipe — just cast and compute, no dynamic scale management\n- Training curves nearly identical to FP32 for most tasks\n\n**BF16 disadvantages:**\n- Lower precision (7 mantissa bits) — more rounding error per operation\n- Not supported on pre-Ampere GPUs (V100 and earlier)\n\nFor LLM training, BF16 has become the default because:\n1. Overflow is **catastrophic** (corrupts the entire step), while rounding errors are **benign** (average out over many operations)\n2. The simpler recipe means fewer bugs and less debugging\n3. Empirically, BF16 training matches FP32 quality for language models\n\nFP16 with loss scaling is still used in some settings (older hardware, inference quantization pipelines) but is no longer the standard for pretraining."
    },
    // Step 12: MC — BF16 vs FP16
    {
      type: "mc",
      question: "A research lab has both V100 GPUs (support FP16 but not BF16) and A100 GPUs (support both). They want to train a 30B model. Which setup is most appropriate?",
      options: [
        "Train on V100s with FP16 and dynamic loss scaling — the higher precision of FP16 (10 mantissa bits) produces better model quality than BF16",
        "Train on A100s with BF16 and no loss scaling — the simpler recipe reduces failure modes, and empirically BF16 matches FP32 quality for LLMs despite lower mantissa precision",
        "Train on V100s with FP32 only — at 30B parameters, the precision of 16-bit formats is insufficient for convergence",
        "Train on A100s with FP16 — the A100's tensor cores compensate for FP16's range limitations, eliminating the need for loss scaling"
      ],
      correct: 1,
      explanation: "A100s with BF16 is the standard choice for modern LLM training. BF16's range eliminates the need for loss scaling (simpler, fewer failure modes), and the slight precision reduction doesn't affect LLM training quality in practice. V100s with FP16 would work but requires loss scaling, adding complexity and potential for instability. FP32-only would be unnecessarily slow and memory-intensive. FP16 on A100 would still need loss scaling — the GPU's tensor cores don't change the numeric format's range limitations."
    },
    // Step 13: Emerging lower-precision methods
    {
      type: "info",
      title: "Beyond 16-Bit: FP8 Training",
      content: "The frontier of mixed precision is moving to **8-bit formats**. NVIDIA's H100 and later GPUs support FP8 tensor core operations in two formats:\n\n**E4M3** (4-bit exponent, 3-bit mantissa): Range $\\pm 448$, ~1.9 decimal digits of precision. Used for **forward pass** computations where the range of activations is moderate.\n\n**E5M2** (5-bit exponent, 2-bit mantissa): Range $\\pm 57{,}344$, ~1.2 decimal digits of precision. Used for **backward pass** gradients where the wider range is needed to capture gradient magnitudes.\n\nFP8 training extends the mixed precision recipe: FP8 for the fast matrix multiplications, FP16/BF16 for accumulation (summing the products), and FP32 for master weights and optimizer states. This provides another 2× speedup on supported hardware.\n\nThe challenge with FP8 is that the representable range is much smaller, requiring more careful **per-tensor scaling** — each tensor gets its own scale factor to map its values into the FP8 range. This is more complex than BF16's \"just cast it\" approach, but the speed gains (up to 2× over BF16) make it worthwhile for large-scale training.\n\nThe trend is clear: each hardware generation reduces the precision of the fast compute path while keeping the accumulation and optimizer updates in higher precision. The master weight / reduced compute split from mixed precision training is the template for all these advances."
    },
    // Step 14: MC — FP8
    {
      type: "mc",
      question: "FP8 E4M3 format has a maximum representable value of 448. During a forward pass, a tensor has values ranging from -2,000 to +2,000. How is this handled in FP8 training?",
      options: [
        "Values outside $[-448, 448]$ are clamped to the nearest representable value, introducing a hard ceiling on activations that the model must learn to stay within",
        "The tensor is multiplied by a per-tensor scale factor (e.g., $448/2000 = 0.224$) before casting to FP8, then the inverse scale is applied after the FP8 computation to recover the correct magnitude",
        "FP8 training automatically switches to BF16 for any tensor whose values exceed the FP8 range, falling back to the safer format on a per-tensor basis",
        "The model architecture is constrained so that no intermediate activation can exceed 448 — this is enforced by using sigmoid or tanh activations throughout"
      ],
      correct: 1,
      explanation: "Per-tensor scaling is the standard approach for FP8 training. Before casting a tensor to FP8, the maximum absolute value is measured and a scale factor is computed to map the values into the FP8 representable range. After the FP8 computation, the inverse scale is applied. This is analogous to loss scaling in FP16 training but applied per-tensor rather than globally. The overhead of computing scale factors is small compared to the 2× speedup from FP8 matrix multiplications."
    }
  ]
};
