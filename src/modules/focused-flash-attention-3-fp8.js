// Focused learning module: FlashAttention-3 FP8 Attention — how FA3 uses
// low-precision (FP8) computation with incoherent processing to maintain
// accuracy while doubling throughput.

export const flashAttention3FP8Learning = {
  id: "G.3-fa3-fp8-learning-hard",
  sectionId: "G.3",
  title: "FlashAttention-3: FP8 Attention with Incoherent Processing",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — Why FP8 for attention
    {
      type: "info",
      title: "The Promise and Problem of FP8 Attention",
      content: "H100 GPUs have dedicated **FP8 tensor cores** that deliver nearly **2× the FLOPS** of FP16 tensor cores: ~1979 TFLOPS (FP8) vs ~989 TFLOPS (FP16). Using FP8 for attention could double throughput.\n\nBut attention has a **precision problem**. The softmax operation amplifies quantization errors:\n\n1. Attention scores $\\mathbf{S} = \\mathbf{QK}^T / \\sqrt{d}$ can span a wide dynamic range\n2. After softmax, small differences in scores become large differences in attention weights\n3. FP8 has very few mantissa bits:\n   - **E4M3** (4-bit exponent, 3-bit mantissa): range $\\pm 448$, precision ~$2^{-3}$ relative\n   - **E5M2** (5-bit exponent, 2-bit mantissa): range $\\pm 57344$, precision ~$2^{-2}$ relative\n\nCompare to FP16's 10-bit mantissa ($2^{-10}$ relative precision) — FP8 E4M3 is 128× less precise.\n\nNaively quantizing Q, K, V to FP8 and running standard FlashAttention produces **significant accuracy degradation**, especially for tasks requiring precise attention patterns (e.g., retrieval, copying). The challenge: get FP8's throughput while maintaining FP16-level accuracy."
    },
    // Step 2: MC — FP8 format tradeoffs
    {
      type: "mc",
      question: "FP8 E4M3 has a 3-bit mantissa while FP8 E5M2 has a 2-bit mantissa. For the $\\mathbf{QK}^T$ matmul in attention, which FP8 format is preferred and why?",
      options: [
        "E4M3 — the extra mantissa bit provides better precision for the accumulated dot products, and the scores are scaled by $1/\\sqrt{d}$ to prevent overflow",
        "E5M2 — the wider exponent range prevents overflow in the dot products, which is the dominant error source",
        "Neither — both formats produce identical results because the matmul accumulates in FP32",
        "E4M3 for Q and E5M2 for K — asymmetric formats balance precision and range across the operands"
      ],
      correct: 0,
      explanation: "The $\\mathbf{QK}^T$ matmul accumulates many products, where precision matters more than range (the $1/\\sqrt{d}$ scaling keeps values manageable). E4M3's extra mantissa bit reduces the per-element quantization error, leading to a more accurate accumulated result. The accumulation itself typically happens in FP32 inside the tensor cores, so the output precision isn't limited to FP8 — it's the input quantization that matters."
    },
    // Step 3: Info — Block quantization
    {
      type: "info",
      title: "Block-wise Quantization for FP8",
      content: "Global quantization (one scale factor per entire tensor) wastes FP8's limited dynamic range. If one element is 100× larger than the others, the scale factor must accommodate the outlier, leaving most values using only a fraction of the representable range.\n\n**Block-wise quantization** uses a separate scale factor per tile:\n\n$$\\mathbf{Q}_i^{\\text{FP8}} = \\text{cast\\_to\\_fp8}\\left(\\frac{\\mathbf{Q}_i}{s_{Q_i}}\\right), \\quad s_{Q_i} = \\frac{\\max|\\mathbf{Q}_i|}{448}$$\n\nwhere 448 is the maximum representable value in E4M3. Each block $\\mathbf{Q}_i$ (e.g., 128 rows × 64 columns) gets its own scale factor.\n\nThis is natural for FlashAttention's tiling: each tile is loaded independently, so the per-tile scale factor is applied during the load from HBM to shared memory. The matmul result is then:\n\n$$\\mathbf{S}_{ij} = s_{Q_i} \\cdot s_{K_j} \\cdot (\\mathbf{Q}_i^{\\text{FP8}} \\cdot \\mathbf{K}_j^{\\text{FP8}^T})$$\n\nThe FP8 matmul runs at 2× the FLOPS of FP16, and the scale factors are applied as a cheap post-processing step.\n\nBlock quantization helps but doesn't fully solve the accuracy problem. The quantization error in each element is still much larger than FP16, and these errors accumulate in the dot products."
    },
    // Step 4: MC — Block quantization
    {
      type: "mc",
      question: "Block-wise quantization assigns one scale factor per tile rather than one per tensor. In FlashAttention's tiling framework, when is the scale factor applied?",
      options: [
        "Before the data is written to HBM during a preprocessing pass, requiring an extra kernel launch",
        "During the TMA copy from HBM to shared memory, using Hopper's built-in format conversion",
        "Inside the softmax function, where it adjusts the temperature to compensate for the scale factor",
        "After the FP8 matmul, by multiplying the FP32 accumulator by $s_{Q_i} \\cdot s_{K_j}$ to recover the correct magnitude"
      ],
      correct: 3,
      explanation: "The FP8 matmul produces a result in a higher-precision accumulator (FP32). The per-block scale factors $s_{Q_i}$ and $s_{K_j}$ are scalar multipliers applied to this FP32 result: $\\mathbf{S}_{ij} = s_{Q_i} \\cdot s_{K_j} \\cdot \\text{fp8\\_matmul}(\\mathbf{Q}_i, \\mathbf{K}_j)$. This is a cheap operation on the already-computed tile. The quantization (casting to FP8 with scaling) can happen when loading data."
    },
    // Step 5: Info — The incoherent processing technique
    {
      type: "info",
      title: "Incoherent Processing: Randomized Rounding on Steroids",
      content: "The core accuracy innovation in FA3's FP8 path is **incoherent processing**, borrowed from quantization theory.\n\n**The problem with coherent quantization**: When Q and K vectors have similar structure (which they do — they're produced by learned projections of the same representation), their quantization errors are **correlated**. Correlated errors in a dot product don't cancel — they accumulate systematically, producing a biased result.\n\n**Incoherent processing** applies a random orthogonal transformation to decorrelate the entries before quantization:\n\n$$\\tilde{\\mathbf{Q}} = \\mathbf{Q} \\cdot \\mathbf{R}, \\quad \\tilde{\\mathbf{K}} = \\mathbf{K} \\cdot \\mathbf{R}$$\n\nwhere $\\mathbf{R}$ is a random rotation matrix. Since $\\mathbf{R}^T\\mathbf{R} = \\mathbf{I}$:\n\n$$\\tilde{\\mathbf{Q}}\\tilde{\\mathbf{K}}^T = \\mathbf{Q}\\mathbf{R}\\mathbf{R}^T\\mathbf{K}^T = \\mathbf{Q}\\mathbf{K}^T$$\n\nThe dot products are **identical** — the rotation cancels out mathematically. But the quantization error behavior changes: the random rotation \"spreads\" each entry's information across all dimensions, making entries more uniform in magnitude. This means:\n1. The scale factor utilizes FP8's range more efficiently (fewer wasted bits)\n2. Quantization errors become **incoherent** (uncorrelated), so they cancel in the dot product like random noise\n\nThe error drops from $O(\\epsilon)$ to $O(\\epsilon / \\sqrt{d})$ where $\\epsilon$ is the per-element quantization error."
    },
    // Step 6: MC — Why incoherent processing helps
    {
      type: "mc",
      question: "Incoherent processing applies a random rotation $\\mathbf{R}$ to Q and K before FP8 quantization. Why does this reduce the dot product error despite not changing the mathematical result?",
      options: [
        "The rotation compresses the vectors, reducing their norm and making them fit better in FP8's limited range",
        "The rotation makes entries more uniform in magnitude, so the per-element quantization errors become uncorrelated and cancel in the dot product sum rather than accumulating coherently",
        "The rotation aligns Q and K vectors with the FP8 grid points, eliminating rounding error entirely",
        "The rotation reduces the effective dimension of the dot product, requiring fewer FP8 multiplications"
      ],
      correct: 1,
      explanation: "Without rotation, Q and K entries may have outlier dimensions that dominate the dot product. Their quantization errors are correlated (same dimensions are poorly quantized in both Q and K), causing systematic error. The random rotation spreads information uniformly across dimensions, making each entry roughly the same magnitude. The quantization errors become independent across dimensions, and by the law of large numbers, they cancel in the $d$-dimensional sum: error scales as $O(\\epsilon/\\sqrt{d})$ instead of $O(\\epsilon)$."
    },
    // Step 7: Info — Practical implementation of incoherent processing
    {
      type: "info",
      title: "Making Incoherent Processing Cheap",
      content: "A random orthogonal matrix $\\mathbf{R} \\in \\mathbb{R}^{d \\times d}$ has a major problem: multiplying by it costs $O(d^2)$ per vector, which is as expensive as the attention matmul itself.\n\nFA3 uses a **Hadamard-based** random rotation instead:\n\n$$\\mathbf{R} = \\frac{1}{\\sqrt{d}} \\mathbf{H}_d \\cdot \\text{diag}(\\mathbf{r})$$\n\nwhere $\\mathbf{H}_d$ is the $d \\times d$ **Hadamard matrix** (a structured orthogonal matrix) and $\\mathbf{r} \\in \\{-1, +1\\}^d$ is a random sign vector.\n\nThe Hadamard transform can be computed in $O(d \\log d)$ via the **Fast Walsh-Hadamard Transform** (FWHT) — similar to how FFT computes the Fourier transform efficiently. For $d = 128$, this is $128 \\times 7 = 896$ operations instead of $128^2 = 16{,}384$ for a general rotation.\n\nThe random signs provide the randomness needed for incoherence. The overall cost of incoherent processing is:\n- One FWHT per Q and K vector: $O(Nd \\log d)$\n- Compare to attention itself: $O(N^2 d)$\n\nSince $N \\gg \\log d$, the overhead is negligible. In practice, the FWHT is applied as a preprocessing step before the FlashAttention kernel."
    },
    // Step 8: MC — Hadamard transform cost
    {
      type: "mc",
      question: "For attention with sequence length $N = 4096$ and head dimension $d = 128$, the Hadamard-based incoherent processing costs $O(Nd \\log d)$ FLOPs. How does this compare to the attention computation's $O(N^2 d)$ FLOPs?",
      options: [
        "The incoherent processing dominates — $Nd \\log d > N^2 d$ for typical values",
        "They are comparable — both are roughly $10^8$ FLOPs for these dimensions",
        "The incoherent processing is negligible — $N \\log d / (N^2) = \\log(128) / 4096 \\approx 0.17\\%$ of attention FLOPs",
        "The comparison depends on whether FP8 or FP16 is used for the Hadamard transform"
      ],
      correct: 2,
      explanation: "Attention FLOPs: $N^2 d = 4096^2 \\times 128 \\approx 2.15 \\times 10^9$. Hadamard FLOPs: $2 \\times N \\times d \\log_2 d = 2 \\times 4096 \\times 128 \\times 7 \\approx 7.3 \\times 10^6$. The ratio is $\\approx 0.34\\%$ — the incoherent processing overhead is negligible. The factor of 2 accounts for applying the transform to both Q and K. As $N$ grows, the ratio shrinks further since attention scales as $N^2$."
    },
    // Step 9: Info — FP8 accuracy results
    {
      type: "info",
      title: "FP8 Attention Accuracy: With and Without Incoherent Processing",
      content: "The accuracy impact of FP8 attention varies significantly depending on whether incoherent processing is used:\n\n**Without incoherent processing** (naive FP8 + block quantization):\n- Attention output error (RMSE vs FP16): $\\sim 10^{-2}$\n- Noticeable quality degradation on downstream tasks\n- Some attention heads produce significantly wrong outputs due to correlated quantization errors\n- Fails especially on tasks requiring precise attention (copying, retrieval)\n\n**With incoherent processing**:\n- Attention output error (RMSE vs FP16): $\\sim 10^{-3}$ to $10^{-4}$\n- Negligible quality difference on most benchmarks\n- Error is uniform across heads (no pathological failures)\n- Within 0.1 perplexity points of FP16 on language modeling tasks\n\nThe improvement is roughly **10–100×** in numerical error, which is the difference between \"visibly broken\" and \"indistinguishable from FP16\" in practice.\n\nCombined with FP8 tensor cores running at 2× the FLOPS of FP16, FA3's FP8 path delivers **close to 1.2 PFLOPS** of effective attention throughput on H100 — approaching 60% of FP8 theoretical peak and nearly 2× the FP16 FA3 throughput."
    },
    // Step 10: MC — Incoherent processing impact
    {
      type: "mc",
      question: "Naive FP8 attention (without incoherent processing) has RMSE ~$10^{-2}$ vs FP16, while with incoherent processing it drops to ~$10^{-3}$–$10^{-4}$. Which downstream effect would you most expect from the naive approach?",
      options: [
        "No observable effect because softmax normalizes away any errors in the attention scores",
        "Complete training divergence within the first 100 steps due to gradient explosion from quantization errors",
        "Slightly higher perplexity but identical generation quality, since language models are robust to small perturbations",
        "Degraded performance specifically on retrieval-heavy tasks (e.g., finding a needle in a haystack) where precise attention patterns matter, while general language modeling is less affected"
      ],
      correct: 3,
      explanation: "Error of $10^{-2}$ in attention output means attention weights are wrong by ~1%. For general language modeling (where attention is distributed), this may be tolerable. But retrieval tasks require placing nearly all attention weight on one specific token — a 1% error can redirect attention to the wrong token entirely. Incoherent processing's $10^{-3}$–$10^{-4}$ error is small enough that even precise attention patterns are preserved. Softmax does not fix input errors; it can amplify them."
    },
    // Step 11: Info — When to use FP8 attention
    {
      type: "info",
      title: "Practical Guidance: When FP8 Attention Helps",
      content: "FP8 attention with incoherent processing is not always the right choice:\n\n**Use FP8 attention when:**\n- Running on Hopper GPUs (H100/H200) — FP8 tensor cores are needed\n- Attention is the computational bottleneck (long sequences, large batch sizes)\n- The application tolerates ~$10^{-3}$ attention error (most generation tasks)\n- Training speed matters more than matching FP16 output exactly\n\n**Stick with FP16/BF16 attention when:**\n- Running on Ampere or older GPUs (no FP8 tensor cores)\n- Short sequences where attention is a small fraction of compute\n- Tasks requiring exact reproducibility or extremely precise attention\n- The Hadamard preprocessing overhead is a concern (rare, but possible for very short sequences)\n\n**Key point for training**: Even when using FP8 for the attention forward pass, the gradients and optimizer states should remain in higher precision (BF16/FP32). FP8 attention reduces the forward/backward attention compute cost but doesn't affect the precision of the overall training loop.\n\nFA3's FP8 path with incoherent processing represents a practical sweet spot: ~2× speedup with negligible accuracy loss, enabled by a theoretically grounded technique (decorrelating quantization errors) that adds minimal overhead."
    },
    // Step 12: MC — FP8 decision
    {
      type: "mc",
      question: "You're training a 70B LLM on H100 GPUs with 32K context length. The training is bottlenecked by attention compute. Should you use FA3's FP8 attention with incoherent processing?",
      options: [
        "Yes — the ~2× attention speedup significantly reduces training time, and incoherent processing keeps accuracy within ~$10^{-3}$ of FP16, with gradients maintained at higher precision",
        "No — the Hadamard preprocessing doubles the total compute, negating the FP8 throughput advantage",
        "No — FP8 introduces quantization errors that will accumulate over thousands of training steps and degrade final model quality",
        "Yes, but only for the first half of training — switch to FP16 attention for the final phase to improve convergence"
      ],
      correct: 0,
      explanation: "This is an ideal scenario for FP8 attention: H100 GPUs provide FP8 tensor cores, 32K context makes attention a major bottleneck, and the model is large enough that training speed matters. Incoherent processing ensures attention output matches FP16 to ~$10^{-3}$ RMSE. The Hadamard preprocessing is $O(Nd\\log d)$ vs attention's $O(N^2d)$ — negligible at 32K context. Gradients stay at BF16/FP32, so training dynamics are unaffected."
    },
    // Step 13: MC — FP8 vs FP16 scaling
    {
      type: "mc",
      question: "As sequence length $N$ increases from 4K to 64K, how does the relative benefit of FP8 vs FP16 FlashAttention change?",
      options: [
        "The benefit decreases — longer sequences mean more quantization error accumulation in the online softmax, degrading accuracy",
        "The benefit stays constant — FP8 provides a fixed 2× throughput advantage regardless of sequence length",
        "The benefit increases — attention becomes a larger fraction of total compute as $N$ grows (due to $O(N^2)$ scaling), so the 2× speedup applies to a larger portion of training time",
        "The benefit reverses — at 64K tokens, attention becomes memory-bound rather than compute-bound, making FP8's compute advantage irrelevant"
      ],
      correct: 2,
      explanation: "Attention compute scales as $O(N^2 d)$ while FFN compute scales as $O(Nd_{\\text{model}}^2)$. As $N$ grows, attention's share of total FLOPs increases. At 4K, attention might be 15% of total compute; at 64K, it could be 60%+. A 2× speedup on 60% of compute gives 1.43× overall speedup, versus 1.08× when attention is only 15%. The accuracy with incoherent processing doesn't degrade with longer sequences because each tile is quantized independently."
    }
  ]
};
