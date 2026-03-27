// Assessment module for C.1: Quantization
// Split from assess-branch-cd.js — per-section test (10 questions)

export const quantizationAssessment = {
  id: "C.1-assess",
  sectionId: "C.1",
  title: "Assessment: Quantization",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In post-training quantization (PTQ), weights are quantized **after** training is complete. Quantization-aware training (QAT) instead:",
      options: ["Trains a separate smaller model from scratch at the target precision, bypassing the pretrained weights entirely", "Simulates quantization effects during training via straight-through estimators so the model learns to be robust to reduced precision", "Quantizes only the optimizer states rather than the model weights, reducing memory without affecting inference precision", "Uses lower learning rates to compensate for precision loss, allowing the model to converge despite the quantization noise"],
      correct: 1,
      explanation: "QAT inserts fake-quantization nodes during training that round weights/activations to the target precision in the forward pass but pass gradients through unmodified (straight-through estimator). The model thus learns weight configurations that are robust to quantization noise. QAT typically recovers 0.5-1.0 perplexity points over PTQ but requires a full training run, making it far more expensive."
    },
    {
      type: "mc",
      question: "Activation quantization is generally harder than weight quantization because:",
      options: ["Activations use more memory than weights in aggregate, so quantization errors accumulate over more values and cause greater total degradation", "The backward pass requires full-precision activations for accurate gradient computation, making any activation quantization incompatible with training", "Activations exhibit **outlier features** — a small number of hidden dimensions have magnitudes 10-100x larger than the rest, making uniform quantization waste most of its range on a few extreme values", "Activations are always stored in float64 by the framework, so quantizing them requires changing the entire compute pipeline rather than just the storage format"],
      correct: 2,
      explanation: "Research (e.g., LLM.int8(), SmoothQuant) showed that transformer activations contain persistent outlier dimensions with magnitudes far exceeding the typical range. A uniform INT8 grid spanning $[-100, 100]$ to accommodate outliers wastes precision for the majority of values clustered near $[-1, 1]$. SmoothQuant addresses this by mathematically migrating the quantization difficulty from activations to weights via per-channel scaling: $Y = (X \\cdot \\text{diag}(s)^{-1}) \\cdot (\\text{diag}(s) \\cdot W)$."
    },
    {
      type: "mc",
      question: "GPTQ is a popular weight quantization method. Its core strategy is:",
      options: ["Training from scratch with quantized weights and a modified loss function that accounts for the reduced precision of each weight matrix", "Pruning weights below a threshold first and then quantizing the surviving weights, combining sparsity with reduced precision", "Quantizing all weights simultaneously with k-means clustering to find the optimal set of centroids for each target bit width", "Quantizing weights **column by column**, using the inverse Hessian to optimally adjust remaining weights to compensate for each column's quantization error"],
      correct: 3,
      explanation: "GPTQ extends the Optimal Brain Quantization framework. It processes weight columns sequentially: after quantizing one column, it uses the inverse Hessian $H^{-1}$ of the layer's reconstruction loss to compute the optimal update to all not-yet-quantized columns, minimizing $\\|WX - \\hat{W}X\\|^2$. This Hessian-guided error compensation is what makes GPTQ achieve much better quality than naive round-to-nearest at 4-bit and below."
    },
    {
      type: "mc",
      question: "AWQ (Activation-Aware Weight Quantization) differs from GPTQ by focusing on:",
      options: ["Training a separate quantization network that learns to map full-precision weights to their optimal quantized representations through end-to-end optimization", "Quantizing activations instead of weights, since activation quantization provides larger memory savings due to the dynamic nature of activation tensors", "Using 8-bit instead of 4-bit quantization for the most sensitive weight matrices, with a learned routing mechanism to determine which layers need higher precision", "Identifying the small fraction of **salient weight channels** (those corresponding to large activation magnitudes) and protecting them with per-channel scaling before quantization, rather than using Hessian-based error compensation"],
      correct: 3,
      explanation: "AWQ observes that only ~1% of weight channels are critical — those connected to activation outlier features. Rather than expensive Hessian computation, AWQ finds per-channel scaling factors $s$ that protect salient channels: it scales weights by $s$ and inversely scales activations, shifting the quantization difficulty away from important channels. This is simpler than GPTQ and often matches or exceeds its quality with faster quantization time."
    },
    {
      type: "mc",
      question: "A model uses mixed-precision quantization: some layers at 4-bit, others at 8-bit. The decision of which layers get higher precision is typically based on:",
      options: ["**Per-layer sensitivity analysis** — layers where quantization causes larger increases in output error or perplexity are assigned higher precision, often measured via Hessian trace, Fisher information, or direct calibration loss", "Parameter count — larger layers get lower precision to achieve greater total memory savings from quantizing their larger weight matrices", "Layer index — earlier layers always need more precision because they establish the representations that all subsequent layers depend on", "Random assignment with a fixed ratio of 4-bit to 8-bit layers, relying on the law of large numbers to average out per-layer quantization errors"],
      correct: 0,
      explanation: "Mixed-precision strategies measure each layer's sensitivity to quantization error, typically by quantizing one layer at a time and measuring the impact on calibration loss. Layers with high Hessian trace ($\\text{tr}(H)$) or large Fisher information are more sensitive. Empirically, attention projection layers and the first/last layers tend to be more sensitive. This yields a constrained optimization: minimize total quality loss subject to a target average bit-width."
    },
    {
      type: "mc",
      question: "SqueezeLLM achieves high-quality ultra-low-bit quantization by combining:",
      options: ["Knowledge distillation with pruning, training a smaller student network that mimics the quantized teacher's behavior while inheriting its sparsity pattern", "Dense-and-sparse decomposition: a low-bit dense representation for the bulk of weights plus a **sparse matrix** storing outlier weights at full precision, keeping the sensitive values exact", "Layer fusion and operator merging that combine adjacent linear layers into single operations, reducing the number of quantization boundaries in the compute graph", "Dynamic quantization at inference time that adapts the bit width per-token based on the activation magnitudes observed during each forward pass"],
      correct: 1,
      explanation: "SqueezeLLM decomposes each weight matrix into a dense low-bit component plus a sparse full-precision component for outlier weights. The key insight is that weight sensitivity follows a heavy-tailed distribution — a small number of weights disproportionately affect output quality. By storing these in a sparse matrix (which adds minimal memory overhead due to sparsity), the dense component can be aggressively quantized to 3 or even 2 bits with minimal degradation."
    },
    {
      type: "mc",
      question: "BitNet b1.58 uses ternary weights $\\{-1, 0, +1\\}$, meaning each weight requires $\\log_2(3) \\approx 1.58$ bits. Compared to standard float16 models of the same size, BitNet claims:",
      options: [
        "Identical accuracy with 10x faster inference at all model sizes, due to replacing floating-point multiplications with simple additions and subtractions",
        "Worse accuracy at all scales but 100x memory savings, making it useful only for deployment on extremely memory-constrained edge devices",
        "Better accuracy than float16 models because ternary weights act as strong regularization that prevents overfitting to noise in the training data",
        "Matching perplexity at the same parameter count starting from ~3B parameters, with matrix multiplications reduced to additions since $w \\in \\{-1, 0, 1\\}$ eliminates the need for floating-point multiply hardware"
      ],
      correct: 3,
      explanation: "With ternary weights, the matrix-vector product $y = Wx$ becomes pure additions and subtractions (multiply by 1, -1, or skip for 0). This eliminates the most expensive operation in inference — floating-point multiplication — and enables dramatically simpler hardware. BitNet b1.58 reports matching LLaMA-equivalent perplexity starting around 3B parameters, suggesting that extreme quantization is viable if applied from the start of training (QAT-style) rather than post-hoc."
    },
    {
      type: "mc",
      question: "A 7B parameter model with float16 weights occupies 14 GB. After GPTQ 4-bit quantization with group size 128, the model size is approximately:",
      options: ["7 GB — quantization halves the storage by sharing weights across pairs of layers, but each weight still requires a full-precision activation map stored alongside", "1.75 GB — exactly $14 \\times (4/16)$ with no overhead, since group-wise scale and zero-point parameters are stored implicitly in the quantization grid itself", "~3.9 GB — each group of 128 weights shares a float16 scale and zero-point, adding overhead: $14 \\times \\frac{4}{16} + \\frac{7 \\times 10^9}{128} \\times 4\\text{ bytes} \\approx 3.5 + 0.22$ GB", "14 GB — the quantized model stores both the 4-bit weights and a full-precision copy used for dequantization during each forward pass, doubling the storage"],
      correct: 2,
      explanation: "4-bit quantization reduces the weight payload to $14 \\times (4/16) = 3.5$ GB. However, each group of 128 weights requires a float16 scale and zero-point (4 bytes per group). With $\\frac{7 \\times 10^9}{128} \\approx 54.7\\text{M}$ groups, that adds $\\sim$219 MB of overhead. Total $\\approx 3.7$-$3.9$ GB depending on metadata. Smaller group sizes improve quality but increase overhead; group size 128 is the standard trade-off."
    },
    {
      type: "mc",
      question: "When quantizing a weight value $w$ to a $b$-bit unsigned integer grid $[0, 2^b - 1]$, the standard affine quantization formula is:",
      options: ["$q = \\text{round}(w \\times 2^b)$ — a direct scaling that maps the weight to the nearest integer on a $[0, 2^b]$ grid without accounting for the actual weight distribution range", "$q = \\text{sign}(w) \\times \\lfloor |w| \\rfloor$ — a symmetric truncation that quantizes magnitudes independently of sign, using the floor function to map to the nearest lower integer", "$q = \\text{clamp}\\left(\\text{round}\\left(\\frac{w - z}{s}\\right), 0, 2^b - 1\\right)$ where $s = \\frac{w_{\\max} - w_{\\min}}{2^b - 1}$ is the scale and $z = w_{\\min}$ is the zero-point", "$q = \\text{round}(w)$ clipped to $b$ bits — a naive rounding scheme that maps each weight to its nearest integer and then truncates to fit within the target bit width"],
      correct: 2,
      explanation: "Affine (asymmetric) quantization maps the floating-point range $[w_{\\min}, w_{\\max}]$ to the integer range $[0, 2^b - 1]$. The scale $s = \\frac{w_{\\max} - w_{\\min}}{2^b - 1}$ determines the step size, and the zero-point $z$ handles asymmetric distributions. Dequantization recovers $\\hat{w} = s \\cdot q + z$. The quantization error per weight is bounded by $s/2$, so fewer bits means larger steps and more error."
    },
    {
      type: "mc",
      question: "A researcher quantizes a 70B model to 2-bit weights and observes catastrophic perplexity degradation. Which approach is LEAST likely to help recover quality?",
      options: [
        "Increasing the batch size during inference to average out the per-sample noise introduced by the aggressive 2-bit weight quantization",
        "Switching to mixed-precision with sensitive layers at 4-bit to protect the most critical weight matrices from extreme quantization",
        "Adding LoRA adapters trained on a small dataset after quantization (QLoRA-style) to compensate for quantization-induced representation errors",
        "Using a larger and more diverse calibration dataset for GPTQ to improve the Hessian estimates used during weight quantization"
      ],
      correct: 0,
      explanation: "Increasing batch size affects throughput but does not change the model's weights or predictions — it cannot recover quality lost to quantization. The other three approaches directly address quantization error: better calibration data improves Hessian estimates in GPTQ, mixed-precision protects sensitive layers, and QLoRA fine-tunes low-rank adapters in float16 on top of quantized weights to compensate for quantization-induced errors. At 2-bit, combining multiple recovery strategies is typically necessary."
    }
  ]
};
