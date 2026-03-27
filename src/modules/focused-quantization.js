// Focused module: Quantization for LLM Inference
// Covers the memory wall, number representations, affine quantization,
// granularity (per-tensor/channel/group), PTQ, GPTQ, and AWQ.
// Grounded in Goodfellow et al. (2016) Ch. 4 (Numerical Computation)
// and the GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023) literature.

export const quantizationLearning = {
  id: "C.1-quantization-learning-easy",
  sectionId: "C.1",
  title: "Quantization for LLM Inference",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "The Memory Wall: Why Quantization Matters",
      content: "Serving a large language model is **memory-bandwidth bound**, not compute-bound. During autoregressive decoding, each token generation requires loading the entire model's weights from GPU memory (HBM) to the compute units. For a 70B parameter model in float16, that's **140 GB** of weights read per token.\n\nModern GPUs like the A100 have ~2 TB/s of HBM bandwidth. Loading 140 GB takes ~70 ms — far longer than the actual matrix multiplications. The GPU's compute units sit idle, waiting for data.\n\nAs Goodfellow et al. (2016, Ch. 4) emphasize, the choice of numerical representation has profound practical consequences. In deep learning inference, representing each weight with fewer bits directly reduces the bytes that must be loaded per token, translating to proportional speedups.\n\n**Quantization** replaces the 16-bit (or 32-bit) floating-point weights with lower-precision representations — 8-bit, 4-bit, or even lower. A 4-bit quantized 70B model occupies roughly **35 GB** instead of 140 GB, fitting on a single GPU and generating tokens 3-4x faster.\n\nThe challenge: how do you reduce precision without destroying the model's capabilities?"
    },
    {
      type: "mc",
      question: "A 7B parameter model in float16 occupies 14 GB. During autoregressive decoding (one token at a time), the primary bottleneck is:",
      options: [
        "The KV-cache lookup, which requires random memory access patterns that defeat the GPU's coalesced read optimizations and dominate latency",
        "The embedding table lookup for the input token, since the full embedding matrix must be scanned linearly to find the correct row",
        "The softmax computation over the entire vocabulary, which requires $O(V)$ exponentiations and a full normalization pass per generated token",
        "Loading 14 GB of weights from HBM per generated token, since the arithmetic intensity (FLOPs per byte) is extremely low at batch size 1"
      ],
      correct: 3,
      explanation: "With batch size 1 (one token per step), each weight element participates in exactly one multiply-add. The arithmetic intensity is ~1 FLOP per 2 bytes loaded — far below the GPU's compute-to-bandwidth ratio of ~150 FLOPs/byte on an A100. The GPU spends most of its time waiting for weight data to arrive from HBM. This is why reducing weight size via quantization directly improves throughput: fewer bytes to load means less waiting."
    },
    {
      type: "info",
      title: "Number Representations: From Float16 to INT4",
      content: "To understand quantization, we need to understand what we're converting between.\n\n**Float16 (FP16)**: 1 sign bit, 5 exponent bits, 10 mantissa bits. Can represent values from $\\sim 6 \\times 10^{-8}$ to $\\sim 65504$ with variable precision — high precision near zero, lower precision for large values. This is the standard training and serving format.\n\n**BFloat16 (BF16)**: 1 sign bit, 8 exponent bits, 7 mantissa bits. Same range as float32 ($\\sim 10^{-38}$ to $\\sim 10^{38}$) but with less precision. Preferred for training because the wider range avoids overflow.\n\n**INT8**: 8-bit integer. Represents exactly 256 values on a uniform grid. With unsigned INT8: values 0 to 255. With signed INT8: values -128 to 127. The key difference from floating-point: the spacing between representable values is **uniform** (constant step size), not adaptive.\n\n**INT4**: 4-bit integer. Only 16 distinct values. This is the frontier of practical quantization — compressing a continuous weight distribution into just 16 bins while preserving model quality.\n\nThe fundamental tradeoff: fewer bits means a coarser grid, and some weight values will be rounded to their nearest grid point with significant error. The art of quantization is minimizing the impact of this rounding error on the model's output."
    },
    {
      type: "mc",
      question: "Float16 can represent values with variable spacing (finer near zero, coarser for large values), while INT8 uses uniform spacing. Why is uniform spacing a challenge for quantizing neural network weights?",
      options: [
        "INT8 cannot represent the value zero exactly, so every bias term and zero-initialized parameter accumulates a persistent rounding offset",
        "Uniform spacing prevents the hardware from using fused multiply-add instructions, negating the throughput benefit of reduced precision",
        "Weight distributions are bell-shaped with most values near zero — uniform spacing wastes grid points on sparse tails while under-resolving the dense center",
        "Neural network weights are strictly positive after ReLU activations, so half the signed INT8 range (negative values) goes unused entirely"
      ],
      correct: 2,
      explanation: "Neural network weights typically follow a roughly Gaussian distribution centered near zero. Most weights are small, with a few outliers. A uniform grid allocates equal resolution across the entire range, but this means the same number of grid points covers the dense center as covers the sparse tails. Floating-point naturally gives more precision near zero (where most weights cluster). Quantization techniques like non-uniform quantization and group quantization address this mismatch."
    },
    {
      type: "info",
      title: "Affine Quantization: Mapping Floats to Integers",
      content: "The standard method to convert a floating-point weight $w$ to a $b$-bit integer is **affine (asymmetric) quantization**:\n\n$$q = \\text{round}\\left(\\frac{w - z}{s}\\right), \\quad q \\in [0, 2^b - 1]$$\n\nwhere:\n- $s = \\frac{w_{\\max} - w_{\\min}}{2^b - 1}$ is the **scale** (step size between adjacent grid points)\n- $z = w_{\\min}$ is the **zero-point** (the floating-point value that maps to integer 0)\n\nTo recover an approximate weight value (dequantization):\n\n$$\\hat{w} = s \\cdot q + z$$\n\nThe maximum quantization error per weight is $s/2$ (half a step). For 8-bit quantization of a weight range $[-1, 1]$: $s = 2/255 \\approx 0.0078$, so the worst-case error is $\\sim 0.004$ per weight. For 4-bit: $s = 2/15 \\approx 0.133$, with worst-case error $\\sim 0.067$ — an order of magnitude larger.\n\n**Symmetric quantization** is a simpler variant where $z = 0$ and the grid is centered: $q = \\text{round}(w / s)$ with $s = \\max(|w|) / (2^{b-1} - 1)$. This wastes some range if the weight distribution is asymmetric, but simplifies the integer arithmetic (no zero-point offset needed)."
    },
    {
      type: "mc",
      question: "A weight tensor has values ranging from $-0.8$ to $+1.2$. With 4-bit unsigned affine quantization (16 grid points, 0 to 15), what is the scale $s$?",
      options: [
        "$s = 1.2 / 15 = 0.08$ — only the positive range matters since most weights are positive",
        "$s = (1.2 - (-0.8)) / 15 = 2.0 / 15 \\approx 0.133$ — the full range divided by the number of intervals",
        "$s = 2.0 / 16 = 0.125$ — the range divided by the number of representable values",
        "$s = 0.8 / 15 \\approx 0.053$ — only the magnitude of the most negative value matters for calibration"
      ],
      correct: 1,
      explanation: "The scale maps the full floating-point range $[w_{\\min}, w_{\\max}] = [-0.8, 1.2]$ to the integer range $[0, 15]$. There are $2^b - 1 = 15$ intervals between 16 grid points, so $s = (1.2 - (-0.8)) / 15 = 2.0 / 15 \\approx 0.133$. Note the denominator is $2^b - 1$ (number of intervals), not $2^b$ (number of values). With this scale, the quantization error per weight is at most $s/2 \\approx 0.067$."
    },
    {
      type: "info",
      title: "Quantization Granularity: Per-Tensor, Per-Channel, Per-Group",
      content: "The scale $s$ and zero-point $z$ determine the quantization grid. Using a **single** $(s, z)$ pair for an entire weight matrix is called **per-tensor quantization**. This is the coarsest approach — all weights share one grid.\n\nThe problem: different rows (or columns) of a weight matrix can have very different ranges. If one row has weights in $[-0.1, 0.1]$ and another has weights in $[-2.0, 2.0]$, the per-tensor scale is set by the wide row, and the narrow row gets only 1-2 of the 16 grid points. Most of the narrow row's weights round to zero.\n\n**Per-channel quantization** uses a separate $(s, z)$ per output channel (row of the weight matrix). Each row gets its own grid matched to its range. This dramatically improves quality, especially for layers with outlier channels.\n\n**Per-group quantization** goes further: each contiguous block of $g$ weights within a channel gets its own $(s, z)$. With group size $g = 128$ (the standard choice), a row of 4096 weights has 32 independent scales. This captures local range variation within a channel.\n\nThe tradeoff: more scales = better quality but more storage overhead. Per-group with $g = 128$ adds ~0.25 bits per weight (the scale and zero-point for each group), a modest cost that yields significant quality improvement at 4-bit and below."
    },
    {
      type: "mc",
      question: "A weight matrix has shape $[4096, 4096]$ and is quantized to 4-bit with group size 128. How many scale/zero-point pairs are stored, and what is the approximate storage overhead compared to the raw 4-bit weights?",
      options: [
        "16,777,216 pairs (one per weight) — each weight gets its own scale/zero-point, so the metadata exceeds the quantized data and negates the compression",
        "$4096 / 128 = 32$ pairs total (one per group across the entire matrix) — minimal overhead because the groups span all rows jointly, sharing a single partition",
        "4,096 pairs (one per row via per-channel quantization) — the 32 groups within each row share one scale, totaling 4,096 float16 pairs or ~16 KB overhead",
        "131,072 pairs ($4096 \\times 4096 / 128$) — each pair is 4 bytes (float16 scale + float16 zero-point), adding ~0.5 MB to the ~8 MB of 4-bit weights (~6%)"
      ],
      correct: 3,
      explanation: "The matrix has $4096 \\times 4096 = 16{,}777{,}216$ weights. With group size 128, there are $16{,}777{,}216 / 128 = 131{,}072$ groups. Each group stores a float16 scale and float16 zero-point (4 bytes). That's $131{,}072 \\times 4 = 524{,}288$ bytes $\\approx 0.5$ MB. The raw 4-bit weights occupy $16{,}777{,}216 \\times 0.5 = 8{,}388{,}608$ bytes $\\approx 8$ MB. So the overhead is about 6% — a small price for much better per-group accuracy."
    },
    {
      type: "info",
      title: "Post-Training Quantization: Round-to-Nearest and Its Limits",
      content: "The simplest quantization method is **round-to-nearest (RTN)**: compute the scale from the weight range, round each weight to the nearest grid point, and use the quantized weights directly. No calibration data needed.\n\nRTN works well at 8-bit. The quantization error per weight ($\\sim 0.004$ for typical ranges) is small enough that the cumulative effect across a matrix multiply is negligible. INT8 RTN with per-channel scaling typically loses $<0.1$ perplexity on language models.\n\nAt **4-bit**, RTN breaks down. The error per weight ($\\sim 0.067$) is 16x larger. In a matrix multiply $y = Wx$ with dimension $d = 4096$, each output element sums 4096 quantized weights. Even if individual errors are small and unbiased, the accumulated error in $y$ can be substantial — and the errors are **not independent** across layers. They compound through the network.\n\nThe failure mode is particularly severe for **outlier weights** — the small number of weights with unusually large magnitudes. These weights are disproportionately important for the model's output (they carry high activation signal), but RTN quantizes them with the same coarse grid as all other weights.\n\nThis is why sophisticated quantization methods like GPTQ and AWQ exist: they minimize the **output error** of each layer, not just the per-weight rounding error."
    },
    {
      type: "mc",
      question: "At 8-bit precision, round-to-nearest (RTN) quantization loses $<0.1$ perplexity. At 4-bit, RTN often degrades perplexity by 1-5 points or more. The primary reason for this disproportionate degradation is:",
      options: [
        "Per-weight error is 16x larger at 4-bit, and these errors accumulate through matrix multiplies and across layers — especially for outlier weights carrying disproportionate signal",
        "4-bit quantization requires specialized hardware instructions that introduce additional rounding at each multiply-accumulate step beyond the initial weight quantization error",
        "The loss landscape is convex near the float16 weights at 8-bit precision but becomes non-convex at 4-bit, causing the quantized model to land in a worse local minimum",
        "4-bit signed integers can only represent values from -8 to 7, so the asymmetric range clips large positive weights that carry most of the learned information"
      ],
      correct: 0,
      explanation: "Going from 8-bit (256 levels) to 4-bit (16 levels) increases the step size by 16x, and thus the maximum per-weight error by 16x. In a $d$-dimensional dot product, the error in the output is roughly $\\sqrt{d}$ times the per-element error (by the central limit theorem for independent rounding errors). With $d = 4096$ and 16x larger per-weight error, the output error grows dramatically. Outlier weights exacerbate this: they set a wide scale, making the grid coarse for all other weights."
    },
    {
      type: "info",
      title: "GPTQ: Hessian-Guided Weight Quantization",
      content: "GPTQ (Frantar et al., 2022) is a post-training quantization method that achieves near-lossless 4-bit quantization. Its key insight: when you quantize one weight, you can **adjust the remaining weights** to compensate for the error.\n\nGPTQ frames quantization as an optimization problem. For a single linear layer with weights $W$ and calibration inputs $X$, it minimizes the **layer-wise reconstruction error**:\n\n$$\\min_{\\hat{W}} \\|WX - \\hat{W}X\\|_F^2$$\n\nwhere $\\hat{W}$ is the quantized weight matrix. This is the Optimal Brain Quantization (OBQ) framework.\n\nThe algorithm processes weight columns one at a time. For each column $w_i$:\n1. **Quantize** $w_i$ to the nearest grid point $\\hat{w}_i$\n2. **Compute the error**: $\\delta_i = w_i - \\hat{w}_i$\n3. **Compensate**: update all not-yet-quantized columns by $\\delta_i \\cdot H^{-1}_{i,:}$, where $H = 2X X^\\top$ is the Hessian of the reconstruction loss\n\nThe Hessian tells us how each weight affects the output. When we introduce error by quantizing $w_i$, the Hessian-guided update distributes this error to other weights in a way that minimizes the overall output distortion.\n\nGPTQ quantizes a 175B model in ~4 hours on a single GPU using a small calibration set (128-256 examples). The result: 4-bit models with $<0.5$ perplexity loss from the float16 baseline."
    },
    {
      type: "mc",
      question: "GPTQ quantizes weight columns sequentially. After quantizing column $w_i$, it updates all remaining columns using $\\delta_i \\cdot H^{-1}_{i,:}$. Why is this compensation step critical?",
      options: [
        "It re-normalizes the remaining columns so the Frobenius norm $\\|W\\|_F$ is preserved after each step, preventing the weight magnitude from drifting during sequential quantization",
        "It updates the calibration data $X$ to reflect the quantized column, allowing subsequent columns to be quantized against the modified input distribution from earlier rounding",
        "It shifts remaining weights to compensate for the error in column $i$, minimizing the layer's output reconstruction error rather than just the per-weight rounding error",
        "It pre-quantizes the remaining columns to the same grid as column $i$, ensuring consistent scale and zero-point parameters across the entire weight matrix"
      ],
      correct: 2,
      explanation: "Without compensation, each column's quantization error is independent and accumulates. GPTQ's compensation step uses the Hessian $H^{-1}$ to find the optimal adjustment to all remaining (unquantized) weights that minimizes the total reconstruction error $\\|WX - \\hat{W}X\\|^2$. Intuitively, if quantizing $w_i$ causes the output to shift in some direction, the compensation slightly adjusts other weights to shift it back. This is why GPTQ far outperforms naive RTN at 4-bit: it globally optimizes the layer output, not individual weight accuracy."
    },
    {
      type: "info",
      title: "AWQ: Protecting Salient Channels",
      content: "AWQ (Activation-Aware Weight Quantization, Lin et al., 2023) takes a different approach from GPTQ. Instead of Hessian-based error compensation, AWQ observes a key empirical fact: **not all weight channels are equally important**.\n\nBy analyzing activations on calibration data, AWQ finds that only ~1% of weight channels carry most of the signal — these are the channels connected to **activation outlier features** (the features with unusually large magnitudes in the hidden representations).\n\nAWQ's strategy: **protect the salient channels by per-channel scaling**. For each weight channel $j$, AWQ computes a scale $s_j$ based on the average activation magnitude:\n\n$$s_j \\propto \\|X_j\\|_2^\\alpha, \\quad \\alpha \\in (0, 1)$$\n\nThe weights are then multiplied by $s_j$ before quantization, and the activations are divided by $s_j$ (preserving the mathematical result). This effectively **narrows the range** of salient weight channels before quantization, giving them more of the quantization grid's resolution.\n\nKey advantages over GPTQ:\n- **Faster**: No Hessian computation or sequential column processing. AWQ quantizes a 70B model in minutes.\n- **Hardware-friendly**: The per-channel scales fold into the preceding layer's output scaling, adding zero overhead at inference time.\n- **Comparable quality**: Matches or slightly exceeds GPTQ quality at 4-bit, particularly for models with strong activation outliers.\n\nThe insight is elegant: rather than fixing quantization error after the fact (GPTQ), prevent it from hitting the weights that matter most (AWQ)."
    },
    {
      type: "mc",
      question: "AWQ scales salient weight channels up and non-salient channels down before quantization (with inverse scaling on activations to preserve correctness). Within a quantization group that mixes salient and non-salient weights, what is the net effect on the model's output quality?",
      options: [
        "No net effect — the scaling and inverse-scaling cancel exactly, so quantization error is merely redistributed across channels without changing the total output error",
        "Salient weights get better relative precision from the shared grid while non-salient weights get worse — but since non-salient channels contribute less to the output, net error decreases",
        "All weights within a group receive equal precision because they share one scale factor, so per-channel scaling applied before grouping has no effect on within-group resolution",
        "The scaling destabilizes inference because modified activation magnitudes cause the layer normalization running statistics to become stale, introducing systematic bias"
      ],
      correct: 1,
      explanation: "Within a quantization group, the scale is determined by the range of the largest weights. By scaling up salient channels, AWQ makes them occupy more of this range — they get more grid points and thus lower relative quantization error. Non-salient channels get compressed into fewer grid points, but they are connected to small activations, so their contribution to the output error is proportionally smaller. The net effect: quantization error is **redistributed** from high-impact channels (where it matters) to low-impact channels (where it doesn't). This is why AWQ achieves strong quality — it's not reducing total quantization error, but putting it where it hurts least."
    }
  ]
};
