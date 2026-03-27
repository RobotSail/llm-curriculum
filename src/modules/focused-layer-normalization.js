// Focused learning module: Layer Normalization
// Section 1.1: Transformer Architecture
// Single concept: how layer normalization stabilizes activations in transformers,
// including LayerNorm vs RMSNorm and Pre-LN vs Post-LN placement.
// Grounded in Ba et al. (2016), Zhang & Sennrich (2019), Xiong et al. (2020).

export const layerNormalizationLearning = {
  id: "1.1-layer-normalization-learning-easy",
  sectionId: "1.1",
  title: "Layer Normalization",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Why normalization is needed
    {
      type: "info",
      title: "Why Transformers Need Normalization",
      content: "Residual connections solve gradient flow but create a new problem: each sub-layer **adds** to the residual stream, so activation magnitudes can grow across layers. Even with proper initialization scaling, the distribution of activations shifts during training as weights update.\n\nWithout normalization, sub-layers receive inputs whose magnitude and distribution vary unpredictably across layers and training steps. This creates two problems:\n\n1. **Unstable optimization**: When input magnitudes vary widely, the same learning rate can be too large for some layers and too small for others. The loss landscape becomes ill-conditioned.\n\n2. **Internal covariate shift**: Each layer's input distribution changes as earlier layers update their weights, forcing later layers to continuously readapt. This slows convergence.\n\n**Normalization layers** address this by standardizing activations to a consistent scale before each sub-layer processes them. The key question is: normalize across which dimension? Across the batch (BatchNorm)? Across features (LayerNorm)? The choice matters for transformers."
    },
    // Step 2: MC \u2014 why not batch normalization
    {
      type: "mc",
      question: "Batch Normalization (Ioffe & Szegedy, 2015) computes mean and variance across the batch dimension for each feature. Why is BatchNorm unsuitable for autoregressive language models?",
      options: [
        "BatchNorm's running statistics computed during training become invalid at inference when processing one sequence at a time, and sequence lengths vary, making the batch statistics unreliable across tokens at different positions",
        "BatchNorm requires computing gradients through the batch statistics, which doubles the memory cost compared to LayerNorm and exceeds GPU capacity for large transformers",
        "BatchNorm was designed for convolutional networks and its mathematical formulation cannot be applied to sequence data of any kind, including transformers",
        "BatchNorm performs identically to LayerNorm for transformers, so the choice is purely historical \u2014 early transformer papers happened to use LayerNorm and the convention stuck"
      ],
      correct: 0,
      explanation: "BatchNorm normalizes each feature across the batch, computing running mean and variance during training for use at inference. For language models, this is problematic: (1) at inference, batch size is often 1, so batch statistics are meaningless; (2) running statistics must be computed per position, but sequence lengths vary; (3) tokens at different positions have different distributional properties, making shared batch statistics noisy. LayerNorm avoids all these issues by normalizing across features within each individual token \u2014 no dependence on batch size or other sequences."
    },
    // Step 3: LayerNorm mechanics
    {
      type: "info",
      title: "Layer Normalization: The Mechanics",
      content: "**Layer Normalization** (Ba et al., 2016) standardizes each activation vector to zero mean and unit variance, then applies a learned affine transform:\n\n$$\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sigma + \\epsilon} + \\beta$$\n\nwhere $\\mu = \\frac{1}{d}\\sum_{i=1}^{d} x_i$ and $\\sigma = \\sqrt{\\frac{1}{d}\\sum_{i=1}^{d}(x_i - \\mu)^2}$ are computed per-token (across the $d$ feature dimensions), and $\\gamma, \\beta \\in \\mathbb{R}^d$ are learned scale and shift parameters.\n\nKey properties:\n- **Per-token normalization**: Each token's $d$-dimensional vector is normalized independently. No dependence on batch size, sequence length, or other tokens.\n- **Removes magnitude information**: After normalization, only the **direction** of the activation vector matters. Information is encoded in relative feature magnitudes, not absolute scale.\n- **Learned affine transform**: $\\gamma$ and $\\beta$ let the model recover any scale and shift it finds useful, so normalization doesn't permanently destroy information.\n- **Epsilon for stability**: The small constant $\\epsilon$ (typically $10^{-5}$ or $10^{-6}$) prevents division by zero when the activation vector has near-zero variance."
    },
    // Step 4: MC \u2014 LayerNorm properties
    {
      type: "mc",
      question: "LayerNorm projects each activation vector onto the unit hypersphere (after mean-centering). If two tokens have activation vectors $x_A = [100, 200, 300]$ and $x_B = [1, 2, 3]$, what is the relationship between $\\text{LayerNorm}(x_A)$ and $\\text{LayerNorm}(x_B)$ (ignoring the learned affine transform)?",
      options: [
        "$\\text{LayerNorm}(x_A)$ has larger entries because the normalization preserves relative magnitude differences between tokens to maintain the model's ability to distinguish high-activation from low-activation tokens",
        "They are unrelated because the nonlinear mean-centering step causes different feature interactions in vectors of different magnitude, even when the ratios are identical",
        "They produce identical outputs because both vectors have the same ratios between components \u2014 LayerNorm removes all absolute scale information, keeping only the direction",
        "$\\text{LayerNorm}(x_B)$ is numerically unstable due to the small values, producing a significantly different result than $x_A$ despite the proportional relationship"
      ],
      correct: 2,
      explanation: "Since $x_A = 100 \\cdot x_B$, both have identical means (proportionally) and identical normalized directions. After subtracting the mean, $x_A - \\mu_A = 100(x_B - \\mu_B)$. Dividing by $\\sigma_A = 100 \\sigma_B$ cancels the factor of 100. LayerNorm is **scale-invariant**: any positive scalar multiple of an input produces the same output. This is why it removes magnitude information \u2014 the model must encode meaning in the angular relationships between features, not their absolute values."
    },
    // Step 5: RMSNorm
    {
      type: "info",
      title: "RMSNorm: A Simpler Alternative",
      content: "**RMSNorm** (Zhang & Sennrich, 2019) simplifies LayerNorm by removing the mean-centering step:\n\n$$\\text{RMSNorm}(x) = \\gamma \\odot \\frac{x}{\\text{RMS}(x)}, \\quad \\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum_{i=1}^{d} x_i^2}$$\n\nRMSNorm normalizes by the root-mean-square of the vector, not the standard deviation. It also drops the learned bias $\\beta$, keeping only the learned scale $\\gamma$.\n\nWhy does removing mean-centering work?\n\n- Empirically, Zhang & Sennrich (2019) showed the mean subtraction contributes minimally to LayerNorm's effectiveness. The key ingredient is **scale normalization** \u2014 preventing activation magnitudes from drifting.\n- The saved compute is modest per layer (one less reduction operation), but across 80+ layers and millions of training steps, it adds up.\n- LLaMA (Touvron et al., 2023) demonstrated equivalent quality with RMSNorm across scales from 7B to 65B parameters.\n\nNearly all modern open-weight LLMs (LLaMA, Mistral, Gemma, Qwen) use RMSNorm. The original LayerNorm with mean-centering is now mainly seen in older architectures (GPT-2, BERT)."
    },
    // Step 6: MC \u2014 RMSNorm
    {
      type: "mc",
      question: "RMSNorm uses $\\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum_i x_i^2}$ while LayerNorm uses $\\sigma = \\sqrt{\\frac{1}{d}\\sum_i (x_i - \\mu)^2}$. For a vector with mean $\\mu = 0$, what is the relationship?",
      options: [
        "RMS and $\\sigma$ are unrelated even when $\\mu = 0$ because the squaring operation interacts differently with the summation in each formula",
        "RMS $= \\sigma$ when $\\mu = 0$ because $\\sum_i x_i^2 = \\sum_i (x_i - 0)^2$ \u2014 the two norms become identical, so RMSNorm and LayerNorm produce the same output for zero-mean inputs",
        "RMS $= 2\\sigma$ when $\\mu = 0$ because RMSNorm counts both positive and negative deviations while standard deviation only counts one direction",
        "RMS $> \\sigma$ always, even when $\\mu = 0$, because the root-mean-square of a vector is always at least twice its standard deviation by the Cauchy-Schwarz inequality"
      ],
      correct: 1,
      explanation: "When $\\mu = 0$: $\\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum_i x_i^2} = \\sqrt{\\frac{1}{d}\\sum_i (x_i - 0)^2} = \\sigma$. The two norms are identical. In practice, activation vectors in transformers often have near-zero mean (especially after many layers of processing), so RMSNorm and LayerNorm produce very similar results. The cases where they differ most \u2014 vectors with large mean offset \u2014 are rare in well-initialized transformers."
    },
    // Step 7: Pre-LN vs Post-LN
    {
      type: "info",
      title: "Pre-LN vs Post-LN: Where to Normalize",
      content: "The placement of normalization relative to the residual connection has significant implications:\n\n**Post-LN** (original Transformer, Vaswani et al., 2017):\n$$x_{l+1} = \\text{LN}(x_l + f_l(x_l))$$\n\nNormalization happens **after** the residual addition. The raw sub-layer output is added to the stream, and then the combined result is normalized.\n\n**Pre-LN** (modern standard):\n$$x_{l+1} = x_l + f_l(\\text{LN}(x_l))$$\n\nNormalization happens **before** the sub-layer processes its input. The residual addition bypasses the normalization entirely.\n\nThe critical difference: in Pre-LN, the residual stream itself is **never normalized** \u2014 only the input to each sub-layer is. This means gradients flowing through the skip path pass through no normalization operations, providing a cleaner gradient highway.\n\nIn Post-LN, gradients must pass through a LayerNorm at every step, and the Jacobian of LayerNorm depends on the activation values \u2014 creating potential for gradient instability, especially in early training before activations have stabilized."
    },
    // Step 8: MC \u2014 Pre-LN vs Post-LN
    {
      type: "mc",
      question: "A researcher trains two identical 48-layer transformers: one with Pre-LN and one with Post-LN. Both use the same hyperparameters \u2014 peak learning rate $3 \\times 10^{-4}$, cosine schedule, NO warmup. What outcome is most likely?",
      options: [
        "Both models train stably because 48 layers is below the critical depth threshold where Pre-LN and Post-LN diverge in behavior, which only matters above 100 layers",
        "The Post-LN model converges to a better final loss because Post-LN provides stronger normalization at every step, and the lack of warmup only affects training speed, not final quality",
        "Both models diverge because $3 \\times 10^{-4}$ is too high for any 48-layer model without warmup, regardless of normalization placement",
        "The Pre-LN model trains stably while the Post-LN model likely diverges in the first few hundred steps due to gradient instability, since Post-LN requires warmup to handle the poorly conditioned early gradients"
      ],
      correct: 3,
      explanation: "This is the key practical difference. Xiong et al. (2020) showed that Pre-LN removes the need for learning rate warmup. The clean gradient path through the unnormalized residual stream provides stable gradients from the start. Post-LN routes gradients through LayerNorm operations at every step, and in early training \u2014 when activations are poorly conditioned \u2014 this can cause gradient explosions. Without warmup, the Post-LN model is very likely to diverge. The $3 \\times 10^{-4}$ learning rate is typical for LLM training but too aggressive for Post-LN without warmup."
    },
    // Step 9: Pre-LN residual growth and the final norm
    {
      type: "info",
      title: "The Final LayerNorm: Taming the Residual Stream",
      content: "Pre-LN has a subtle consequence: since the residual stream is never directly normalized, its magnitude can grow across layers. Each sub-layer adds contributions, and while initialization scaling controls the initial growth, during training the contributions can increase.\n\nTo handle this, Pre-LN architectures add a **final LayerNorm** (or RMSNorm) after the last transformer block, before the output projection:\n\n$$\\text{logits} = W_{\\text{unembed}} \\cdot \\text{LN}(x_{\\text{final}})$$\n\nThis final norm ensures the input to the unembedding matrix has consistent scale, regardless of how much the residual stream grew. Without it, the logit magnitudes would be unpredictable, making the softmax temperature effectively uncontrolled.\n\nPost-LN doesn't need this final norm because every residual addition is already followed by normalization. This is one reason Post-LN sometimes achieves slightly better final quality in careful experiments \u2014 every layer sees properly normalized inputs.\n\nSome recent architectures (like DeepNorm from Microsoft) attempt to combine Post-LN's quality benefits with Pre-LN's training stability, using carefully tuned residual scaling factors."
    },
    // Step 10: MC \u2014 final norm and practical choices
    {
      type: "mc",
      question: "In Pre-LN, the residual stream is never directly normalized \u2014 only the sub-layer inputs are. What potential issue does this create, and how is it addressed in practice?",
      options: [
        "The unnormalized residual stream causes numerical overflow in float16, requiring all transformers to use float32 for the residual path while keeping sub-layer computations in float16",
        "The residual stream activations can grow in magnitude across layers since each sub-layer adds to the stream without normalization \u2014 this is addressed by the GPT-2 initialization scaling of $1/\\sqrt{2L}$ on output projections and by a final LayerNorm before the output head",
        "The residual stream values drift toward infinity during long sequence generation, which is why Pre-LN transformers cannot handle sequences longer than their training length without encountering NaN values",
        "Pre-LN has no issues with residual stream magnitude because the LayerNorm on sub-layer inputs implicitly constrains the sub-layer outputs, which prevents the stream from growing"
      ],
      correct: 1,
      explanation: "With $2L$ sub-layers each adding contributions to the stream, the variance of the residual stream can grow as $O(L)$ across layers. Two mechanisms control this: (1) Output projections in attention and FFN are initialized with std $\\propto 1/\\sqrt{2L}$, so each contribution has variance $\\propto 1/(2L)$ and the total variance remains $O(1)$ at initialization. (2) A final LayerNorm before the output head normalizes the final stream values before computing logits. During training, the model also implicitly learns to control stream magnitude through the learned parameters."
    },
    // Step 11: QK-Norm and modern variants
    {
      type: "info",
      title: "Beyond Pre-LN: Modern Normalization Strategies",
      content: "The normalization story doesn't end at Pre-LN + RMSNorm. Modern architectures add normalization at additional points:\n\n**QK-Norm** (Dehghani et al., 2023): Applies normalization to the query and key vectors *before* computing attention scores:\n\n$$\\text{Attn}(Q, K, V) = \\text{softmax}\\left(\\frac{\\text{LN}(Q) \\cdot \\text{LN}(K)^T}{\\sqrt{d_k}}\\right) V$$\n\nWhy? Without QK-Norm, the dot product $q \\cdot k$ can grow unboundedly as the model trains, causing attention logits to become very large. This makes the softmax sharpen to near-one-hot distributions, effectively \"collapsing\" attention. QK-Norm prevents this by keeping query and key magnitudes controlled.\n\nThis was critical for training very large ViTs (22B parameters) and is now used in several LLM architectures to improve training stability at scale.\n\n**Sandwich Norm**: Some architectures apply normalization both before and after the sub-layer, providing extra stability at the cost of additional compute.\n\nThe general principle: normalization is a tool for controlling the scale of intermediate representations. Where instability occurs, adding normalization can help \u2014 but each additional norm slightly constrains the model's representational freedom."
    },
    // Step 12: MC \u2014 modern normalization
    {
      type: "mc",
      question: "A team training a 70B parameter model observes that attention entropy (measuring how spread out the attention weights are) steadily decreases during training until many heads attend to only 1-2 tokens. What normalization technique most directly addresses this?",
      options: [
        "Switching from Pre-LN to Post-LN, which normalizes after each residual addition and prevents the attention logits from growing during the forward pass",
        "Increasing the $\\epsilon$ in RMSNorm from $10^{-6}$ to $10^{-3}$, which adds more noise to the normalization and prevents any single feature from dominating the attention computation",
        "Applying QK-Norm \u2014 normalizing queries and keys before the dot product prevents attention logits from growing unboundedly, keeping the softmax from collapsing to near-one-hot distributions",
        "Doubling the number of attention heads, which forces each head to specialize on fewer patterns and naturally prevents any single head from collapsing to attending one token"
      ],
      correct: 2,
      explanation: "Attention entropy collapse occurs when $q \\cdot k$ values grow large, causing softmax to produce near-one-hot distributions. This happens because query and key vectors can grow in magnitude during training without any constraint. QK-Norm directly addresses the root cause: by normalizing $Q$ and $K$ before the dot product, the attention logits are bounded regardless of training dynamics. Pre-LN normalizes the *input* to the attention sub-layer but doesn't prevent $W_Q$ and $W_K$ from producing large-magnitude outputs. Increasing $\\epsilon$ would not meaningfully affect the query/key magnitudes."
    }
  ]
};
