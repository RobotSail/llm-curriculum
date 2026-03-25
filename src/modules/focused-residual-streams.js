// Focused learning module: Residual Connections and Layer Normalization
// Section 1.1: Transformer Architecture
// Covers: residual connections as skip paths, the residual stream view,
// LayerNorm mechanics, Pre-LN vs Post-LN, and gradient flow.
// Grounded in Goodfellow et al. (2016) Ch. 8.7.1 (skip connections),
// and He et al. (2016) for residual networks.

export const residualStreamsLearning = {
  id: "1.1-residual-streams-learning-easy",
  sectionId: "1.1",
  title: "Residual Connections and Layer Normalization",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: The depth problem
    {
      type: "info",
      title: "The Depth Problem: Why Deep Networks Are Hard to Train",
      content: "Stacking more layers should make a network more expressive — deeper networks can represent more complex functions. But in practice, naively stacking layers causes training to fail.\n\nThe core issue is **gradient flow**. During backpropagation, gradients pass through every layer in reverse. In a network with $L$ layers, the gradient of the loss with respect to layer $l$'s parameters involves a product of $L - l$ Jacobian matrices:\n\n$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta_l} \\propto \\prod_{j=l+1}^{L} \\frac{\\partial h_j}{\\partial h_{j-1}}$$\n\nIf these Jacobians consistently have spectral norm $> 1$, gradients **explode** exponentially. If $< 1$, gradients **vanish** exponentially. Either way, early layers receive useless gradient signals.\n\nFor a 96-layer transformer (like GPT-3), gradients must flow through 96 such products. Without mitigation, training is essentially impossible — early layers would receive gradients that are either astronomically large or indistinguishably close to zero (Goodfellow et al., 2016, §8.2.5)."
    },
    // Step 2: MC — gradient flow
    {
      type: "mc",
      question: "A 48-layer network without residual connections has layer Jacobians with average spectral norm 0.95. What is the approximate magnitude of the gradient signal reaching layer 1 relative to layer 48?",
      options: [
        "About $0.95 \\times 48 \\approx 46$ — linear decay with depth, so the gradient is roughly half its original magnitude",
        "About $0.95^{48} \\approx 0.085$ — the gradient is reduced to about 8.5% of its value at the last layer, making early-layer learning very slow",
        "About $0.95^{48} \\approx 0.085$, but adaptive optimizers like Adam fully compensate for this by normalizing per-parameter gradients, so it has no practical effect",
        "Exactly $0.95$ regardless of depth, because each layer independently scales the gradient and the total effect is determined by the worst single layer"
      ],
      correct: 1,
      explanation: "The gradient magnitude scales as $\\prod_{j} \\|J_j\\| \\approx 0.95^{47} \\approx 0.085$. This is the vanishing gradient problem: even with spectral norms close to 1, the exponential product over many layers causes severe attenuation. Adam helps by adapting per-parameter (dividing by $\\sqrt{v_t}$), but it cannot fully compensate — if the raw gradient is near zero, Adam's estimate of $v_t$ is also near zero, and the ratio can be noisy or unstable. Residual connections are the structural solution."
    },
    // Step 3: Residual connections
    {
      type: "info",
      title: "Residual Connections: The Identity Shortcut",
      content: "A **residual connection** (He et al., 2016) adds a skip path that bypasses each sub-layer:\n\n$$h_{l+1} = h_l + f_l(h_l)$$\n\ninstead of $h_{l+1} = f_l(h_l)$, where $f_l$ is the sub-layer's computation (attention or FFN).\n\nThis simple change has a profound effect on gradient flow. The Jacobian of the residual block is:\n\n$$\\frac{\\partial h_{l+1}}{\\partial h_l} = I + \\frac{\\partial f_l}{\\partial h_l}$$\n\nThe identity matrix $I$ guarantees that gradients can flow through the skip path unchanged, regardless of what $f_l$ does. Even if $\\frac{\\partial f_l}{\\partial h_l} \\approx 0$ (the sub-layer produces negligible gradients), the gradient still passes through via the identity path.\n\nOver $L$ layers, the gradient includes a **direct path** from the loss back to any layer — the product of identity matrices, which is just $I$. Each sub-layer can add or subtract from this gradient, but it cannot block it.\n\nIn transformers, there are **two residual connections per layer**: one around the attention sub-layer and one around the FFN sub-layer. A 32-layer transformer thus has 64 residual additions."
    },
    // Step 4: MC — residual mechanics
    {
      type: "mc",
      question: "In the residual connection $h_{l+1} = h_l + f_l(h_l)$, the sub-layer $f_l$ learns to compute a **correction** to the input $h_l$. At initialization, $f_l$ outputs near-zero values (due to small random weights). What does this mean for the network's behavior at the start of training?",
      options: [
        "The network is effectively a random function because the small random outputs from each $f_l$ accumulate across layers, creating a random mapping from input to output",
        "The network approximately implements the identity function — each layer passes its input through nearly unchanged, and the model output is close to a simple projection of the input embeddings",
        "The network cannot learn because the near-zero sub-layer outputs produce near-zero gradients, creating a chicken-and-egg problem where the network is stuck at initialization",
        "The network behaves like a single-layer model because only the last sub-layer has non-negligible output, and all earlier layers are effectively bypassed"
      ],
      correct: 1,
      explanation: "With $f_l \\approx 0$ at initialization, $h_{l+1} \\approx h_l + 0 = h_l$. The input representations flow through all layers nearly unchanged — the deep network starts as an approximate identity function. This is beneficial: the model begins from a well-behaved starting point (identity) and gradually learns layer-by-layer corrections. Without residual connections, the initialized network would apply a product of random transformations, producing chaotic outputs. The residual structure means the network can never be *worse* than identity at initialization."
    },
    // Step 5: The residual stream
    {
      type: "info",
      title: "The Residual Stream: A Shared Communication Bus",
      content: "Anthropic's \"circuits\" framework provides an elegant reinterpretation of residual connections. Instead of thinking of layers as sequential processors, view the $d_{\\text{model}}$-dimensional vector as a **residual stream** — a shared bus that flows through the entire network.\n\nEach sub-layer **reads** from and **writes** to this stream:\n\n$$x_0 \\xrightarrow{+\\text{attn}_1} x_1 \\xrightarrow{+\\text{ffn}_1} x_2 \\xrightarrow{+\\text{attn}_2} x_3 \\xrightarrow{+\\text{ffn}_2} x_4 \\cdots$$\n\nwhere $x_{2l+1} = x_{2l} + \\text{attn}_l(x_{2l})$ and $x_{2l+2} = x_{2l+1} + \\text{ffn}_l(x_{2l+1})$.\n\nThe output at the final layer is the sum of all contributions:\n\n$$x_{\\text{final}} = x_0 + \\sum_{l=1}^{L} \\text{attn}_l(\\cdot) + \\sum_{l=1}^{L} \\text{ffn}_l(\\cdot)$$\n\nThis means **every sub-layer can directly influence the output** — not through a chain of intermediaries, but by writing to a stream that is read by the final projection. Sub-layers that are later in the network don't have privileged access; they simply write last.\n\nThis also means sub-layers can communicate: attention in layer 5 can read what the FFN in layer 3 wrote to the stream, enabling multi-step computations composed from independent modules."
    },
    // Step 6: MC — residual stream reasoning
    {
      type: "mc",
      question: "In the residual stream view, the final output is $x_{\\text{final}} = x_0 + \\sum_l \\text{attn}_l(\\cdot) + \\sum_l \\text{ffn}_l(\\cdot)$. If you could ablate (set to zero) the contribution of a single attention head in layer 10, what would happen?",
      options: [
        "All subsequent layers would receive corrupted input, causing a cascade of failures that completely destroys the model's output quality",
        "Only the direct effect of that head is removed from the final output, but indirect effects (where later layers used that head's output) would also be affected — both direct and indirect contributions are lost",
        "Nothing would change because individual attention heads have negligible impact — the model has hundreds of heads and is robust to single-head removal",
        "Only the output of layer 10 changes — layers 11+ are computed from scratch using the new residual stream and are unaffected by the original head's contribution"
      ],
      correct: 1,
      explanation: "Ablating a head removes two types of contribution: (1) its **direct effect** — what it wrote to the stream that the final projection reads, and (2) its **indirect effects** — what later sub-layers read from the stream that included that head's contribution. A head that computes useful features may be relied upon by FFN layers in later blocks. In practice, importance varies enormously: some heads have large direct and indirect effects (e.g., induction heads), while others can be removed with minimal impact. This is the basis for attention head pruning."
    },
    // Step 7: Layer Normalization
    {
      type: "info",
      title: "Layer Normalization: Controlling Activation Magnitude",
      content: "Residual connections solve gradient flow but create a new problem: each sub-layer **adds** to the stream, so activation magnitudes can grow without bound across layers.\n\n**Layer Normalization** (Ba et al., 2016) stabilizes activations by normalizing each vector to zero mean and unit variance, then applying a learned affine transform:\n\n$$\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sigma + \\epsilon} + \\beta$$\n\nwhere $\\mu = \\frac{1}{d}\\sum_{i=1}^{d} x_i$ and $\\sigma = \\sqrt{\\frac{1}{d}\\sum_{i=1}^{d}(x_i - \\mu)^2}$ are computed per-token (across the feature dimension), and $\\gamma, \\beta \\in \\mathbb{R}^d$ are learned scale and shift parameters.\n\nKey properties:\n- **Per-token normalization**: Unlike batch normalization, LayerNorm normalizes across features for each individual token, making it independent of batch size and other sequences\n- **Removes magnitude information**: After normalization, only the **direction** of the activation vector matters — the model must encode information in relative feature magnitudes, not absolute scale\n- **Learned affine transform**: $\\gamma$ and $\\beta$ let the model recover any scale and shift it finds useful, so normalization doesn't permanently destroy information"
    },
    // Step 8: MC — LayerNorm mechanics
    {
      type: "mc",
      question: "RMSNorm (Zhang & Sennrich, 2019), used in LLaMA and most modern LLMs, simplifies LayerNorm by removing the mean-centering step: $\\text{RMSNorm}(x) = \\gamma \\odot \\frac{x}{\\text{RMS}(x)}$ where $\\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum_i x_i^2}$. Why is this simplification effective?",
      options: [
        "The mean of hidden activations is always exactly zero in transformers due to the symmetric initialization, so subtracting the mean is a no-op that wastes compute",
        "Removing the mean subtraction reduces the parameter count by half (no $\\beta$ needed), and the model can achieve the same effect by using the bias terms in the preceding linear layers",
        "The re-centering in standard LayerNorm is empirically unnecessary — the model's representational power comes primarily from the scale normalization (dividing by the norm), and removing mean subtraction saves compute with no measurable quality loss",
        "RMSNorm is mathematically equivalent to standard LayerNorm when the learned shift $\\beta = 0$, which is what standard LayerNorm converges to anyway during training"
      ],
      correct: 2,
      explanation: "Zhang & Sennrich (2019) showed empirically that the re-centering (mean subtraction) contributes minimally to LayerNorm's effectiveness. The key ingredient is scale normalization — projecting activations onto the unit hypersphere. RMSNorm removes the mean computation and the learned bias $\\beta$, saving both compute and parameters. In practice, LLaMA showed that RMSNorm produces equivalent model quality to full LayerNorm across model scales from 7B to 65B. The compute savings are modest per layer but meaningful across 80+ layers and millions of training steps."
    },
    // Step 9: Pre-LN vs Post-LN
    {
      type: "info",
      title: "Pre-LN vs Post-LN: Where to Normalize",
      content: "The placement of LayerNorm relative to the residual connection has significant implications:\n\n**Post-LN** (original Transformer):\n$$x_{l+1} = \\text{LN}(x_l + f_l(x_l))$$\n\nNormalization happens **after** the residual addition. The raw sub-layer output is added to the stream, and then the combined result is normalized.\n\n**Pre-LN** (modern standard):\n$$x_{l+1} = x_l + f_l(\\text{LN}(x_l))$$\n\nNormalization happens **before** the sub-layer processes its input. The residual addition bypasses the normalization entirely.\n\nThe critical difference: in Pre-LN, the residual stream itself is **never normalized** — only the input to each sub-layer is. This means gradients flowing through the skip path pass through no normalization operations, providing a cleaner gradient highway.\n\nIn Post-LN, gradients must pass through a LayerNorm at every step, and the Jacobian of LayerNorm depends on the activation values — creating potential for gradient instability, especially in early training before activations have stabilized.\n\nXiong et al. (2020) showed that Pre-LN allows stable training without learning rate warmup, while Post-LN requires careful warmup to avoid early divergence. Nearly all modern LLMs use Pre-LN (with a final LayerNorm before the output projection)."
    },
    // Step 10: MC — Pre-LN vs Post-LN
    {
      type: "mc",
      question: "In Pre-LN, the residual stream is never directly normalized — only the sub-layer inputs are. What potential issue does this create, and how is it addressed in practice?",
      options: [
        "The residual stream activations can grow in magnitude across layers since each sub-layer adds to the stream without normalization — this is addressed by the GPT-2 initialization scaling of $1/\\sqrt{2L}$ on output projections and by a final LayerNorm before the output head",
        "The unnormalized residual stream causes numerical overflow in float16, requiring all transformers to use float32 for the residual path while keeping sub-layer computations in float16",
        "The residual stream values drift toward infinity during long sequence generation, which is why Pre-LN transformers cannot handle sequences longer than their training length without encountering NaN values",
        "Pre-LN has no issues with residual stream magnitude because the LayerNorm on sub-layer inputs implicitly constrains the sub-layer outputs, which prevents the stream from growing"
      ],
      correct: 0,
      explanation: "With $2L$ sub-layers each adding contributions to the stream, the variance of the residual stream can grow as $O(L)$ across layers. Two mechanisms control this: (1) Output projections in attention and FFN are initialized with std $\\propto 1/\\sqrt{2L}$, so each contribution has variance $\\propto 1/(2L)$ and the total variance remains $O(1)$ at initialization. (2) A final LayerNorm before the output head normalizes the final stream values before computing logits. During training, the model also implicitly learns to control stream magnitude through the learned parameters."
    },
    // Step 11: Integration
    {
      type: "info",
      title: "The Complete Transformer Block",
      content: "Putting it all together, a Pre-LN transformer block processes each token as:\n\n$$a_l = x_l + \\text{MultiHeadAttn}(\\text{LN}_1(x_l))$$\n$$x_{l+1} = a_l + \\text{FFN}(\\text{LN}_2(a_l))$$\n\nThe information flow in a single block:\n\n1. **LayerNorm** normalizes the residual stream for stable processing\n2. **Multi-head attention** reads from all positions and writes inter-token information\n3. **Residual add** merges attention output back into the stream\n4. **LayerNorm** normalizes again for the FFN\n5. **FFN** processes each position independently (per-token computation)\n6. **Residual add** merges FFN output back into the stream\n\nAfter $L$ blocks, a **final LayerNorm** normalizes the stream, and a **linear projection** (the \"unembedding\" or \"LM head\") maps from $d_{\\text{model}}$ to vocabulary size $V$.\n\nThis architecture — residual connections for gradient flow, LayerNorm for activation stability, attention for inter-token communication, FFN for per-token processing — has proven remarkably robust across scales from 100M to 400B+ parameters."
    },
    // Step 12: MC — integration question
    {
      type: "mc",
      question: "A researcher trains two identical 48-layer transformers: one with Pre-LN and one with Post-LN. Both use the same hyperparameters — peak learning rate $3 \\times 10^{-4}$, cosine schedule, NO warmup. What outcome is most likely?",
      options: [
        "Both models train stably because 48 layers is below the critical depth threshold where Pre-LN and Post-LN diverge in behavior, which only matters above 100 layers",
        "The Pre-LN model trains stably while the Post-LN model likely diverges in the first few hundred steps due to gradient instability, since Post-LN requires warmup to handle the poorly conditioned early gradients",
        "The Post-LN model converges to a better final loss because Post-LN provides stronger normalization at every step, and the lack of warmup only affects training speed, not final quality",
        "Both models diverge because $3 \\times 10^{-4}$ is too high for any 48-layer model without warmup, regardless of normalization placement"
      ],
      correct: 1,
      explanation: "This is the key practical difference. Pre-LN was specifically shown by Xiong et al. (2020) to remove the need for learning rate warmup. The clean gradient path through the unnormalized residual stream provides stable gradients from the start. Post-LN routes gradients through LayerNorm operations at every step, and in early training — when activations are poorly conditioned — this can cause gradient explosions. Without warmup, the Post-LN model is very likely to diverge. The $3 \\times 10^{-4}$ learning rate is typical for LLM training but too aggressive for Post-LN without warmup."
    }
  ]
};
