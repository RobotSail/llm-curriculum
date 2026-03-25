// Focused module: Adam optimizer from first principles.
// Covers momentum, adaptive learning rates, bias correction, memory footprint,
// and the fundamental per-element limitation that motivates matrix-aware optimizers.

export const adamLearning = {
  id: "0.3-adam-learning-easy",
  sectionId: "0.3",
  title: "Adam Optimizer: First Principles",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Momentum: Smoothing the Gradient Signal",
      content: "SGD updates parameters using the raw gradient: $\\theta_{t+1} = \\theta_t - \\alpha g_t$. The problem is that gradients on individual mini-batches are **noisy** — they point roughly toward the minimum but jitter around the true direction.\n\n**Momentum** fixes this by maintaining a running average of past gradients:\n\n$$m_t = \\beta_1 \\, m_{t-1} + (1 - \\beta_1) \\, g_t$$\n$$\\theta_{t+1} = \\theta_t - \\alpha \\, m_t$$\n\nThe parameter $\\beta_1$ (typically 0.9) controls how much history to keep. $m_t$ is an **exponential moving average (EMA)** of gradients — recent gradients count more, but the entire history contributes.\n\nMomentum has two effects:\n1. **Noise reduction**: Random jitter in $g_t$ averages out across steps\n2. **Acceleration**: Consistent gradient directions accumulate velocity, speeding up traversal of flat regions"
    },
    {
      type: "mc",
      question: "With momentum $\\beta_1 = 0.9$, roughly how many recent steps contribute meaningfully to $m_t$?",
      options: [
        "About 2 steps — the EMA forgets very quickly with $\\beta_1 < 1$",
        "About 10 steps — the effective window of an EMA with decay $\\beta_1$ is approximately $1/(1 - \\beta_1)$",
        "About 90 steps — $\\beta_1 = 0.9$ means 90% of steps are retained",
        "All previous steps contribute equally because the EMA never fully discards history"
      ],
      correct: 1,
      explanation: "The effective window of an EMA with decay $\\beta_1$ is $1/(1 - \\beta_1) = 1/0.1 = 10$ steps. After 10 steps, a gradient's contribution has decayed to roughly $0.9^{10} \\approx 0.35$, and after 20 steps to $0.9^{20} \\approx 0.12$. So the last ~10 steps dominate, older ones fade rapidly."
    },
    {
      type: "info",
      title: "Adaptive Learning Rates: The Second Moment",
      content: "Different parameters may need different learning rates. A parameter with consistently large gradients is in a steep region — it needs a smaller step. A parameter with small gradients is in a flat region — it needs a larger step.\n\nAdam tracks a **second moment** estimate (EMA of squared gradients) to adapt per-parameter:\n\n$$v_t = \\beta_2 \\, v_{t-1} + (1 - \\beta_2) \\, g_t^2$$\n\nThe update then divides by $\\sqrt{v_t}$:\n\n$$\\theta_{t+1} = \\theta_t - \\alpha \\frac{m_t}{\\sqrt{v_t} + \\epsilon}$$\n\nParameters with large $v_t$ (steep, noisy) get smaller effective steps. Parameters with small $v_t$ (flat, quiet) get larger effective steps. The $\\epsilon$ (typically $10^{-8}$) prevents division by zero.\n\nThis is **per-parameter** — each scalar entry $\\theta_i$ has its own $m_i$ and $v_i$. Adam treats every parameter independently."
    },
    {
      type: "mc",
      question: "In a transformer, the output projection matrix has gradients 100x larger than a mid-layer attention matrix. Under Adam, how do their effective learning rates compare?",
      options: [
        "The output projection gets a ~10x larger effective learning rate because Adam amplifies large gradient signals",
        "Both get approximately the same effective learning rate because Adam's $1/\\sqrt{v_t}$ scaling cancels out gradient magnitude differences",
        "The output projection gets a ~10x smaller effective learning rate because $\\sqrt{v_t}$ scales with gradient magnitude",
        "The ratio depends entirely on $\\beta_2$ and cannot be determined from gradient magnitudes alone"
      ],
      correct: 2,
      explanation: "If gradients are 100x larger, $v_t \\propto g^2$ is $10{,}000$x larger, and $\\sqrt{v_t}$ is 100x larger. The effective step $\\alpha \\cdot m_t / \\sqrt{v_t}$ is roughly $100/100 = 1$x for the gradient signal, but the raw effective learning rate $\\alpha/\\sqrt{v_t}$ is 100x smaller for the large-gradient layer. Adam dampens parameters with large gradients and amplifies those with small gradients. This is roughly a 10x ratio after considering $\\sqrt{}$."
    },
    {
      type: "info",
      title: "Bias Correction",
      content: "There is a subtle initialization problem: $m_0 = 0$ and $v_0 = 0$. During the first few steps, both estimates are biased toward zero because they haven't accumulated enough history.\n\nAdam applies **bias correction**:\n\n$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n\nAt $t = 1$: $1 - \\beta_1^1 = 0.1$, so $\\hat{m}_1 = m_1 / 0.1 = 10 \\cdot m_1$ — this compensates for $m_1$ being almost entirely the first gradient (not a true average). As $t \\to \\infty$, $\\beta^t \\to 0$ and the correction vanishes.\n\nThe correction for $v_t$ is especially important because $\\beta_2 = 0.999$ makes the second moment estimate warm up very slowly — it takes ~1000 steps for $v_t$ to stabilize. Without bias correction, early training steps would have wildly inflated effective learning rates."
    },
    {
      type: "mc",
      question: "With $\\beta_2 = 0.999$, approximately how many training steps does it take for the second moment bias correction factor $1/(1 - \\beta_2^t)$ to be within 10% of 1?",
      options: [
        "About 1,000 steps",
        "About 100 steps",
        "About 10 steps",
        "About 10,000 steps"
      ],
      correct: 0,
      explanation: "We need $1/(1 - 0.999^t) < 1.1$, which means $0.999^t < 1/11 \\approx 0.09$. Taking logs: $t \\cdot \\ln(0.999) < \\ln(0.09)$, so $t > \\ln(0.09)/\\ln(0.999) \\approx -2.41 / -0.001 \\approx 2400$. More approximately, the effective window is $1/(1-0.999) = 1000$ steps, and the correction stabilizes around that scale. ~1,000 steps is the right order of magnitude."
    },
    {
      type: "info",
      title: "What Adam Misses: Matrix Structure",
      content: "Adam's fundamental design treats each parameter as an independent scalar. For a weight matrix $W \\in \\mathbb{R}^{m \\times n}$, Adam maintains $m \\times n$ independent first moments and $m \\times n$ independent second moments. It knows nothing about the relationships between entries.\n\nBut weight matrices in transformers have **geometric meaning**: rows correspond to output neuron directions, columns to input neuron directions. The gradient $G = \\nabla_W L$ is also a matrix whose singular value decomposition $G = U\\Sigma V^T$ reveals which input→output direction mappings the loss wants to change most.\n\nAdam processes $G_{ij}$ independently, so it cannot reason about this directional structure. Concretely:\n- If the top singular direction of $G$ has value 100 and the bottom has value 0.01, Adam scales them separately but does NOT equalize them\n- Two correlated columns of $G$ (indicating a common input direction) are processed independently\n- The effective update has **no guarantee** of being well-conditioned as a matrix\n\nThis limitation motivated optimizers like Shampoo and **Muon** that operate on the matrix structure of gradients rather than individual entries."
    },
    {
      type: "mc",
      question: "A gradient matrix $G$ has singular values $[100, 50, 1, 0.01]$. After Adam processes it (assuming $v_t$ has converged), the effective update for each entry is approximately $\\text{sign}(G_{ij})$. What are the approximate singular values of Adam's update matrix?",
      options: [
        "$[1, 1, 1, 1]$ — Adam equalizes all singular values since $\\text{sign}$ has uniform magnitude",
        "$[100, 50, 1, 0.01]$ — Adam preserves the original singular value structure",
        "$[10, 7.1, 1, 0.1]$ — Adam takes the square root of each singular value",
        "They depend on the specific entries of $G$, not just its singular values, because Adam operates element-wise"
      ],
      correct: 3,
      explanation: "This is the key insight. Adam's element-wise $\\text{sign}(G_{ij})$ produces a matrix of $\\pm 1$ entries, but the singular values of a sign matrix depend on the **pattern** of signs, not on the original singular values. The sign matrix's spectral structure is essentially uncontrolled — it depends on which entries of $G$ are positive vs negative, which has no simple relationship to $G$'s SVD. This is fundamentally different from Muon, which explicitly controls the spectral structure."
    },
    {
      type: "info",
      title: "Adam's Memory Footprint",
      content: "For a model with $N$ parameters, Adam/AdamW stores:\n\n- **First moment** $m$: $N$ floats (same size as the model)\n- **Second moment** $v$: $N$ floats (same size as the model)\n- **Parameters** $\\theta$: $N$ floats\n\nTotal optimizer state: $2N$ floats. For a 70B parameter model in float32, that is $70B \\times 2 \\times 4$ bytes $= 560$ GB just for optimizer state, plus 280 GB for the model itself.\n\nThis is why large-scale training uses mixed precision: parameters and optimizer states in float32 (\"master weights\"), forward/backward pass in bfloat16. Even so, Adam's $2N$ overhead is the dominant memory cost at scale.\n\nMuon replaces the two-buffer system with a single momentum buffer for most parameters, cutting optimizer memory by roughly 35-40%. At 70B scale, that saves ~200 GB — enough to matter for what fits on a training cluster."
    },
    {
      type: "mc",
      question: "A team trains a 13B model with AdamW in float32. The optimizer state alone requires ~104 GB ($13B \\times 2 \\times 4$ bytes). They switch to Muon, which uses one momentum buffer for the 98% of parameters that are 2D matrices and Adam for the remaining 2%. Approximately how much optimizer memory is needed?",
      options: [
        "~104 GB — Muon needs the same total memory but distributes it differently",
        "~54 GB — one buffer for Muon parameters plus two buffers for the 2% using Adam",
        "~26 GB — Muon uses half-precision for its momentum buffer",
        "~78 GB — Muon saves one buffer but adds Newton-Schulz intermediate storage"
      ],
      correct: 1,
      explanation: "Muon parameters (98%): $0.98 \\times 13B \\times 1 \\times 4 = 51$ GB (one momentum buffer). Adam parameters (2%): $0.02 \\times 13B \\times 2 \\times 4 = 2.1$ GB (two buffers). Total: ~53 GB, approximately half of Adam's 104 GB. Newton-Schulz intermediates are temporary and don't persist in optimizer state."
    }
  ]
};
