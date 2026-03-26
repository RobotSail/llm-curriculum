// Focused learning module: Training Instabilities in Large Language Models
// Section 1.3: Pretraining Objectives & Dynamics
// Covers: loss spikes, gradient explosions, learning rate warmup, gradient clipping,
// the role of initialization, and practical mitigation strategies.
// Single-concept module building from first principles.
// Grounded in Goodfellow et al. (2016) Ch. 8 (optimization) and Ch. 10 (sequence models).

export const trainingInstabilitiesLearning = {
  id: "1.3-training-instabilities-learning-easy",
  sectionId: "1.3",
  title: "Training Instabilities in LLM Pretraining",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why training instabilities matter
    {
      type: "info",
      title: "The Stakes: When Training Goes Wrong",
      content: "Training a large language model costs millions of dollars in compute. A single training run for a frontier model might use thousands of GPUs for weeks or months. If the loss suddenly spikes or diverges midway through, you face a painful choice: restart from a checkpoint (losing days of compute) or try to recover (which may not work).\n\nThese **training instabilities** — sudden loss spikes, gradient explosions, or slow divergence — are not rare edge cases. They are a central practical challenge in LLM pretraining. The PaLM paper (Chowdhery et al., 2022) reported dozens of loss spikes during training that required manual intervention. OPT-175B's training log documented frequent instabilities requiring restarts.\n\nAs Goodfellow et al. (2016, Ch. 8) emphasize, optimization of deep neural networks is fundamentally different from convex optimization. The loss landscape is non-convex, with saddle points, sharp minima, and regions where the curvature changes dramatically. Understanding what causes instabilities — and how to prevent them — is essential practical knowledge for anyone training large models."
    },
    // Step 2: MC — stakes
    {
      type: "mc",
      question: "A team training a 70B parameter model on 8,192 GPUs observes a sudden loss spike at step 50,000. They restart from their most recent checkpoint at step 48,000. Assuming 2 minutes per training step, approximately how much compute is lost?",
      options: [
        "About 33 hours — 2,000 steps × 2 minutes, but only on a single GPU since checkpoints save all other GPU states",
        "About 67 hours — 2,000 steps × 2 minutes × 8,192 GPUs worth of compute, but wall-clock time lost is only ~67 minutes",
        "About 2 minutes — checkpoints save the exact optimizer state, so only the single step that caused the spike is lost",
        "About 33 hours of wall-clock time — 2,000 steps × 2 minutes — and that time multiplied by 8,192 GPUs worth of compute is wasted"
      ],
      correct: 3,
      explanation: "Restarting from step 48,000 means re-doing steps 48,000 through 50,000. That's 2,000 steps × 2 min/step = 4,000 minutes ≈ 67 hours of wall-clock time. But since all 8,192 GPUs were running in parallel, the total compute wasted is 67 hours × 8,192 GPUs. The wall-clock loss alone is ~67 hours (correcting the arithmetic: 2,000 × 2 min = 4,000 min ≈ 67 hours). This illustrates why preventing instabilities is critical — each spike can cost days of wall-clock time and enormous compute budgets."
    },
    // Step 3: Gradient explosion and vanishing
    {
      type: "info",
      title: "Gradient Explosion: The Root Cause",
      content: "The fundamental source of training instabilities is the **gradient explosion problem**. In a deep network with $L$ layers, backpropagation computes the gradient of the loss with respect to early-layer parameters by multiplying Jacobians:\n\n$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta_1} = \\frac{\\partial \\mathcal{L}}{\\partial h_L} \\cdot \\frac{\\partial h_L}{\\partial h_{L-1}} \\cdot \\frac{\\partial h_{L-1}}{\\partial h_{L-2}} \\cdots \\frac{\\partial h_2}{\\partial h_1} \\cdot \\frac{\\partial h_1}{\\partial \\theta_1}$$\n\nThis is a product of $L$ matrices. If the typical spectral norm of each Jacobian $\\left\\|\\frac{\\partial h_{l+1}}{\\partial h_l}\\right\\|$ is slightly greater than 1 — say 1.01 — then after 100 layers the product grows as $1.01^{100} \\approx 2.7$. Manageable. But if it's 1.1, then $1.1^{100} \\approx 13{,}781$. The gradients explode.\n\nAs Goodfellow et al. (Ch. 8.2.4) explain, the gradient can grow exponentially with depth. For transformers, which routinely have 32-128 layers, even a small per-layer amplification factor can cause catastrophic gradient magnitudes. This is why the design of every component — initialization, normalization, residual connections — must carefully control gradient flow."
    },
    // Step 4: MC — gradient explosion
    {
      type: "mc",
      question: "A 96-layer transformer has per-layer Jacobian spectral norms averaging 1.05. The gradient magnitude at the last layer is 1.0. What is the approximate gradient magnitude at the first layer?",
      options: [
        "About 96 × 1.05 = 100.8 — the gradient grows linearly with depth times the amplification factor",
        "About $1.05^{96} \\approx 115$ — the gradient grows exponentially because backpropagation multiplies per-layer Jacobians",
        "About 1.0 — residual connections in transformers perfectly normalize the gradient at each layer",
        "About $96^{1.05} \\approx 120$ — the gradient grows as a power law in the number of layers"
      ],
      correct: 1,
      explanation: "Backpropagation multiplies per-layer Jacobians, so the gradient at layer 1 is approximately $1.0 \\times 1.05^{96} \\approx 115$. This exponential growth is why even slightly-above-1.0 spectral norms are dangerous in deep networks. The growth is multiplicative (exponential in depth), not additive (linear). Residual connections help but do not perfectly normalize — they add an identity term that keeps the spectral norm close to 1, but deviations still compound."
    },
    // Step 5: Loss spikes in practice
    {
      type: "info",
      title: "Loss Spikes: What They Look Like",
      content: "In practice, gradient explosion manifests as **loss spikes** — sudden, dramatic increases in the training loss that can be 10-100× the normal value. The loss may recover on its own, or it may diverge permanently (the model \"blows up\").\n\nLoss spikes happen because:\n\n1. **Bad data batches**: A batch containing unusually long sequences, rare tokens, or corrupted data can produce anomalously large activations. These propagate through the network and create extreme gradients.\n\n2. **Sharp loss landscape regions**: The loss surface of a neural network is non-convex with varying curvature. The model may wander into a region where the loss surface is very steep (high curvature). A normal-sized gradient step in a high-curvature region overshoots dramatically.\n\n3. **Compounding numerical errors**: With mixed-precision training (FP16/BF16), small numerical errors can accumulate. A sequence of operations that individually are fine can produce intermediate values that overflow the representable range.\n\nThe PaLM team found that loss spikes correlated with specific data batches — skipping those batches during a restart prevented the spike from recurring. This suggests that data-triggered instabilities interact with the current model state: the same batch might be fine early in training but catastrophic later."
    },
    // Step 6: MC — loss spikes
    {
      type: "mc",
      question: "During training of a 65B model, a loss spike occurs at step 120,000. The team restarts from the step 119,500 checkpoint and skips the batch that was processed at step 120,000. The spike does not recur. What does this tell us?",
      options: [
        "The spike was caused purely by the data batch — any model encountering that batch at any training step would experience the same spike",
        "The spike resulted from an interaction between the specific data batch and the model's parameter state at that point — the batch triggered instability in a region of parameter space that the model happened to occupy",
        "The model's random seed was the cause — restarting with a different random state is what prevented the spike, not skipping the batch",
        "The spike was caused by a hardware error (bit flip) that corrupted the gradient computation for that specific batch"
      ],
      correct: 1,
      explanation: "If the batch alone were the cause, it would cause spikes whenever encountered. If the model state alone were the cause, skipping the batch wouldn't help — other batches would trigger the same spike at similar parameter states. The fact that skipping the specific batch at the specific step prevents recurrence shows it's an interaction: the model reached a region of parameter space where this particular batch produced extreme gradients. This is consistent with the non-convex loss landscape — certain parameter configurations are more vulnerable to certain input patterns."
    },
    // Step 7: Learning rate warmup
    {
      type: "info",
      title: "Learning Rate Warmup: Starting Gently",
      content: "One of the most important stability techniques is **learning rate warmup**: starting training with a very small learning rate and linearly increasing it over the first few thousand steps.\n\nWhy is warmup necessary? At initialization, the model's weights are random. The gradients computed on random weights are high-variance and point in directions that reflect the random initialization more than the true loss landscape geometry. Taking large steps based on these noisy early gradients can push the model into a bad region of parameter space from which it never recovers.\n\nMore precisely, the Adam optimizer (Kingma & Ba, 2015) maintains exponential moving averages of the first moment $m_t$ and second moment $v_t$:\n\n$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n\nAt the start of training, both $m_t$ and $v_t$ are initialized to zero and are heavily biased toward zero. Adam applies **bias correction**: $\\hat{m}_t = m_t / (1 - \\beta_1^t)$. But in the first few steps, $1 - \\beta_1^t$ is very small (e.g., $1 - 0.9^1 = 0.1$), so the bias-corrected estimate amplifies noise by up to $10\\times$.\n\nWarmup counteracts this: by keeping the learning rate small during the period when Adam's moment estimates are unreliable, it prevents the amplified noise from causing large destructive updates. Typical warmup schedules use 1,000-2,000 steps for large models."
    },
    // Step 8: MC — warmup
    {
      type: "mc",
      question: "A team trains a model with Adam ($\\beta_1 = 0.9$) and no warmup, jumping directly to the peak learning rate of $3 \\times 10^{-4}$. At step 1, Adam's bias correction divides $m_1$ by $(1 - 0.9^1) = 0.1$. What is the effective amplification of the gradient signal at step 1?",
      options: [
        "No amplification — Adam's denominator $\\sqrt{\\hat{v}_t}$ cancels out the numerator amplification exactly",
        "10× amplification of the first moment estimate, combined with a full-sized learning rate, produces an effective step that is far larger than intended",
        "0.1× attenuation — the bias correction shrinks the gradient to prevent early instability",
        "Exactly 1× — the bias correction is designed to produce an unbiased estimate, so the effective magnitude matches the true gradient moment"
      ],
      correct: 1,
      explanation: "At step 1, the bias-corrected first moment is $\\hat{m}_1 = m_1 / (1 - 0.9^1) = m_1 / 0.1 = 10 \\cdot m_1$. The correction is mathematically correct — it produces an unbiased estimate of the true mean gradient. But the *variance* of that estimate is also amplified by 10×. When combined with a full learning rate (no warmup), this high-variance, amplified estimate produces update steps that are effectively much larger than intended. Warmup reduces the learning rate during this period so the large effective multiplier doesn't cause destructive updates."
    },
    // Step 9: Gradient clipping
    {
      type: "info",
      title: "Gradient Clipping: A Safety Net",
      content: "**Gradient clipping** (Pascanu et al., 2013) is a direct defense against gradient explosion. Before applying the optimizer update, we compute the global gradient norm $\\|g\\| = \\sqrt{\\sum_i g_i^2}$ across all parameters, and if it exceeds a threshold $\\tau$, we rescale:\n\n$$g \\leftarrow g \\cdot \\frac{\\tau}{\\|g\\|} \\quad \\text{if } \\|g\\| > \\tau$$\n\nThis preserves the **direction** of the gradient but caps its magnitude. The gradient direction is the most reliable information — it tells us which way is downhill. The magnitude is the unreliable part — in regions of high curvature, the magnitude can be misleading.\n\nAs Goodfellow et al. (Ch. 10.11.1) explain, gradient clipping effectively adapts the learning rate based on local curvature. In flat regions (small gradient), the full learning rate is used. In steep regions (large gradient), clipping reduces the effective learning rate, preventing overshooting.\n\nTypical clipping thresholds for LLM training are $\\tau = 1.0$. The choice of $\\tau$ matters: too small and you slow down training by constantly clipping; too large and clipping never activates when you need it. Monitoring the **gradient norm** during training is standard practice — a sudden spike in gradient norm often precedes a loss spike by a few steps, giving an early warning."
    },
    // Step 10: MC — gradient clipping
    {
      type: "mc",
      question: "During training, the gradient norm suddenly jumps from its typical value of 0.5 to 25.0 (with clipping threshold $\\tau = 1.0$). After clipping, the gradient direction is preserved but the norm becomes 1.0. What is the effective learning rate reduction from clipping at this step?",
      options: [
        "The effective learning rate is reduced by a factor of 25× — from $\\eta$ to $\\eta/25$ — because the gradient magnitude is scaled down by $\\tau / \\|g\\| = 1/25$",
        "The effective learning rate is unchanged — clipping only affects the gradient direction, not the step size",
        "The effective learning rate is reduced by a factor of $\\sqrt{25} \\approx 5\\times$ because clipping operates on the squared gradient norm",
        "The effective learning rate drops to zero because clipping signals that the current step should be skipped entirely"
      ],
      correct: 0,
      explanation: "The update step is $\\theta \\leftarrow \\theta - \\eta \\cdot g_{\\text{clipped}}$. Without clipping, $\\|g\\| = 25$, so the step magnitude is $\\eta \\times 25$. With clipping, $\\|g_{\\text{clipped}}\\| = 1.0$, so the step magnitude is $\\eta \\times 1.0$. The effective step is 25× smaller — equivalent to temporarily dividing the learning rate by 25. This is exactly what we want: in steep regions (large gradients), take smaller steps. Clipping preserves the descent direction while preventing catastrophically large updates."
    },
    // Step 11: Initialization and normalization
    {
      type: "info",
      title: "Initialization: Setting the Stage for Stability",
      content: "Proper initialization is the first line of defense against instability. The goal is to ensure that, at the start of training, each layer neither amplifies nor attenuates its inputs — keeping the spectral norm of each layer's Jacobian close to 1.\n\nFor linear layers $h = Wx$, if $W \\in \\mathbb{R}^{m \\times n}$ has entries drawn i.i.d. from $\\mathcal{N}(0, \\sigma^2)$, the expected squared norm of the output is:\n\n$$\\mathbb{E}[\\|h\\|^2] = n \\sigma^2 \\cdot \\|x\\|^2$$\n\nTo keep $\\|h\\| \\approx \\|x\\|$, we need $\\sigma^2 = 1/n$. This is **Xavier/Glorot initialization** (Glorot & Bengio, 2010), designed for linear activations.\n\nFor transformers specifically, GPT-2 and subsequent models use a refined approach: the residual stream accumulates contributions from $L$ layers, so the output projection of each residual block is initialized with $\\sigma = \\frac{1}{\\sqrt{2L}}$ (the factor of 2 accounts for both attention and MLP sub-layers). This ensures the total variance contribution from all $L$ layers stays bounded.\n\nNormalization layers (**LayerNorm** or **RMSNorm**) provide a complementary safeguard: they explicitly re-normalize activations at each layer, preventing the gradual drift that initialization alone cannot eliminate over the course of training."
    },
    // Step 12: MC — initialization
    {
      type: "mc",
      question: "A 48-layer transformer uses the GPT-2 initialization scheme, scaling residual block output projections by $1/\\sqrt{2L}$. If this scaling were omitted (all layers initialized at standard Xavier scale), what would happen to the residual stream variance after all 48 layers?",
      options: [
        "The variance would stay approximately constant because LayerNorm at each layer renormalizes the activations regardless of the initialization scale",
        "The variance would grow proportionally to $2L = 96$, since each of the 96 sub-layers (48 attention + 48 MLP) adds an expected variance contribution of 1.0 without the scaling",
        "The variance would decrease exponentially because Xavier initialization is designed for networks without residual connections",
        "The variance would be unaffected — the $1/\\sqrt{2L}$ factor only matters for gradient flow during backpropagation, not for forward-pass activation magnitudes"
      ],
      correct: 1,
      explanation: "In a residual network, the output is $x + f(x)$. If $f(x)$ has variance roughly equal to $\\text{Var}(x)$ (which Xavier initialization produces), then $\\text{Var}(x + f(x)) \\approx 2 \\cdot \\text{Var}(x)$ (assuming independence). After 96 sub-layers, variance grows by a factor of ~96. The $1/\\sqrt{2L}$ scaling makes each sub-layer contribute variance $1/(2L)$ of the input variance, so the total added variance across all 96 sub-layers is approximately 1.0 — keeping the residual stream variance bounded. LayerNorm helps but is applied before the residual addition, so it doesn't prevent the accumulation."
    },
    // Step 13: Putting it all together
    {
      type: "info",
      title: "The Stability Toolkit: Layered Defenses",
      content: "Modern LLM training combines multiple stability techniques in layers:\n\n**1. Architecture design** (prevents instabilities by construction):\n- Pre-LayerNorm (normalize before attention/MLP, not after) — ensures bounded inputs to each sub-layer\n- Residual connections with scaled initialization — bounded variance growth\n- RMSNorm instead of LayerNorm — simpler, more stable (removes mean centering)\n\n**2. Optimization configuration** (absorbs transient instabilities):\n- Learning rate warmup (1,000-2,000 steps) — protects against noisy early gradients\n- Gradient clipping ($\\tau = 1.0$) — prevents single-step catastrophes\n- $\\beta_2 = 0.95$ instead of 0.999 — faster adaptation of Adam's second moment, reducing stale variance estimates\n\n**3. Training infrastructure** (detects and recovers):\n- Gradient norm monitoring — early warning of impending spikes\n- Frequent checkpointing — minimizes compute lost on restart\n- Automatic spike detection and batch skipping — some teams skip batches where the loss exceeds a threshold (e.g., 3× the running average)\n\nThese defenses are not independent — they interact. For example, lowering $\\beta_2$ makes Adam more responsive to gradient changes (good for tracking curvature shifts) but also more sensitive to noise (potentially bad). The art of stable training is balancing these tradeoffs for a specific model size and data distribution."
    },
    // Step 14: MC — putting it together
    {
      type: "mc",
      question: "A team observes recurring loss spikes every ~10,000 steps during training. Gradient norms spike 2-3 steps before each loss spike. They have gradient clipping at $\\tau = 1.0$ but spikes still occur. Which intervention is most likely to help?",
      options: [
        "Removing gradient clipping entirely — the clipping is distorting the gradient direction and causing the optimizer to oscillate",
        "Increasing the learning rate — the current rate is too small, causing the optimizer to get stuck in high-curvature regions where spikes occur",
        "Lowering the clipping threshold to $\\tau = 0.3$ and reducing the peak learning rate — the current threshold is too permissive, allowing destructive updates even after clipping",
        "Switching from Pre-LN to Post-LN architecture — Post-LN produces more stable gradients at the cost of slower convergence"
      ],
      correct: 2,
      explanation: "If gradient clipping at $\\tau = 1.0$ isn't preventing spikes, the clipped gradients are still producing updates large enough to push the model into unstable regions. Lowering the threshold to $\\tau = 0.3$ (a common value for large models) and reducing the peak learning rate together reduce the maximum possible step size. This is the standard first response to persistent instability. Removing clipping would make things worse. Increasing the learning rate would amplify instability. Post-LN is generally less stable than Pre-LN for deep transformers."
    }
  ]
};
