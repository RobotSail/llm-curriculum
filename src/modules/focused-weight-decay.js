// Focused module: Weight Decay Regularization
// Covers L2 regularization, why it breaks under adaptive optimizers,
// decoupled weight decay (AdamW), and practical implications for LLM training.

export const weightDecayLearning = {
  id: "0.3-weight-decay-learning-easy",
  sectionId: "0.3",
  title: "Weight Decay Regularization",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Why Regularize Weights?",
      content: "Neural networks are highly overparameterized — a 7B model has billions of parameters but may train on only trillions of tokens. Without regularization, networks can memorize training data rather than learning generalizable patterns.\n\n**Weight decay** is the most common regularization technique for LLMs. The idea is simple: at each step, shrink all weights slightly toward zero:\n\n$$\\theta_{t+1} = (1 - \\lambda)\\theta_t - \\alpha g_t$$\n\nThe factor $(1 - \\lambda)$ multiplies every weight by a number slightly less than 1 (e.g., $\\lambda = 0.01$ gives a 1% shrink per step). This creates a constant pressure toward smaller weights.\n\nThe effect: weights that are actively reinforced by gradients survive. Weights that are not needed slowly decay to zero. This prevents the network from relying on large, fragile weight values and encourages distributed representations."
    },
    {
      type: "mc",
      question: "A weight $\\theta_i = 5.0$ receives zero gradient for 100 consecutive steps. With weight decay $\\lambda = 0.01$ and no gradient updates, what approximately happens to $\\theta_i$?",
      options: [
        "$\\theta_i$ stays at 5.0 because weight decay only acts when gradients are nonzero",
        "$\\theta_i$ decays to $(1 - 0.01)^{100} \\times 5.0 \\approx 0.99^{100} \\times 5.0 \\approx 1.83$ — exponential shrinkage",
        "$\\theta_i$ decays linearly to $5.0 - 100 \\times 0.01 = 4.0$",
        "$\\theta_i$ immediately jumps to zero after the first step with zero gradient"
      ],
      correct: 1,
      explanation: "With zero gradient, the update is $\\theta_{t+1} = (1 - \\lambda)\\theta_t = 0.99 \\theta_t$. After 100 steps: $\\theta_{100} = 0.99^{100} \\times 5.0 \\approx 0.366 \\times 5.0 \\approx 1.83$. Weight decay acts continuously regardless of gradients — it is a multiplicative shrinkage at every step, producing exponential decay toward zero."
    },
    {
      type: "info",
      title: "L2 Regularization: The Loss-Based View",
      content: "An alternative way to shrink weights is **L2 regularization**: add a penalty term to the loss function:\n\n$$\\tilde{L}(\\theta) = L(\\theta) + \\frac{\\lambda}{2}\\|\\theta\\|^2$$\n\nThe gradient of the penalty term is $\\lambda\\theta$, so the gradient update becomes:\n\n$$\\theta_{t+1} = \\theta_t - \\alpha(g_t + \\lambda\\theta_t) = (1 - \\alpha\\lambda)\\theta_t - \\alpha g_t$$\n\nWith vanilla SGD, L2 regularization and weight decay produce **mathematically identical updates** (up to reparameterization of $\\lambda$). This is why the terms are often used interchangeably.\n\nThe key equivalence: decoupled weight decay with strength $\\lambda_d$ equals L2 with $\\lambda_{L2} = \\lambda_d / \\alpha$ under SGD. They are truly the same thing.\n\nBut this equivalence **breaks** with adaptive optimizers like Adam. The reason is fundamental: L2 regularization modifies the gradient, and adaptive optimizers transform gradients before applying them."
    },
    {
      type: "mc",
      question: "Under vanilla SGD with learning rate $\\alpha$, which of the following produces the same parameter update as L2 regularization with coefficient $\\lambda$?",
      options: [
        "Weight decay with strength $\\lambda$ — the decay coefficient maps directly regardless of learning rate",
        "Weight decay with strength $\\alpha\\lambda$ — because the L2 gradient $\\lambda\\theta$ gets multiplied by $\\alpha$",
        "Weight decay with strength $\\lambda/\\alpha$ — because the $\\alpha$ in the SGD update cancels one factor",
        "No weight decay setting can match L2 because they are fundamentally different regularizers"
      ],
      correct: 1,
      explanation: "L2 gives update $(1 - \\alpha\\lambda)\\theta_t - \\alpha g_t$. Decoupled weight decay with strength $\\lambda_d$ gives $(1 - \\lambda_d)\\theta_t - \\alpha g_t$. Setting $\\lambda_d = \\alpha\\lambda$ makes them identical. Under SGD, L2 and weight decay are equivalent up to this rescaling. This equivalence is so clean that many textbooks don't distinguish the two."
    },
    {
      type: "info",
      title: "How Adam Breaks the Equivalence",
      content: "With Adam, the L2 regularization gradient $\\lambda\\theta_t$ passes through Adam's adaptive machinery just like the task gradient.\n\nRecall Adam's update: $\\theta_{t+1} = \\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$\n\nWhen using L2, the gradient becomes $g_t + \\lambda\\theta_t$, and both the first moment $m_t$ and second moment $v_t$ absorb this combined signal. The regularization gradient gets **scaled by $1/\\sqrt{v_t}$** along with everything else.\n\nThis creates a problem:\n- Parameters with **large task gradients** → large $v_t$ → regularization gradient is **dampened** → weight decay is weakened exactly where weights tend to be large\n- Parameters with **small task gradients** → small $v_t$ → regularization gradient is **amplified** → weight decay is strengthened where weights are already small\n\nAdam's adaptivity, which is helpful for the task gradient, is actively harmful for the regularization gradient. The weights that most need regularization get the least."
    },
    {
      type: "mc",
      question: "A parameter $\\theta_i$ has consistently large task gradients, so $v_i$ is large. Another parameter $\\theta_j$ has small task gradients, so $v_j$ is small. Both have the same weight magnitude $|\\theta_i| = |\\theta_j|$. Under Adam with L2 regularization, which parameter experiences stronger effective weight decay?",
      options: [
        "$\\theta_i$ — large gradients mean more total gradient signal including the regularization term",
        "$\\theta_j$ — its small $v_j$ amplifies the regularization gradient $\\lambda\\theta_j$ through the $1/\\sqrt{v_j}$ scaling",
        "Both experience equal weight decay because L2 adds the same $\\lambda\\theta$ regardless of gradient history",
        "Neither experiences meaningful decay because Adam's momentum cancels the regularization signal over time"
      ],
      correct: 1,
      explanation: "The effective regularization update for $\\theta_i$ is proportional to $\\lambda\\theta_i / \\sqrt{v_i}$, and similarly for $\\theta_j$. Since $v_i \\gg v_j$ but $|\\theta_i| = |\\theta_j|$, the effective decay on $\\theta_j$ is much stronger. This is backwards: $\\theta_j$ (with small gradients, likely less important) gets more regularization than $\\theta_i$ (with large gradients, where unchecked growth is most dangerous)."
    },
    {
      type: "info",
      title: "AdamW: Decoupled Weight Decay",
      content: "Loshchilov & Hutter (2019) proposed a simple fix: apply weight decay **directly to the parameters**, completely outside Adam's adaptive mechanism:\n\n$$\\theta_{t+1} = (1 - \\alpha\\lambda)\\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$\n\nThe key difference from L2 regularization through Adam:\n- The decay term $(1 - \\alpha\\lambda)\\theta_t$ is applied **uniformly** to all parameters\n- It does NOT pass through the first or second moment estimates\n- Every parameter with the same weight magnitude receives the same absolute decay, regardless of its gradient history\n\nThis is called **decoupled** weight decay because the regularization is decoupled from the adaptive gradient processing.\n\nAdamW is now the standard optimizer for LLM pretraining. Essentially every major language model (GPT, LLaMA, Gemma, etc.) uses AdamW or a close variant."
    },
    {
      type: "mc",
      question: "A team switches from Adam+L2 to AdamW (decoupled weight decay) with equivalent regularization strength. They keep all other hyperparameters the same. What is the most likely effect on training?",
      options: [
        "No measurable difference — L2 and decoupled weight decay are interchangeable for all practical purposes",
        "Training diverges because AdamW applies too much regularization to parameters with large gradients",
        "Better generalization — weight decay is now applied uniformly, so large weights are properly regularized regardless of gradient magnitude",
        "Faster convergence but worse generalization because uniform decay prevents Adam from specializing learning rates"
      ],
      correct: 2,
      explanation: "AdamW's uniform decay ensures that large weights are regularized proportionally to their magnitude, not inversely proportional to their gradient magnitude (as with L2 through Adam). This leads to better generalization in practice. The Loshchilov & Hutter paper demonstrated consistent improvements across benchmarks, and AdamW became the standard precisely because of this generalization benefit."
    },
    {
      type: "info",
      title: "Weight Decay in Practice: LLM Training",
      content: "Standard AdamW settings for LLM pretraining:\n\n- **Weight decay coefficient**: $\\lambda \\in [0.01, 0.1]$, commonly 0.1\n- **Applied to**: Weight matrices (2D parameters) only\n- **Not applied to**: Biases, layer norm parameters, embedding tables\n\nWhy exclude biases and norms? These parameters are low-dimensional — a bias vector for a 4096-dim layer has only 4096 entries vs. millions in the weight matrix. Regularizing them provides negligible benefit and can hurt performance by constraining the network's ability to shift activations.\n\n**Weight decay interacts with learning rate scheduling.** Since AdamW applies decay as $(1 - \\alpha\\lambda)$, the effective decay shrinks as $\\alpha$ decreases during cosine annealing. Some implementations use a fixed decay rate independent of $\\alpha$ to avoid this coupling.\n\nAt scale, the weight decay coefficient $\\lambda$ is often one of the most impactful hyperparameters. Too little decay → overfitting and training instability. Too much → underfitting and difficulty learning long-range dependencies."
    },
    {
      type: "mc",
      question: "During cosine learning rate decay, the learning rate $\\alpha$ drops from $3 \\times 10^{-4}$ to $3 \\times 10^{-5}$ over training. With AdamW using $\\lambda = 0.1$ and decay applied as $(1 - \\alpha\\lambda)$, how does the effective per-step weight decay change?",
      options: [
        "It stays constant at $\\lambda = 0.1$ per step because decoupled weight decay is independent of learning rate by definition",
        "It decreases by 10x — from $(1 - 3 \\times 10^{-5})$ per step to $(1 - 3 \\times 10^{-6})$ per step, tracking the learning rate schedule",
        "It increases by 10x because smaller learning rates mean less gradient signal to counteract the decay",
        "It oscillates with the cosine schedule because decay and learning rate are phase-coupled"
      ],
      correct: 1,
      explanation: "In the standard AdamW formulation, the per-step decay is $(1 - \\alpha\\lambda)$, so it scales linearly with $\\alpha$. When $\\alpha$ drops by 10x, the per-step decay strength also drops by 10x. This coupling means late-stage training has very little effective regularization. Some practitioners prefer a formulation where decay is $(1 - \\lambda)$ independent of $\\alpha$ to maintain consistent regularization throughout training."
    },
    {
      type: "mc",
      question: "You are training a 7B parameter model with AdamW. Layer norm parameters ($\\gamma$, $\\beta$) make up about 0.01% of total parameters. A colleague suggests applying weight decay to layer norm parameters too, arguing \"regularization everywhere is more consistent.\" What is the most likely outcome?",
      options: [
        "Significant generalization improvement because layer norm parameters are currently unregularized outliers",
        "Negligible change or slight degradation — decaying $\\gamma$ toward zero destabilizes normalization without meaningful regularization benefit",
        "Training diverges because weight decay on $\\gamma$ forces activations to collapse to zero",
        "Identical training dynamics because 0.01% of parameters cannot affect the loss measurably"
      ],
      correct: 1,
      explanation: "Layer norm has $\\gamma$ (scale) initialized to 1 and $\\beta$ (shift) initialized to 0. Decaying $\\gamma$ toward 0 actively fights the normalization mechanism — it shrinks the normalized activations toward zero. Decaying $\\beta$ toward 0 is mostly harmless but pointless. The number of parameters is tiny (no regularization benefit), and the potential harm to training stability outweighs any theoretical consistency gain. This is why standard practice excludes norms and biases from weight decay."
    }
  ]
};
