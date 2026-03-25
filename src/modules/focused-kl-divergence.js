// Focused, first-principles modules for KL divergence.
// Each module covers exactly ONE concept deeply.

export const forwardKLLearning = {
  id: "0.2-forward-kl-learning-easy",
  sectionId: "0.2",
  title: "Forward KL Divergence",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "What Does KL Divergence Measure?",
      content: "KL divergence measures how one probability distribution $q$ differs from a reference distribution $p$. It answers: **how much extra information do I need if I use $q$ to encode data that actually comes from $p$?**\n\nThe key insight is that KL divergence is **not symmetric**: $\\text{KL}(p \\| q) \\neq \\text{KL}(q \\| p)$. These two directions have completely different behaviors, and choosing the wrong one changes what your model learns. This module focuses entirely on the **forward** direction."
    },
    {
      type: "mc",
      question: "KL divergence $\\text{KL}(p \\| q)$ is zero when:",
      options: [
        "$p$ and $q$ have the same support (non-zero regions)",
        "$p$ and $q$ are identical distributions everywhere",
        "$q$ assigns higher probability than $p$ to every outcome",
        "$p$ and $q$ have the same mean and variance"
      ],
      correct: 1,
      explanation: "$\\text{KL}(p \\| q) = 0$ if and only if $p = q$ almost everywhere. Having the same support, same moments, or $q > p$ everywhere is not sufficient — the distributions must match exactly."
    },
    {
      type: "info",
      title: "The Forward KL Formula",
      content: "**Forward KL divergence** is defined as:\n\n$$\\text{KL}(p \\| q) = \\sum_x p(x) \\log \\frac{p(x)}{q(x)}$$\n\nNotice where $p$ appears: it is both the **weighting distribution** (in front of the log) and part of the **ratio inside the log**. This means forward KL only cares about regions where $p(x) > 0$ — if $p$ assigns zero probability to an outcome, that outcome contributes nothing to the sum regardless of what $q$ does.\n\nThe critical consequence: if $p(x) > 0$ but $q(x) \\approx 0$, the ratio $p(x)/q(x)$ explodes, making KL very large. Forward KL **severely penalizes $q$ for missing any region where $p$ has mass**."
    },
    {
      type: "mc",
      question: "Suppose $p$ is a mixture of two Gaussians with modes at $x = -3$ and $x = 3$. You are fitting $q$ (a single Gaussian) by minimizing $\\text{KL}(p \\| q)$. What will $q$ look like?",
      options: [
        "A narrow Gaussian centered at $x = -3$ (the taller mode)",
        "A narrow Gaussian centered at $x = 3$ (the wider mode)",
        "A broad Gaussian centered near $x = 0$ covering both modes",
        "Two separate narrow Gaussians, one at each mode"
      ],
      correct: 2,
      explanation: "Forward KL penalizes $q$ for any region where $p > 0$ but $q \\approx 0$. If $q$ concentrated on just one mode, it would leave the other mode uncovered, causing a huge penalty. So $q$ spreads out to cover both modes — even though it wastes probability mass between them. This is the **mode-covering** behavior of forward KL. (Option D is impossible since $q$ is a single Gaussian.)"
    },
    {
      type: "info",
      title: "Mode-Covering Behavior",
      content: "Forward KL produces **mode-covering** approximations. When $q$ cannot perfectly match $p$, minimizing $\\text{KL}(p \\| q)$ forces $q$ to spread out and cover all regions where $p$ has significant mass.\n\nWhy? Because the penalty structure is asymmetric:\n- **$p > 0, q \\approx 0$**: Catastrophic. The log ratio $\\log(p/q) \\to \\infty$, weighted by $p > 0$. This produces an enormous penalty.\n- **$p \\approx 0, q > 0$**: Harmless. Even if $q$ wastes probability mass in regions where $p$ is small, $p \\approx 0$ means this term contributes almost nothing.\n\nSo forward KL would rather $q$ be **too broad** (wasting mass) than **too narrow** (missing mass)."
    },
    {
      type: "mc",
      question: "A language model $q$ is trained by minimizing forward KL against the true data distribution $p$. Which failure mode is forward KL most likely to produce?",
      options: [
        "The model repeats the same high-quality sentence over and over",
        "The model refuses to generate any output at all",
        "The model generates only the single most common sentence in the training data",
        "The model generates diverse but sometimes incoherent or low-quality text"
      ],
      correct: 3,
      explanation: "Forward KL's mode-covering behavior means the model tries to assign probability to everything the true distribution covers. This leads to high diversity but also means $q$ places mass on low-probability regions between modes — generating text that is varied but sometimes incoherent. Repetitive or single-output behavior would indicate mode-seeking (reverse KL)."
    },
    {
      type: "info",
      title: "Forward KL = Maximum Likelihood",
      content: "Here is a critical connection: **minimizing forward KL is equivalent to maximum likelihood estimation (MLE)**.\n\nWhen we train a neural network with cross-entropy loss on data from $p$, we minimize:\n\n$$\\text{KL}(p \\| q_\\theta) = \\underbrace{H(p)}_{\\text{constant}} + \\underbrace{\\left(-\\sum_x p(x) \\log q_\\theta(x)\\right)}_{\\text{cross-entropy}}$$\n\nSince $H(p)$ is fixed, minimizing $\\text{KL}(p \\| q_\\theta)$ is the same as minimizing cross-entropy, which is the same as maximizing the likelihood of the data under $q_\\theta$.\n\nThis means every time you train an LLM with next-token prediction (cross-entropy loss), you are implicitly minimizing forward KL. The mode-covering tendencies of forward KL are baked into standard pretraining."
    },
    {
      type: "mc",
      question: "A researcher replaces their LLM's standard cross-entropy pretraining loss with a loss that minimizes $\\text{KL}(q_\\theta \\| p)$ instead. Compared to standard training, the resulting model would most likely:",
      options: [
        "Learn a broader vocabulary but make more grammatical errors",
        "Produce higher-perplexity text that covers more diverse topics",
        "Generate more repetitive but higher-quality text by focusing on common patterns",
        "Achieve identical performance since both KL directions are equivalent for training"
      ],
      correct: 2,
      explanation: "Switching from $\\text{KL}(p \\| q_\\theta)$ (forward, mode-covering) to $\\text{KL}(q_\\theta \\| p)$ (reverse, mode-seeking) would make the model focus on confidently matching the highest-probability patterns rather than covering everything. This leads to more repetitive but locally higher-quality output. The two directions are NOT equivalent — this is the fundamental asymmetry of KL divergence."
    },
    {
      type: "info",
      title: "When Forward KL Goes Wrong",
      content: "Forward KL's mode-covering behavior has concrete failure modes in language modeling:\n\n**Oversmoothing**: The model spreads probability across too many tokens at each position, producing \"safe\" but bland predictions. Entropy is too high.\n\n**Hallucination of probability mass**: The model assigns non-trivial probability to token sequences that never appear in the real distribution, because it is trying to cover all modes and inevitably fills in gaps.\n\n**Poor calibration on rare events**: Forward KL treats all regions where $p > 0$ as important, but it has finite capacity. The model may sacrifice accuracy on common patterns to avoid the catastrophic penalty of missing rare ones.\n\nThese issues motivate the use of **reverse KL** in fine-tuning and alignment, where we want models to be confident and precise rather than maximally broad."
    },
    {
      type: "mc",
      question: "In the context of LLM pretraining, the oversmoothing problem caused by forward KL manifests as:",
      options: [
        "The model assigns near-uniform probability across all tokens at every position",
        "The model's per-token entropy is higher than the true distribution's entropy in contexts where few continuations are valid",
        "The model consistently predicts the same token regardless of context",
        "The model's loss converges to exactly $\\log(V)$ where $V$ is the vocabulary size"
      ],
      correct: 1,
      explanation: "Oversmoothing means the model's predicted distribution is broader than the true distribution — it assigns too much probability to incorrect tokens. This shows up as higher per-token entropy than the true distribution, especially in constrained contexts (like after 'The capital of France is' where only a few completions are valid). It does NOT converge to uniform ($\\log V$) — that would be zero learning. And it's the opposite of always predicting the same token."
    },
    {
      type: "info",
      title: "Forward KL in RLHF",
      content: "In RLHF, the KL penalty term constraining the policy $\\pi_\\theta$ to stay near the reference policy $\\pi_{\\text{ref}}$ is typically:\n\n$$\\mathbb{E}_{x \\sim \\pi_\\theta}\\left[\\text{KL}(\\pi_\\theta(\\cdot | x) \\| \\pi_{\\text{ref}}(\\cdot | x))\\right]$$\n\nThis is **reverse** KL from the policy's perspective (we will cover this in the next module). But the supervised fine-tuning (SFT) step that creates $\\pi_{\\text{ref}}$ uses **forward** KL (cross-entropy on human demonstrations).\n\nSo the full RLHF pipeline uses **both** directions:\n1. **SFT stage**: Forward KL → mode-covering → the reference model tries to imitate all human demonstrations broadly\n2. **RL stage**: Reverse KL penalty → mode-seeking → the policy is encouraged to stay within confident regions of $\\pi_{\\text{ref}}$\n\nUnderstanding this asymmetry is essential for understanding alignment training dynamics."
    },
    {
      type: "mc",
      question: "Why does RLHF use reverse KL ($\\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$) rather than forward KL ($\\text{KL}(\\pi_{\\text{ref}} \\| \\pi_\\theta)$) as its penalty term?",
      options: [
        "Reverse KL is computationally cheaper to estimate from policy samples",
        "Forward KL would require sampling from $\\pi_{\\text{ref}}$, but the RL loop already generates samples from $\\pi_\\theta$, making reverse KL naturally estimable on-policy",
        "Forward KL always produces larger gradient magnitudes, making training unstable",
        "There is no practical difference; the choice is a historical convention"
      ],
      correct: 1,
      explanation: "In the RL loop, we generate rollouts from $\\pi_\\theta$ (the current policy). Reverse KL $\\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$ takes expectations under $\\pi_\\theta$, which means we can estimate it directly from our on-policy samples. Forward KL would require expectations under $\\pi_{\\text{ref}}$, needing separate samples from the reference policy. The choice is driven by what's naturally estimable in the on-policy RL setting."
    }
  ]
};

export const reverseKLLearning = {
  id: "0.2-reverse-kl-learning-easy",
  sectionId: "0.2",
  title: "Reverse KL Divergence",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "Reverse KL: The Other Direction",
      content: "**Reverse KL divergence** swaps the roles of $p$ and $q$:\n\n$$\\text{KL}(q \\| p) = \\sum_x q(x) \\log \\frac{q(x)}{p(x)}$$\n\nCompare with forward KL: $\\text{KL}(p \\| q) = \\sum_x p(x) \\log \\frac{p(x)}{q(x)}$.\n\nThe crucial change: **$q$ is now the weighting distribution**. Reverse KL only cares about regions where $q(x) > 0$. If $q$ assigns zero probability somewhere, it doesn't matter what $p$ does there — that term vanishes from the sum.\n\nThis single change produces completely opposite behavior."
    },
    {
      type: "mc",
      question: "In reverse KL $\\text{KL}(q \\| p)$, which scenario creates the largest penalty?",
      options: [
        "$p(x) = 0.5$ and $q(x) = 0.001$ for some outcome $x$",
        "$q(x) = 0.3$ and $p(x) = 0.001$ for some outcome $x$",
        "$p(x) = 0$ and $q(x) = 0$ for some outcome $x$",
        "$p(x) = 0.5$ and $q(x) = 0$ for some outcome $x$"
      ],
      correct: 1,
      explanation: "Reverse KL weights by $q(x)$, so the penalty is $q(x) \\log(q(x)/p(x))$. When $q(x) = 0.3$ and $p(x) = 0.001$, the log ratio is $\\log(300) \\approx 5.7$, weighted by $0.3$, giving $\\approx 1.7$. Option A gives $0.001 \\times \\log(0.002) \\approx 0$ (small $q$ weight). Option C contributes $0$ ($q = 0$). Option D also contributes $0$ ($q = 0$). The biggest penalty comes when $q$ puts substantial mass where $p$ is near zero."
    },
    {
      type: "info",
      title: "Mode-Seeking Behavior",
      content: "Reverse KL produces **mode-seeking** (or **zero-avoiding**) approximations.\n\nWhen $q$ cannot perfectly match $p$, minimizing $\\text{KL}(q \\| p)$ forces $q$ to concentrate on a **subset** of $p$'s mass rather than trying to cover everything.\n\nThe penalty structure:\n- **$q > 0, p \\approx 0$**: Catastrophic. The ratio $q/p \\to \\infty$, and this is weighted by $q > 0$. The model is severely penalized for putting mass where the target has none.\n- **$q \\approx 0, p > 0$**: Harmless. If $q$ ignores a region where $p$ has mass, the $q \\approx 0$ weight makes this term vanish.\n\nSo reverse KL would rather $q$ be **too narrow** (ignoring some modes of $p$) than **too broad** (placing mass in wrong places). It locks onto one mode and gets it right."
    },
    {
      type: "mc",
      question: "You fit a single Gaussian $q$ to a bimodal target $p$ (modes at $x = -3$ and $x = 3$) by minimizing reverse KL $\\text{KL}(q \\| p)$. What happens?",
      options: [
        "$q$ centers near $x = 0$ to split the difference between modes",
        "$q$ oscillates between both modes during optimization, never converging",
        "$q$ becomes a very broad, nearly uniform distribution",
        "$q$ collapses onto one of the two modes with low variance"
      ],
      correct: 3,
      explanation: "Reverse KL's mode-seeking behavior means $q$ will find one mode and concentrate there. Placing mass at $x = 0$ (between modes) would put $q$ mass where $p$ is near zero — a huge penalty. Spreading broadly would do the same. Instead, $q$ locks onto whichever mode it finds first, getting that mode right while completely ignoring the other. Which mode it picks depends on initialization."
    },
    {
      type: "info",
      title: "Reverse KL in Variational Inference",
      content: "Variational inference (VI) is the most classical application of reverse KL. Given a true posterior $p(z|x)$, we approximate it with a tractable $q_\\phi(z)$ by minimizing:\n\n$$\\text{KL}(q_\\phi(z) \\| p(z|x))$$\n\nThis is reverse KL, and it yields the **Evidence Lower Bound (ELBO)**:\n\n$$\\log p(x) \\geq \\mathbb{E}_{q_\\phi}[\\log p(x|z)] - \\text{KL}(q_\\phi(z) \\| p(z))$$\n\nThe mode-seeking behavior of reverse KL is why variational approximations tend to **underestimate posterior uncertainty** — they capture one mode of the posterior well but miss others. This is a known limitation of standard VI."
    },
    {
      type: "mc",
      question: "A VAE trained with the standard ELBO objective (which uses reverse KL) on a dataset of handwritten digits produces blurry reconstructions. This is most directly explained by:",
      options: [
        "The reverse KL penalty causes the encoder to map all inputs to the same latent code",
        "The decoder likelihood term $\\mathbb{E}_{q}[\\log p(x|z)]$ averages over a $q$ that is too narrow, failing to cover the full posterior",
        "Reverse KL forces the approximate posterior to cover all modes, spreading latent codes too thin",
        "The ELBO objective has nothing to do with blurriness — it is caused by the choice of Gaussian decoder"
      ],
      correct: 3,
      explanation: "VAE blurriness is primarily caused by the Gaussian decoder assumption (option D is closest to the practical truth), but among the KL-specific options, the issue is that reverse KL makes $q_\\phi(z)$ mode-seeking — it captures a narrow region of the posterior. When the decoder must reconstruct from these limited latent samples, it averages over uncertainty, producing blurry outputs. Option C describes forward KL behavior, not reverse."
    },
    {
      type: "info",
      title: "Reverse KL as the RLHF Penalty",
      content: "In RLHF, the optimization objective is:\n\n$$\\max_{\\pi_\\theta} \\; \\mathbb{E}_{x \\sim \\pi_\\theta}[r(x)] - \\beta \\, \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$$\n\nThe penalty term is **reverse KL** — it measures how far $\\pi_\\theta$ deviates from $\\pi_{\\text{ref}}$, weighted by where $\\pi_\\theta$ puts mass.\n\nThis is mode-seeking with respect to $\\pi_{\\text{ref}}$:\n- $\\pi_\\theta$ is **severely penalized** for generating tokens that $\\pi_{\\text{ref}}$ would never generate (high $\\pi_\\theta$, low $\\pi_{\\text{ref}}$)\n- $\\pi_\\theta$ is **not penalized** for ignoring things $\\pi_{\\text{ref}}$ can generate (low $\\pi_\\theta$, high $\\pi_{\\text{ref}}$)\n\nThis asymmetry is intentional: the policy can **narrow** its behavior (becoming more selective) but cannot **invent** behavior the reference model doesn't support."
    },
    {
      type: "mc",
      question: "During RLHF with a reverse KL penalty, a policy $\\pi_\\theta$ discovers that a particular response style scores very high reward but has near-zero probability under $\\pi_{\\text{ref}}$. What happens?",
      options: [
        "The policy generates this response frequently because high reward overrides the KL penalty",
        "The KL penalty is near zero since $\\pi_{\\text{ref}}$ assigns low probability, so the reward signal dominates",
        "The KL penalty becomes very large because $\\pi_\\theta$ has high mass where $\\pi_{\\text{ref}}$ has near-zero mass, likely preventing this behavior",
        "The KL penalty drives the policy to assign exactly $\\pi_{\\text{ref}}$'s probability to this response, maintaining the original frequency"
      ],
      correct: 2,
      explanation: "Reverse KL = $\\sum \\pi_\\theta \\log(\\pi_\\theta / \\pi_{\\text{ref}})$. When $\\pi_\\theta$ is high and $\\pi_{\\text{ref}}$ is near zero, the ratio $\\pi_\\theta/\\pi_{\\text{ref}} \\to \\infty$, making the penalty enormous. Unless the reward signal is extraordinarily high AND $\\beta$ is very small, the KL penalty prevents the policy from putting mass on responses the reference model wouldn't generate. This is exactly how the KL penalty prevents reward hacking."
    },
    {
      type: "info",
      title: "The $\\beta$ Parameter Controls the Tradeoff",
      content: "The strength of the reverse KL penalty is controlled by $\\beta$:\n\n$$\\max_{\\pi_\\theta} \\; \\mathbb{E}[r(x)] - \\beta \\, \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$$\n\n- **Large $\\beta$**: Strong constraint. $\\pi_\\theta$ stays close to $\\pi_{\\text{ref}}$, sacrificing reward for safety. The policy barely moves.\n- **Small $\\beta$**: Weak constraint. $\\pi_\\theta$ can deviate far from $\\pi_{\\text{ref}}$ to chase reward. Risk of reward hacking.\n- **$\\beta = 0$**: No constraint. Pure reward maximization. The policy will exploit any flaw in the reward model.\n\nIn practice, $\\beta$ is often **annealed** during training — starting large (safe) and gradually decreasing (allowing more reward optimization). The optimal $\\beta$ depends on how trustworthy the reward model is."
    },
    {
      type: "mc",
      question: "A team trains two RLHF policies with identical setups except $\\beta_A = 0.01$ and $\\beta_B = 0.5$. After training, they evaluate both on a held-out set of reward model queries. Which is most likely?",
      options: [
        "Policy A achieves higher reward scores but shows more reward hacking artifacts when evaluated by humans",
        "Policy B achieves higher reward scores because the stronger KL penalty acts as beneficial regularization",
        "Both policies converge to the same behavior because the reward model dominates regardless of $\\beta$",
        "Policy A stays closer to $\\pi_{\\text{ref}}$ because lower $\\beta$ means less deviation is allowed"
      ],
      correct: 0,
      explanation: "Lower $\\beta$ (policy A) means weaker KL constraint, so the policy can deviate further from $\\pi_{\\text{ref}}$ to maximize reward. This typically means higher reward model scores but more exploitation of reward model weaknesses (reward hacking). Higher $\\beta$ (policy B) keeps the policy closer to $\\pi_{\\text{ref}}$, sacrificing some reward for stability. Option D reverses the relationship — lower $\\beta$ means MORE deviation is allowed, not less."
    },
    {
      type: "mc",
      question: "You notice that DPO (Direct Preference Optimization) implicitly minimizes a **forward KL** variant with respect to the optimal policy, while PPO-based RLHF uses a **reverse KL** penalty. This difference predicts that:",
      options: [
        "DPO-trained models will be more diverse but potentially less precise, while PPO models will be more focused and confident",
        "DPO models will be more focused on high-reward modes, while PPO models will cover more of the distribution",
        "Both produce identical policies at convergence regardless of the KL direction",
        "DPO will converge faster because forward KL has simpler gradients"
      ],
      correct: 0,
      explanation: "Forward KL is mode-covering (tries to cover everything the target does), so DPO-trained models tend to be more diverse but may place mass on lower-quality outputs. Reverse KL is mode-seeking (concentrates on a subset), so PPO models tend to be more focused and confident. This theoretical distinction matches empirical observations: DPO models often produce more varied outputs while PPO models tend to be more consistent. They do NOT converge to the same thing."
    }
  ]
};
