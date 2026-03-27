// Focused learning module: Direct Preference Optimization (DPO)
// Section A.4: Direct Alignment Methods
// Covers: the RLHF objective, DPO derivation from the closed-form optimal policy,
// the DPO loss, gradient dynamics, and comparison with PPO-based RLHF.
// Single-concept module building the derivation from first principles.

export const dpoLearning = {
  id: "A.4-dpo-learning-easy",
  sectionId: "A.4",
  title: "Direct Preference Optimization (DPO)",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: The RLHF objective
    {
      type: "info",
      title: "Starting Point: The RLHF Objective",
      content: "To align a language model with human preferences, RLHF optimizes:\n\n$$\\max_{\\pi_\\theta} \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_\\theta(\\cdot|x)}\\left[r(x, y)\\right] - \\beta \\, D_{\\text{KL}}\\left(\\pi_\\theta(\\cdot|x) \\| \\pi_{\\text{ref}}(\\cdot|x)\\right)$$\n\nwhere $r(x, y)$ is a learned reward model, $\\pi_{\\text{ref}}$ is the reference policy (typically the SFT model), and $\\beta > 0$ controls the KL penalty.\n\nThe first term says: **generate responses that get high reward**. The second term says: **don't stray too far from the reference policy**. Without the KL penalty, the policy would collapse to degenerate outputs that exploit the reward model (reward hacking).\n\nThe standard RLHF pipeline to optimize this objective:\n1. Train a reward model $r(x, y)$ on preference data\n2. Use PPO (a reinforcement learning algorithm) to optimize $\\pi_\\theta$ against $r$\n\nThis pipeline is complex: it requires maintaining four models simultaneously (policy, reference, reward model, value function) and is notoriously unstable. **DPO's insight**: there's a shortcut that eliminates the reward model and the RL loop entirely."
    },
    // Step 2: MC — RLHF objective understanding
    {
      type: "mc",
      question: "In the RLHF objective, what happens as $\\beta \\to 0$ (the KL penalty weight approaches zero)?",
      options: [
        "The policy converges to the reference policy $\\pi_{\\text{ref}}$, since without a KL penalty the reward term has no anchor and optimization defaults to the prior",
        "The policy maximizes the reward model without constraint, likely exploiting reward model artifacts to generate degenerate high-reward outputs (reward hacking)",
        "The objective reduces to supervised fine-tuning on the highest-reward responses, since removing KL makes the optimization equivalent to maximum-likelihood on top-$k$ samples",
        "Training becomes numerically unstable because the KL divergence computation involves a $\\log(\\beta)$ term that diverges as $\\beta \\to 0$, producing NaN gradients"
      ],
      correct: 1,
      explanation: "With $\\beta \\to 0$, the KL penalty vanishes and the policy is free to maximize reward without any constraint tying it to $\\pi_{\\text{ref}}$. The reward model is an imperfect proxy for true human preferences — it has exploitable patterns. Without the KL regularizer, the policy finds reward-hacking strategies: repetitive text, sycophantic outputs, or formatting tricks that score high on the reward model but are low-quality. The $\\beta$ parameter controls this tradeoff: larger $\\beta$ keeps the policy conservative, smaller $\\beta$ allows more reward optimization."
    },
    // Step 3: The closed-form solution
    {
      type: "info",
      title: "The Closed-Form Optimal Policy",
      content: "The RLHF objective has a known analytical solution. For a fixed reward function $r$, the optimal policy is:\n\n$$\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp\\!\\left(\\frac{r(x,y)}{\\beta}\\right)$$\n\nwhere $Z(x) = \\sum_y \\pi_{\\text{ref}}(y|x) \\exp(r(x,y)/\\beta)$ is the partition function (normalizing constant).\n\nThis is an energy-based model: the optimal policy re-weights the reference policy by $\\exp(r/\\beta)$, amplifying high-reward responses and suppressing low-reward ones. The temperature $\\beta$ controls how aggressively: small $\\beta$ makes the re-weighting extreme, large $\\beta$ keeps it moderate.\n\nWe can verify this is optimal by noting it satisfies the KKT conditions for the constrained optimization. The derivation uses calculus of variations on the functional $J[\\pi] = \\mathbb{E}_\\pi[r] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ with the constraint that $\\pi(\\cdot|x)$ is a valid distribution.\n\nThe problem: we can't directly compute $Z(x)$ because it sums over **all possible responses** $y$ — an astronomically large set. This is why RLHF uses RL (PPO) instead — it approximates the optimization without needing $Z(x)$."
    },
    // Step 4: MC — optimal policy
    {
      type: "mc",
      question: "The optimal policy $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(x,y)/\\beta)$ can be interpreted as: starting from the reference policy, which responses get amplified?",
      options: [
        "Only responses above the mean reward $\\bar{r}(x) = \\mathbb{E}_{\\pi_{\\text{ref}}}[r(x,y)]$ — all below-mean responses are set to exactly zero probability by the partition function normalization",
        "All responses equally — the exponential factors are absorbed into the normalizing constant $Z(x)$, so the optimal policy is identical to the reference distribution",
        "Higher-reward responses get exponentially more mass relative to the reference, but all responses retain nonzero probability since $\\exp(r/\\beta) > 0$ for any finite $r$",
        "Only the single highest-reward response retains probability mass, because the exponential amplification creates a winner-take-all dynamic for any finite $\\beta$"
      ],
      correct: 2,
      explanation: "Since $\\exp(r/\\beta) > 0$ for all finite $r$, every response that had nonzero probability under $\\pi_{\\text{ref}}$ retains nonzero probability under $\\pi^*$. But the ratio of probabilities for two responses changes exponentially with their reward difference: $\\pi^*(y_1)/\\pi^*(y_2) = (\\pi_{\\text{ref}}(y_1)/\\pi_{\\text{ref}}(y_2)) \\cdot \\exp((r_1 - r_2)/\\beta)$. Small $\\beta$ amplifies this exponentially, concentrating mass on high-reward responses. Large $\\beta$ keeps the distribution closer to the reference. No response is zeroed out — this is the \"soft\" nature of KL-regularized optimization."
    },
    // Step 5: DPO's key rearrangement
    {
      type: "info",
      title: "DPO's Key Insight: Rearranging the Optimal Policy",
      content: "Here is where DPO (Rafailov et al., 2023) makes its breakthrough. Start from the optimal policy and **solve for the reward**:\n\n$$\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp\\!\\left(\\frac{r(x,y)}{\\beta}\\right)$$\n\nTake the log:\n$$\\log \\pi^*(y|x) = \\log \\pi_{\\text{ref}}(y|x) + \\frac{r(x,y)}{\\beta} - \\log Z(x)$$\n\nRearrange for $r$:\n$$r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$$\n\nThis expresses the reward as a function of the **log-probability ratio** between the optimal policy and the reference, plus the partition function.\n\nNow substitute this into the **Bradley-Terry model** for human preferences, which says:\n$$P(y_w \\succ y_l | x) = \\sigma(r(x, y_w) - r(x, y_l))$$\n\nwhere $\\sigma$ is the sigmoid function. The crucial cancellation: $\\beta \\log Z(x)$ appears in both $r(x, y_w)$ and $r(x, y_l)$, so it **cancels in the difference**. The intractable partition function disappears completely."
    },
    // Step 6: MC — the derivation
    {
      type: "mc",
      question: "Why does the partition function $Z(x)$ cancel when the reward is substituted into the Bradley-Terry preference model $P(y_w \\succ y_l) = \\sigma(r_w - r_l)$?",
      options: [
        "Bradley-Terry depends only on the reward difference $r_w - r_l$, and $\\beta \\log Z(x)$ is an additive prompt-level constant that contributes equally to both rewards, so it cancels exactly",
        "The partition function $Z(x)$ equals exactly 1 whenever the reference policy is properly normalized, so $\\log Z(x) = 0$ and there is nothing that needs to cancel in the first place",
        "The sigmoid $\\sigma$ satisfies $\\sigma(a + c) - \\sigma(b + c) = \\sigma(a) - \\sigma(b)$ for any constant $c$, so it absorbs the partition function term into its output bias",
        "DPO approximates $Z(x) \\approx 1$ via a first-order Taylor expansion around the reference policy, making the cancellation approximate but negligible in practice"
      ],
      correct: 0,
      explanation: "The reward difference is: $r_w - r_l = \\beta \\log \\frac{\\pi^*(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} + \\beta \\log Z(x) - \\beta \\log \\frac{\\pi^*(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} - \\beta \\log Z(x)$. The $\\beta \\log Z(x)$ terms cancel exactly. This works because Bradley-Terry depends only on the reward DIFFERENCE, and $Z(x)$ is a prompt-level constant that contributes equally to both rewards. The cancellation is exact, not approximate — this is the mathematical elegance at the heart of DPO."
    },
    // Step 7: The DPO loss
    {
      type: "info",
      title: "The DPO Loss Function",
      content: "After the partition function cancels, we replace the optimal policy $\\pi^*$ with our trainable policy $\\pi_\\theta$ and get the **DPO loss**:\n\n$$\\mathcal{L}_{\\text{DPO}}(\\theta) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}}\\left[\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$\n\nDefine the **implicit reward** of a response as:\n$$\\hat{r}_\\theta(x, y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$$\n\nThen the loss simplifies to:\n$$\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\hat{r}_\\theta(x, y_w) - \\hat{r}_\\theta(x, y_l)\\right)\\right]$$\n\nThis is just binary cross-entropy: the loss is low when the implicit reward of the preferred response $y_w$ exceeds that of the dispreferred response $y_l$. No reward model is needed — the policy itself serves as an implicit reward model via the log-probability ratio.\n\nThe inputs are simply preference pairs $(x, y_w, y_l)$: a prompt, a preferred response, and a dispreferred response. The training loop is pure supervised learning — forward pass through $\\pi_\\theta$ and $\\pi_{\\text{ref}}$, compute log-probabilities, apply the loss."
    },
    // Step 8: MC — loss interpretation
    {
      type: "mc",
      question: "The DPO implicit reward is $\\hat{r}_\\theta(x, y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$. If $\\pi_\\theta = \\pi_{\\text{ref}}$ (training has not started), what is the implicit reward for any response?",
      options: [
        "It equals the true reward $r(x, y)$ from the reward model, since at initialization the policy perfectly reflects the ground-truth preference ordering",
        "It is undefined because dividing by $\\pi_{\\text{ref}}$ when $\\pi_\\theta = \\pi_{\\text{ref}}$ creates a $0/0$ indeterminate form for low-probability responses",
        "It is exactly zero for every response, since $\\log(\\pi_{\\text{ref}}/\\pi_{\\text{ref}}) = \\log(1) = 0$ — meaning the model starts with no preference between any responses",
        "It is a large negative number for all responses because the initial policy has not learned any preferences yet"
      ],
      correct: 2,
      explanation: "When $\\pi_\\theta = \\pi_{\\text{ref}}$: $\\hat{r} = \\beta \\log(\\pi_{\\text{ref}}/\\pi_{\\text{ref}}) = \\beta \\log(1) = 0$ for all $(x, y)$. Every response has the same implicit reward of zero. The DPO loss at initialization becomes $-\\log \\sigma(0 - 0) = -\\log(0.5) = \\log 2 \\approx 0.693$ — equivalent to random guessing on the preference. Training then moves the policy to assign higher log-probability (relative to the reference) to preferred responses and lower to dispreferred ones."
    },
    // Step 9: DPO gradient dynamics
    {
      type: "info",
      title: "How DPO Learns: Gradient Dynamics",
      content: "The DPO gradient has an elegant structure. For a preference pair $(x, y_w, y_l)$:\n\n$$\\nabla_\\theta \\mathcal{L}_{\\text{DPO}} = -\\underbrace{\\left(1 - \\sigma(\\hat{r}_w - \\hat{r}_l)\\right)}_{\\text{weighting}} \\cdot \\left[\\beta \\nabla_\\theta \\log \\pi_\\theta(y_w|x) - \\beta \\nabla_\\theta \\log \\pi_\\theta(y_l|x)\\right]$$\n\nThe gradient does two things simultaneously:\n- **Increases** $\\log \\pi_\\theta(y_w|x)$ — makes the preferred response more likely\n- **Decreases** $\\log \\pi_\\theta(y_l|x)$ — makes the dispreferred response less likely\n\nThe weighting factor $1 - \\sigma(\\hat{r}_w - \\hat{r}_l)$ is an **implicit curriculum**:\n- When the model already correctly ranks the pair ($\\hat{r}_w \\gg \\hat{r}_l$): the weight $\\approx 0$, gradient vanishes — no need to learn this pair further\n- When the model incorrectly ranks the pair ($\\hat{r}_w \\ll \\hat{r}_l$): the weight $\\approx 1$, full gradient signal — focus learning here\n\nThis means DPO automatically concentrates learning effort on the pairs the model currently gets wrong. Already-learned preferences don't waste gradient budget."
    },
    // Step 10: MC — gradient dynamics
    {
      type: "mc",
      question: "During DPO training, the model's implicit reward gap $\\hat{r}_w - \\hat{r}_l$ for a specific pair is $+5.0$ (strongly preferring the correct response). What happens to the gradient for this pair?",
      options: [
        "The gradient is at maximum magnitude because the model is confident, and DPO continues to reinforce confident correct predictions to increase the reward margin further",
        "The gradient is moderately large at $(1 - \\sigma(5)) \\approx 0.007$, providing a steady but slowly diminishing learning signal that fine-tunes the margin between preferred and dispreferred",
        "The gradient flips sign because the model is overconfident — DPO applies an implicit regularization penalty to prevent the reward margin from growing beyond a stability threshold",
        "The gradient is near zero at $(1 - \\sigma(5)) \\approx 0.007$, so this pair contributes almost no learning signal — DPO focuses effort on pairs the model still ranks incorrectly"
      ],
      correct: 3,
      explanation: "$\\sigma(5) \\approx 0.993$, so the weighting is $1 - 0.993 = 0.007$. The gradient for this pair is effectively zero — the model has already learned this preference, so DPO moves on. This is an elegant property: learning is self-paced without any explicit curriculum design. In contrast, a naive MSE loss on preference margins would still produce gradients on correctly-ranked pairs, wasting capacity. However, this property can also be a weakness: DPO can become too confident on easy pairs early in training, ignoring harder pairs."
    },
    // Step 11: DPO vs PPO-based RLHF
    {
      type: "info",
      title: "DPO vs PPO: Simplicity vs Flexibility",
      content: "DPO and PPO-based RLHF optimize the same KL-constrained objective, but through very different mechanisms:\n\n| Aspect | PPO-based RLHF | DPO |\n|---|---|---|\n| **Models needed** | 4 (policy, ref, reward, value) | 2 (policy, ref) |\n| **Training loop** | Generate → Score → RL update | Forward pass → Loss |\n| **Data** | On-policy (regenerated each step) | Offline (fixed preference pairs) |\n| **Stability** | Notoriously finicky | Relatively stable |\n| **Distribution shift** | None (on-policy) | Yes (offline data) |\n\n**DPO's advantages**: Simpler pipeline, fewer hyperparameters, no reward model training, no RL instabilities, lower memory (2 models instead of 4).\n\n**PPO's advantages**: On-policy — the model trains on its own outputs, avoiding distribution shift. As $\\pi_\\theta$ evolves, PPO generates fresh responses that reflect the current policy. DPO's offline data was generated by a different policy (usually the SFT model), and the log-probabilities $\\log \\pi_\\theta(y|x)$ become less meaningful for responses the current $\\pi_\\theta$ would never generate.\n\nThis distribution shift issue has led to **online DPO** (also called iterative DPO): periodically regenerate preference pairs using the current policy, combining DPO's simpler loss with PPO's on-policy property."
    },
    // Step 12: MC — DPO vs PPO tradeoffs
    {
      type: "mc",
      question: "A team trains a model with offline DPO on 100K preference pairs. After training, they evaluate and find that the model has learned to assign high implicit reward to the preferred responses in the training set, but the model's actual generation quality hasn't improved much. What is the most likely explanation?",
      options: [
        "The preference pairs are too easy — both $y_w$ and $y_l$ are high-quality, so distinguishing them teaches the model nothing about how to improve its own generations",
        "Distribution shift: the model learned correct implicit rewards for the training responses (from a different policy), but these preferences don't transfer to its own generation distribution",
        "The $\\beta$ parameter is too large, causing the KL penalty to dominate and prevent any meaningful divergence from the reference policy's generation behavior during inference",
        "DPO fundamentally cannot improve generation quality — it only modifies the implicit reward function without affecting the autoregressive sampling distribution at inference time"
      ],
      correct: 1,
      explanation: "This is the classic distribution shift problem. The model learns to assign correct implicit rewards to the training responses (which were generated by a different policy), but its own generation distribution has shifted away from those responses. At inference, the model generates responses unlike $y_w$ or $y_l$, and the learned preference signal doesn't transfer. The model effectively memorized the preferences on out-of-distribution data. Online DPO addresses this by regenerating pairs from the current policy, keeping the training distribution aligned with the model's actual generation behavior."
    },
    // Step 13: The reference model's role
    {
      type: "info",
      title: "The Reference Model: Anchor and Regularizer",
      content: "The reference model $\\pi_{\\text{ref}}$ plays a crucial dual role in DPO:\n\n**1. Implicit regularization**: The implicit reward $\\hat{r} = \\beta \\log(\\pi_\\theta / \\pi_{\\text{ref}})$ measures how much the policy has changed from the reference. Large implicit rewards require large probability shifts, which the KL penalty resists. This prevents the policy from degenerating.\n\n**2. Reward anchoring**: The reference policy defines what \"normal\" generation looks like. The implicit reward is measured relative to this baseline — a response that the reference model already considered likely gets less implicit reward credit than one the reference considered unlikely but the trained policy now favors.\n\nIn practice, $\\pi_{\\text{ref}}$ is a frozen copy of the SFT model. It's loaded alongside the policy during training, doubling memory requirements. This is one of DPO's main costs.\n\nThe $\\beta$ parameter controls how tightly the policy is anchored:\n- **Large $\\beta$**: Strong regularization, conservative updates, policy stays close to reference\n- **Small $\\beta$**: Weak regularization, aggressive optimization, policy can diverge significantly\n\nTypical values: $\\beta \\in [0.1, 0.5]$. Values below 0.05 risk reward hacking; values above 1.0 produce negligible alignment effect."
    },
    // Step 14: MC — integration
    {
      type: "mc",
      question: "A researcher wants to align a 70B model using DPO. They have 50K preference pairs, 8 A100 GPUs, and must keep the reference model loaded during training. What is the primary engineering challenge?",
      options: [
        "The 50K preference pairs are insufficient — DPO requires at least 500K pairs for models above 10B parameters to achieve convergence on the implicit reward landscape",
        "DPO's gradient computation is $O(n^3)$ in sequence length due to the log-probability ratio calculation, making it prohibitive for the long responses typical of 70B models",
        "The 70B model's vocabulary is too large for the softmax in the implicit reward, requiring sparse approximation techniques to compute per-token log-probabilities efficiently",
        "Memory: both $\\pi_\\theta$ and $\\pi_{\\text{ref}}$ must fit in GPU memory simultaneously — at ~140 GB each in FP16, that's ~280 GB for weights alone before optimizer states"
      ],
      correct: 3,
      explanation: "DPO requires forward passes through both $\\pi_\\theta$ and $\\pi_{\\text{ref}}$ on the same batch to compute the log-probability ratio. For a 70B model in FP16: each model is $\\sim$140 GB. Both together: $\\sim$280 GB. Plus optimizer states for $\\pi_\\theta$ ($\\sim$840 GB with Adam). With 8 A100s (640 GB HBM total), this requires FSDP/ZeRO-3 to shard both models across GPUs, and techniques like LoRA or quantizing $\\pi_{\\text{ref}}$ to reduce memory further. Some implementations compute $\\pi_{\\text{ref}}$ log-probs offline and cache them, eliminating the need to load $\\pi_{\\text{ref}}$ during training."
    }
  ]
};
