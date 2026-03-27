// Section A.3: RLHF & Policy Optimization Assessment

export const rlhfAssessment = {
  id: "A.3-assess",
  sectionId: "A.3",
  title: "Assessment: RLHF & Policy Optimization",
  difficulty: "hard",
  estimatedMinutes: 16,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The PPO clipped surrogate objective is $L^{\\text{CLIP}} = \\mathbb{E}_t\\left[\\min\\left(\\rho_t \\hat{A}_t, \; \\text{clip}(\\rho_t, 1 - \\epsilon, 1 + \\epsilon) \\hat{A}_t\\right)\\right]$ where $\\rho_t = \\frac{\\pi_\\theta(a_t \\mid s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t \\mid s_t)}$. When the advantage $\\hat{A}_t > 0$ (good action), what does the clipping achieve?",
      options: ["It caps $\\rho_t$ at $1 + \\epsilon$, preventing the policy from increasing the probability of a good action *too much* in a single update — limiting the step size", "It prevents the probability ratio from dropping below $1 - \\epsilon$, ensuring the policy does not decrease the probability of good actions during optimization", "It forces $\\rho_t = 1$ for all good actions by normalizing the policy ratio, effectively reverting any probability increase beyond the old policy's value", "It removes all gradient signal for good actions to prevent positive feedback loops that would otherwise cause the policy to collapse to a single response"],
      correct: 0,
      explanation: "When $\\hat{A}_t > 0$, we want to increase $\\pi_\\theta(a_t | s_t)$, which increases $\\rho_t$. But $\\min(\\rho_t \\hat{A}_t, (1+\\epsilon) \\hat{A}_t)$ caps the objective at $(1+\\epsilon)\\hat{A}_t$ — beyond $\\rho_t = 1 + \\epsilon$, there's no further incentive to increase the probability. This prevents catastrophically large policy updates that could destabilize training. Symmetrically, when $\\hat{A}_t < 0$, clipping prevents $\\rho_t$ from dropping below $1 - \\epsilon$."
    },
    {
      type: "mc",
      question: "Generalized Advantage Estimation (GAE) defines $\\hat{A}_t^{\\text{GAE}(\\gamma, \\lambda)} = \\sum_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta_{t+l}$ where $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$. The parameter $\\lambda$ trades off:",
      options: [
        "Exploration vs. exploitation: $\\lambda = 0$ favors exploitation of known reward regions, $\\lambda = 1$ encourages broader exploration of the action space",
        "Bias vs. variance: $\\lambda = 0$ gives low-variance but high-bias (1-step TD), $\\lambda = 1$ gives high-variance but unbiased (Monte Carlo)",
        "Learning rate vs. batch size: $\\lambda = 0$ behaves like a small learning rate with large batches, $\\lambda = 1$ like a large learning rate with small batches",
        "Reward scale vs. KL penalty strength: $\\lambda = 0$ upweights the KL term relative to reward, $\\lambda = 1$ upweights reward relative to the KL penalty"
      ],
      correct: 1,
      explanation: "At $\\lambda = 0$: $\\hat{A}_t = \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, the 1-step TD error. This has low variance (uses the value function baseline) but is biased if $V$ is inaccurate. At $\\lambda = 1$: $\\hat{A}_t = \\sum_l \\gamma^l r_{t+l} - V(s_t)$, the Monte Carlo return minus baseline. This is unbiased but high variance. Intermediate $\\lambda$ (commonly 0.95) interpolates, providing a practical bias-variance tradeoff. In RLHF, $\\lambda \\approx 0.95$ is standard."
    },
    {
      type: "mc",
      question: "In the RLHF objective $\\max_\\pi \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi(\\cdot | x)}[r(x, y)] - \\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, increasing $\\beta$ has what effect?",
      options: ["Increases reward but decreases the KL penalty term, allowing the policy to explore more aggressively while still remaining on the reference distribution", "Always improves both reward and KL simultaneously by tightening the constraint surface, making the optimization landscape easier to navigate for the policy", "Has no effect on the policy because $\\beta$ cancels out when the KL and reward gradient terms are combined, leaving the effective update direction unchanged", "Makes the policy more conservative — it stays closer to $\\pi_{\\text{ref}}$ at the cost of less reward optimization, reducing reward hacking but limiting improvement"],
      correct: 3,
      explanation: "Higher $\\beta$ increases the cost of diverging from $\\pi_{\\text{ref}}$. The optimal policy is $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. As $\\beta \\to \\infty$, $\\pi^* \\to \\pi_{\\text{ref}}$ (no adaptation). As $\\beta \\to 0$, $\\pi^*$ concentrates on the reward-maximizing response (maximum reward hacking risk). In practice, $\\beta$ is the primary knob for controlling the reward-quality tradeoff, and is often scheduled (starting high, decreasing)."
    },
    {
      type: "mc",
      question: "The RLHF penalty uses **forward KL** $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, not reverse KL $\\text{KL}(\\pi_{\\text{ref}} \\| \\pi)$. What is the practical reason for this choice?",
      options: ["Forward KL is easier to compute because it only requires samples from the policy $\\pi$, whereas reverse KL requires expensive samples from the frozen reference distribution $\\pi_{\\text{ref}}$", "Forward and reverse KL are identical for autoregressive language models since both distributions share the same support over the full token vocabulary at each position", "Forward KL penalizes $\\pi$ for placing mass where $\\pi_{\\text{ref}}$ has low mass — preventing the policy from generating implausible text that the base model would never produce", "Reverse KL causes numerical overflow in all practical cases because the log-ratio $\\log \\frac{\\pi_{\\text{ref}}}{\\pi}$ explodes whenever the reference assigns higher probability than the policy"],
      correct: 2,
      explanation: "Forward KL: $\\text{KL}(\\pi \\| \\pi_{\\text{ref}}) = \\mathbb{E}_\\pi[\\log \\pi / \\pi_{\\text{ref}}]$. The expectation is under $\\pi$, so if $\\pi$ generates text $y$ where $\\pi_{\\text{ref}}(y) \\approx 0$, the log-ratio explodes. This directly prevents the policy from discovering adversarial outputs that the base model would never produce. Reverse KL $\\text{KL}(\\pi_{\\text{ref}} \\| \\pi)$ would instead penalize $\\pi$ for *not covering* modes of $\\pi_{\\text{ref}}$, which is the wrong inductive bias for alignment."
    },
    {
      type: "mc",
      question: "GRPO (Group Relative Policy Optimization), used in DeepSeek-R1, eliminates the value network by:",
      options: ["Sampling a group of responses per prompt, computing rewards, and using the group-normalized advantage $\\hat{A}_i = \\frac{r_i - \\text{mean}(\\mathbf{r})}{\\text{std}(\\mathbf{r})}$ as the baseline instead of a learned critic", "Using a fixed constant reward for all responses and relying solely on the per-token KL penalty to differentiate between high-quality and low-quality generated outputs", "Training the policy and value function with fully shared parameters so that the value head provides an implicit baseline without requiring a separate critic network", "Using Monte Carlo tree search over the token space instead of a value function, with the reward model scoring complete leaf-node sequences for backpropagation"],
      correct: 0,
      explanation: "GRPO samples $G$ completions $\\{y_1, \\ldots, y_G\\}$ for each prompt $x$, scores them with the RM, then normalizes: $\\hat{A}_i = (r_i - \\mu_G) / \\sigma_G$. This group-level normalization serves as a variance-reducing baseline without needing a learned value function. Benefits: (1) removes the value network (halving GPU memory), (2) avoids value function approximation error, (3) naturally handles reward scale differences across prompts. The key insight is that relative ranking within a group is sufficient for policy improvement."
    },
    {
      type: "mc",
      question: "The importance sampling ratio $\\rho_t = \\frac{\\pi_\\theta(a_t | s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t | s_t)}$ in PPO can cause training instability when:",
      options: [
        "$\\rho_t$ is always exactly 1.0 throughout training, which provides zero gradient signal and completely prevents the policy from updating in any direction",
        "The ratio becomes very large or very small, indicating significant divergence from the old policy — sampled trajectories are no longer representative, causing high-variance gradients",
        "The ratio becomes negative for certain token sequences, which inverts the advantage signal and causes the policy to reinforce bad actions while suppressing good ones",
        "The ratio is complex-valued when the policy assigns imaginary log-probabilities to out-of-vocabulary tokens that are encountered during the generation rollout phase"
      ],
      correct: 1,
      explanation: "Importance sampling corrects for the mismatch between the sampling policy ($\\pi_{\\text{old}}$) and the current policy ($\\pi_\\theta$). When $\\rho_t \\gg 1$, an action that was unlikely under $\\pi_{\\text{old}}$ is now likely under $\\pi_\\theta$ — the correction factor amplifies this sample's contribution, introducing high variance. The variance of importance-weighted estimators scales with $\\mathbb{E}[\\rho^2]$, which can diverge. PPO's clipping directly addresses this by bounding $\\rho_t \\in [1-\\epsilon, 1+\\epsilon]$."
    },
    {
      type: "mc",
      question: "In RLHF for language models, the \"state\" $s_t$ and \"action\" $a_t$ in the MDP formulation are typically defined as:",
      options: ["State = the entire training dataset including all prompts seen so far, Action = the model's weight update vector computed at each gradient optimization step", "State = the reward model's scalar output for the current partial sequence, Action = the per-token KL divergence adjustment applied at each decoding position", "State = the hidden state of the transformer at the final layer for the current position, Action = the attention pattern selected for the next decoding step", "State = the prompt plus all tokens generated so far $(x, y_{<t})$, Action = the next token $y_t$ — the episode terminates when EOS is generated"],
      correct: 3,
      explanation: "RLHF treats autoregressive generation as a token-level MDP: the state is the concatenation of the prompt and all generated tokens so far, and the action is choosing the next token from the vocabulary. The reward is typically sparse — assigned only at the end of generation (from the RM). The episode starts with the prompt and ends at EOS. This framing makes the action space $|V|$ (vocabulary size, typically 32K–100K), and episodes are typically 100–2000 steps long."
    },
    {
      type: "mc",
      question: "A common source of PPO instability in RLHF is the interaction between the value function and the policy. Specifically:",
      options: ["The value function converges too quickly to an inaccurate estimate, locking in a biased baseline that the policy then overfits against throughout the remainder of training", "The value function and policy always converge to identical parameters due to their shared SFT initialization, creating a degenerate feedback loop in the optimization dynamics", "The value function must estimate sequence-level returns from token-level states — its errors propagate through GAE into advantage estimates, causing noisy gradients that can spiral into divergence", "PPO never uses a value function in practice for language model alignment, relying instead on Monte Carlo return estimates that are computed over full-episode rollouts"],
      correct: 2,
      explanation: "In RLHF, the value function $V_\\phi(s_t)$ must predict the expected return (RM score + KL penalties for remaining tokens). This is challenging: (1) the reward is sparse (only at episode end), (2) the value function must generalize across diverse prompts, (3) as the policy changes, the value function's training data shifts. Errors in $V$ directly corrupt advantage estimates via $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, leading to wrong policy gradients. This is why some approaches (GRPO, REINFORCE-based) eliminate the value function entirely."
    },
    {
      type: "mc",
      question: "In the PPO objective for RLHF, the per-token reward is typically defined as $r_t = -\\beta \\log \\frac{\\pi_\\theta(y_t | x, y_{<t})}{\\pi_{\\text{ref}}(y_t | x, y_{<t})}$ for $t < T$ and $r_T = R_{\\text{RM}}(x, y) - \\beta \\log \\frac{\\pi_\\theta(y_T | x, y_{<T})}{\\pi_{\\text{ref}}(y_T | x, y_{<T})}$. Why is the KL penalty applied per-token rather than as a single sequence-level penalty?",
      options: ["Per-token KL provides denser reward signal, enabling better credit assignment — the value function and GAE can attribute KL costs to specific tokens rather than lumping them at sequence end", "Per-token KL is cheaper to compute because it avoids the expensive full-sequence log-probability marginalization that the sequence-level formulation requires at every training step", "Sequence-level KL is not mathematically well-defined for autoregressive models, because the conditional factorization makes the joint distribution intractable to evaluate or differentiate", "Per-token and sequence-level KL penalties produce identical gradient signals and have the same effect on optimization dynamics, so the choice is purely an implementation convenience"],
      correct: 0,
      explanation: "The sequence-level KL decomposes as $\\text{KL}(\\pi \\| \\pi_{\\text{ref}}) = \\sum_t \\mathbb{E}[\\log \\frac{\\pi(y_t | x, y_{<t})}{\\pi_{\\text{ref}}(y_t | x, y_{<t})}]$, so the per-token formulation is mathematically equivalent. However, placing the KL penalty at each token step is crucial for the RL optimization: it provides dense intermediate rewards, making the value function estimation problem much easier and enabling GAE to assign credit at the token level. Without this, the value function must predict the entire future KL from each state, which is much harder."
    },
    {
      type: "mc",
      question: "Consider a simplified RLHF setup with discrete reward $r \\in \\{0, 1\\}$ and a policy $\\pi_\\theta$ parameterized by a single scalar $\\theta$ controlling the probability of a \"good\" action: $\\pi_\\theta(\\text{good}) = \\sigma(\\theta)$. The REINFORCE gradient estimator is $\\nabla_\\theta J = \\mathbb{E}_{a \\sim \\pi_\\theta}[r(a) \\nabla_\\theta \\log \\pi_\\theta(a)]$. With a single sample $a$ and no baseline, the variance of this estimator is high because:",
      options: ["The gradient is always zero in expectation for binary rewards, because the positive and negative action contributions cancel out over repeated samples from the policy distribution", "When $r(a) = 0$ (bad action), the gradient is zero regardless of how informative the sample is, and when $r(a) = 1$ (good action), the gradient magnitude depends on $\\pi_\\theta(a)$ — high reward but low probability actions produce large, rare gradient spikes", "The estimator is biased for binary rewards because the discrete reward structure violates the continuous differentiability assumptions that the log-derivative trick requires for unbiased estimation", "REINFORCE cannot be applied to discrete action spaces with binary rewards because the probability ratio gradient requires a continuous action density for proper normalization of the likelihood"],
      correct: 1,
      explanation: "REINFORCE with $r \\in \\{0, 1\\}$: if the sampled action has $r = 0$, we get $\\nabla = 0$ — no learning signal, even though a bad outcome is informative. If $r = 1$, we get $\\nabla = \\nabla_\\theta \\log \\pi_\\theta(a)$, which is large when $\\pi_\\theta(a)$ is small (rare good actions produce gradient spikes). The variance is $\\text{Var}[r \\nabla \\log \\pi] = \\mathbb{E}[r^2 (\\nabla \\log \\pi)^2] - (\\mathbb{E}[r \\nabla \\log \\pi])^2$, which can be enormous. A baseline $b$ (e.g., $V(s)$) replaces $r$ with $r - b$, drastically reducing variance without introducing bias."
    }
  ]
};
