// Focused module: Generalized Advantage Estimation (GAE)
// Covers TD errors, n-step returns, the GAE formula, and the bias-variance tradeoff via λ.

export const gaeLearning = {
  id: "A.3-gae-learning-medium",
  sectionId: "A.3",
  title: "Generalized Advantage Estimation (GAE)",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Advantage Estimation Problem",
      content: "Policy gradient methods need an estimate of the **advantage** $A^\\pi(s_t, a_t)$ — how much better was this action compared to the average action? The advantage appears directly in the policy gradient:\n\n$$\\nabla_\\theta J = \\mathbb{E}\\left[\\sum_t \\hat{A}_t \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t)\\right]$$\n\nThe quality of $\\hat{A}_t$ determines training stability. A noisy advantage estimate produces noisy gradients; a biased one pushes the policy in the wrong direction.\n\nTwo extreme approaches exist:\n\n**Monte Carlo (MC)**: Use the actual return $\\hat{A}_t^{MC} = G_t - V(s_t) = \\sum_{k=0}^{T-t} \\gamma^k r_{t+k} - V(s_t)$. This is **unbiased** — on average, it equals the true advantage. But it has **high variance** because a single trajectory's return depends on every future action and reward.\n\n**1-step TD**: Use the TD error $\\hat{A}_t^{TD} = \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$. This has **low variance** (only depends on one step) but is **biased** when $V$ is imperfect — if $V(s_{t+1})$ is wrong, the advantage estimate is systematically off.\n\n**GAE** interpolates between these extremes using a single parameter $\\lambda$."
    },
    {
      type: "mc",
      question: "A perfectly accurate value function $V = V^\\pi$ would make the 1-step TD advantage $\\delta_t = r_t + \\gamma V^\\pi(s_{t+1}) - V^\\pi(s_t)$ have what property?",
      options: [
        "It would be unbiased — its expectation equals the true advantage — but still have variance because the immediate reward $r_t$ and next state $s_{t+1}$ are stochastic",
        "It would have zero variance since the value function deterministically predicts future returns at every state",
        "It would be both unbiased and have zero variance, making GAE and all other advantage estimators unnecessary",
        "It would be biased toward zero because the TD error subtracts the value from itself, canceling out the advantage signal"
      ],
      correct: 0,
      explanation: "With a perfect value function, $\\mathbb{E}[\\delta_t | s_t, a_t] = A^\\pi(s_t, a_t)$ — the TD error is unbiased. But variance remains because $r_t$ and the specific $s_{t+1}$ reached are still stochastic (in general MDPs). In the LLM case, transitions are deterministic given the action, and reward is typically sparse, so the remaining variance comes from the sampled action's effect on future rewards. The key insight: even a perfect $V$ doesn't eliminate variance from the 1-step estimator."
    },
    {
      type: "info",
      title: "N-step Returns: Between MC and TD",
      content: "Before GAE, let's understand **n-step returns** — they interpolate between TD and MC by looking ahead $n$ steps before bootstrapping from $V$:\n\n$$\\hat{A}_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k r_{t+k} + \\gamma^n V(s_{t+n}) - V(s_t)$$\n\n- $n = 1$: $\\hat{A}_t^{(1)} = r_t + \\gamma V(s_{t+1}) - V(s_t) = \\delta_t$ — the TD error\n- $n = 2$: $\\hat{A}_t^{(2)} = r_t + \\gamma r_{t+1} + \\gamma^2 V(s_{t+2}) - V(s_t)$\n- $n = T - t$: Full Monte Carlo return $G_t - V(s_t)$ — no bootstrapping\n\nAs $n$ increases:\n- **Bias decreases**: we use more actual rewards and rely less on the potentially inaccurate $V$\n- **Variance increases**: we accumulate more stochastic terms\n\nBut which $n$ is best? The answer depends on how accurate $V$ is (lower $n$ if $V$ is good) and how noisy trajectories are (lower $n$ if rewards are noisy). Rather than choosing a single $n$, GAE takes a **weighted average** over all possible $n$."
    },
    {
      type: "mc",
      question: "In RLHF where reward is sparse (only at the final token), the 1-step TD error at intermediate token positions is $\\delta_t = 0 + \\gamma V(s_{t+1}) - V(s_t)$ (since $r_t = 0$ for $t < T$). What does this imply?",
      options: [
        "The 1-step TD advantage is always zero at intermediate positions, so the policy receives no gradient signal during most of the sequence",
        "Sparse rewards make TD learning impossible because the Bellman equation requires non-zero immediate rewards at every timestep to propagate value information",
        "The TD error becomes negative at all intermediate steps because $V(s_{t+1}) < V(s_t)$ must hold when no reward is received",
        "The 1-step TD advantage depends entirely on how the value function changes between consecutive states — it measures whether appending token $y_t$ increased or decreased the predicted return"
      ],
      correct: 3,
      explanation: "With $r_t = 0$, $\\delta_t = \\gamma V(s_{t+1}) - V(s_t)$. This is the change in predicted value from one token to the next. If the value function predicts that adding token $y_t$ improved the expected final reward ($V(s_{t+1}) > V(s_t)/\\gamma$), then $\\delta_t > 0$ and that token gets reinforced. The value function essentially provides per-token credit assignment by tracking how each token changes the predicted outcome. This is exactly why an accurate $V$ is so valuable in sparse-reward settings."
    },
    {
      type: "info",
      title: "The GAE Formula",
      content: "**Generalized Advantage Estimation** defines the advantage as an exponentially-weighted average of n-step advantages:\n\n$$\\hat{A}_t^{GAE(\\gamma, \\lambda)} = \\sum_{l=0}^{T-t} (\\gamma \\lambda)^l \\delta_{t+l}$$\n\nwhere $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ is the TD error at each step.\n\nExpanding this:\n$$\\hat{A}_t^{GAE} = \\delta_t + (\\gamma\\lambda)\\delta_{t+1} + (\\gamma\\lambda)^2 \\delta_{t+2} + \\cdots$$\n\nThe parameter $\\lambda \\in [0, 1]$ controls the tradeoff:\n\n- **$\\lambda = 0$**: $\\hat{A}_t = \\delta_t$ — pure 1-step TD. Low variance, potentially high bias.\n- **$\\lambda = 1$**: $\\hat{A}_t = \\sum_{l=0}^{T-t} \\gamma^l \\delta_{t+l} = G_t - V(s_t)$ — Monte Carlo return minus baseline. Unbiased, high variance.\n- **$\\lambda = 0.95$** (standard in PPO for RLHF): mostly looks at the full trajectory but gently downweights distant TD errors, providing a practical bias-variance sweet spot.\n\nThe exponential weighting $(\\gamma\\lambda)^l$ means recent TD errors contribute more than distant ones. This is analogous to an exponential moving average — recent information is trusted more."
    },
    {
      type: "mc",
      question: "GAE with $\\lambda = 0.95$ and $\\gamma = 1.0$ weights the TD error at $l$ steps ahead by $(\\gamma\\lambda)^l = 0.95^l$. After how many steps does the weight drop below 10% of the immediate TD error's weight?",
      options: [
        "About 5 steps — $0.95^5 \\approx 0.77$, but the compound effect with the reward terms brings the effective weight below 10%",
        "About 45 steps — $0.95^{45} \\approx 0.099$, so TD errors beyond 45 tokens ahead contribute less than 10% weight",
        "About 22 steps — solving $0.95^l < 0.1$ gives $l > \\ln(0.1)/\\ln(0.95) \\approx 44.9$, but per-token effects halve this",
        "About 100 steps — the exponential decay is very slow at $\\lambda = 0.95$, maintaining above 10% weight for nearly the full context window"
      ],
      correct: 1,
      explanation: "Solving $0.95^l < 0.1$: $l > \\ln(0.1)/\\ln(0.95) = (-2.303)/(-0.0513) \\approx 44.9$. So at $l=45$, $0.95^{45} \\approx 0.099$. For a 200-token response, the first ~45 future TD errors carry most of the weight. This is a reasonable horizon — it captures enough future context for credit assignment without the full Monte Carlo variance. For longer sequences, this effective window means GAE focuses on relatively local information."
    },
    {
      type: "info",
      title: "Computing GAE Efficiently: Backward Pass",
      content: "GAE is computed via a simple backward recursion, which is much more efficient than naively summing over all future TD errors:\n\n$$\\hat{A}_T = \\delta_T$$\n$$\\hat{A}_t = \\delta_t + \\gamma \\lambda \\cdot \\hat{A}_{t+1}$$\n\nStarting from the last timestep and working backward, each advantage is the current TD error plus a discounted version of the next advantage. This is $O(T)$ — linear in the sequence length.\n\nIn code (pseudocode):\n```\ndelta[t] = reward[t] + gamma * V[t+1] - V[t]\ngae = 0\nfor t = T, T-1, ..., 0:\n    gae = delta[t] + gamma * lambda * gae\n    advantage[t] = gae\n```\n\nThis recursive structure mirrors how the exponential weights work: each step inherits the accumulated future information, discounted by $\\gamma\\lambda$. The implementation is nearly trivial once you have the TD errors — just one backward loop over the sequence.\n\nNote: the value predictions $V(s_t)$ are computed in a single forward pass through the value network before GAE computation begins. The GAE calculation itself is pure arithmetic — no neural network calls."
    },
    {
      type: "mc",
      question: "The backward recursion $\\hat{A}_t = \\delta_t + \\gamma\\lambda \\hat{A}_{t+1}$ processes a 512-token response. What is the computational complexity of GAE relative to a forward pass through the value network?",
      options: [
        "GAE dominates the cost because the recursion requires $O(T^2)$ operations due to the expanding sum at each position",
        "GAE is negligible — it requires one pass of $O(T)$ scalar additions and multiplications, while the value network forward pass involves $O(T \\cdot d^2)$ matrix operations across transformer layers",
        "GAE and the value forward pass have equal cost because both process $T$ tokens sequentially with the same amount of computation per token",
        "GAE is more expensive because it requires backpropagation through the value network at each of the $T$ timesteps to compute TD error gradients"
      ],
      correct: 1,
      explanation: "The GAE computation is 512 scalar multiply-and-add operations — essentially free compared to a transformer forward pass, which involves matrix multiplications of dimension $d \\times d$ (e.g., $d=4096$) across dozens of layers. The value network forward pass is $O(T \\cdot L \\cdot d^2)$ where $L$ is the number of layers. GAE computation is negligible in wall-clock time. Note: GAE does not require backpropagation through $V$ — it uses the value predictions as fixed targets."
    },
    {
      type: "info",
      title: "How $\\lambda$ Affects Training in Practice",
      content: "Choosing $\\lambda$ is one of the most important hyperparameter decisions in PPO-based RLHF:\n\n**$\\lambda$ too low (e.g., 0.5)**:\n- Heavy reliance on value function predictions\n- If $V$ is inaccurate (common early in training), advantages are systematically biased\n- The policy may learn to exploit value function errors rather than maximize actual reward\n- Credit assignment is too local — the model doesn't learn long-range dependencies in response quality\n\n**$\\lambda$ too high (e.g., 1.0)**:\n- Monte Carlo-like behavior — advantages have high variance\n- Training is noisy, requiring larger batch sizes to compensate\n- But: zero bias from value function errors, which can be valuable when $V$ is unreliable\n\n**$\\lambda = 0.95$** (the standard for LLM RLHF):\n- Effective lookahead of ~45 tokens (as we computed)\n- Tolerant of moderate value function errors\n- Empirically stable across a wide range of model sizes and tasks\n\nNote that $\\lambda$ interacts with $\\gamma$: the effective discount per step is $\\gamma\\lambda$. With $\\gamma = 1.0$ (common in LLM RL), $\\lambda$ alone controls the bias-variance tradeoff. With $\\gamma = 0.99$ and $\\lambda = 0.95$, the effective per-step discount is $0.99 \\times 0.95 = 0.9405$."
    },
    {
      type: "mc",
      question: "A team observes that their PPO training diverges after a few hundred steps. The value function loss is still high (the critic hasn't converged). Which $\\lambda$ adjustment would most likely help?",
      options: [
        "Decrease $\\lambda$ from 0.95 to 0.5 to rely more on the value function, which will force the critic to converge faster through stronger gradient signal",
        "Set $\\lambda = 0$ to use pure TD errors, completely eliminating variance from future trajectory noise",
        "Increase $\\lambda$ toward 1.0 to reduce dependence on the inaccurate value function, accepting higher variance but removing the bias from a poorly-trained critic",
        "Remove $\\lambda$ entirely and switch to a fixed 10-step return, which provides a compromise without the complexity of exponential weighting"
      ],
      correct: 2,
      explanation: "When the value function is inaccurate, low $\\lambda$ is dangerous: $\\hat{A}_t \\approx \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ feeds wrong $V$ predictions directly into advantages, creating biased gradients that push the policy in the wrong direction. Increasing $\\lambda$ toward 1.0 makes GAE behave more like Monte Carlo (using actual returns), which is unbiased regardless of $V$'s quality. The higher variance can be managed with larger batch sizes. Decreasing $\\lambda$ would make the problem worse."
    },
    {
      type: "info",
      title: "GAE's Role in the PPO Training Loop",
      content: "GAE fits into the PPO training loop at a specific point:\n\n1. **Rollout phase**: Generate responses using the current policy $\\pi_{\\theta_{\\text{old}}}$. Collect states, actions, and rewards.\n\n2. **Value prediction**: Run all states through the value network $V_\\phi$ in one batched forward pass.\n\n3. **GAE computation**: Compute TD errors $\\delta_t = r_t + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)$, then run the backward recursion to get $\\hat{A}_t$ for every token. Also compute **return targets** $\\hat{R}_t = \\hat{A}_t + V_\\phi(s_t)$ for training the value function.\n\n4. **Optimization phase**: Run multiple epochs of gradient descent on the PPO clipped objective using the computed advantages. Simultaneously update $V_\\phi$ to minimize $(V_\\phi(s_t) - \\hat{R}_t)^2$.\n\nThe advantages and return targets are computed **once** per rollout and reused across optimization epochs. The value function is updated using the return targets from step 3, which creates a moving target problem — but PPO's clipping and limited epochs (typically 1-4) keep this stable.\n\nThis is where GRPO simplifies things: it replaces steps 2-3 entirely with group-normalized rewards."
    },
    {
      type: "mc",
      question: "In step 4 of the PPO loop, the value function is updated to minimize $(V_\\phi(s_t) - \\hat{R}_t)^2$ where $\\hat{R}_t = \\hat{A}_t + V_{\\phi_{\\text{old}}}(s_t)$. Why is $V_{\\phi_{\\text{old}}}$ (the value function from the rollout phase) used in the target, not the current $V_\\phi$?",
      options: [
        "Using the current $V_\\phi$ in the target would create a fixed point where the loss is always zero, regardless of whether the predictions are actually accurate",
        "The old value function is more accurate because it was trained on more data, while the current one has been partially overwritten by the policy gradient updates",
        "The return targets must be fixed during optimization to prevent a moving-target problem — if targets changed each epoch, the value function could chase its own tail and diverge",
        "This is purely for computational efficiency — recomputing GAE at each optimization epoch would require additional forward passes through the value network"
      ],
      correct: 2,
      explanation: "If the target were $\\hat{R}_t = \\hat{A}_t + V_\\phi(s_t)$ with the current $V_\\phi$, then as $V_\\phi$ changes during optimization, the targets would shift. This creates instability: the value function chases a moving target. By fixing $\\hat{R}_t$ using the old value predictions, we give the optimizer a stable regression target. This is analogous to target networks in DQN. The computational efficiency argument is secondary — the stability argument is the fundamental reason."
    },
    {
      type: "mc",
      question: "Which statement correctly describes the relationship between GAE, GRPO, and REINFORCE with baseline?",
      options: [
        "GAE is a generalization of both GRPO and REINFORCE — setting the right hyperparameters recovers either method exactly",
        "All three methods require a learned value function but differ in how they use it: GAE for per-token advantages, GRPO for group normalization, REINFORCE for the baseline subtraction",
        "GRPO is strictly superior to GAE because it achieves zero bias while GAE always introduces bias through the value function bootstrap",
        "GAE provides per-token advantage estimates using a value function and $\\lambda$-weighted TD errors; REINFORCE with baseline uses a value function but with sequence-level MC returns; GRPO bypasses the value function entirely using group statistics"
      ],
      correct: 3,
      explanation: "GAE uses a value function to compute per-token TD errors, then combines them with $\\lambda$-weighting for bias-variance control. REINFORCE with baseline subtracts $V(s)$ from the Monte Carlo return (equivalent to GAE with $\\lambda=1$). GRPO eliminates the value function entirely, using the group mean reward as a sequence-level baseline. Each has different tradeoffs: GAE gives fine-grained credit assignment but needs a good $V$; GRPO is simpler but coarser. GRPO is not strictly superior — its per-sequence advantages sacrifice the per-token discrimination that GAE provides."
    }
  ]
};
