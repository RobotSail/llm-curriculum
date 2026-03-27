// Focused module: Markov Decision Process Foundations for RL in LLMs
// Covers states, actions, rewards, transitions, returns, value functions, and Bellman equations.

export const mdpFoundationsLearning = {
  id: "A.3-mdp-foundations-learning-easy",
  sectionId: "A.3",
  title: "MDP Foundations for Language Model RL",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "Why RL Needs a Formal Framework",
      content: "Reinforcement learning for language models — PPO, GRPO, REINFORCE — all rely on the same underlying mathematical framework: the **Markov Decision Process (MDP)**. Before diving into specific algorithms, we need to understand the MDP abstraction that makes them all possible.\n\nAn MDP formalizes sequential decision-making: an agent takes actions in an environment, receives rewards, and transitions between states. The goal is to find a **policy** (a strategy for choosing actions) that maximizes cumulative reward.\n\nFor language models, the mapping is concrete:\n- **State**: the prompt plus all tokens generated so far\n- **Action**: choosing the next token from the vocabulary\n- **Reward**: a scalar signal (from a reward model, human feedback, or rule-based verifier)\n- **Policy**: the language model itself — $\\pi_\\theta(a_t | s_t)$ is the probability of generating token $a_t$ given context $s_t$\n\nThis module builds the MDP framework from first principles and connects every concept to LLM fine-tuning."
    },
    {
      type: "mc",
      question: "In the MDP formulation of autoregressive generation, what constitutes a single \"action\" by the agent?",
      options: [
        "Selecting one token from the vocabulary at a single decoding step",
        "Generating the entire response sequence from start to end-of-sequence token",
        "Updating the model's parameters via one gradient descent step",
        "Computing the attention scores across all previous tokens in the context"
      ],
      correct: 0,
      explanation: "In the token-level MDP, each action is selecting one token from the vocabulary $\\mathcal{V}$ (typically 32K-128K options). The episode consists of many such actions in sequence. Generating the full response would be the entire episode, not a single action. Parameter updates and attention computations are internal model mechanics, not MDP actions."
    },
    {
      type: "info",
      title: "The MDP Tuple: $(\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)$",
      content: "An MDP is defined by five components:\n\n1. **State space** $\\mathcal{S}$: All possible states the agent can be in. For an LLM, $s_t = (x, y_1, \\dots, y_{t-1})$ — the prompt $x$ concatenated with all tokens generated so far. The state space is enormous: every possible partial sequence.\n\n2. **Action space** $\\mathcal{A}$: The set of actions available at each state. For an LLM, $\\mathcal{A} = \\mathcal{V}$ — the full vocabulary at every step.\n\n3. **Transition function** $P(s_{t+1} | s_t, a_t)$: The probability of moving to state $s_{t+1}$ after taking action $a_t$ in state $s_t$. For autoregressive generation, this is **deterministic**: if you're in state $(x, y_1, \\dots, y_{t-1})$ and take action $y_t$, the next state is always $(x, y_1, \\dots, y_t)$. The token just gets appended.\n\n4. **Reward function** $R(s_t, a_t)$: The immediate reward for taking action $a_t$ in state $s_t$. In RLHF, reward is typically **sparse** — zero for all intermediate tokens, with the reward model score assigned only at the final token.\n\n5. **Discount factor** $\\gamma \\in [0, 1]$: How much future rewards are worth relative to immediate rewards. In LLM RL, $\\gamma = 1$ is common since episodes are finite and relatively short."
    },
    {
      type: "mc",
      question: "The transition function in the LLM-as-MDP formulation is deterministic: given state $s_t$ and action $a_t$, the next state $s_{t+1}$ is fully determined. Where does the stochasticity in RL for LLMs come from?",
      options: [
        "From the reward model, which outputs random scores for the same input with each evaluation",
        "From the policy $\\pi_\\theta(a_t | s_t)$, which samples tokens probabilistically — the randomness is in action selection, not state transitions",
        "From the transition function, which randomly drops tokens with a small probability to regularize generation",
        "From the discount factor $\\gamma$, which is sampled uniformly at each timestep to encourage exploration"
      ],
      correct: 1,
      explanation: "The MDP transition is deterministic (appending a token), but the policy is stochastic — the LLM samples from a categorical distribution over the vocabulary at each step. All randomness comes from this sampling. This is a key feature of the LLM MDP: the environment is simple and deterministic, but the agent's behavior is stochastic. This contrasts with robotics or games where the environment itself is often stochastic."
    },
    {
      type: "info",
      title: "The Markov Property",
      content: "The \"Markov\" in MDP refers to the **Markov property**: the future depends only on the current state, not on how we got there.\n\n$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \\dots) = P(s_{t+1} | s_t, a_t)$$\n\nFor language models, this property holds naturally because the state $s_t = (x, y_1, \\dots, y_{t-1})$ includes the **entire history**. The state is the full context — prompt plus all generated tokens. Given this context, the next state depends only on which token is chosen next, not on the order in which the model \"decided\" to generate previous tokens.\n\nThis is different from, say, a chess game where the state (board position) does not encode how the pieces got there. In the LLM MDP, the state is the full sequence, so the Markov property is automatically satisfied.\n\nThe Markov property is what makes dynamic programming and Bellman equations possible — we can reason about optimal behavior from any state forward, without needing to know the full trajectory history."
    },
    {
      type: "mc",
      question: "A partially observed MDP (POMDP) violates the Markov property because the agent cannot see the full state. Why is the standard LLM generation MDP typically NOT a POMDP?",
      options: [
        "Because LLMs use attention mechanisms that can attend to all previous tokens, giving them perfect memory of the entire generated sequence",
        "Because the vocabulary is finite, which guarantees full observability in any discrete action space MDP",
        "Because the reward model provides complete information about the environment's internal state at each timestep",
        "Because the state is defined as the full sequence $(x, y_{<t})$, which is directly observable — the agent sees its complete history at every step"
      ],
      correct: 3,
      explanation: "The key insight is definitional: we define the state as the full sequence $(x, y_1, \\dots, y_{t-1})$, and this is exactly what the model conditions on. The agent fully observes its state because the state IS the complete history. Attention mechanisms are an implementation detail of how the model processes this state, not the reason it's fully observed. A finite action space doesn't guarantee full observability."
    },
    {
      type: "info",
      title: "Returns and the Discount Factor",
      content: "The agent's goal is to maximize the **return** — the cumulative reward from timestep $t$ onward:\n\n$$G_t = r_t + \\gamma r_{t+1} + \\gamma^2 r_{t+2} + \\cdots = \\sum_{k=0}^{T-t} \\gamma^k r_{t+k}$$\n\nThe discount factor $\\gamma$ controls the tradeoff between immediate and future rewards:\n- $\\gamma = 0$: completely myopic — only care about immediate reward\n- $\\gamma = 1$: no discounting — all future rewards are equally important\n- $\\gamma = 0.99$: slightly prefer sooner rewards (standard in many RL settings)\n\n**In RLHF for LLMs**, $\\gamma = 1$ is standard because:\n1. Episodes are finite (generation ends at EOS or max length)\n2. With sparse reward (RM score only at the end), discounting would shrink the only reward signal\n3. We want the model to optimize for final response quality, not rush to finish\n\nHowever, when per-token KL penalties are used as shaping rewards, $\\gamma < 1$ (typically 0.99-1.0) can help stabilize training by downweighting distant KL penalties."
    },
    {
      type: "mc",
      question: "In RLHF with sparse reward (RM score only at the final token), using $\\gamma = 0.5$ instead of $\\gamma = 1.0$ would:",
      options: [
        "Have no effect because the reward is only at the terminal state, so there are no future rewards to discount at any intermediate step",
        "Double the effective reward signal by compressing future rewards into a higher immediate value at each token position",
        "Shrink the RM reward by $0.5^T$ when viewed from the first token, making early tokens receive almost no gradient signal for optimizing response quality",
        "Improve training stability by preventing the RM reward from dominating the per-token KL penalty terms in the advantage estimate"
      ],
      correct: 2,
      explanation: "With sparse terminal reward $r_T$ and $\\gamma = 0.5$, the return at timestep $t$ is $G_t = \\gamma^{T-t} r_T = 0.5^{T-t} r_T$. For a 200-token response, the first token sees the reward discounted by $0.5^{200} \\approx 10^{-60}$ — essentially zero. This destroys credit assignment for early tokens. With $\\gamma = 1$, every token sees the full reward $G_t = r_T$, giving all tokens equal responsibility for the outcome."
    },
    {
      type: "info",
      title: "Value Functions: $V^\\pi$ and $Q^\\pi$",
      content: "Value functions predict expected future return. They are the core tool for evaluating and improving policies.\n\n**State-value function** $V^\\pi(s)$: The expected return starting from state $s$ and following policy $\\pi$:\n$$V^\\pi(s) = \\mathbb{E}_{\\pi}\\left[G_t \\mid s_t = s\\right] = \\mathbb{E}_{\\pi}\\left[\\sum_{k=0}^{\\infty} \\gamma^k r_{t+k} \\mid s_t = s\\right]$$\n\n**Action-value function** $Q^\\pi(s, a)$: The expected return from taking action $a$ in state $s$, then following $\\pi$:\n$$Q^\\pi(s, a) = \\mathbb{E}_{\\pi}\\left[G_t \\mid s_t = s, a_t = a\\right]$$\n\nThe relationship between them:\n$$V^\\pi(s) = \\mathbb{E}_{a \\sim \\pi(\\cdot|s)}[Q^\\pi(s, a)] = \\sum_a \\pi(a|s) Q^\\pi(s, a)$$\n\nIn RLHF, the value function $V_\\phi(s_t)$ is a learned neural network that predicts: \"given this prompt and partial response, what RM score do I expect the finished response to get?\" This prediction is used as a **baseline** in policy gradient methods to reduce variance."
    },
    {
      type: "mc",
      question: "The advantage function is defined as $A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s)$. What does a positive advantage $A^\\pi(s_t, a_t) > 0$ mean in the context of token generation?",
      options: [
        "Token $a_t$ leads to higher expected return than the average token the policy would have sampled — it is better than the policy's default behavior at this position",
        "Token $a_t$ will always produce a higher-quality response than any other token, regardless of what comes after it",
        "The value function is underestimating the true return, indicating the critic network needs more training on states similar to $s_t$",
        "The reward model assigns a positive score to partial sequences ending in token $a_t$, independent of the remaining generation"
      ],
      correct: 0,
      explanation: "The advantage $A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s)$ measures how much better action $a$ is compared to the policy's average behavior. $V^\\pi(s)$ is the expected return under $\\pi$ (averaging over all tokens the policy might sample). If $Q^\\pi(s, a) > V^\\pi(s)$, then this specific token leads to better outcomes than average. The advantage can be positive or negative — it's a relative measure, not an absolute quality judgment."
    },
    {
      type: "info",
      title: "The Bellman Equation",
      content: "The Bellman equation is the recursive relationship that value functions satisfy. It says: the value of a state equals the immediate reward plus the discounted value of the next state.\n\n$$V^\\pi(s) = \\mathbb{E}_{a \\sim \\pi, s' \\sim P}\\left[r(s, a) + \\gamma V^\\pi(s')\\right]$$\n\nExpanded for the LLM MDP (deterministic transitions, $s' = s \\oplus a$):\n$$V^\\pi(s_t) = \\sum_{a \\in \\mathcal{V}} \\pi_\\theta(a | s_t) \\left[r(s_t, a) + \\gamma V^\\pi(s_t \\oplus a)\\right]$$\n\nwhere $s_t \\oplus a$ means appending token $a$ to the sequence.\n\nThe **Bellman optimality equation** defines the optimal value function $V^*(s) = \\max_\\pi V^\\pi(s)$:\n$$V^*(s) = \\max_a \\left[r(s, a) + \\gamma V^*(s')\\right]$$\n\nIn classical RL (small state spaces), we solve these equations directly via dynamic programming. In LLM RL, the state space is far too large — instead, we approximate $V^\\pi$ with a neural network and use the Bellman equation to define the **temporal difference (TD) error**:\n$$\\delta_t = r_t + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)$$\n\nThe TD error is the building block of advantage estimation (GAE) used in PPO."
    },
    {
      type: "mc",
      question: "The TD error $\\delta_t = r_t + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)$ is zero when:",
      options: [
        "The reward $r_t$ is exactly zero, regardless of the accuracy of the value function estimates",
        "The policy has converged to the optimal policy and no further improvement is possible via any gradient update",
        "The discount factor $\\gamma$ is set to zero, which eliminates all dependence on future state values from the TD computation",
        "The value function perfectly satisfies the Bellman equation at state $s_t$ — the predicted value equals the reward plus the discounted next-state value"
      ],
      correct: 3,
      explanation: "The TD error measures the Bellman residual: how much the value function's prediction $V_\\phi(s_t)$ deviates from the one-step Bellman target $r_t + \\gamma V_\\phi(s_{t+1})$. When $\\delta_t = 0$, the value function is locally consistent with the Bellman equation. Note that this doesn't mean the value function is globally correct — it could be consistently wrong. It also doesn't require the policy to be optimal, just that the value function accurately predicts returns under the current policy."
    },
    {
      type: "info",
      title: "From MDPs to LLM RL Algorithms",
      content: "Every RL algorithm for LLMs maps onto the MDP framework:\n\n**REINFORCE**: Uses the return $G_t$ directly as a signal. No value function needed, but high variance because $G_t$ is a single Monte Carlo sample of the return.\n\n**PPO**: Learns a value function $V_\\phi(s)$ to compute advantages via GAE (which uses TD errors). The clipped objective bounds policy updates. Requires both a policy network and a value network — roughly doubling memory.\n\n**GRPO** (DeepSeek-R1): Eliminates the value network entirely. Instead of learning $V(s)$, it samples a group of responses per prompt and uses the group mean reward as a baseline. The advantage becomes $\\hat{A}_i = (r_i - \\mu) / \\sigma$ — no Bellman equations, no TD errors.\n\n**DPO**: Sidesteps the MDP formulation almost entirely by showing that the optimal policy under the KL-constrained objective can be extracted directly from preference data, without ever computing rewards or returns.\n\nThe progression from REINFORCE → PPO → GRPO → DPO represents increasing simplification: each method removes a component of the MDP machinery while maintaining (or improving) training stability."
    },
    {
      type: "mc",
      question: "GRPO eliminates the value network by using group-normalized rewards as advantages. Which MDP component does this remove the need to estimate?",
      options: [
        "The transition function $P(s'|s,a)$, since group sampling reveals the environment dynamics empirically across multiple rollouts",
        "The state-value function $V^\\pi(s)$, since the group mean reward serves as a prompt-level baseline without needing per-state value predictions",
        "The reward function $R(s,a)$, since normalized advantages replace absolute rewards with relative rankings among completions",
        "The policy $\\pi_\\theta(a|s)$, since GRPO uses a fixed sampling distribution instead of the current policy for generating candidate responses"
      ],
      correct: 1,
      explanation: "GRPO's key innovation is replacing the learned value function $V_\\phi(s)$ with the empirical group mean reward $\\mu_G$. In PPO, the value function estimates $V^\\pi(s_t)$ at every token position to compute TD errors and advantages via GAE. GRPO skips all of this: it scores complete responses, computes $(r_i - \\mu_G)/\\sigma_G$ as the advantage, and applies PPO-style clipping. The reward function is still needed (from the RM), the transition function is still deterministic, and the policy is still $\\pi_\\theta$."
    }
  ]
};
