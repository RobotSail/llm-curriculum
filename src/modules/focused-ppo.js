// Focused module: PPO Mechanics for LLM Fine-Tuning
// Covers trust regions, clipping, the PPO-clip objective, and LLM-specific considerations.

export const ppoMechanicsLearning = {
  id: "A.3-ppo-learning-medium",
  sectionId: "A.3",
  title: "PPO Mechanics for LLM Fine-Tuning",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "The Problem with Vanilla Policy Gradients",
      content: "REINFORCE gives us an unbiased gradient, but there is a deeper problem: **how big should each update be?**\n\nWith supervised learning, if you take a step that is too large, the loss goes up and you correct next step. The loss landscape is relatively smooth near the data.\n\nWith policy gradients, a large update changes $\\pi_\\theta$, which changes the **data distribution** (since the model generates its own training data). A bad step can push the policy into a region where it generates terrible responses, receives bad rewards, and the resulting gradients push it further away. This is a **catastrophic collapse** — the policy degenerates and cannot recover.\n\nThe core issue: we estimated the gradient using samples from $\\pi_{\\theta_{\\text{old}}}$, but the update moves us to $\\pi_{\\theta_{\\text{new}}}$. If $\\pi_{\\theta_{\\text{new}}}$ is far from $\\pi_{\\theta_{\\text{old}}}$, our gradient estimate is stale and unreliable."
    },
    {
      type: "mc",
      question: "A policy gradient update increases the probability of a high-reward response from 0.1 to 0.8 in a single step. Why is this dangerous, even though the response had high reward?",
      options: [
        "The reward model's score was probably miscalibrated for such a rare response",
        "The gradient was estimated when this response had probability 0.1 — the update is valid for small changes near 0.1, not for jumping to 0.8, and the policy's behavior in other regions is now unpredictable",
        "A probability of 0.8 for any single response means the model has collapsed to always producing that response",
        "The advantage estimate used a value function trained on the old policy, which is no longer applicable"
      ],
      correct: 1,
      explanation: "The gradient $\\nabla_\\theta \\log \\pi_\\theta(y)$ is a local linear approximation valid near the current parameters. A step that changes a probability from 0.1 to 0.8 is far outside this linear regime. The model's behavior on all other responses may have changed in unpredictable ways. This is why trust region methods constrain step size — to keep updates within the region where our gradient estimate is reliable."
    },
    {
      type: "info",
      title: "Trust Regions: Constraining the Step",
      content: "The idea behind trust region methods: **only apply the gradient update within a region where we trust our estimate**.\n\nTRPO (Trust Region Policy Optimization) formalized this by solving:\n\n$$\\max_\\theta \\; \\mathbb{E}_{y \\sim \\pi_{\\text{old}}}\\left[\\frac{\\pi_\\theta(y)}{\\pi_{\\text{old}}(y)} A(y)\\right] \\quad \\text{s.t.} \\; \\text{KL}(\\pi_{\\text{old}} \\| \\pi_\\theta) \\leq \\delta$$\n\nThe ratio $\\pi_\\theta(y) / \\pi_{\\text{old}}(y)$ measures how much the new policy differs from the old one for response $y$. The KL constraint keeps the overall policy close.\n\nTRPO works well but requires computing second-order KL derivatives (Fisher information matrix) and conjugate gradient solves — expensive for large models.\n\n**PPO** replaces the hard KL constraint with a **clipping** mechanism in the objective itself, achieving similar trust-region behavior with only first-order gradients."
    },
    {
      type: "mc",
      question: "In the TRPO objective, the importance weight $\\pi_\\theta(y) / \\pi_{\\text{old}}(y)$ reweights advantages from the old policy to apply to the new policy. If this ratio equals 3.0 for a response, what does that mean?",
      options: [
        "The new policy is 3x more likely to generate this response than the old policy — it is being heavily upweighted",
        "The reward for this response is 3x the average reward in the batch",
        "The response was sampled 3 times during data collection",
        "The new policy generates responses 3x faster than the old policy"
      ],
      correct: 0,
      explanation: "The ratio $\\pi_\\theta(y) / \\pi_{\\text{old}}(y) = 3.0$ means the new policy assigns 3x the probability to response $y$ compared to the policy that generated it. This large ratio means the policy has changed substantially for this response — which is exactly what trust region methods try to limit, since our advantage estimate $A(y)$ was computed under $\\pi_{\\text{old}}$."
    },
    {
      type: "info",
      title: "PPO's Clipped Objective",
      content: "PPO defines the clipped surrogate objective:\n\n$$L^{\\text{CLIP}}(\\theta) = \\mathbb{E}\\left[\\min\\left(r_t(\\theta) \\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t\\right)\\right]$$\n\nwhere $r_t(\\theta) = \\pi_\\theta(a_t | s_t) / \\pi_{\\text{old}}(a_t | s_t)$ and $\\epsilon$ is typically 0.1 or 0.2.\n\nThe clipping works differently depending on the sign of the advantage:\n\n**Positive advantage** ($\\hat{A}_t > 0$, good action): We want to increase $r_t$, but the clip caps it at $1 + \\epsilon$. Beyond that, there is no further incentive — the gradient becomes zero. This prevents over-reinforcing a single good response.\n\n**Negative advantage** ($\\hat{A}_t < 0$, bad action): We want to decrease $r_t$, but the clip floors it at $1 - \\epsilon$. Beyond that, the gradient becomes zero. This prevents over-suppressing a single bad response.\n\nThe $\\min$ ensures we take the more **pessimistic** (conservative) estimate. If the unclipped term suggests a larger improvement than the clipped term, we trust the clipped version."
    },
    {
      type: "mc",
      question: "With PPO clip $\\epsilon = 0.2$, a response has positive advantage $\\hat{A} = 1.5$ and the ratio has grown to $r_t = 1.4$ during the current optimization epoch. What is the effective gradient from the PPO objective for this response?",
      options: [
        "The gradient is proportional to $\\hat{A} = 1.5$ because the ratio is within the clip range $[0.8, 1.2]$",
        "The gradient is zero because $r_t = 1.4 > 1 + \\epsilon = 1.2$ and the advantage is positive, so the clipped term is active and flat",
        "The gradient is proportional to $1.2 \\times 1.5 = 1.8$ from the clipped term",
        "The gradient is negative because the ratio has exceeded the trust region"
      ],
      correct: 1,
      explanation: "Since $\\hat{A} > 0$, the clipped term is $\\min(r_t, 1.2) \\times \\hat{A} = 1.2 \\times 1.5$. The unclipped term is $1.4 \\times 1.5$. The $\\min$ selects the clipped term ($1.8 < 2.1$). Since $r_t > 1.2$, the clipped term $1.2 \\times \\hat{A}$ is constant with respect to $\\theta$ — its gradient is zero. The policy has already been reinforced enough for this response; PPO stops pushing."
    },
    {
      type: "info",
      title: "PPO for LLMs: The Training Loop",
      content: "Applying PPO to LLM fine-tuning follows a specific pattern:\n\n**1. Rollout phase**: Given a batch of prompts, generate complete responses from $\\pi_\\theta$. This requires full autoregressive decoding — the most expensive step.\n\n**2. Reward computation**: Score each response with a reward model $r(x, y)$. In RLHF, this is a trained classifier. In other settings (code, math), it can be a verifier.\n\n**3. Advantage estimation**: Use a value function $V_\\phi(x, y_{\\leq t})$ to compute per-token advantages via GAE. The value function is a separate model (or a head on the policy) that predicts expected future reward.\n\n**4. Optimization phase**: Perform $K$ gradient steps on the PPO objective using the collected rollouts. Typically $K = 1$-$4$ epochs. The clipping prevents the policy from moving too far from $\\pi_{\\text{old}}$ even across multiple epochs.\n\n**5. Update reference**: Set $\\pi_{\\text{old}} \\leftarrow \\pi_\\theta$ and repeat.\n\nThe key cost: each iteration requires **generating** complete responses (slow autoregressive sampling), **scoring** them (reward model forward pass), and **optimizing** (multiple backward passes). This is 5-10x more expensive per iteration than SFT."
    },
    {
      type: "mc",
      question: "During PPO's optimization phase, the policy is updated for $K = 4$ epochs on the same batch of rollouts. The rollouts were generated by $\\pi_{\\text{old}}$ at the start. By epoch 4, the importance ratios $r_t$ for some responses have moved outside the clip range. What is the net effect of the clipping on these late-epoch updates?",
      options: [
        "The clipped responses contribute zero gradient, so epochs 3-4 effectively only update parameters for responses still within the clip range",
        "The clipped responses are removed from the batch entirely, reducing the effective batch size",
        "The clipping is relaxed in later epochs to allow the policy to keep improving",
        "All responses contribute equally regardless of clipping because the gradients are averaged across epochs"
      ],
      correct: 0,
      explanation: "When a response's ratio exits the clip range in the direction of its advantage (high ratio for positive advantage, low ratio for negative), the PPO objective becomes flat — zero gradient for that response. In later epochs, more responses hit this ceiling, so the effective learning signal decreases. This is by design: it automatically reduces the update magnitude as the policy moves away from $\\pi_{\\text{old}}$, implementing a soft trust region."
    },
    {
      type: "info",
      title: "The KL Penalty in RLHF-PPO",
      content: "Standard PPO for games uses clipping alone to constrain updates. RLHF adds an additional constraint: a **KL penalty** against the reference policy $\\pi_{\\text{ref}}$ (typically the SFT model).\n\nThe combined objective:\n\n$$\\max_{\\pi_\\theta} \\; \\mathbb{E}_{y \\sim \\pi_\\theta}\\left[r(y) - \\beta \\, \\text{KL}(\\pi_\\theta(\\cdot|x) \\| \\pi_{\\text{ref}}(\\cdot|x))\\right]$$\n\noptimized using PPO's clipped surrogate. This gives **two levels of constraint**:\n\n1. **PPO clipping**: Constrains how far $\\pi_\\theta$ moves from $\\pi_{\\text{old}}$ within each rollout iteration (short-term stability)\n2. **KL penalty**: Constrains how far $\\pi_\\theta$ drifts from $\\pi_{\\text{ref}}$ across all of training (long-term anchoring)\n\nThe clip prevents catastrophic single-step collapses. The KL penalty prevents slow drift away from the reference over many iterations. Both are necessary: clipping alone would allow the policy to wander far from $\\pi_{\\text{ref}}$ through many small steps, while KL alone would not prevent single-step collapses."
    },
    {
      type: "mc",
      question: "A team trains with PPO-RLHF using clip $\\epsilon = 0.2$ and KL coefficient $\\beta = 0.1$. After 10,000 steps, the KL divergence from $\\pi_{\\text{ref}}$ has grown to 15 nats. They decide to increase $\\beta$ to 0.5. What is the expected short-term effect?",
      options: [
        "The policy immediately snaps back to $\\pi_{\\text{ref}}$ because the penalty now dominates the reward",
        "The policy gradually drifts back toward $\\pi_{\\text{ref}}$ as the stronger penalty outweighs the marginal reward gain from further divergence",
        "Nothing changes because the KL penalty is evaluated against $\\pi_{\\text{old}}$, not $\\pi_{\\text{ref}}$",
        "The PPO clipping becomes ineffective because the KL gradient overwhelms the reward gradient"
      ],
      correct: 1,
      explanation: "Increasing $\\beta$ makes the KL penalty term larger relative to the reward. Since KL(π_θ || π_ref) = 15 nats, the penalty is now $0.5 \\times 15 = 7.5$ per sample, which may exceed the marginal reward from further divergence. The policy will gradually move back toward $\\pi_{\\text{ref}}$ as the penalty outweighs reward gains. It won't snap back instantly — gradient descent moves gradually. The clipping and KL penalty are independent constraints (clip is vs $\\pi_{\\text{old}}$, KL is vs $\\pi_{\\text{ref}}$)."
    },
    {
      type: "info",
      title: "Why PPO Matters for Optimizer Research",
      content: "PPO's specific structure creates distinctive gradient patterns that interact with optimizer choice:\n\n**Clipping creates sparse gradients**: When responses hit the clip boundary, they contribute zero gradient. This means a fraction of each batch produces no signal, making the effective gradient sparser than in SFT.\n\n**Multiple epochs create correlated updates**: Optimizing $K$ epochs on the same data means the optimizer sees correlated gradient estimates. Adam's momentum and second-moment estimates are updated $K$ times on related signals, which can cause over-adaptation.\n\n**The KL penalty adds a consistent direction**: The $\\nabla_\\theta \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$ term always points back toward $\\pi_{\\text{ref}}$, adding a constant-direction component to the gradient. An optimizer that handles this directional bias well (rather than dampening it through second-moment scaling) may maintain better alignment stability.\n\n**Token-level advantage weighting**: Different tokens in the same response get different advantage weights, creating within-batch gradient diversity that is qualitatively different from SFT's uniform token weighting."
    },
    {
      type: "mc",
      question: "In PPO for LLMs, the clipping mechanism zeros out gradients for responses where $r_t$ exits $[1-\\epsilon, 1+\\epsilon]$. An optimizer that aggressively adapts per-parameter learning rates (like Adam) may respond to this sparsity by:",
      options: [
        "Reducing all learning rates uniformly because the average gradient magnitude decreases",
        "Increasing effective learning rates for parameters that receive non-zero gradients from unclipped responses, potentially causing those parameters to be over-updated",
        "Ignoring the clipped responses entirely, which is the intended behavior with no side effects",
        "Accumulating a large momentum in the KL penalty direction since it is never clipped"
      ],
      correct: 1,
      explanation: "When some samples are clipped (zero gradient), Adam's second moment $v_t$ for affected parameters decreases (fewer large gradient entries). Smaller $v_t$ means larger effective learning rate $\\alpha/(\\sqrt{v_t} + \\epsilon)$. When those parameters do receive gradients from unclipped samples in subsequent steps, the inflated learning rate can cause larger-than-intended updates. This interaction between PPO's clipping and Adam's adaptivity is a real practical concern."
    }
  ]
};
