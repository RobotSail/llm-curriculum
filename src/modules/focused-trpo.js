// Focused module: Trust Region Policy Optimization (TRPO)
// Covers the monotonic improvement guarantee, surrogate objective, KL constraint,
// natural gradient connection, conjugate gradient, and the path to PPO.

export const trpoLearning = {
  id: "A.3-trpo-learning-medium",
  sectionId: "A.3",
  title: "Trust Region Policy Optimization (TRPO)",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Policy Update Problem",
      content: "Policy gradient methods face a fundamental instability: we compute a gradient at the current parameters $\\theta_{\\text{old}}$, then step to new parameters $\\theta_{\\text{new}}$. But the gradient was only a **local linear approximation** — valid in a tiny neighborhood around $\\theta_{\\text{old}}$.\n\nIn parameter space, a small change in $\\theta$ can cause a **large change in the policy** $\\pi_\\theta$. Neural networks are highly nonlinear: perturbing a single weight in an attention layer can radically change the output distribution for many inputs.\n\nThe question TRPO addresses: **how large a step can we safely take?** We want the largest update that is guaranteed to improve the policy, without overshooting into a catastrophic region.\n\nThe key idea (Schulman et al., 2015): measure the step size not in parameter space ($\\|\\theta_{\\text{new}} - \\theta_{\\text{old}}\\|$) but in **policy space** ($\\text{KL}(\\pi_{\\text{old}} \\| \\pi_{\\text{new}})$). A step that is small in KL divergence keeps the new policy close to the old one in the sense that matters: the distribution over actions."
    },
    {
      type: "mc",
      question: "Two parameter updates both change $\\theta$ by the same Euclidean distance $\\|\\Delta\\theta\\| = 0.01$. Update A changes $\\text{KL}(\\pi_{\\text{old}} \\| \\pi_{\\text{new}}) = 0.001$ and Update B changes $\\text{KL}(\\pi_{\\text{old}} \\| \\pi_{\\text{new}}) = 0.5$. Which update is riskier, and why?",
      options: [
        "Update A is riskier because the small KL change means the policy barely moved, suggesting the gradient direction is nearly orthogonal to useful updates",
        "Both are equally risky since the parameter-space distance is identical and KL divergence is just a different metric on the same change",
        "Update B is riskier because the large KL change means the policy's behavior shifted substantially, making the gradient estimate unreliable for that region",
        "Neither is risky because $\\|\\Delta\\theta\\| = 0.01$ is a small step that cannot cause meaningful policy changes regardless of the KL"
      ],
      correct: 2,
      explanation: "Update B moved the policy's output distribution much further (KL = 0.5 vs 0.001), even though both moved the same distance in parameter space. The gradient was estimated under $\\pi_{\\text{old}}$ — it is only accurate in a region where $\\pi_{\\text{new}} \\approx \\pi_{\\text{old}}$. With KL = 0.5, the new policy assigns substantially different probabilities to actions, meaning the advantage estimates used to compute the gradient are stale. TRPO constrains the KL, not the parameter norm, precisely because KL measures the behaviorally relevant change."
    },
    {
      type: "info",
      title: "The Monotonic Improvement Guarantee",
      content: "TRPO is grounded in a theoretical result by Kakade & Langford (2002). Define the **expected advantage** of a new policy $\\tilde{\\pi}$ relative to the old policy $\\pi$:\n\n$$\\eta(\\tilde{\\pi}) = \\eta(\\pi) + \\mathbb{E}_{s \\sim d^{\\tilde{\\pi}}, a \\sim \\tilde{\\pi}}[A^\\pi(s, a)]$$\n\nwhere $\\eta(\\pi)$ is the expected return of policy $\\pi$, and $d^{\\tilde{\\pi}}$ is the state distribution under the **new** policy. This says: the new policy's performance equals the old policy's performance plus the expected advantage of the new policy's actions.\n\nThe problem: evaluating this requires sampling states from $d^{\\tilde{\\pi}}$ — the new policy's state distribution — which we do not have (we only have data from $\\pi_{\\text{old}}$).\n\nThe key theoretical insight: if we instead use the **old** policy's state distribution $d^\\pi$ and bound how far the new policy can deviate from the old one, we get a **guaranteed lower bound** on improvement:\n\n$$\\eta(\\tilde{\\pi}) \\geq L_\\pi(\\tilde{\\pi}) - C \\cdot \\max_s \\text{KL}(\\pi(\\cdot|s) \\| \\tilde{\\pi}(\\cdot|s))$$\n\nwhere $L_\\pi(\\tilde{\\pi})$ is the surrogate objective (advantage under old state distribution) and $C$ is a constant. Maximizing this lower bound guarantees improvement."
    },
    {
      type: "mc",
      question: "The monotonic improvement bound replaces states sampled from the new policy $d^{\\tilde{\\pi}}$ with states from the old policy $d^\\pi$, adding a KL penalty. Why is the KL term necessary?",
      options: [
        "The old and new policies visit different states, so the KL bounds how much the state distributions can differ, correcting the mismatch",
        "The KL term regularizes the policy to prevent memorizing specific high-reward responses in the training batch",
        "The KL term ensures the new policy has high entropy, preventing premature convergence to a deterministic strategy",
        "The KL penalty replaces the reward model score, providing a differentiable proxy for response quality"
      ],
      correct: 0,
      explanation: "When $\\tilde{\\pi} \\neq \\pi$, the new policy visits different states than the old one. The surrogate $L_\\pi(\\tilde{\\pi})$ evaluates the new policy's actions at the old policy's states, which is only accurate if both policies visit similar states. The KL constraint bounds how different the policies can be pointwise, which in turn bounds how different their state distributions can be (via a coupling argument). If KL is small, $d^{\\tilde{\\pi}} \\approx d^\\pi$ and the surrogate is a good proxy for the true performance."
    },
    {
      type: "info",
      title: "The TRPO Optimization Problem",
      content: "The theoretical bound uses a max-KL constraint, which is conservative. In practice, TRPO uses the **average** KL divergence, leading to the constrained optimization:\n\n$$\\max_\\theta \\; \\underbrace{\\mathbb{E}_{s \\sim d^{\\pi_{\\text{old}}}, a \\sim \\pi_{\\text{old}}}\\left[\\frac{\\pi_\\theta(a|s)}{\\pi_{\\text{old}}(a|s)} \\hat{A}^{\\pi_{\\text{old}}}(s, a)\\right]}_{\\text{surrogate objective } L_{\\theta_{\\text{old}}}(\\theta)}$$\n$$\\text{subject to} \\quad \\overline{\\text{KL}}(\\pi_{\\theta_{\\text{old}}}, \\pi_\\theta) \\leq \\delta$$\n\nThe surrogate objective reweights the old policy's advantages by the importance ratio $\\pi_\\theta / \\pi_{\\text{old}}$. When $\\theta = \\theta_{\\text{old}}$, all ratios are 1 and the gradient of the surrogate equals the standard policy gradient.\n\nThe constraint $\\delta$ is a hyperparameter (typically 0.01) controlling the trust region size. Within this region, the surrogate is a reliable proxy for the true objective."
    },
    {
      type: "mc",
      question: "At the current parameters $\\theta = \\theta_{\\text{old}}$, the importance ratio $\\pi_\\theta(a|s)/\\pi_{\\text{old}}(a|s) = 1$ for all state-action pairs. What does this imply about the surrogate objective's gradient at this point?",
      options: [
        "The gradient is zero because all ratios equal 1, so there is no direction of improvement in the surrogate",
        "The gradient is undefined because the importance ratio is constant and independent of $\\theta$ at this point",
        "The gradient depends only on the KL constraint, not on the advantages, since the surrogate is maximized when ratios equal 1",
        "The gradient equals the standard policy gradient $\\mathbb{E}[\\hat{A} \\nabla_\\theta \\log \\pi_\\theta(a|s)]$, making TRPO a locally-justified extension of REINFORCE"
      ],
      correct: 3,
      explanation: "At $\\theta = \\theta_{\\text{old}}$, $\\nabla_\\theta L_{\\theta_{\\text{old}}}(\\theta) = \\mathbb{E}[\\hat{A} \\nabla_\\theta (\\pi_\\theta / \\pi_{\\text{old}})] = \\mathbb{E}[\\hat{A} \\nabla_\\theta \\log \\pi_\\theta]$ (since $\\nabla_\\theta \\pi_\\theta / \\pi_{\\text{old}}|_{\\theta=\\theta_{\\text{old}}} = \\nabla_\\theta \\log \\pi_\\theta$). This is exactly the policy gradient. TRPO's contribution is not a new gradient direction but a principled way to determine the step size: take the largest step in this direction that stays within the KL trust region."
    },
    {
      type: "info",
      title: "Connection to the Natural Gradient",
      content: "To solve the constrained optimization, TRPO approximates both the objective and constraint using Taylor expansions around $\\theta_{\\text{old}}$:\n\n- **Objective** (first-order): $L(\\theta) \\approx g^T (\\theta - \\theta_{\\text{old}})$ where $g = \\nabla_\\theta L|_{\\theta_{\\text{old}}}$ is the policy gradient.\n- **Constraint** (second-order): $\\overline{\\text{KL}}(\\pi_{\\theta_{\\text{old}}}, \\pi_\\theta) \\approx \\frac{1}{2}(\\theta - \\theta_{\\text{old}})^T F (\\theta - \\theta_{\\text{old}})$ where $F$ is the **Fisher information matrix**.\n\nThe Fisher matrix $F = \\mathbb{E}[\\nabla_\\theta \\log \\pi_\\theta \\, (\\nabla_\\theta \\log \\pi_\\theta)^T]$ measures the local curvature of the KL divergence. It captures how sensitive the policy distribution is to each parameter direction.\n\nSolving the linearized problem gives the **natural gradient** update:\n\n$$\\theta_{\\text{new}} = \\theta_{\\text{old}} + \\sqrt{\\frac{2\\delta}{g^T F^{-1} g}} \\; F^{-1} g$$\n\nThe natural gradient $F^{-1} g$ is the policy gradient **preconditioned by the inverse Fisher**. It rescales the gradient so that each direction moves the policy distribution by an equal amount, regardless of the parameterization."
    },
    {
      type: "mc",
      question: "The Fisher information matrix $F$ is large along directions where small parameter changes cause large distributional shifts. The natural gradient $F^{-1}g$ scales down the gradient in these directions. Why is this desirable?",
      options: [
        "Sensitive directions correspond to frequently-used tokens, and we want to update rare token probabilities more aggressively for faster convergence",
        "It prevents the optimizer from exploiting parameterization artifacts — directions where $\\theta$ moves the distribution a lot should take smaller steps to maintain the KL budget",
        "Large Fisher eigenvalues indicate poor conditioning, and the inverse dampens these modes to stabilize the matrix computations",
        "Scaling by $F^{-1}$ converts the gradient from log-probability space to probability space, which is the natural representation for policy updates"
      ],
      correct: 1,
      explanation: "The natural gradient equalizes the effect of each parameter direction on the policy distribution. Without preconditioning, a direction in $\\theta$-space that happens to change the distribution a lot (large $F$ eigenvalue) would dominate the update, potentially exceeding the trust region. The inverse Fisher scales these directions down so that every direction contributes equally in KL-divergence terms. This makes the update **reparameterization-invariant** — the same policy change regardless of how the neural network happens to represent it."
    },
    {
      type: "info",
      title: "Conjugate Gradient and Line Search",
      content: "For a model with $N$ parameters, the Fisher matrix $F$ is $N \\times N$. For a 7B-parameter LLM, that is a $7 \\times 10^9 \\times 7 \\times 10^9$ matrix — impossible to store or invert.\n\nTRPO avoids forming $F$ explicitly using two tricks:\n\n**1. Conjugate gradient (CG)**: To compute $F^{-1}g$, we only need **matrix-vector products** $Fv$ for arbitrary vectors $v$. The Fisher-vector product can be computed efficiently using two backpropagation passes (one forward, one backward), costing $O(N)$ rather than $O(N^2)$. CG iteratively solves $Fx = g$ using ~10-20 such products.\n\n**2. Backtracking line search**: The Taylor approximation is only valid locally. After computing the natural gradient direction $s = F^{-1}g$, TRPO performs a line search: try step size $\\alpha$, check if the actual (not approximate) KL constraint is satisfied and the actual objective improves, and halve $\\alpha$ if not.\n\nThis makes TRPO significantly more expensive per update than standard gradient descent:\n- ~20 backward passes for CG (vs 1 for SGD)\n- Multiple forward passes for line search\n- Each rollout requires full response generation\n\nFor LLMs with billions of parameters, this overhead is prohibitive."
    },
    {
      type: "mc",
      question: "TRPO uses conjugate gradient to compute $F^{-1}g$ without forming the full Fisher matrix. If each Fisher-vector product costs about as much as one backward pass, and CG runs for 15 iterations, how does TRPO's per-update cost compare to vanilla policy gradient (one backward pass)?",
      options: [
        "TRPO is roughly 15x more expensive in backward passes alone, plus additional forward passes for the line search",
        "TRPO is roughly 2x more expensive because CG only runs a few iterations before converging to sufficient accuracy",
        "TRPO has the same backward-pass cost because CG reuses intermediate computations from the policy gradient calculation",
        "TRPO is cheaper per update because the natural gradient takes larger steps, requiring fewer total updates to converge"
      ],
      correct: 0,
      explanation: "Each CG iteration requires one Fisher-vector product, which costs approximately one backward pass. With 15 CG iterations, that is ~15 backward passes just for the direction computation. The line search adds several forward passes to verify the constraint and objective. Compare this to vanilla policy gradient: one backward pass for the gradient, one step. TRPO's per-update cost is 15-20x higher. It may still require fewer total updates to reach the same performance (larger, safer steps), but for billion-parameter LLMs, the constant factor is too expensive."
    },
    {
      type: "info",
      title: "From TRPO to PPO: Simplifying the Trust Region",
      content: "TRPO achieves reliable policy improvement but is complex and expensive. Schulman et al. (2017) introduced PPO as a practical simplification.\n\nTRPO enforces the trust region via a **hard KL constraint** solved with second-order methods. PPO replaces this with two simpler alternatives:\n\n**PPO-Clip**: Clip the importance ratio directly in the objective:\n$$L^{\\text{CLIP}} = \\mathbb{E}\\left[\\min\\left(r_t \\hat{A}_t, \\; \\text{clip}(r_t, 1{-}\\epsilon, 1{+}\\epsilon) \\hat{A}_t\\right)\\right]$$\n\nThis caps the ratio at $[1-\\epsilon, 1+\\epsilon]$ (typically $\\epsilon = 0.2$), achieving a soft trust region with only first-order gradients.\n\n**PPO-Penalty** (less common): Add a KL penalty to the objective with an adaptive coefficient $\\beta$:\n$$L^{\\text{KL}} = \\mathbb{E}\\left[r_t \\hat{A}_t - \\beta \\, \\text{KL}(\\pi_{\\text{old}} \\| \\pi_\\theta)\\right]$$\n\nwhere $\\beta$ is increased if KL exceeds a target and decreased if KL is too small.\n\nThe practical impact: PPO requires only standard gradient descent with clipping — no conjugate gradient, no Fisher matrix, no line search. This makes it feasible for billion-parameter models, which is why PPO became the standard for RLHF."
    },
    {
      type: "mc",
      question: "TRPO uses a hard KL constraint $\\text{KL} \\leq \\delta$ while PPO-Clip restricts the importance ratio to $[1-\\epsilon, 1+\\epsilon]$. When might PPO-Clip's trust region fail to approximate TRPO's KL constraint?",
      options: [
        "When the batch size is small, causing the sample-estimated KL to diverge from the population KL in both methods equally",
        "When the advantage estimates are noisy, causing both methods to take suboptimal steps regardless of the trust region mechanism",
        "When the learning rate is too small, preventing either method from reaching the boundary of their respective trust regions",
        "When a few tokens have extreme ratio changes within the allowed range that collectively produce a large KL divergence across the full response distribution"
      ],
      correct: 3,
      explanation: "PPO-Clip constrains each individual importance ratio to $[0.8, 1.2]$ with $\\epsilon = 0.2$. But the KL divergence is an average over the entire distribution. Many tokens each changing their ratio by a modest amount (all within the clip range) can collectively produce a large total KL divergence. TRPO's constraint directly bounds the KL, preventing this. In practice, PPO-Clip works well because the clipping interacts with the optimization dynamics to keep KL controlled, but there is no formal guarantee — it is a practical heuristic rather than a principled bound."
    },
    {
      type: "info",
      title: "Why TRPO Still Matters",
      content: "Despite PPO's dominance in practice, understanding TRPO matters for several reasons:\n\n**Theoretical foundation**: TRPO provides the monotonic improvement guarantee that justifies trust region methods. PPO lacks this guarantee — it works empirically but can violate the KL constraint. Understanding TRPO explains *why* PPO works.\n\n**Natural gradient insight**: TRPO reveals that policy optimization is best done in **distribution space**, not parameter space. The Fisher information matrix adapts the geometry to match the policy manifold. This insight connects to:\n- **Adam**: Can be viewed as a diagonal approximation to the natural gradient\n- **K-FAC**: Kronecker-factored approximation to the Fisher for more efficient natural gradient computation\n- **Muon optimizer**: Uses orthogonalization to approximate the natural gradient on weight matrices\n\n**Debugging PPO failures**: When PPO training is unstable, it often helps to monitor the actual KL divergence between updates. If KL spikes despite clipping, the trust region is being violated — understanding TRPO clarifies what went wrong.\n\n**Algorithm design**: Methods like GRPO inherit TRPO's clipped surrogate objective. Understanding the theoretical motivation behind clipping helps when designing new algorithms or tuning hyperparameters."
    },
    {
      type: "mc",
      question: "A researcher observes that PPO training with $\\epsilon = 0.2$ produces a KL divergence of 0.8 between consecutive policy updates, despite all importance ratios being within $[0.8, 1.2]$. Based on TRPO's theoretical framework, what does this indicate?",
      options: [
        "The reward model is assigning inconsistent scores, causing the policy to oscillate between high-KL regions",
        "The critic's value estimates are poor, leading to noisy advantages that push the policy in many conflicting directions simultaneously",
        "Many tokens each shifted within the clip range, but the cumulative distributional shift is large — the clip is not controlling the aggregate policy change",
        "The KL measurement is wrong because KL divergence should be bounded by $\\epsilon^2 / 2 \\approx 0.02$ when ratios are clipped"
      ],
      correct: 2,
      explanation: "This is exactly the failure mode that TRPO's KL constraint prevents. Each token's ratio is within $[0.8, 1.2]$, satisfying the per-token clip constraint. But KL divergence aggregates across the full vocabulary at every position — a response of 500 tokens each shifting modestly can produce a large total KL. TRPO would directly constrain this aggregate KL to $\\delta \\approx 0.01$. PPO-Clip trades this guarantee for computational simplicity. In practice, this can be addressed by reducing $\\epsilon$, lowering the learning rate, or using fewer optimization epochs per rollout."
    },
    {
      type: "mc",
      question: "Which statement best describes the relationship between the natural policy gradient, TRPO, and PPO?",
      options: [
        "They are three independent algorithms that achieve trust-region behavior through completely different mechanisms and theoretical foundations",
        "TRPO is a special case of PPO where the clip range $\\epsilon$ approaches zero, recovering the hard KL constraint",
        "Natural gradient computes the optimal direction in distribution space, TRPO adds safeguards (CG + line search) to enforce the KL constraint, and PPO replaces the KL constraint with a simpler clipping heuristic",
        "All three are equivalent — they compute the same update direction but differ only in learning rate selection"
      ],
      correct: 2,
      explanation: "The progression is: the natural policy gradient gives the steepest ascent direction in KL-divergence geometry (solve $Fx = g$). TRPO takes this direction and adds practical safeguards — conjugate gradient for scalable $F^{-1}g$ computation, and line search to verify the KL constraint and actual improvement. PPO observes that the hard KL constraint is expensive to enforce and replaces it with ratio clipping, which empirically achieves similar stability with only first-order computation. Each step simplifies the previous one while retaining the core trust-region intuition."
    }
  ]
};
