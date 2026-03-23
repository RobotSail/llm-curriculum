// Assessment: KL Divergence & Divergence Measures
// Section 0.2: Diagnostic test — KL divergence, f-divergences, JS, Hellinger, Rényi
// Pure assessment to gauge depth of understanding

export const divergencesAssessment = {
  id: "0.2-assess-divergences",
  sectionId: "0.2",
  title: "Assessment: KL Divergence & Divergence Measures",
  difficulty: "medium",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: KL Divergence & Divergence Measures",
      content: "This is a **diagnostic assessment** covering KL divergence, f-divergences, and related distance measures between distributions.\n\nThese measures are central to LLM training (cross-entropy = forward KL), RLHF (KL penalties), GANs (Jensen-Shannon), and evaluation (distributional comparison).\n\nIf you score below 70%, review these topics carefully — they appear everywhere in modern ML."
    },
    {
      type: "mc",
      question: "The KL divergence $\\text{KL}(P \\| Q) = \\mathbb{E}_P\\left[\\log \\frac{P(x)}{Q(x)}\\right]$ is NOT a true metric because:",
      options: [
        "It can be negative",
        "It is asymmetric ($\\text{KL}(P\\|Q) \\neq \\text{KL}(Q\\|P)$) and does not satisfy the triangle inequality",
        "It is always infinite",
        "It is only defined for Gaussian distributions"
      ],
      correct: 1,
      explanation: "KL divergence fails two metric properties: (1) asymmetry — $\\text{KL}(P \\| Q) \\neq \\text{KL}(Q \\| P)$ in general, and (2) no triangle inequality. It IS non-negative ($\\text{KL} \\geq 0$ by Gibbs' inequality) and equals zero iff $P = Q$. Despite not being a metric, it's the natural measure arising from likelihood-based training."
    },
    {
      type: "mc",
      question: "In RLHF, the objective is $\\max_\\pi \\mathbb{E}_{\\pi}[r(x)] - \\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$. The KL term $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ is **forward KL** from $\\pi$ to $\\pi_{\\text{ref}}$. What behavior does this penalty enforce?",
      options: ["Wherever $\\pi$ places probability mass, $\\pi_{\\text{ref}}$ must also have mass — preventing $\\pi$ from generating text that $\\pi_{\\text{ref}}$ considers highly unlikely", "It forces $\\pi$ to match a single mode of $\\pi_{\\text{ref}}$", "It ensures $\\pi_{\\text{ref}}$ covers all modes of $\\pi$", "It minimizes the entropy of $\\pi$"],
      correct: 0,
      explanation: "$\\text{KL}(\\pi \\| \\pi_{\\text{ref}}) = \\mathbb{E}_\\pi[\\log \\pi / \\pi_{\\text{ref}}]$ — the expectation is under $\\pi$. If $\\pi(x) > 0$ but $\\pi_{\\text{ref}}(x) \\approx 0$, the ratio explodes. So $\\pi$ is penalized for putting mass where the reference doesn't. This prevents \"reward hacking\" — generating text that scores high reward but is completely unlike natural text."
    },
    {
      type: "mc",
      question: "The **Jensen-Shannon divergence** $\\text{JS}(P \\| Q) = \\frac{1}{2}\\text{KL}(P \\| M) + \\frac{1}{2}\\text{KL}(Q \\| M)$ where $M = \\frac{1}{2}(P + Q)$ has which advantages over KL?",
      options: ["It is always larger than KL divergence", "It is easier to compute than KL", "It is symmetric, bounded ($0 \\leq \\text{JS} \\leq \\log 2$), and well-defined even when supports don't fully overlap", "It converges faster during training"],
      correct: 2,
      explanation: "JS divergence is symmetric ($\\text{JS}(P\\|Q) = \\text{JS}(Q\\|P)$), bounded by $\\log 2$, and always finite even when $P$ and $Q$ have disjoint support (because $M$ covers both). Its square root is a true metric. The original GAN objective minimizes JS divergence, but the bounded/finite property actually causes problems (vanishing gradients when $P$ and $Q$ are far apart)."
    },
    {
      type: "mc",
      question: "**Reverse KL** $\\text{KL}(Q \\| P)$ is called \"mode-seeking\" because when fitting $Q$ to a multimodal $P$:",
      options: ["$Q$ covers all modes of $P$ uniformly", "$Q$ becomes uniform regardless of $P$", "$Q$ places mass only between modes", "$Q$ tends to concentrate on a single mode of $P$, fitting it precisely while ignoring others"],
      correct: 3,
      explanation: "In reverse KL, the expectation is under $Q$: $\\mathbb{E}_Q[\\log Q/P]$. If $Q$ places mass where $P(x) \\approx 0$, $\\log(Q/P) \\to \\infty$, so $Q$ avoids regions where $P$ is small. But $Q$ pays no penalty for *ignoring* modes of $P$ (since it doesn't sample there). So $Q$ \"seeks\" a single mode and fits it tightly. This is the behavior of variational inference with reverse KL."
    },
    {
      type: "mc",
      question: "The **Rényi divergence** of order $\\alpha$ is $D_\\alpha(P \\| Q) = \\frac{1}{\\alpha - 1} \\log \\mathbb{E}_Q\\left[\\left(\\frac{P(x)}{Q(x)}\\right)^\\alpha\\right]$. As $\\alpha \\to 1$, it converges to:",
      options: [
        "The total variation distance",
        "The KL divergence $\\text{KL}(P \\| Q)$",
        "The Hellinger distance",
        "Zero for all distributions"
      ],
      correct: 1,
      explanation: "Rényi divergence is a one-parameter family that includes KL as a special case at $\\alpha \\to 1$ (by L'Hôpital's rule). At $\\alpha = 1/2$ it relates to the Hellinger distance, and at $\\alpha \\to \\infty$ it becomes the max-divergence $\\log \\max_x P(x)/Q(x)$. Different $\\alpha$ values weight tail behavior differently, making Rényi divergences useful for robust training objectives."
    },
    {
      type: "mc",
      question: "The **total variation distance** $\\text{TV}(P, Q) = \\frac{1}{2} \\sum_x |P(x) - Q(x)|$ relates to KL divergence through **Pinsker's inequality**:",
      options: ["$\\text{TV}(P, Q) \\leq \\sqrt{\\frac{1}{2} \\text{KL}(P \\| Q)}$", "$\\text{TV}(P, Q) \\leq \\text{KL}(P \\| Q)$", "$\\text{KL}(P \\| Q) \\leq \\text{TV}(P, Q)$", "$\\text{TV}(P, Q) = \\text{KL}(P \\| Q)$ always"],
      correct: 0,
      explanation: "Pinsker's inequality: $\\text{TV}(P, Q) \\leq \\sqrt{\\frac{1}{2} \\text{KL}(P \\| Q)}$. This means small KL implies small TV, but not vice versa — TV can be small while KL is large (when the ratio $P/Q$ is extreme in regions with small mass). This inequality is fundamental in PAC-Bayes bounds and differential privacy proofs."
    },
    {
      type: "mc",
      question: "When two distributions $P$ and $Q$ have **disjoint supports** (no overlap), what happens to KL divergence, JS divergence, and total variation?",
      options: ["All three are zero", "All three are infinite", "KL is undefined ($\\infty$), JS = $\\log 2$ (finite), TV = 1 (maximum)", "KL is finite, JS is infinite, TV = 0"],
      correct: 2,
      explanation: "With disjoint supports: KL is $+\\infty$ (dividing by zero in the log-ratio), JS is $\\log 2$ (its maximum, finite value), and TV is 1 (its maximum). This is why JS is preferred in some contexts — it gracefully handles non-overlapping distributions. However, this bounded behavior means JS gradients vanish when distributions are far apart, which was the original GAN training problem solved by Wasserstein distance."
    },
    {
      type: "mc",
      question: "The **f-divergence** framework defines $D_f(P \\| Q) = \\mathbb{E}_Q[f(P(x)/Q(x))]$ for convex $f$ with $f(1) = 0$. The **variational representation** $D_f(P \\| Q) = \\sup_T \\{\\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[f^*(T(x))]\\}$ (where $f^*$ is the Fenchel conjugate) is important because:",
      options: ["It allows exact computation of any f-divergence", "It eliminates the need for density ratio estimation", "It proves all f-divergences are equal", "It converts divergence estimation into an optimization problem that a neural network (discriminator/critic) can solve — this is the foundation of f-GANs"],
      correct: 3,
      explanation: "The variational representation turns f-divergence computation into a supremum over functions $T$ — which a neural network can approximate. The original GAN uses this for JS divergence (where $T$ is the discriminator), and f-GANs generalize this to arbitrary f-divergences. This is a powerful trick: you never need to know the density ratio explicitly; instead, you train a network to estimate it."
    },
    {
      type: "mc",
      question: "In DPO (Direct Preference Optimization), the loss involves $\\log \\frac{\\pi_\\theta(y_w \\mid x)}{\\pi_{\\text{ref}}(y_w \\mid x)} - \\log \\frac{\\pi_\\theta(y_l \\mid x)}{\\pi_{\\text{ref}}(y_l \\mid x)}$ where $y_w$ is preferred over $y_l$. The log-ratios $\\log \\frac{\\pi_\\theta}{\\pi_{\\text{ref}}}$ are called **implicit rewards**. This formulation implicitly uses which divergence concept?",
      options: ["Jensen-Shannon divergence between preferred and dispreferred completions", "The density ratio $\\pi_\\theta / \\pi_{\\text{ref}}$ that appears in KL divergence — DPO reparameterizes the RLHF objective's KL-constrained optimization into a supervised loss using the closed-form solution involving log-ratios", "Total variation distance between old and new policies", "Wasserstein distance in token space"],
      correct: 1,
      explanation: "DPO derives from the RLHF objective $\\max_\\pi \\mathbb{E}[r(x)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, whose closed-form solution is $\\pi^*(y \\mid x) \\propto \\pi_{\\text{ref}}(y \\mid x) \\exp(r(y,x)/\\beta)$. Rearranging gives $r(y,x) = \\beta \\log \\frac{\\pi^*(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)} + \\text{const}$. DPO substitutes this into the Bradley-Terry preference model, yielding a supervised loss that implicitly optimizes the KL-constrained objective."
    }
  ]
};
