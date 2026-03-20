// Module: Information Theory — f-Divergences
// Section 0.2: Probability, statistics & information theory
// Three difficulty tracks building on the user's KL → cross-entropy derivation

export const easyModule = {
  id: "0.2-easy",
  sectionId: "0.2",
  title: "The f-Divergence Family",
  difficulty: "easy",
  estimatedMinutes: 15,
  steps: [
    {
      type: "info",
      title: "Review: Cross-Entropy Minimizes KL",
      content: "You've already derived the key result: the cross-entropy loss is secretly a KL minimization.\n\nStarting from $\\text{KL}(P \\| Q) = \\mathbb{E}_P\\!\\left[\\log \\frac{P(x)}{Q(x)}\\right]$, you split this into:\n\n$$\\text{KL}(P \\| Q) = \\underbrace{\\mathbb{E}_P[\\log P(x)]}_{\\text{constant w.r.t. } Q} - \\mathbb{E}_P[\\log Q(x)]$$\n\nSo minimizing KL over $Q$ is equivalent to maximizing $\\mathbb{E}_P[\\log Q(y \\mid x)]$ — which is exactly the cross-entropy training objective.\n\nBut KL is just **one** way to measure distributional distance. What if a different divergence gave you better properties for your specific problem?"
    },
    {
      type: "mc",
      question: "Which of the following is a known limitation of KL divergence that motivates looking for alternatives?",
      options: [
        "It requires both distributions to be discrete",
        "It is asymmetric: $\\text{KL}(P \\| Q) \\neq \\text{KL}(Q \\| P)$",
        "It is always bounded between 0 and 1",
        "It cannot handle continuous distributions"
      ],
      correct: 1,
      explanation: "KL divergence is **asymmetric** — swapping the arguments gives a fundamentally different quantity. This asymmetry has deep practical consequences: $\\text{KL}(P \\| Q)$ and $\\text{KL}(Q \\| P)$ penalize different types of mismatch, leading to different model behaviors."
    },
    {
      type: "info",
      title: "The Two Faces of KL: Mode-Covering vs. Mode-Seeking",
      content: "The asymmetry of KL creates two distinct optimization behaviors:\n\n**Forward KL** $\\text{KL}(P \\| Q) = \\mathbb{E}_P[\\log P/Q]$: the expectation is under $P$, so wherever $P(x) > 0$, the model $Q$ is forced to assign non-negligible probability. If $Q(x) \\to 0$ where $P(x) > 0$, the log ratio explodes. Result: $Q$ **covers all modes** of $P$, even at the cost of spreading mass into low-density regions.\n\n**Reverse KL** $\\text{KL}(Q \\| P) = \\mathbb{E}_Q[\\log Q/P]$: the expectation is under $Q$, so $Q$ only pays a penalty where *it* places mass. It can safely ignore modes of $P$ by placing zero mass there. Result: $Q$ **seeks a single mode** of $P$ and fits it precisely.\n\nIn RLHF, the KL penalty $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ is forward KL from $\\pi$'s perspective — it prevents the policy from collapsing modes that the reference model covers."
    },
    {
      type: "mc",
      question: "A target distribution $P$ is bimodal with two well-separated peaks. You fit $Q$ by minimizing **forward KL** $\\text{KL}(P \\| Q)$. What behavior do you expect?",
      options: [
        "$Q$ locks onto one peak and ignores the other",
        "$Q$ spreads to cover both peaks, possibly placing mass between them",
        "$Q$ assigns uniform probability everywhere",
        "$Q$ exactly matches $P$"
      ],
      correct: 1,
      explanation: "Forward KL is **mode-covering**: $Q$ must assign probability wherever $P$ has mass (otherwise $\\log P/Q \\to \\infty$). For a bimodal $P$, $Q$ will cover both modes — but if $Q$ is a unimodal family (e.g., a single Gaussian), it may place significant mass in the valley between the peaks. This is the classic failure mode of variational inference with forward KL."
    },
    {
      type: "info",
      title: "Generalizing: The f-Divergence Framework",
      content: "KL divergence measures distributional distance using $\\log(P/Q)$. But what if we used a different function of the density ratio?\n\nThe **f-divergence** family unifies many divergence measures:\n\n$$D_f(P \\| Q) = \\mathbb{E}_Q\\!\\left[f\\!\\left(\\frac{P(x)}{Q(x)}\\right)\\right] = \\sum_x Q(x) \\cdot f\\!\\left(\\frac{P(x)}{Q(x)}\\right)$$\n\nwhere $f: \\mathbb{R}_+ \\to \\mathbb{R}$ is a **convex** function with $f(1) = 0$.\n\nWhy these constraints?\n\n- **$f(1) = 0$** ensures $D_f(P \\| Q) = 0$ when $P = Q$ (the ratio is 1 everywhere).\n- **Convexity** ensures $D_f \\geq 0$, by Jensen's inequality: $\\mathbb{E}_Q[f(P/Q)] \\geq f(\\mathbb{E}_Q[P/Q]) = f(1) = 0$."
    },
    {
      type: "mc",
      question: "To verify that KL divergence is an f-divergence, we need to find the generator $f$. With $f(t) = t \\log t$, expand $D_f(P \\| Q) = \\sum_x Q(x) \\cdot f(P(x)/Q(x))$. What do you get?",
      options: [
        "$\\sum_x P(x) \\log Q(x)$",
        "$\\sum_x P(x) \\log \\frac{P(x)}{Q(x)}$",
        "$\\sum_x Q(x) \\log \\frac{Q(x)}{P(x)}$",
        "$\\sum_x Q(x) \\log Q(x)$"
      ],
      correct: 1,
      explanation: "Substituting: $\\sum_x Q(x) \\cdot \\frac{P(x)}{Q(x)} \\log \\frac{P(x)}{Q(x)}$. The $Q(x)$ cancels: $= \\sum_x P(x) \\log \\frac{P(x)}{Q(x)} = \\text{KL}(P \\| Q)$. So KL divergence is the f-divergence with generator $f(t) = t \\log t$."
    },
    {
      type: "info",
      title: "The Family Members",
      content: "Different choices of $f$ give different divergences, each with distinct properties:\n\n**KL divergence**: $f(t) = t \\log t$ — asymmetric, unbounded, undefined when $Q(x) = 0$ where $P(x) > 0$.\n\n**Reverse KL**: $f(t) = -\\log t$ — gives $\\text{KL}(Q \\| P)$, asymmetric, mode-seeking.\n\n**Chi-squared** ($\\chi^2$): $f(t) = (t-1)^2$ — asymmetric, directly measures importance sampling variance (you'll see this in the Medium module).\n\n**Jensen-Shannon** (JS): defined as $\\text{JS}(P \\| Q) = \\frac{1}{2}\\text{KL}(P \\| M) + \\frac{1}{2}\\text{KL}(Q \\| M)$ where $M = \\frac{P+Q}{2}$ — **symmetric**, **bounded** in $[0, \\log 2]$. This is what the original GAN minimizes.\n\n**Total Variation** (TV): $f(t) = \\frac{1}{2}|t - 1|$ — symmetric, bounded in $[0, 1]$. Pinsker's inequality: $\\text{TV}(P, Q) \\leq \\sqrt{\\text{KL}(P \\| Q) / 2}$.\n\n**Hellinger**: $f(t) = (\\sqrt{t} - 1)^2$ — symmetric, bounded."
    },
    {
      type: "mc",
      question: "You need a divergence that is both **symmetric** and **bounded** (won't blow up to infinity). Which would you choose?",
      options: [
        "KL divergence ($f(t) = t \\log t$)",
        "Chi-squared divergence ($f(t) = (t-1)^2$)",
        "Jensen-Shannon divergence",
        "Reverse KL divergence ($f(t) = -\\log t$)"
      ],
      correct: 2,
      explanation: "Jensen-Shannon is the only option that is both symmetric ($\\text{JS}(P \\| Q) = \\text{JS}(Q \\| P)$) and bounded ($0 \\leq \\text{JS} \\leq \\log 2$). This is why it was used in the original GAN formulation — bounded values prevent infinite losses, and symmetry means neither distribution is privileged."
    },
    {
      type: "info",
      title: "Why the Choice of Divergence Matters",
      content: "Each f-divergence has different **sensitivity** to where two distributions disagree:\n\n**KL** is extremely sensitive to **support mismatch**: if $Q(x) \\approx 0$ where $P(x) > 0$, the divergence explodes. This forces coverage but makes optimization unstable when distributions have different supports.\n\n**Chi-squared** is sensitive to the **square** of density ratios. Since importance sampling variance equals $\\chi^2(P \\| Q)$ (as you'll derive in the Medium module), chi-squared directly tells you when importance weights will be unreliable — critical for off-policy RLHF.\n\n**Jensen-Shannon** handles non-overlapping supports gracefully (it's bounded), but its gradients vanish when distributions don't overlap — the Wasserstein GAN was invented specifically to fix this.\n\n**Total Variation** is the most conservative: $\\text{TV}(P, Q) = \\frac{1}{2}\\sum_x |P(x) - Q(x)|$ is the largest probability mass you could assign to any event differently. Pinsker's inequality $\\text{TV} \\leq \\sqrt{\\text{KL}/2}$ links it to KL.\n\nThe divergence you choose as your training objective determines your model's failure modes."
    },
    {
      type: "mc",
      question: "During language model training with cross-entropy loss (= forward KL minimization), the training data $P$ has support on certain token sequences. If $Q$ (your model) assigns near-zero probability to a sequence that $P$ covers, what happens?",
      options: [
        "The KL divergence stays bounded and training proceeds smoothly",
        "The gradient signal for that sequence vanishes",
        "The loss for that sequence becomes very large, creating a strong gradient signal",
        "The model ignores that sequence and focuses on higher-probability ones"
      ],
      correct: 2,
      explanation: "Since $\\text{KL}(P \\| Q) = \\sum P(x) \\log(P(x)/Q(x))$, when $Q(x) \\to 0$ for some $x$ where $P(x) > 0$, the term $\\log(P/Q) \\to \\infty$. This creates a large loss and strong gradient, forcing the model to assign *some* probability to every sequence in the training data. This is forward KL's mode-covering property at work — it's also why language models sometimes hallucinate: they've been forced to cover the full support of $P$, including rare patterns."
    },
    {
      type: "info",
      title: "Summary: From KL to a Toolkit",
      content: "You started with one divergence — KL — and now have a family of them, each defined by a convex function $f$ with $f(1) = 0$:\n\n$$D_f(P \\| Q) = \\mathbb{E}_Q\\!\\left[f\\!\\left(\\frac{P(x)}{Q(x)}\\right)\\right]$$\n\nKey takeaways:\n\n**1.** KL's asymmetry isn't a bug — it's a feature. Forward vs. reverse KL are different tools for different jobs (mode-covering vs. mode-seeking).\n\n**2.** Different f-divergences have different sensitivity profiles. The right choice depends on whether you care about support mismatch (KL), importance weight variance (chi-squared), or robust optimization (JS).\n\n**3.** Properties like symmetry and boundedness aren't just mathematical curiosities — they determine whether a divergence is suitable as a training objective.\n\nIn the **Medium** module, you'll see how chi-squared divergence is literally the variance of importance sampling weights (explaining why off-policy RL is hard), and how Jensen-Shannon gives you the original GAN objective."
    }
  ]
};

export const mediumModule = {
  id: "0.2-medium",
  sectionId: "0.2",
  title: "Chi-Squared, Importance Sampling & GANs",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Chi-Squared Divergence: Definition",
      content: "The chi-squared divergence uses the generator $f(t) = (t-1)^2$:\n\n$$\\chi^2(P \\| Q) = \\sum_x Q(x) \\left(\\frac{P(x)}{Q(x)} - 1\\right)^{\\!2} = \\mathbb{E}_Q\\!\\left[\\left(\\frac{P(x)}{Q(x)} - 1\\right)^{\\!2}\\right]$$\n\nExpanding the square and using $\\sum P(x) = \\sum Q(x) = 1$:\n\n$$\\chi^2(P \\| Q) = \\mathbb{E}_Q\\!\\left[\\left(\\frac{P(x)}{Q(x)}\\right)^{\\!2}\\right] - 1$$\n\nThis formula should remind you of something from statistics: it looks like a **variance**."
    },
    {
      type: "mc",
      question: "The chi-squared divergence is $\\chi^2(P \\| Q) = \\mathbb{E}_Q[(P/Q)^2] - 1$. Recall that $\\text{Var}[X] = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$. What is $\\mathbb{E}_Q[P(x)/Q(x)]$?",
      options: [
        "$0$",
        "$1$",
        "It depends on $P$ and $Q$",
        "$\\text{KL}(P \\| Q)$"
      ],
      correct: 1,
      explanation: "$\\mathbb{E}_Q[P/Q] = \\sum_x Q(x) \\cdot \\frac{P(x)}{Q(x)} = \\sum_x P(x) = 1$. The importance weight $P/Q$ always has expectation 1 under $Q$. This is the fundamental property that makes importance sampling work — and it means $\\chi^2(P \\| Q) = \\mathbb{E}_Q[(P/Q)^2] - 1 = \\text{Var}_Q[P/Q]$."
    },
    {
      type: "info",
      title: "The Key Identity: Chi-Squared = IS Variance",
      content: "This is the central result:\n\n$$\\chi^2(P \\| Q) = \\text{Var}_Q\\!\\left[\\frac{P(x)}{Q(x)}\\right]$$\n\nThe chi-squared divergence between $P$ and $Q$ **is** the variance of the importance weight $P/Q$ under $Q$.\n\nWhy does this matter? In **importance sampling**, we estimate $\\mathbb{E}_P[h(x)]$ using samples from $Q$:\n\n$$\\mathbb{E}_P[h(x)] = \\mathbb{E}_Q\\!\\left[\\frac{P(x)}{Q(x)} h(x)\\right]$$\n\nThe variance of this estimator is governed by the variance of $P/Q$. When $\\chi^2(P \\| Q)$ is large, the importance weights have high variance, and your IS estimates become unreliable — high-weight samples dominate the estimate, and you need exponentially more samples for convergence.\n\nThis is not an abstract concern. It's the core reason why **off-policy methods in RLHF** are fragile."
    },
    {
      type: "mc",
      question: "In off-policy RLHF, you collect rollouts from a behavior policy $\\pi_{\\text{old}}$ and use them to update the current policy $\\pi$. The importance weight is $w(x) = \\pi(x)/\\pi_{\\text{old}}(x)$. What quantity directly measures the reliability of this off-policy estimate?",
      options: [
        "$\\text{KL}(\\pi \\| \\pi_{\\text{old}})$",
        "$\\text{KL}(\\pi_{\\text{old}} \\| \\pi)$",
        "$\\chi^2(\\pi \\| \\pi_{\\text{old}}) = \\text{Var}_{\\pi_{\\text{old}}}[\\pi/\\pi_{\\text{old}}]$",
        "$\\text{JS}(\\pi \\| \\pi_{\\text{old}})$"
      ],
      correct: 2,
      explanation: "The chi-squared divergence $\\chi^2(\\pi \\| \\pi_{\\text{old}})$ is exactly the variance of the importance weights $\\pi/\\pi_{\\text{old}}$ under $\\pi_{\\text{old}}$. When this is large, individual samples can have outsized influence on the gradient estimate, making training unstable. PPO's clipping ($\\epsilon$-clipping the ratio) is a crude but effective way to bound this variance."
    },
    {
      type: "info",
      title: "PPO Clipping Through the Lens of Chi-Squared",
      content: "PPO clips the importance ratio: $\\min(r_t(\\theta) A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t)$ where $r_t = \\pi_\\theta / \\pi_{\\text{old}}$.\n\nClipping $r_t \\in [1-\\epsilon, 1+\\epsilon]$ effectively **bounds the chi-squared divergence**: since $\\chi^2 = \\text{Var}[r]$ and $r$ is constrained near 1, the variance can't explode.\n\nThe KL penalty approach (used in the original RLHF paper) controls a *different* quantity. KL divergence $\\text{KL}(\\pi \\| \\pi_{\\text{old}})$ penalizes log-density ratios, while chi-squared penalizes squared density ratios. Because $(t-1)^2$ grows faster than $t \\log t$ for large $t$, chi-squared is **more sensitive to large importance weights** than KL.\n\nThis means:\n- **KL penalty**: smoother constraint, but allows occasional large importance weights\n- **Clipping**: harder constraint on individual weights, directly controls chi-squared\n\nIn practice, PPO's clipping is more robust precisely because it directly addresses the variance problem."
    },
    {
      type: "mc",
      question: "For large density ratios $t = P(x)/Q(x) \\gg 1$, which grows faster — the KL contribution $t \\log t$ or the chi-squared contribution $(t-1)^2$?",
      options: [
        "$t \\log t$ grows faster (KL is more sensitive to large ratios)",
        "$(t-1)^2$ grows faster (chi-squared is more sensitive to large ratios)",
        "They grow at the same rate",
        "It depends on the specific distributions"
      ],
      correct: 1,
      explanation: "For $t \\gg 1$: $(t-1)^2 \\approx t^2$ (quadratic) while $t \\log t$ is superlinear but subquadratic. So **chi-squared is more sensitive** to large importance weights. This means chi-squared will flag distributional mismatch that KL might underweight — which is why directly bounding the ratio (PPO clipping) is more conservative than a KL penalty."
    },
    {
      type: "info",
      title: "Jensen-Shannon: A Symmetric, Bounded Divergence",
      content: "The Jensen-Shannon divergence symmetrizes KL by comparing both distributions to their average:\n\n$$\\text{JS}(P \\| Q) = \\frac{1}{2}\\text{KL}(P \\| M) + \\frac{1}{2}\\text{KL}(Q \\| M), \\quad M = \\frac{P + Q}{2}$$\n\nKey properties:\n\n**Symmetric**: $\\text{JS}(P \\| Q) = \\text{JS}(Q \\| P)$ — neither distribution is privileged.\n\n**Bounded**: $0 \\leq \\text{JS}(P \\| Q) \\leq \\log 2$ — the divergence never explodes, even when distributions have disjoint support.\n\n**Well-defined everywhere**: Unlike KL, JS is finite even when $P$ and $Q$ don't overlap, because both are compared to the mixture $M$ which has support wherever either $P$ or $Q$ does.\n\nThese properties made JS attractive as a training objective — and it's exactly what the original GAN minimizes."
    },
    {
      type: "mc",
      question: "If $P$ and $Q$ have completely disjoint support (no overlap at all), what is $\\text{JS}(P \\| Q)$?",
      options: [
        "$0$",
        "$\\log 2$",
        "$\\infty$",
        "Undefined"
      ],
      correct: 1,
      explanation: "When $P$ and $Q$ don't overlap: $M = (P+Q)/2$ has support everywhere either does. Then $\\text{KL}(P \\| M) = \\sum P \\log \\frac{P}{P/2} = \\sum P \\log 2 = \\log 2$, and similarly $\\text{KL}(Q \\| M) = \\log 2$. So $\\text{JS} = \\frac{1}{2}\\log 2 + \\frac{1}{2}\\log 2 = \\log 2$. JS saturates at its maximum — it tells you the distributions are maximally different, but **the gradient of this constant is zero**. This is the vanishing gradient problem that motivated Wasserstein GANs."
    },
    {
      type: "info",
      title: "The GAN Objective Is JS Minimization",
      content: "The original GAN training objective is:\n\n$$\\min_G \\max_D \\;\\mathbb{E}_{x \\sim P}[\\log D(x)] + \\mathbb{E}_{x \\sim Q_G}[\\log(1 - D(x))]$$\n\nwhere $P$ is the data distribution and $Q_G$ is the generator's distribution.\n\nAt the optimal discriminator $D^*(x) = \\frac{P(x)}{P(x) + Q_G(x)}$, the inner maximization evaluates to:\n\n$$2\\,\\text{JS}(P \\| Q_G) - 2\\log 2$$\n\nSo the generator minimizes JS divergence between real and generated data. This explains both GAN successes and failures:\n\n**Success**: JS is bounded and symmetric, so training is stable when distributions overlap.\n\n**Failure**: When $P$ and $Q_G$ don't overlap (common early in training when the generator is poor), JS saturates at $\\log 2$ and gradients vanish. The generator gets no learning signal."
    },
    {
      type: "mc",
      question: "Early in GAN training, the generator produces low-quality samples far from the data distribution. The discriminator easily distinguishes real from fake (near-perfect classification). What happens to the JS divergence gradient?",
      options: [
        "The gradient is very large, providing a strong learning signal",
        "The gradient is moderate and training proceeds normally",
        "The gradient nearly vanishes because JS saturates at $\\log 2$",
        "The gradient oscillates unpredictably"
      ],
      correct: 2,
      explanation: "When the discriminator is near-perfect, $P$ and $Q_G$ are effectively non-overlapping. JS saturates at its maximum $\\log 2$, and $\\nabla_G \\text{JS} \\approx 0$. The generator receives almost no gradient signal. This is the **GAN training instability** that led to alternatives: Wasserstein GAN uses Earth Mover's distance (not an f-divergence) which provides gradients even for non-overlapping distributions."
    },
    {
      type: "info",
      title: "Mutual Information as KL Divergence",
      content: "**Mutual information** measures how much knowing one variable tells you about another:\n\n$$I(X; Z) = \\text{KL}\\big(P(X, Z) \\,\\|\\, P(X)\\,P(Z)\\big)$$\n\nIt's the KL divergence between the joint distribution and the product of marginals. If $X$ and $Z$ are independent, the joint *equals* the product of marginals, so $I(X; Z) = 0$.\n\nIn representation learning, $Z$ is a learned representation of input $X$. Then $I(X; Z)$ quantifies **how much information about $X$ the representation captures**.\n\n**CLIP** (contrastive learning) implicitly maximizes a lower bound on $I(\\text{image}; \\text{text})$ — it learns representations where images and their captions are maximally informative about each other.\n\n**The information bottleneck** minimizes $I(X; Z)$ (compress the representation) while maximizing $I(Z; Y)$ (preserve task-relevant information). This trade-off has the same mathematical structure as the RLHF objective — you'll see this connection in the Hard module."
    },
    {
      type: "mc",
      question: "A representation $Z = f(X)$ is a deterministic function of input $X$. If $I(X; Z) = 0$, what can you conclude about $Z$?",
      options: [
        "$Z$ perfectly encodes all information in $X$",
        "$Z$ is independent of $X$ — it carries no information about the input",
        "$Z$ encodes only task-relevant information",
        "Nothing — MI can be zero even for useful representations"
      ],
      correct: 1,
      explanation: "$I(X; Z) = 0$ means $P(X, Z) = P(X)P(Z)$, i.e., $X$ and $Z$ are independent. The representation carries **no information** about the input — it's useless. High MI means $Z$ is informative about $X$; low MI means $Z$ has discarded information. The information bottleneck asks: what's the minimum MI $I(X; Z)$ you can get away with while still predicting the target $Y$ well?"
    }
  ]
};

export const hardModule = {
  id: "0.2-hard",
  sectionId: "0.2",
  title: "Variational Bounds & the GAN Connection",
  difficulty: "hard",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "The Fenchel Conjugate",
      content: "To derive the GAN objective from f-divergences, we need one tool from convex analysis: the **Fenchel conjugate** (also called the convex conjugate).\n\nFor a convex function $f$, its conjugate is:\n\n$$f^*(u) = \\sup_{t \\in \\text{dom}(f)} \\{u \\cdot t - f(t)\\}$$\n\nGeometrically, $f^*(u)$ is the maximum gap between the linear function $ut$ and $f(t)$. The supremum is achieved where $f'(t) = u$, giving $t^* = (f')^{-1}(u)$.\n\nThe key property: **$f^{**} = f$** for convex $f$ (the conjugate of the conjugate recovers $f$). This means:\n\n$$f(t) = \\sup_u \\{u \\cdot t - f^*(u)\\}$$\n\nThis \"variational\" representation of $f$ is what lets us turn f-divergences into optimization problems — which is exactly what a GAN discriminator does."
    },
    {
      type: "mc",
      question: "Compute the Fenchel conjugate of $f(t) = t \\log t$ (the KL generator). Set $\\frac{d}{dt}[ut - t\\log t] = 0$ to find the optimal $t^*$, then substitute back.",
      options: [
        "$f^*(u) = u$",
        "$f^*(u) = e^{u-1}$",
        "$f^*(u) = u^2 / 2$",
        "$f^*(u) = \\log(1 + e^u)$"
      ],
      correct: 1,
      explanation: "Setting $\\frac{d}{dt}[ut - t\\log t] = u - \\log t - 1 = 0$ gives $t^* = e^{u-1}$. Substituting: $f^*(u) = u \\cdot e^{u-1} - e^{u-1} \\cdot (u-1) = e^{u-1}(u - u + 1) = e^{u-1}$. So the conjugate of $t \\log t$ is $e^{u-1}$."
    },
    {
      type: "info",
      title: "The Variational Representation of f-Divergences",
      content: "Substituting $f(t) = \\sup_u\\{ut - f^*(u)\\}$ into the f-divergence definition:\n\n$$D_f(P \\| Q) = \\mathbb{E}_Q\\!\\left[\\sup_u\\left\\{u \\cdot \\frac{P}{Q} - f^*(u)\\right\\}\\right]$$\n\nSwapping the sup outside the expectation (which gives a lower bound) and allowing $u$ to be a function $T(x)$:\n\n$$D_f(P \\| Q) \\geq \\sup_T\\left\\{\\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[f^*(T(x))]\\right\\}$$\n\nThis bound is **tight** — equality holds when $T^*(x) = f'(P(x)/Q(x))$.\n\nThis is the **variational lower bound** on f-divergences, and it transforms a divergence computation into an optimization problem over functions $T$. The function $T$ that maximizes the bound is exactly what a **GAN discriminator** computes."
    },
    {
      type: "mc",
      question: "In the variational bound $D_f(P \\| Q) \\geq \\sup_T\\{\\mathbb{E}_P[T] - \\mathbb{E}_Q[f^*(T)]\\}$, the function $T$ is optimized to tighten the bound. In a GAN, what plays the role of $T$?",
      options: [
        "The generator network",
        "The discriminator network",
        "The loss function",
        "The latent noise vector"
      ],
      correct: 1,
      explanation: "The **discriminator** maximizes the variational bound — it's the function $T$ that distinguishes real from generated data. The generator then minimizes the resulting (approximate) f-divergence. This is the adversarial game: the discriminator tightens the lower bound on the divergence, and the generator minimizes it."
    },
    {
      type: "info",
      title: "Deriving the GAN Objective",
      content: "For Jensen-Shannon, let's see how the variational bound gives the standard GAN loss.\n\nThe JS divergence can be written as an f-divergence. Applying the variational bound with a specific parameterization $T(x) = \\log D(x)$ and working through the conjugate, the discriminator objective becomes:\n\n$$\\max_D\\; \\mathbb{E}_{x \\sim P}[\\log D(x)] + \\mathbb{E}_{x \\sim Q_G}[\\log(1 - D(x))]$$\n\nThis is exactly the original GAN objective! The discriminator $D$ maximizes a lower bound on $\\text{JS}(P \\| Q_G)$.\n\nThe **f-GAN framework** (Nowozin et al., 2016) generalized this: choose *any* f-divergence, compute its conjugate $f^*$, and plug into the variational bound to get a valid GAN training objective.\n\nDifferent choices of $f$ give different GAN variants:\n- $f(t) = t \\log t$ (KL) → KL-GAN\n- $f(t) = -(t+1)\\log\\frac{t+1}{2} + t\\log t$ (JS) → Original GAN\n- $f(t) = (t-1)^2$ (chi-squared) → Pearson GAN $\\approx$ Least-Squares GAN"
    },
    {
      type: "mc",
      question: "Least-Squares GAN (LSGAN) replaces the log-loss with squared error: $\\min_G \\max_D\\; \\mathbb{E}_P[(D(x)-1)^2] + \\mathbb{E}_{Q_G}[D(x)^2]$. Based on the f-GAN framework, which f-divergence does LSGAN approximately minimize?",
      options: [
        "KL divergence",
        "Jensen-Shannon divergence",
        "Pearson chi-squared divergence",
        "Total variation distance"
      ],
      correct: 2,
      explanation: "LSGAN's squared-error objective corresponds to the **Pearson chi-squared divergence** ($f(t) = (t-1)^2$). The conjugate of $(t-1)^2$ is $f^*(u) = u + u^2/4$, which gives a quadratic (least-squares) discriminator loss. This was shown by Mao et al. (2017) and provides a theoretical justification for why LSGAN training is often more stable than the original — chi-squared has nicer gradient properties near saturation."
    },
    {
      type: "info",
      title: "Entropy as a Training Diagnostic",
      content: "Switching from divergences between distributions to a property of a *single* distribution: **entropy** is a powerful diagnostic during training.\n\nThe entropy of a discrete distribution $p$ is $H(p) = -\\sum_i p_i \\log p_i$. Maximum entropy = uniform distribution; minimum entropy = all mass on one outcome.\n\nIn transformers, two key entropy diagnostics:\n\n**Attention entropy**: Each attention head produces a distribution over keys. If $H(\\text{attn}) \\to 0$, the attention is collapsing to a **one-hot** pattern (attending to a single token). This is called **entropy collapse** and can indicate the head is over-specializing.\n\n**Output entropy**: The softmax output distribution's entropy reflects model confidence. Very low entropy = model is extremely confident (possibly overconfident). Temperature $\\tau$ in $\\text{softmax}(z/\\tau)$ directly controls output entropy: higher $\\tau$ → more uniform → higher entropy.\n\nDuring training, monitoring attention entropy across layers and heads reveals which components are learning meaningful patterns vs. collapsing. Loss spikes often correlate with sudden entropy drops in specific attention heads."
    },
    {
      type: "mc",
      question: "During training, you observe that several attention heads in the same layer have attention entropy dropping to near zero. What is the most likely consequence?",
      options: [
        "Those heads are learning precise, useful attention patterns",
        "Those heads are collapsing to attend to a single position (e.g., [BOS] token), effectively wasting model capacity",
        "The model is converging faster than expected",
        "The learning rate should be increased to prevent stagnation"
      ],
      correct: 1,
      explanation: "Attention entropy near zero means the attention distribution is nearly one-hot — the head always attends to the same position regardless of input. This is **entropy collapse**: the head has stopped being useful and is wasting parameters. The $1/\\sqrt{d_k}$ scaling in attention exists precisely to prevent softmax saturation that leads to this collapse. If multiple heads collapse simultaneously, it often indicates the learning rate is too high or a training instability is developing."
    },
    {
      type: "info",
      title: "The Information Bottleneck: Same Math as RLHF",
      content: "The **information bottleneck** (IB) asks: what's the best compressed representation $Z$ of input $X$ for predicting target $Y$?\n\n$$\\min_{p(Z|X)} \\; I(X; Z) - \\beta \\cdot I(Z; Y)$$\n\n- $I(X; Z)$: how much $Z$ remembers about $X$ (want to **minimize** — compress)\n- $I(Z; Y)$: how useful $Z$ is for predicting $Y$ (want to **maximize** — preserve task info)\n- $\\beta$: controls the compression/performance trade-off\n\nNow compare with the RLHF objective:\n\n$$\\max_\\pi \\; \\mathbb{E}[R(x)] - \\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$$\n\n- $\\mathbb{E}[R(x)]$: reward (want to **maximize**)\n- $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$: distance from reference (want to **minimize** — stay close)\n- $\\beta$: controls the reward/constraint trade-off\n\nBoth are **KL-constrained optimization** problems. The $\\beta$ parameter plays the same role: it traces a Pareto frontier between competing objectives. Increase $\\beta$ → more conservative (tighter constraint); decrease $\\beta$ → more aggressive (chase the objective harder).\n\nThis isn't a coincidence — both are instances of the **rate-distortion** framework from information theory."
    },
    {
      type: "mc",
      question: "In both the information bottleneck and RLHF, $\\beta$ controls a trade-off. In the IB, increasing $\\beta$ makes the model retain more task-relevant information. In RLHF, increasing $\\beta$ makes the policy:",
      options: [
        "Chase higher rewards more aggressively, diverging further from $\\pi_{\\text{ref}}$",
        "Stay closer to $\\pi_{\\text{ref}}$, sacrificing potential reward",
        "Have no effect — $\\beta$ only matters in the IB setting",
        "Increase the entropy of the policy distribution"
      ],
      correct: 1,
      explanation: "In the RLHF objective $\\max \\mathbb{E}[R] - \\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, a larger $\\beta$ increases the penalty for deviating from $\\pi_{\\text{ref}}$. The policy stays closer to the reference model and sacrifices potential reward for safety/coherence. This is exactly analogous to the IB: higher $\\beta$ → stronger constraint → more conservative behavior. The Pareto frontier traced by varying $\\beta$ is called the **rate-distortion curve** in information theory."
    }
  ]
};
