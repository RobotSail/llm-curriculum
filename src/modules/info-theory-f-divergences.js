// Module: Information Theory — f-Divergences
// Section 0.2: Probability, statistics & information theory
// Three difficulty tracks building on the user's KL → cross-entropy derivation

export const easyModule = {
  id: "0.2-easy",
  sectionId: "0.2",
  title: "The f-Divergence Family",
  difficulty: "easy",
  moduleType: "learning",
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
      options: ["It requires both distributions to be discrete for the divergence to be well-defined", "It cannot handle continuous distributions without first discretizing the support", "It is always bounded between 0 and 1, limiting its sensitivity to large mismatches", "It is asymmetric: $\\text{KL}(P \\| Q) \\neq \\text{KL}(Q \\| P)$ in general for $P \\neq Q$"],
      correct: 3,
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
      options: ["$Q$ spreads to cover both peaks, possibly placing significant mass between them", "$Q$ locks onto one peak and assigns near-zero probability to the other peak", "$Q$ converges toward a uniform distribution that ignores the bimodal structure", "$Q$ exactly matches $P$ because forward KL guarantees recovery of the true distribution"],
      correct: 0,
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
        "KL divergence ($f(t) = t \\log t$), which is asymmetric but unbounded",
        "Chi-squared divergence ($f(t) = (t-1)^2$), which is asymmetric but bounded",
        "Jensen-Shannon divergence, which is symmetric and bounded in $[0, \\log 2]$",
        "Reverse KL divergence ($f(t) = -\\log t$), which is asymmetric but bounded"
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
      options: ["The KL divergence stays bounded and training proceeds smoothly for that sequence", "The gradient signal for that sequence vanishes, so the model never learns to cover it", "The model ignores that sequence entirely and focuses on more likely training sequences", "The loss for that sequence becomes very large, creating a strong corrective gradient signal"],
      correct: 3,
      explanation: "Since $\\text{KL}(P \\| Q) = \\sum P(x) \\log(P(x)/Q(x))$, when $Q(x) \\to 0$ for some $x$ where $P(x) > 0$, the term $\\log(P/Q) \\to \\infty$. This creates a large loss and strong gradient, forcing the model to assign *some* probability to every sequence in the training data. This is forward KL's mode-covering property at work — it's also why language models sometimes hallucinate: they've been forced to cover the full support of $P$, including rare patterns."
    },
    {
      type: "info",
      title: "Summary: From KL to a Toolkit",
      content: "You started with one divergence — KL — and now have a family of them, each defined by a convex function $f$ with $f(1) = 0$:\n\n$$D_f(P \\| Q) = \\mathbb{E}_Q\\!\\left[f\\!\\left(\\frac{P(x)}{Q(x)}\\right)\\right]$$\n\nKey takeaways:\n\n**1.** KL's asymmetry isn't a bug — it's a feature. Forward vs. reverse KL are different tools for different jobs (mode-covering vs. mode-seeking).\n\n**2.** Different f-divergences have different sensitivity profiles. The right choice depends on whether you care about support mismatch (KL), importance weight variance (chi-squared), or robust optimization (JS).\n\n**3.** Properties like symmetry and boundedness aren't just mathematical curiosities — they determine whether a divergence is suitable as a training objective.\n\nIn the **Medium** module, you'll see how chi-squared divergence is literally the variance of importance sampling weights (explaining why off-policy RL is hard), and how Jensen-Shannon gives you the original GAN objective."
    }
  ]
};

export const hardModule = {
  id: "0.2-hard",
  sectionId: "0.2",
  title: "Variational Bounds & the GAN Connection",
  difficulty: "hard",
  moduleType: "learning",
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
        "$f^*(u) = u$ (linear in $u$)",
        "$f^*(u) = e^{u-1}$ (exponential)",
        "$f^*(u) = u^2 / 2$ (quadratic)",
        "$f^*(u) = \\log(1 + e^u)$ (softplus)"
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
      options: ["The generator network, which produces samples to fool the bound", "The loss function, which defines the divergence being estimated", "The discriminator network, which tightens the variational bound", "The latent noise vector, which provides the randomness for sampling"],
      correct: 2,
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
      options: ["KL divergence — the log-loss corresponds to minimizing the forward KL between distributions", "Jensen-Shannon divergence — the squared error is equivalent to the symmetrized KL formulation", "Total variation distance — the squared error directly measures absolute probability differences", "Pearson chi-squared divergence — the squared error corresponds to penalizing squared density ratios"],
      correct: 3,
      explanation: "LSGAN's squared-error objective corresponds to the **Pearson chi-squared divergence** ($f(t) = (t-1)^2$). The conjugate of $(t-1)^2$ is $f^*(u) = u + u^2/4$, which gives a quadratic (least-squares) discriminator loss. This was shown by Mao et al. (2017) and provides a theoretical justification for why LSGAN training is often more stable than the original — chi-squared has nicer gradient properties near saturation."
    },
    {
      type: "info",
      title: "Why the Bound Is Tight — and What Breaks It",
      content: "The variational bound $D_f(P \\| Q) \\geq \\sup_T\\{\\mathbb{E}_P[T] - \\mathbb{E}_Q[f^*(T)]\\}$ is tight when $T^*(x) = f'(P(x)/Q(x))$ — the optimal critic recovers the true density ratio.\n\nIn practice, three things weaken the bound:\n\n**1. Finite capacity**: A neural network $T_\\theta$ can only approximate $T^*$. Deeper, wider networks give tighter bounds but are harder to optimize.\n\n**2. Finite samples**: We estimate $\\mathbb{E}_P[T]$ and $\\mathbb{E}_Q[f^*(T)]$ with minibatch averages. High-variance estimates of these expectations produce noisy gradients for the generator.\n\n**3. Optimization dynamics**: The generator and discriminator are updated alternately, not jointly. If the discriminator is too strong (tight bound), the generator receives vanishing gradients through saturated activations. If too weak (loose bound), the generator gets uninformative gradients.\n\nThis is the fundamental instability of GAN training: the bound must be tight enough to guide the generator, but not so tight that gradients vanish. The choice of f-divergence directly affects this balance."
    },
    {
      type: "mc",
      question: "In the original GAN (JS divergence), when the discriminator becomes near-optimal and the generator distribution $Q_G$ has little overlap with the real distribution $P$, what happens to the generator's gradients?",
      options: [
        "The gradients become large and unstable, causing the generator to overshoot and produce increasingly unrealistic samples each iteration",
        "The gradients remain informative because JS divergence is bounded, so the generator always receives a useful learning signal",
        "The gradients vanish because JS divergence saturates at $\\log 2$ when distributions don't overlap, giving near-zero signal",
        "The gradients oscillate between large positive and negative values, preventing the generator from converging to a stable solution"
      ],
      correct: 2,
      explanation: "When $P$ and $Q_G$ have disjoint supports, $\\text{JS}(P \\| Q_G) = \\log 2$ (its maximum). The discriminator achieves perfect classification, and $D(x) \\to 0$ for generated samples. The generator gradient $\\nabla_\\theta \\log(1 - D(G(z)))$ vanishes because $\\log(1 - D) \\to 0$ when $D \\to 0$. This **vanishing gradient** problem motivated Wasserstein GAN, which uses the Earth Mover's distance (not an f-divergence) to provide useful gradients even with non-overlapping supports."
    },
    {
      type: "info",
      title: "Beyond f-Divergences: Wasserstein and IPMs",
      content: "The vanishing gradient problem of JS-GAN revealed a fundamental limitation of f-divergences: they all depend on **density ratios** $P(x)/Q(x)$, which are undefined or degenerate when distributions live on low-dimensional manifolds in high-dimensional space (as images do).\n\nThe **Wasserstein-1 distance** (Earth Mover's distance) takes a different approach:\n\n$$W_1(P, Q) = \\inf_{\\gamma \\in \\Pi(P,Q)} \\mathbb{E}_{(x,y) \\sim \\gamma}[\\|x - y\\|]$$\n\nBy the Kantorovich-Rubinstein duality, this has its own variational form:\n\n$$W_1(P, Q) = \\sup_{\\|T\\|_L \\leq 1} \\{\\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[T(x)]\\}$$\n\nwhere $T$ is restricted to 1-Lipschitz functions. Compare this to the f-divergence bound — the structure is similar (sup over a function class of the difference in expectations), but with a **Lipschitz constraint** instead of $f^*$.\n\nThis family of distances — **Integral Probability Metrics** (IPMs) — provides meaningful gradients even when distributions don't overlap, because they measure how much *mass must move* rather than how density ratios differ."
    },
    {
      type: "mc",
      question: "The WGAN critic objective is $\\sup_{\\|T\\|_L \\leq 1}\\{\\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[T(x)]\\}$. In the original WGAN paper, the Lipschitz constraint was enforced by weight clipping. Why was this approach problematic, leading to WGAN-GP?",
      options: [
        "Weight clipping biased the critic toward very simple functions (low capacity), providing weak gradients and slow convergence",
        "Weight clipping made the critic too powerful, causing it to memorize training examples rather than learn distributional structure",
        "Weight clipping violated the Kantorovich-Rubinstein duality conditions, producing systematically invalid Wasserstein distance estimates",
        "Weight clipping caused the critic to always output the same value regardless of input, making training impossible to converge"
      ],
      correct: 0,
      explanation: "Clipping weights to $[-c, c]$ forces the critic to use only a small fraction of its capacity — it biases toward very simple, nearly linear functions. This means the critic provides **weak, uninformative gradients** to the generator. WGAN-GP (Gulrajani et al., 2017) replaced weight clipping with a **gradient penalty**: $\\lambda \\mathbb{E}_{\\hat{x}}[(\\|\\nabla_{\\hat{x}} T(\\hat{x})\\| - 1)^2]$, which directly enforces the Lipschitz constraint at interpolated points $\\hat{x}$ between real and generated samples. This allows the critic to use its full capacity while satisfying the constraint."
    },
    {
      type: "info",
      title: "Summary: A Unified View of Distributional Distances",
      content: "You've now seen the full arc from f-divergences to their variational representations to practical GAN training:\n\n**1. f-Divergences** measure distributional distance via density ratios and a convex generator $f$. Each choice of $f$ gives a different divergence (KL, JS, chi-squared, etc.) with distinct sensitivity properties.\n\n**2. The variational bound** $D_f \\geq \\sup_T\\{\\mathbb{E}_P[T] - \\mathbb{E}_Q[f^*(T)]\\}$ transforms divergence estimation into an optimization problem — the discriminator/critic tightens this bound.\n\n**3. The original GAN** is the variational bound applied to JS divergence. The f-GAN framework generalizes this to any f-divergence.\n\n**4. f-Divergence limitations**: All f-divergences depend on density ratios and can produce vanishing gradients when distributions have disjoint supports (common in high dimensions).\n\n**5. Wasserstein/IPMs** replace density ratios with function-class constraints (Lipschitz), providing useful gradients even without distributional overlap. This is why WGAN training is often more stable.\n\nThe choice of divergence or distance defines your training objective's **failure modes**: KL gives mode-covering, JS gives vanishing gradients with disjoint supports, Wasserstein gives stable but slower training. Understanding these trade-offs is essential for both GAN design and RLHF objective selection."
    },
    {
      type: "mc",
      question: "You're designing a training objective for aligning a language model. You want the objective to (a) penalize the model for deviating from a reference distribution, and (b) provide informative gradients even when the model's distribution is far from the reference. Which approach best satisfies both requirements?",
      options: [
        "Combine reverse KL $\\text{KL}(\\pi \\| P_{\\text{ref}})$ with a Wasserstein regularizer — KL penalizes deviation, Wasserstein provides gradients in low-overlap regions",
        "Use JS divergence $\\text{JS}(\\pi \\| P_{\\text{ref}})$ — it's symmetric and bounded, handling both directions of distributional mismatch equally well",
        "Use forward KL $\\text{KL}(P_{\\text{ref}} \\| \\pi)$ — its mode-covering property ensures strong gradients whenever the model misses reference probability mass",
        "Use chi-squared divergence $\\chi^2(\\pi \\| P_{\\text{ref}})$ — it directly measures importance weight variance, the most relevant quantity for alignment stability"
      ],
      correct: 0,
      explanation: "Forward KL satisfies (a) but can have infinite values that destabilize optimization. JS satisfies (b) via boundedness but actually *loses* gradient signal when distributions don't overlap (the vanishing gradient problem). Chi-squared is sensitive to density ratio variance but doesn't specifically address low-overlap gradients. Combining reverse KL (which directly penalizes the model for placing mass where the reference doesn't) with a Wasserstein term (which provides geometric gradient information even in low-overlap regimes) addresses both requirements. This hybrid approach reflects the general principle: different distances have complementary strengths, and practical objectives often combine them."
    }
  ]
};
