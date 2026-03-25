// Focused learning module: Chi-squared divergence and its identity
// with importance sampling variance. Covers ESS, high-dimensional IS failure,
// Rényi divergence interpolation, self-normalized IS, and IWAE/ELBO connections.

export const chiSquaredLearning = {
  id: "0.2-chi-squared-learning-medium",
  sectionId: "0.2",
  title: "Chi-Squared Divergence and Importance Sampling",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    // --- Step 1: Definition ---
    {
      type: "info",
      title: "Chi-Squared Divergence: Definition",
      content: "The **chi-squared divergence** is a member of the f-divergence family with generator $f(t) = (t - 1)^2$. Plugging this into the f-divergence template $D_f(P \\| Q) = \\mathbb{E}_Q[f(P/Q)]$ gives:\n\n$$\\chi^2(P \\| Q) = \\mathbb{E}_Q\\!\\left[\\left(\\frac{P}{Q} - 1\\right)^{\\!2}\\right] = \\mathbb{E}_Q\\!\\left[\\left(\\frac{P}{Q}\\right)^{\\!2}\\right] - 1$$\n\nThe second form follows by expanding the square and using the normalization identity $\\mathbb{E}_Q[P/Q] = 1$. Unlike KL divergence, chi-squared divergence is a **polynomial** function of the likelihood ratio — it is not bounded from above and grows quadratically in regions where $P/Q$ is large.\n\nThis quadratic growth makes $\\chi^2$ especially sensitive to **tail mismatch**: even a few points where $P$ is much larger than $Q$ can dominate the entire divergence."
    },
    // --- Step 2: Normalization identity ---
    {
      type: "mc",
      question: "Why does $\\mathbb{E}_Q[P(x)/Q(x)] = 1$ hold for any distributions $P$ and $Q$ with the same support?",
      options: [
        "Because $P/Q$ is a valid probability density, so it must integrate to 1",
        "Because $\\sum_x Q(x) \\cdot \\frac{P(x)}{Q(x)} = \\sum_x P(x) = 1$ — the $Q$ cancels and we sum $P$",
        "Because Jensen's inequality applied to the convex function $t \\mapsto t$ gives this bound",
        "Because $P$ and $Q$ are assumed to have the same mean, so their ratio averages to 1"
      ],
      correct: 1,
      explanation: "The $Q(x)$ in the expectation cancels with the $Q(x)$ in the denominator of the ratio: $\\mathbb{E}_Q[P/Q] = \\sum_x Q(x) \\cdot P(x)/Q(x) = \\sum_x P(x) = 1$. This is a direct consequence of both being normalized probability distributions. It does NOT require equal means or any special relationship between $P$ and $Q$ beyond shared support."
    },
    // --- Step 3: The key identity ---
    {
      type: "info",
      title: "The Key Identity: Chi-Squared = IS Variance",
      content: "Since $\\mathbb{E}_Q[P/Q] = 1$, the chi-squared divergence is literally the **variance** of the importance weight $w(x) = P(x)/Q(x)$ under $Q$:\n\n$$\\chi^2(P \\| Q) = \\text{Var}_Q\\!\\left[\\frac{P}{Q}\\right]$$\n\nThis identity is the central result of this module. It tells us that chi-squared divergence directly measures the **reliability of importance sampling** (IS) when we use samples from $Q$ to estimate expectations under $P$.\n\nIn an IS estimator $\\hat{\\mu} = \\frac{1}{n}\\sum_{i=1}^n \\frac{P(x_i)}{Q(x_i)} g(x_i)$ with $x_i \\sim Q$, the variance of the importance weights controls how noisy the estimate is. High $\\chi^2(P \\| Q)$ means a few samples will have enormous weights, dominating the estimate and making it unreliable.\n\nFor **off-policy RL and RLHF**, this is critical: when you evaluate a new policy $\\pi$ using trajectories collected under an old policy $\\pi_{\\text{old}}$, you are doing importance sampling with ratio $\\pi/\\pi_{\\text{old}}$."
    },
    // --- Step 4: Off-policy RLHF ---
    {
      type: "mc",
      question: "In off-policy RLHF, a new policy $\\pi$ is evaluated using trajectories from $\\pi_{\\text{old}}$. Which quantity directly measures the variance of the per-step importance weights?",
      options: [
        "$\\text{KL}(\\pi \\| \\pi_{\\text{old}})$, because KL controls the expected log-ratio",
        "$\\text{JS}(\\pi, \\pi_{\\text{old}})$, because Jensen-Shannon is symmetric and bounded",
        "$\\chi^2(\\pi \\| \\pi_{\\text{old}}) = \\text{Var}_{\\pi_{\\text{old}}}[\\pi / \\pi_{\\text{old}}]$, by the chi-squared–variance identity",
        "$\\text{TV}(\\pi, \\pi_{\\text{old}})$, because total variation bounds the maximum weight"
      ],
      correct: 2,
      explanation: "By the key identity, $\\chi^2(\\pi \\| \\pi_{\\text{old}}) = \\text{Var}_{\\pi_{\\text{old}}}[\\pi/\\pi_{\\text{old}}]$. This is exactly the variance of the importance sampling ratio. KL divergence measures the expected log-ratio (not variance), Jensen-Shannon is related but does not equal the IS variance, and total variation bounds probability differences rather than weight variance."
    },
    // --- Step 5: PPO clipping through chi-squared lens ---
    {
      type: "info",
      title: "PPO Clipping Through the Lens of Chi-Squared",
      content: "PPO clips the importance ratio $r = \\pi(a|s)/\\pi_{\\text{old}}(a|s)$ to the interval $[1 - \\epsilon, 1 + \\epsilon]$. What does this look like through the chi-squared lens?\n\nClipping forces $(r - 1)^2 \\leq \\epsilon^2$ for every action, which directly **upper-bounds the per-sample contribution to $\\chi^2$**. The total chi-squared divergence is bounded by $\\epsilon^2$, guaranteeing low IS variance and stable policy updates.\n\nCompare the f-divergence generators:\n- **KL**: $f(t) = t \\log t$ — grows as $t \\log t$ for large $t$\n- **Chi-squared**: $f(t) = (t-1)^2$ — grows as $t^2$ for large $t$\n\nFor large likelihood ratios ($t \\gg 1$), $t^2$ dominates $t \\log t$. This means chi-squared divergence is **more sensitive to outlier ratios** than KL. A policy that produces even a few actions with very high probability ratios will have moderate KL but potentially enormous chi-squared divergence — and correspondingly unreliable IS estimates.\n\nThis is why PPO's hard clipping is so effective: it directly controls the worst-case contributor to IS variance, whereas a soft KL penalty allows rare large ratios that can destabilize training."
    },
    // --- Step 6: Sensitivity comparison ---
    {
      type: "mc",
      question: "A single action has probability ratio $r = \\pi(a|s)/\\pi_{\\text{old}}(a|s) = 10$. How do the per-sample f-divergence contributions compare?",
      options: [
        "KL contributes $10 \\ln 10 \\approx 23$ and chi-squared contributes $(10-1)^2 = 81$; chi-squared is far more sensitive",
        "Both contribute approximately the same amount because $10 \\log 10 \\approx 100 \\approx (10-1)^2$",
        "KL contributes $(10-1)^2 = 81$ and chi-squared contributes $10 \\ln 10 \\approx 23$; KL is more sensitive",
        "Chi-squared contributes $\\log(81) \\approx 4.4$ because it applies a log transform to $(t-1)^2$"
      ],
      correct: 0,
      explanation: "The KL generator is $f(t) = t \\log t$, giving $10 \\ln 10 \\approx 23$. The chi-squared generator is $f(t) = (t-1)^2$, giving $81$. Chi-squared is roughly 3.5x larger for this single outlier ratio. As $t$ grows, the gap widens because $t^2$ dominates $t \\log t$ asymptotically. This demonstrates why chi-squared (and IS variance) is so much more sensitive to large ratios than KL."
    },
    // --- Step 7: Effective Sample Size ---
    {
      type: "info",
      title: "Effective Sample Size (ESS)",
      content: "When importance sampling with $n$ samples from $Q$, not all samples contribute equally. The **effective sample size** quantifies how many \"equivalent\" unweighted samples you have:\n\n$$\\text{ESS} = \\frac{\\left(\\sum_{i=1}^n w_i\\right)^2}{\\sum_{i=1}^n w_i^2}$$\n\nwhere $w_i = P(x_i)/Q(x_i)$ are the unnormalized importance weights. ESS always satisfies $1 \\leq \\text{ESS} \\leq n$.\n\nThe connection to chi-squared divergence is direct. For the population version with normalized weights $\\bar{w}_i = w_i / \\sum_j w_j$:\n\n$$\\text{ESS}_{\\text{pop}} = \\frac{n}{1 + \\chi^2(P \\| Q)}$$\n\nWhen $\\chi^2(P \\| Q) = 0$ (identical distributions), $\\text{ESS} = n$ — every sample is fully useful. As $\\chi^2$ grows, ESS shrinks. If $\\chi^2 = 99$, only $1\\%$ of your samples are effectively contributing — the estimate is dominated by a handful of high-weight samples.\n\nIn RLHF, if your new policy has drifted enough that $\\chi^2(\\pi \\| \\pi_{\\text{old}}) = 9$, your effective batch size is only $n/10$. A batch of 1000 trajectories gives you the statistical power of just 100."
    },
    // --- Step 8: ESS quiz ---
    {
      type: "mc",
      question: "You collect 2000 rollouts from $\\pi_{\\text{old}}$ and want to evaluate $\\pi$. You compute $\\chi^2(\\pi \\| \\pi_{\\text{old}}) \\approx 3$. Approximately how many effective samples do you have?",
      options: [
        "2000, because chi-squared of 3 is negligibly small",
        "667, because $\\text{ESS} = n / \\chi^2 = 2000 / 3$",
        "1000, because $\\text{ESS} = n / 2$ whenever chi-squared is finite",
        "500, because $\\text{ESS} = n / (1 + \\chi^2) = 2000 / 4$"
      ],
      correct: 3,
      explanation: "The population ESS formula gives $\\text{ESS} = n / (1 + \\chi^2) = 2000 / (1 + 3) = 500$. Note the $+1$ in the denominator: it comes from expanding $\\mathbb{E}[w^2] = \\text{Var}[w] + (\\mathbb{E}[w])^2 = \\chi^2 + 1$. So 75% of your samples are effectively wasted due to weight variance."
    },
    // --- Step 9: High-dimensional IS failure ---
    {
      type: "info",
      title: "Why Importance Sampling Fails in High Dimensions",
      content: "Importance sampling becomes catastrophically unreliable in high dimensions, and chi-squared divergence explains exactly why.\n\nConsider two $d$-dimensional Gaussians $P = \\mathcal{N}(\\mu, I)$ and $Q = \\mathcal{N}(0, I)$ with $\\|\\mu\\| = \\delta$. The chi-squared divergence is:\n\n$$\\chi^2(P \\| Q) = e^{\\delta^2 \\cdot d} - 1$$\n\nThis grows **exponentially in the dimension** $d$, even for a tiny shift $\\delta$. The ESS therefore shrinks exponentially: $\\text{ESS} \\approx n \\cdot e^{-\\delta^2 d}$.\n\nFor a concrete example: with $\\delta = 0.1$ and $d = 1000$, $\\chi^2 \\approx e^{10} - 1 \\approx 22{,}025$. You would need tens of thousands of samples before even one contributes meaningfully. With $d = 10{,}000$ (a modest token-level action space), the situation is hopeless.\n\nThis is the fundamental reason why **naive off-policy correction does not scale** to large action spaces. PPO's clipping, TRPO's trust regions, and other constrained update methods are not just conveniences — they are necessities because unconstrained IS breaks down exponentially."
    },
    // --- Step 10: High-dimensional quiz ---
    {
      type: "mc",
      question: "For two $d$-dimensional Gaussians differing only by a mean shift of $\\delta$ per dimension, why does importance sampling require exponentially many samples as $d$ grows?",
      options: [
        "Because KL divergence grows linearly with $d$, making the log-weights too spread out",
        "Because the total variation distance approaches 1, making the distributions non-overlapping",
        "Because the normalizing constants of high-dimensional Gaussians overflow numerically",
        "Because $\\chi^2 = e^{\\delta^2 d} - 1$ grows exponentially, so ESS $\\approx n \\cdot e^{-\\delta^2 d}$ collapses"
      ],
      correct: 3,
      explanation: "The chi-squared divergence between these Gaussians equals $e^{\\delta^2 d} - 1$, which is exponential in dimension. Since $\\text{ESS} = n/(1 + \\chi^2) \\approx n e^{-\\delta^2 d}$, the effective sample size collapses exponentially. KL does grow linearly with $d$ (option A is partially true), but it is the exponential growth of chi-squared that directly controls IS variance and ESS. This is a sharper characterization of IS failure than either total variation or KL alone."
    },
    // --- Step 11: Rényi divergence interpolation ---
    {
      type: "info",
      title: "Rényi Divergence: Interpolating KL and Chi-Squared",
      content: "The **Rényi divergence** of order $\\alpha > 0$ ($\\alpha \\neq 1$) provides a continuous interpolation between KL and chi-squared:\n\n$$D_\\alpha(P \\| Q) = \\frac{1}{\\alpha - 1} \\log \\mathbb{E}_Q\\!\\left[\\left(\\frac{P}{Q}\\right)^{\\!\\alpha}\\right]$$\n\nKey special cases:\n- $\\alpha \\to 1$: Rényi converges to **KL divergence** $\\text{KL}(P \\| Q)$\n- $\\alpha = 2$: Rényi becomes $D_2(P \\| Q) = \\log(1 + \\chi^2(P \\| Q))$\n\nSo $D_2$ is just the log of the second moment of importance weights, while $\\chi^2$ is the second central moment (variance). They are monotonically related — bounding one bounds the other.\n\nHigher $\\alpha$ makes Rényi increasingly sensitive to the **worst-case** ratio $P/Q$, approaching the max-divergence $D_\\infty = \\log \\max_x (P(x)/Q(x))$ as $\\alpha \\to \\infty$. In practice, $D_2$ (chi-squared) is the most commonly used order because it directly corresponds to IS variance and has clean analytical forms for exponential families."
    },
    // --- Step 12: Rényi quiz ---
    {
      type: "mc",
      question: "The Rényi divergence of order $\\alpha = 2$ satisfies $D_2(P \\| Q) = \\log(1 + \\chi^2(P \\| Q))$. If $D_2(P \\| Q) = \\log 5$, what is the effective sample size when using $n$ samples from $Q$?",
      options: [
        "$n / 5$, because $\\chi^2 = 5$ and $\\text{ESS} = n / (1 + \\chi^2) = n/6$ — wait, that doesn't match",
        "$n / 4$, because $\\chi^2 = e^{\\log 5} - 1 = 4$ and $\\text{ESS} = n / (1 + 4) = n/5$ — wait, that gives $n/5$",
        "$n / 5$, because $1 + \\chi^2 = e^{D_2} = 5$, so $\\text{ESS} = n / 5$",
        "$n / \\log 5$, because ESS divides by the Rényi divergence directly"
      ],
      correct: 2,
      explanation: "From $D_2 = \\log(1 + \\chi^2)$, we get $1 + \\chi^2 = e^{D_2} = e^{\\log 5} = 5$, so $\\chi^2 = 4$. The ESS formula gives $\\text{ESS} = n/(1 + \\chi^2) = n/5$. Note that ESS divides by $1 + \\chi^2$, which equals $e^{D_2}$ directly — so you can go from $D_2$ to ESS without computing $\\chi^2$ as an intermediate step."
    },
    // --- Step 13: Self-normalized IS ---
    {
      type: "info",
      title: "Practical Variance Reduction: Self-Normalized IS",
      content: "Standard (unnormalized) IS estimates $\\mathbb{E}_P[g(x)]$ as $\\frac{1}{n}\\sum_i w_i g(x_i)$, where $w_i = P(x_i)/Q(x_i)$. This is unbiased but has variance proportional to $\\chi^2(P \\| Q)$.\n\n**Self-normalized IS** divides by the sum of weights:\n\n$$\\hat{\\mu}_{\\text{SN}} = \\frac{\\sum_i w_i \\, g(x_i)}{\\sum_i w_i}$$\n\nThis introduces a small bias (of order $1/n$) but dramatically reduces variance. The intuition: if one sample has an enormous weight, unnormalized IS lets it dominate the entire estimate, but self-normalization caps its influence at a fraction of the total weight.\n\nSelf-normalized IS is **consistent** regardless of the chi-squared divergence — it converges to the correct answer as $n \\to \\infty$ even when $\\chi^2(P \\| Q)$ is large. However, convergence is still slow when chi-squared is high: you need many samples before the law of large numbers kicks in for both numerator and denominator.\n\nIn practice, most implementations of importance-weighted objectives (like those in off-policy RL) use self-normalized weights, often combined with clipping or truncation for additional stability."
    },
    // --- Step 14: IWAE and self-normalized IS quiz ---
    {
      type: "mc",
      question: "The importance-weighted ELBO (IWAE) used in variational autoencoders computes $\\mathcal{L}_k = \\mathbb{E}\\!\\left[\\log \\frac{1}{k}\\sum_{i=1}^k \\frac{p(x, z_i)}{q(z_i | x)}\\right]$ with $z_i \\sim q(z|x)$. As $k \\to \\infty$, which statement is correct?",
      options: [
        "$\\mathcal{L}_k \\to \\log p(x)$ because the IS estimate of $p(x)$ becomes exact, but convergence depends on $\\chi^2(p(z|x) \\| q(z|x))$",
        "$\\mathcal{L}_k$ remains a strict lower bound that never reaches $\\log p(x)$, regardless of sample count",
        "$\\mathcal{L}_k$ converges to the standard single-sample ELBO because additional samples provide diminishing returns",
        "$\\mathcal{L}_k \\to \\text{KL}(q \\| p)$ because the IWAE objective reduces to the KL divergence in the infinite-sample limit"
      ],
      correct: 0,
      explanation: "IWAE uses IS with proposal $q(z|x)$ to estimate the marginal $p(x) = \\mathbb{E}_{q(z|x)}[p(x,z)/q(z|x)]$. As $k \\to \\infty$, $\\frac{1}{k}\\sum_i w_i \\to \\mathbb{E}_q[w] = p(x)$, so $\\mathcal{L}_k \\to \\log p(x)$. However, the rate of convergence is controlled by the variance of $w = p(x,z)/q(z|x)$ under $q$, which is the chi-squared divergence $\\chi^2(p(z|x) \\| q(z|x))$. A poor variational approximation (high chi-squared) needs many more samples to approach the true log-likelihood."
    },
  ]
};
