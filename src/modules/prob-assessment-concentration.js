// Assessment: Concentration Inequalities & Tail Bounds
// Section 0.2: Diagnostic test — Markov, Chebyshev, Hoeffding, Chernoff, sub-Gaussian
// Pure assessment to gauge depth of understanding

export const concentrationAssessment = {
  id: "0.2-assess-concentration",
  sectionId: "0.2",
  title: "Assessment: Concentration Inequalities & Tail Bounds",
  difficulty: "hard",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "info",
      title: "Diagnostic: Concentration Inequalities & Tail Bounds",
      content: "This is a **diagnostic assessment** covering concentration inequalities — the mathematical tools for bounding how much a random variable deviates from its expectation.\n\nThese inequalities underpin: generalization bounds, PAC learning theory, why large-batch training works, confidence intervals for evaluation metrics, and the theory of gradient estimation in SGD.\n\nIf you score below 70%, review these — they form the backbone of statistical learning theory."
    },
    {
      type: "mc",
      question: "**Markov's inequality** states that for a non-negative random variable $X$ and $a > 0$: $P(X \\geq a) \\leq \\frac{\\mathbb{E}[X]}{a}$. This bound is:",
      options: [
        "Tight for all distributions",
        "Very loose in general (it only uses the mean), but important because it requires minimal assumptions — just non-negativity",
        "Only valid for Gaussian distributions",
        "Tighter than Chebyshev's inequality"
      ],
      correct: 1,
      explanation: "Markov's inequality is the weakest but most general bound — it only requires $X \\geq 0$ and finite mean. It's rarely used directly because it's very loose, but it's the foundation from which all stronger bounds are derived (Chebyshev applies Markov to $(X - \\mu)^2$, Chernoff applies Markov to $e^{tX}$). Its value is conceptual: it shows that heavy-tailed behavior is constrained by the mean."
    },
    {
      type: "mc",
      question: "**Chebyshev's inequality** states $P(|X - \\mu| \\geq t) \\leq \\frac{\\sigma^2}{t^2}$. For a sample mean $\\bar{X}_n$ of $n$ i.i.d. variables with variance $\\sigma^2$, what does Chebyshev give for $P(|\\bar{X}_n - \\mu| \\geq \\epsilon)$?",
      options: [
        "$\\frac{\\sigma^2}{\\epsilon^2}$ — independent of $n$",
        "$\\frac{\\sigma^2}{n \\epsilon^2}$ — the bound tightens linearly with $n$, giving a $1/n$ convergence rate",
        "$\\frac{\\sigma}{\\sqrt{n} \\epsilon}$",
        "$e^{-n\\epsilon^2 / 2\\sigma^2}$"
      ],
      correct: 1,
      explanation: "$\\text{Var}(\\bar{X}_n) = \\sigma^2/n$, so Chebyshev gives $P(|\\bar{X}_n - \\mu| \\geq \\epsilon) \\leq \\sigma^2 / (n\\epsilon^2)$. This is a **polynomial** (not exponential) bound in $n$. To get $P \\leq \\delta$, you need $n \\geq \\sigma^2 / (\\epsilon^2 \\delta)$ — which can be impractically large. This motivates sub-Gaussian bounds that give exponential concentration."
    },
    {
      type: "mc",
      question: "**Hoeffding's inequality** for i.i.d. bounded random variables $X_i \\in [a, b]$ states $P(|\\bar{X}_n - \\mu| \\geq t) \\leq 2\\exp\\left(-\\frac{2nt^2}{(b-a)^2}\\right)$. Compared to Chebyshev, this gives:",
      options: [
        "A weaker bound that uses more assumptions",
        "An **exponentially** decreasing tail bound in $n$ (vs. polynomial for Chebyshev) — the price is requiring bounded variables",
        "The same rate but with a better constant",
        "A bound that doesn't depend on the range $[a, b]$"
      ],
      correct: 1,
      explanation: "Hoeffding gives $e^{-\\Theta(n)}$ decay vs. Chebyshev's $1/n$ decay. To achieve failure probability $\\delta$, Hoeffding needs $n = O(\\log(1/\\delta) / t^2)$ samples — logarithmic in $1/\\delta$ vs. Chebyshev's linear $O(1/\\delta)$. This exponential concentration is crucial in practice: it means evaluation metrics converge quickly and generalization bounds are meaningful."
    },
    {
      type: "mc",
      question: "A random variable $X$ is **sub-Gaussian** with parameter $\\sigma$ if $\\mathbb{E}[e^{t(X - \\mu)}] \\leq e^{\\sigma^2 t^2 / 2}$ for all $t$. Why is the sub-Gaussian property important in deep learning?",
      options: [
        "All neural network weights are sub-Gaussian",
        "It implies exponential tail concentration $P(|X - \\mu| \\geq t) \\leq 2e^{-t^2/(2\\sigma^2)}$, which applies to bounded random variables, Gaussian variables, and sums thereof — covering most quantities we care about in training and evaluation",
        "It guarantees convergence of SGD",
        "It means the distribution is exactly Gaussian"
      ],
      correct: 1,
      explanation: "Sub-Gaussianity is a tail condition: the tails decay at least as fast as a Gaussian with variance $\\sigma^2$. Bounded variables ($[a,b]$) are sub-Gaussian with $\\sigma = (b-a)/2$, and sums of sub-Gaussians are sub-Gaussian. This covers sample means, bounded losses, and many gradient estimators. The tail bound gives sharp concentration — the foundation for PAC-Bayes bounds, uniform convergence, and confidence intervals."
    },
    {
      type: "mc",
      question: "In SGD, the gradient estimate $g = \\nabla L(x_i; \\theta)$ for a random mini-batch is an unbiased estimator of $\\nabla L(\\theta)$. The **variance** of this estimate relates to concentration via:",
      options: [
        "Higher variance means faster convergence due to exploration",
        "Gradient variance $\\text{Var}(g)$ scales as $\\sigma^2_g / B$ for batch size $B$ — doubling batch size halves the variance but also halves the number of parameter updates per epoch",
        "Variance is irrelevant because SGD converges regardless",
        "Variance only matters for the final few iterations"
      ],
      correct: 1,
      explanation: "The mini-batch gradient is a sample mean, so $\\text{Var}(\\bar{g}_B) = \\sigma^2_g / B$. This means larger batches give more concentrated (less noisy) gradient estimates. But the total computation is $B \\times \\text{updates}$, and doubling $B$ halves updates per epoch. The \"critical batch size\" is where further increases in $B$ no longer improve wall-clock convergence — below this, gradient noise is too high; above it, you're wasting compute on variance reduction that doesn't help."
    },
    {
      type: "mc",
      question: "**McDiarmid's inequality** generalizes Hoeffding to functions of independent variables: if changing any one input $x_i$ changes $f(x_1, \\ldots, x_n)$ by at most $c_i$, then $P(f - \\mathbb{E}[f] \\geq t) \\leq \\exp\\left(-\\frac{2t^2}{\\sum_i c_i^2}\\right)$. This is useful in ML for bounding:",
      options: [
        "The training loss only",
        "The **generalization gap** — since train/test metrics are functions of i.i.d. data points, and changing one datapoint has bounded effect (bounded differences), McDiarmid gives exponential concentration of empirical risk around its expectation",
        "The norm of the weight matrix",
        "The number of epochs needed"
      ],
      correct: 1,
      explanation: "The empirical risk $\\hat{R} = \\frac{1}{n} \\sum_i L(f(x_i), y_i)$ is a function of $n$ i.i.d. data points. Changing one point changes $\\hat{R}$ by at most $c_i = M/n$ (if the loss is bounded by $M$). McDiarmid gives $P(|\\hat{R} - R| \\geq t) \\leq 2e^{-2nt^2/M^2}$. This is a key ingredient in VC theory and PAC learning bounds."
    },
    {
      type: "mc",
      question: "The **Chernoff bound** technique optimizes over the parameter in the moment generating function: $P(X \\geq a) \\leq \\min_{t > 0} e^{-ta} \\mathbb{E}[e^{tX}]$. This gives the **tightest exponential bound** because:",
      options: [
        "It uses all moments of the distribution simultaneously (through the MGF), not just the mean or variance — the optimization over $t$ selects the tightest exponential rate for each specific tail probability",
        "It only requires the mean",
        "It works for any distribution including heavy-tailed ones",
        "It is always tighter than the exact probability"
      ],
      correct: 0,
      explanation: "The Chernoff method applies Markov's inequality to $e^{tX}$ for any $t > 0$, then optimizes over $t$. Since $\\mathbb{E}[e^{tX}]$ encodes all moments (via Taylor expansion: $\\sum_k t^k \\mathbb{E}[X^k]/k!$), the optimization over $t$ extracts the best exponential bound the moment information can provide. For sub-Gaussian and sub-exponential variables, this gives sharp rates. It fails for truly heavy-tailed distributions where the MGF doesn't exist."
    },
    {
      type: "mc",
      question: "When evaluating an LLM on a benchmark with $n = 1000$ binary-scored questions, your model gets 72% accuracy. Using Hoeffding's inequality, a 95% confidence interval ($\\delta = 0.05$) for the true accuracy is approximately:",
      options: [
        "$72\\% \\pm 0.1\\%$ — very precise with 1000 samples",
        "$72\\% \\pm 2.7\\%$ — from $\\epsilon = \\sqrt{\\frac{\\log(2/\\delta)}{2n}} \\approx 0.027$ for bounded $[0,1]$ variables",
        "$72\\% \\pm 10\\%$ — 1000 samples is not enough",
        "$72\\% \\pm 1.4\\%$ — using the normal approximation"
      ],
      correct: 1,
      explanation: "Hoeffding gives $P(|\\hat{p} - p| \\geq \\epsilon) \\leq 2e^{-2n\\epsilon^2}$. Setting $2e^{-2(1000)\\epsilon^2} = 0.05$ gives $\\epsilon = \\sqrt{\\log(40)/2000} \\approx 0.043$, so about $\\pm 4.3\\%$. The tighter Clopper-Pearson or normal approximation ($\\pm 1.96\\sqrt{p(1-p)/n} \\approx \\pm 2.8\\%$) gives a narrower interval. The point: even 1000 samples leave meaningful uncertainty in benchmark scores, which is why comparing models that differ by 1-2% is statistically dubious."
    }
  ]
};
