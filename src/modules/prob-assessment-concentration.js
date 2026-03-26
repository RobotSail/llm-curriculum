// Assessment: Concentration Inequalities & Tail Bounds
// Section 0.2: Diagnostic test — Markov, Chebyshev, Hoeffding, Chernoff, sub-Gaussian
// Pure assessment to gauge depth of understanding

export const concentrationAssessment = {
  id: "0.2-assess-concentration",
  sectionId: "0.2",
  title: "Assessment: Concentration Inequalities & Tail Bounds",
  difficulty: "hard",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Concentration Inequalities & Tail Bounds",
      content: "This is a **diagnostic assessment** covering concentration inequalities — the mathematical tools for bounding how much a random variable deviates from its expectation.\n\nThese inequalities underpin: generalization bounds, PAC learning theory, why large-batch training works, confidence intervals for evaluation metrics, and the theory of gradient estimation in SGD.\n\nIf you score below 70%, review these — they form the backbone of statistical learning theory."
    },
    {
      type: "mc",
      question: "**Markov's inequality** states that for a non-negative random variable $X$ and $a > 0$: $P(X \\geq a) \\leq \\frac{\\mathbb{E}[X]}{a}$. This bound is:",
      options: ["Very loose in general (it only uses the mean), but important because it requires minimal assumptions — just non-negativity and a finite first moment", "Tight for all distributions with finite variance, since it leverages both the mean and the second moment to constrain the tail probability", "Only valid for Gaussian distributions or distributions that can be approximated by a Gaussian through the central limit theorem", "Tighter than Chebyshev's inequality because it avoids the squared term in the denominator that loosens higher-order moment bounds"],
      correct: 0,
      explanation: "Markov's inequality is the weakest but most general bound — it only requires $X \\geq 0$ and finite mean. It's rarely used directly because it's very loose, but it's the foundation from which all stronger bounds are derived (Chebyshev applies Markov to $(X - \\mu)^2$, Chernoff applies Markov to $e^{tX}$). Its value is conceptual: it shows that heavy-tailed behavior is constrained by the mean."
    },
    {
      type: "mc",
      question: "**Chebyshev's inequality** states $P(|X - \\mu| \\geq t) \\leq \\frac{\\sigma^2}{t^2}$. For a sample mean $\\bar{X}_n$ of $n$ i.i.d. variables with variance $\\sigma^2$, what does Chebyshev give for $P(|\\bar{X}_n - \\mu| \\geq \\epsilon)$?",
      options: [
        "$\\frac{\\sigma^2}{\\epsilon^2}$ — independent of $n$, since the variance bound does not account for averaging effects across samples",
        "$\\frac{\\sigma^2}{n \\epsilon^2}$ — the bound tightens linearly with $n$, giving a $1/n$ polynomial convergence rate for the sample mean",
        "$\\frac{\\sigma}{\\sqrt{n} \\epsilon}$ — a square-root scaling that arises from applying the CLT approximation to the tail probability directly",
        "$e^{-n\\epsilon^2 / 2\\sigma^2}$ — an exponential bound that follows from the sub-Gaussian property of bounded sample means"
      ],
      correct: 1,
      explanation: "$\\text{Var}(\\bar{X}_n) = \\sigma^2/n$, so Chebyshev gives $P(|\\bar{X}_n - \\mu| \\geq \\epsilon) \\leq \\sigma^2 / (n\\epsilon^2)$. This is a **polynomial** (not exponential) bound in $n$. To get $P \\leq \\delta$, you need $n \\geq \\sigma^2 / (\\epsilon^2 \\delta)$ — which can be impractically large. This motivates sub-Gaussian bounds that give exponential concentration."
    },
    {
      type: "mc",
      question: "**Hoeffding's inequality** for i.i.d. bounded random variables $X_i \\in [a, b]$ states $P(|\\bar{X}_n - \\mu| \\geq t) \\leq 2\\exp\\left(-\\frac{2nt^2}{(b-a)^2}\\right)$. Compared to Chebyshev, this gives:",
      options: [
        "A weaker bound that uses stronger assumptions — the boundedness requirement restricts applicability without improving the convergence rate over Chebyshev",
        "An exponentially decreasing tail bound in $n$ (vs. polynomial for Chebyshev) — the price is requiring bounded variables rather than just finite variance",
        "The same asymptotic rate as Chebyshev but with a tighter multiplicative constant, providing marginal improvement for moderate sample sizes only",
        "A bound that does not depend on the range $[a, b]$ at all, since the exponential decay rate is determined entirely by the sample size and deviation"
      ],
      correct: 1,
      explanation: "Hoeffding gives $e^{-\\Theta(n)}$ decay vs. Chebyshev's $1/n$ decay. To achieve failure probability $\\delta$, Hoeffding needs $n = O(\\log(1/\\delta) / t^2)$ samples — logarithmic in $1/\\delta$ vs. Chebyshev's linear $O(1/\\delta)$. This exponential concentration is crucial in practice: it means evaluation metrics converge quickly and generalization bounds are meaningful."
    },
    {
      type: "mc",
      question: "A random variable $X$ is **sub-Gaussian** with parameter $\\sigma$ if $\\mathbb{E}[e^{t(X - \\mu)}] \\leq e^{\\sigma^2 t^2 / 2}$ for all $t$. Why is the sub-Gaussian property important in deep learning?",
      options: ["All neural network weights are sub-Gaussian after standard initialization schemes, which guarantees that forward-pass activations remain bounded throughout training", "It guarantees convergence of SGD to a global minimum in non-convex landscapes, provided the gradient noise satisfies the sub-Gaussian tail condition at each step", "It implies exponential tail concentration $P(|X - \\mu| \\geq t) \\leq 2e^{-t^2/(2\\sigma^2)}$, which applies to bounded variables, Gaussians, and their sums — covering most training quantities", "It means the distribution is exactly Gaussian in its first two moments, allowing closed-form computation of expected loss and gradient variance during optimization"],
      correct: 2,
      explanation: "Sub-Gaussianity is a tail condition: the tails decay at least as fast as a Gaussian with variance $\\sigma^2$. Bounded variables ($[a,b]$) are sub-Gaussian with $\\sigma = (b-a)/2$, and sums of sub-Gaussians are sub-Gaussian. This covers sample means, bounded losses, and many gradient estimators. The tail bound gives sharp concentration — the foundation for PAC-Bayes bounds, uniform convergence, and confidence intervals."
    },
    {
      type: "mc",
      question: "In SGD, the gradient estimate $g = \\nabla L(x_i; \\theta)$ for a random mini-batch is an unbiased estimator of $\\nabla L(\\theta)$. The **variance** of this estimate relates to concentration via:",
      options: ["Higher variance means faster convergence due to implicit exploration of the loss landscape, which helps SGD escape sharp local minima more effectively", "Variance only matters for the final few iterations of training when the model is near convergence and noise prevents settling into the exact optimum", "Variance is irrelevant to convergence because SGD's unbiasedness alone guarantees convergence regardless of second-moment properties of the gradient estimator", "Gradient variance $\\text{Var}(g)$ scales as $\\sigma^2_g / B$ for batch size $B$ — doubling the batch size halves the variance but also halves the parameter updates per epoch"],
      correct: 3,
      explanation: "The mini-batch gradient is a sample mean, so $\\text{Var}(\\bar{g}_B) = \\sigma^2_g / B$. This means larger batches give more concentrated (less noisy) gradient estimates. But the total computation is $B \\times \\text{updates}$, and doubling $B$ halves updates per epoch. The \"critical batch size\" is where further increases in $B$ no longer improve wall-clock convergence — below this, gradient noise is too high; above it, you're wasting compute on variance reduction that doesn't help."
    },
    {
      type: "mc",
      question: "**McDiarmid's inequality** generalizes Hoeffding to functions of independent variables: if changing any one input $x_i$ changes $f(x_1, \\ldots, x_n)$ by at most $c_i$, then $P(f - \\mathbb{E}[f] \\geq t) \\leq \\exp\\left(-\\frac{2t^2}{\\sum_i c_i^2}\\right)$. This is useful in ML for bounding:",
      options: ["The generalization gap — since train/test metrics are functions of i.i.d. data points, and changing one point has bounded effect, McDiarmid gives exponential concentration of empirical risk", "The training loss at each iteration, providing a deterministic upper bound on the per-step decrease that guarantees monotonic convergence of the optimization procedure", "The spectral norm of the weight matrix after each gradient update, which controls the Lipschitz constant of the network and thus its generalization capacity", "The number of epochs needed to reach a target accuracy, since the bounded-differences condition translates directly into a convergence rate for gradient descent"],
      correct: 0,
      explanation: "The empirical risk $\\hat{R} = \\frac{1}{n} \\sum_i L(f(x_i), y_i)$ is a function of $n$ i.i.d. data points. Changing one point changes $\\hat{R}$ by at most $c_i = M/n$ (if the loss is bounded by $M$). McDiarmid gives $P(|\\hat{R} - R| \\geq t) \\leq 2e^{-2nt^2/M^2}$. This is a key ingredient in VC theory and PAC learning bounds."
    },
    {
      type: "mc",
      question: "The **Chernoff bound** technique optimizes over the parameter in the moment generating function: $P(X \\geq a) \\leq \\min_{t > 0} e^{-ta} \\mathbb{E}[e^{tX}]$. This gives the **tightest exponential bound** because:",
      options: ["It only requires the mean of the distribution, making it the most broadly applicable bound that still achieves exponential decay in the tail probability", "It uses all moments of the distribution simultaneously (through the MGF), not just the mean or variance — the optimization over $t$ selects the tightest exponential rate", "It works for any distribution including heavy-tailed ones where the MGF may not exist, by truncating the moment generating function at a finite order", "It is always tighter than the exact probability because the exponential tilting introduces a bias that systematically underestimates the true tail mass"],
      correct: 1,
      explanation: "The Chernoff method applies Markov's inequality to $e^{tX}$ for any $t > 0$, then optimizes over $t$. Since $\\mathbb{E}[e^{tX}]$ encodes all moments (via Taylor expansion: $\\sum_k t^k \\mathbb{E}[X^k]/k!$), the optimization over $t$ extracts the best exponential bound the moment information can provide. For sub-Gaussian and sub-exponential variables, this gives sharp rates. It fails for truly heavy-tailed distributions where the MGF doesn't exist."
    },
    {
      type: "mc",
      question: "The **empirical Bernstein bound** uses the sample variance $\\hat{\\sigma}^2$ instead of the range $(b-a)^2$ to get a tighter bound when the data has low variance. In the context of evaluating an LLM, when would empirical Bernstein provide the greatest improvement over Hoeffding?",
      options: [
        "When the model gets close to 50% accuracy on binary questions, because the binomial variance $p(1-p)$ is maximized and the empirical Bernstein bound exploits this to tighten the interval",
        "When the model gets close to 100% or 0% accuracy, because the true variance $p(1-p)$ is much smaller than the worst-case $0.25$ that Hoeffding effectively assumes for $[0,1]$-bounded variables",
        "When the evaluation dataset is very small ($n < 50$), because empirical Bernstein corrects for the finite-sample bias that makes Hoeffding too conservative on small datasets",
        "When the questions have non-binary scores (e.g., partial credit on a 0-10 scale), because Hoeffding cannot handle non-binary bounded variables while empirical Bernstein can"
      ],
      correct: 1,
      explanation: "Hoeffding uses the range $(b-a)^2 = 1$ for $[0,1]$-bounded variables, implicitly assuming worst-case variance of $0.25$ (the variance of a Bernoulli with $p=0.5$). When the true accuracy is near 0% or 100%, the actual variance $p(1-p)$ is much smaller. The empirical Bernstein bound estimates the sample variance and uses it instead, giving a tighter bound precisely when the variance is low relative to the range. A model at 98% accuracy has variance $0.98 \\times 0.02 = 0.0196$, far below the $0.25$ Hoeffding assumes."
    },
    {
      type: "mc",
      question: "In SGD for LLM training, the gradient estimate from a mini-batch of size $B$ is a sum of i.i.d. per-example gradients. If the per-example gradient is sub-Gaussian with parameter $\\sigma_g$, how does the concentration of the mini-batch gradient estimate scale, and what does this imply for the critical batch size?",
      options: [
        "Concentration scales as $e^{-B^2 t^2 / (2\\sigma_g^2)}$ (quadratic in $B$), meaning tiny batch sizes are sufficient and the critical batch size is always 1 for sub-Gaussian gradients",
        "Concentration does not improve with $B$ for sub-Gaussian gradients because the sub-Gaussian parameter $\\sigma_g$ also increases with $B$ due to correlated training examples within each mini-batch",
        "Concentration scales as $e^{-Bt^2 / (2\\sigma_g^2)}$ (linear in $B$) — the critical batch size is the point where further increasing $B$ reduces gradient noise below the scale of the curvature, so additional variance reduction no longer speeds up convergence",
        "Concentration is independent of $B$ entirely because the learning rate is scaled proportionally to $B$ (linear scaling rule), perfectly canceling any variance reduction from larger batches"
      ],
      correct: 2,
      explanation: "The sample mean of $B$ i.i.d. sub-Gaussian($\\sigma_g$) variables is sub-Gaussian($\\sigma_g / \\sqrt{B}$), giving tail bound $e^{-Bt^2 / (2\\sigma_g^2)}$. This linear scaling in $B$ means doubling the batch halves the variance. The critical batch size $B_{\\text{crit}}$ is where gradient noise variance equals the \"signal\" from the expected gradient direction — beyond this, extra variance reduction doesn't help because the noise is already small relative to the curvature scale. McCandlish et al. (2018) showed $B_{\\text{crit}}$ scales with the loss value for LLM training."
    },
    {
      type: "mc",
      question: "When evaluating an LLM on a benchmark with $n = 1000$ binary-scored questions, your model gets 72% accuracy. Using Hoeffding's inequality, a 95% confidence interval ($\\delta = 0.05$) for the true accuracy is approximately:",
      options: [
        "$72\\% \\pm 0.1\\%$ — extremely precise, since 1000 samples with exponential concentration yield a negligibly small confidence radius",
        "$72\\% \\pm 2.7\\%$ — from $\\epsilon = \\sqrt{\\frac{\\log(2/\\delta)}{2n}} \\approx 0.027$ for bounded $[0,1]$ variables with Hoeffding's bound",
        "$72\\% \\pm 10\\%$ — because 1000 samples is insufficient for meaningful concentration when the true accuracy is far from 50\\%",
        "$72\\% \\pm 1.4\\%$ — using the normal approximation $1.96\\sqrt{p(1-p)/n}$ which gives a tighter interval than Hoeffding for binary outcomes"
      ],
      correct: 1,
      explanation: "Hoeffding gives $P(|\\hat{p} - p| \\geq \\epsilon) \\leq 2e^{-2n\\epsilon^2}$. Setting $2e^{-2(1000)\\epsilon^2} = 0.05$ gives $\\epsilon = \\sqrt{\\log(40)/2000} \\approx 0.043$, so about $\\pm 4.3\\%$. The tighter Clopper-Pearson or normal approximation ($\\pm 1.96\\sqrt{p(1-p)/n} \\approx \\pm 2.8\\%$) gives a narrower interval. The point: even 1000 samples leave meaningful uncertainty in benchmark scores, which is why comparing models that differ by 1-2% is statistically dubious."
    }
  ]
};
