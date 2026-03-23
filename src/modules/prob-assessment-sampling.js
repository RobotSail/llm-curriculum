// Assessment: Sampling Methods & Monte Carlo
// Section 0.2: Diagnostic test — importance sampling, MCMC, rejection sampling
// Pure assessment to gauge depth of understanding

export const samplingAssessment = {
  id: "0.2-assess-sampling",
  sectionId: "0.2",
  title: "Assessment: Sampling Methods & Monte Carlo",
  difficulty: "medium",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Sampling Methods & Monte Carlo",
      content: "This is a **diagnostic assessment** covering sampling and Monte Carlo methods.\n\nThese methods appear everywhere in LLMs: token sampling strategies (top-k, nucleus), importance sampling in off-policy RL, MCMC for Bayesian approaches, and Monte Carlo estimation of intractable expectations.\n\nIf you score below 70%, review these foundations — they're essential for understanding generation and training."
    },
    {
      type: "mc",
      question: "**Monte Carlo estimation** approximates $\\mathbb{E}_P[f(x)] \\approx \\frac{1}{N} \\sum_{i=1}^N f(x_i)$ where $x_i \\sim P$. The variance of this estimator decreases as:",
      options: ["$O(1/N)$ — halving the variance requires doubling the number of samples, regardless of the dimensionality of the space", "$O(1/N^2)$ — the variance decreases quadratically, making larger sample sizes dramatically more effective per sample", "$O(1/\\sqrt{N})$ — the standard error and variance both decrease at the same square-root rate with more samples", "$O(e^{-N})$ — the variance decreases exponentially fast, meaning a few hundred samples suffice for arbitrary precision"],
      correct: 0,
      explanation: "By the CLT, the variance of the sample mean is $\\text{Var}[f(X)] / N$, so it decreases as $O(1/N)$. The standard error decreases as $O(1/\\sqrt{N})$. Crucially, this rate is **dimension-independent** — this is why Monte Carlo methods scale to high dimensions where grid-based methods fail exponentially. This $O(1/N)$ rate drives many design choices in how many samples to use."
    },
    {
      type: "mc",
      question: "**Importance sampling** estimates $\\mathbb{E}_P[f(x)]$ using samples from a different distribution $Q$ via $\\mathbb{E}_P[f(x)] = \\mathbb{E}_Q\\left[f(x) \\frac{P(x)}{Q(x)}\\right]$. The ratio $w(x) = P(x)/Q(x)$ is called the importance weight. When can this go badly wrong?",
      options: ["When $P$ and $Q$ are identical distributions, because the importance weights degenerate to a constant and provide no variance reduction benefit", "When $f(x)$ is a constant function, because the importance weights amplify noise without contributing any useful signal to the estimate", "When $Q$ has lighter tails than $P$ — the weights $P(x)/Q(x)$ can become enormous in the tails, causing high variance and unstable estimates", "When $N$ is very large, because accumulated floating-point errors in the importance weight computation eventually dominate the estimate"],
      correct: 2,
      explanation: "If $Q$ has lighter tails than $P$, then in regions where $P(x) \\gg Q(x)$, the weight $P(x)/Q(x)$ explodes. A few samples may dominate the entire estimate, giving high variance. This is the \"weight degeneracy\" problem. In off-policy RL, this manifests when the learned policy differs significantly from the behavior policy — importance weights become degenerate, which is why PPO clips them."
    },
    {
      type: "mc",
      question: "In **PPO** (Proximal Policy Optimization), the clipped surrogate objective involves $\\min\\left(r_t(\\theta) A_t, \\, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) A_t\\right)$ where $r_t = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$ is an importance weight. The clipping addresses:",
      options: [
        "Numerical overflow in the softmax layer that occurs when logits become very large during policy optimization steps",
        "The high-variance problem of importance sampling — by preventing the policy ratio from deviating too far, it bounds the variance of the gradient estimate",
        "The mode-seeking behavior of reverse KL divergence, which would otherwise cause the policy to collapse onto a single high-reward action",
        "The computational cost of computing the importance ratio, which becomes prohibitively expensive when policies diverge significantly"
      ],
      correct: 1,
      explanation: "Without clipping, if $\\pi_\\theta$ diverges far from $\\pi_{\\theta_{\\text{old}}}$, the importance ratio $r_t$ can become very large, causing destructively large policy updates. Clipping $r_t$ to $[1-\\epsilon, 1+\\epsilon]$ bounds the effective step size, keeping the new policy close to the old. This is a practical solution to the importance sampling variance problem, complementary to the KL penalty approach in earlier TRPO."
    },
    {
      type: "mc",
      question: "**Top-k sampling** from an LLM restricts sampling to the $k$ highest-probability tokens and redistributes probability mass. From a sampling theory perspective, this is equivalent to:",
      options: ["Importance sampling with a uniform proposal distribution over the entire vocabulary, reweighted by the model's token probabilities", "Gibbs sampling over the vocabulary that iteratively updates each token position conditioned on all other positions in the sequence", "Rejection sampling where tokens drawn from the full distribution are rejected if they fall outside the top-$k$ probability set", "Sampling from a truncated version of the original distribution — zeroing out low-probability tokens and renormalizing the top-$k$ probabilities"],
      correct: 3,
      explanation: "Top-k sets $P(w) = 0$ for all tokens outside the top $k$, then renormalizes: $P'(w) = P(w) / \\sum_{w' \\in \\text{top-k}} P(w')$ for $w \\in \\text{top-k}$. This is distribution truncation. Nucleus (top-p) sampling is similar but adaptive — it truncates at the smallest set whose cumulative probability exceeds $p$, making it more robust to varying entropy across positions."
    },
    {
      type: "mc",
      question: "**MCMC** (Markov Chain Monte Carlo) methods construct a Markov chain whose stationary distribution is the target $P$. The **Metropolis-Hastings** acceptance probability $\\alpha = \\min\\left(1, \\frac{P(x') Q(x|x')}{P(x) Q(x'|x)}\\right)$ ensures:",
      options: ["**Detailed balance**: the chain is reversible with respect to $P$, guaranteeing that $P$ is the unique stationary distribution of the chain", "That the Markov chain converges to the target distribution within a finite and bounded number of transition steps", "That all states in the state space are visited with exactly equal frequency, ensuring uniform coverage of the support", "That the proposal distribution $Q$ converges to match the target $P$ exactly as the number of accepted samples increases"],
      correct: 0,
      explanation: "The acceptance ratio enforces detailed balance: $P(x) T(x'|x) = P(x') T(x|x')$ where $T$ is the transition kernel. Detailed balance implies $P$ is stationary (but is stronger — it also implies reversibility). Note that we only need the ratio $P(x')/P(x)$, so we don't need to know the normalizing constant of $P$ — this is why MCMC works for Bayesian posteriors where the evidence is intractable."
    },
    {
      type: "mc",
      question: "The **Gumbel-max trick** provides exact samples from a categorical distribution: $\\arg\\max_i (\\log p_i + G_i)$ where $G_i \\sim \\text{Gumbel}(0, 1)$ are i.i.d. The **Gumbel-Softmax** relaxation replaces argmax with softmax to make this:",
      options: ["More numerically stable by avoiding the overflow issues that arise when computing argmax over large logit vectors with extreme values", "Faster to compute by replacing the sequential argmax scan with a parallelizable softmax operation across all vocabulary entries", "Differentiable — enabling gradient-based optimization through discrete sampling operations using a continuous relaxation with temperature $\\tau$", "Exact rather than approximate, eliminating the sampling noise inherent in the stochastic Gumbel perturbations of the original method"],
      correct: 2,
      explanation: "The argmax is non-differentiable, blocking backpropagation. Gumbel-Softmax replaces it with $\\text{softmax}((\\log p_i + G_i)/\\tau)$, which is differentiable and approaches a one-hot vector as $\\tau \\to 0$. This enables end-to-end training of models with discrete choices (e.g., hard attention, discrete latent variables). The temperature $\\tau$ controls the bias-variance trade-off: low $\\tau$ is more accurate but higher variance."
    },
    {
      type: "mc",
      question: "When estimating $\\mathbb{E}_P[f(x)]$ with MCMC samples, the samples are **correlated** (not i.i.d.). The **effective sample size** (ESS) accounts for this. If you draw 10,000 MCMC samples but the ESS is 500, this means:",
      options: [
        "Exactly 9,500 of the samples should be discarded as burn-in because the chain had not yet reached its stationary distribution",
        "The correlated chain provides the same statistical power as 500 independent samples — autocorrelation has reduced the information content by 20×",
        "The chain has failed to converge to the target distribution and the samples should not be used for any downstream estimation",
        "You need to draw exactly 500 additional samples from an independent chain to reach the minimum threshold for reliable estimates"
      ],
      correct: 1,
      explanation: "ESS = $N / (1 + 2\\sum_{k=1}^\\infty \\rho_k)$ where $\\rho_k$ is the autocorrelation at lag $k$. High autocorrelation (slow mixing) drastically reduces ESS. An ESS of 500 from 10,000 samples means consecutive samples are highly correlated — the chain is \"exploring slowly.\" This is why mixing time and thinning (keeping every $k$-th sample) matter. In practice, ESS guides how long to run the chain."
    },
    {
      type: "mc",
      question: "**Rejection sampling** draws $x \\sim Q$, then accepts with probability $\\frac{P(x)}{M \\cdot Q(x)}$ where $M \\geq \\sup_x \\frac{P(x)}{Q(x)}$. In high dimensions, this method:",
      options: ["Becomes more efficient because the law of large numbers ensures better concentration of the acceptance probability around its mean", "Is naturally replaced by importance sampling, which suffers from the same exponential inefficiency due to weight degeneracy in high dimensions", "Works with identical efficiency as in low dimensions because the acceptance probability depends only on the density ratio, not dimensionality", "Becomes exponentially inefficient — the acceptance rate drops exponentially with dimension because $M$ must be exponentially large to bound $P/Q$ everywhere"],
      correct: 3,
      explanation: "In $d$ dimensions, the acceptance rate $1/M$ typically decays as $e^{-\\Theta(d)}$ because the proposal $Q$ must cover the tails of $P$ in all dimensions simultaneously. This is the \"curse of dimensionality\" for rejection sampling. MCMC avoids this by not requiring a global bound — it only needs local moves. This is why rejection sampling is practical only in low dimensions, while MCMC and variational methods scale to millions of parameters."
    },
    {
      type: "mc",
      question: "**Speculative decoding** uses a small draft model to generate $k$ candidate tokens, then verifies them against the large target model in parallel. The verification uses a form of:",
      options: ["Rejection sampling — each draft token is accepted with probability $\\min(1, P_{\\text{target}}(w_t) / P_{\\text{draft}}(w_t))$, and on rejection we resample from a corrected distribution to match the target exactly", "Beam search with pruning, where the draft model proposes candidate beams and the target model scores and selects among the top-scoring continuations", "Importance sampling with the draft model as the proposal distribution, reweighting each candidate token by the ratio of target to draft probabilities", "Gibbs sampling that alternates between the draft and target model distributions, iteratively refining each token position conditioned on its neighbors"],
      correct: 0,
      explanation: "Speculative decoding is mathematically exact: the output distribution matches what the target model would produce with standard autoregressive sampling. The trick is that acceptance probability $\\min(1, P_{\\text{target}}/P_{\\text{draft}})$ is high when the draft model is good, so most tokens are accepted without running the large model sequentially. On rejection, sampling from the residual $(P_{\\text{target}} - P_{\\text{draft}})_+$ corrects for the draft model's errors."
    }
  ]
};
