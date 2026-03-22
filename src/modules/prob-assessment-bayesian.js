// Assessment: Bayesian Inference & Variational Methods
// Section 0.2: Diagnostic test — priors, posteriors, ELBO, variational inference
// Pure assessment to gauge depth of understanding

export const bayesianAssessment = {
  id: "0.2-assess-bayesian",
  sectionId: "0.2",
  title: "Assessment: Bayesian Inference & Variational Methods",
  difficulty: "hard",
  estimatedMinutes: 14,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Bayesian Inference & Variational Methods",
      content: "This is a **diagnostic assessment** covering Bayesian reasoning, posterior inference, and variational methods.\n\nVariational inference underpins VAEs, the theoretical derivation of DPO, the ELBO in latent variable models, and the Bayesian perspective on regularization. These concepts bridge \"fitting a model\" to \"reasoning about uncertainty.\"\n\nIf you score below 70%, invest time in these fundamentals — they're harder to patch later."
    },
    {
      type: "mc",
      question: "Bayes' theorem states $P(\\theta \\mid \\mathcal{D}) = \\frac{P(\\mathcal{D} \\mid \\theta) P(\\theta)}{P(\\mathcal{D})}$. The denominator $P(\\mathcal{D}) = \\int P(\\mathcal{D} \\mid \\theta) P(\\theta) d\\theta$ is called the **evidence** or **marginal likelihood**. Why is it typically intractable?",
      options: [
        "Because $P(\\mathcal{D} \\mid \\theta)$ is unknown",
        "Because the integral is over the entire parameter space, which is high-dimensional for neural networks — making it impossible to evaluate analytically or numerically",
        "Because the prior $P(\\theta)$ is always improper",
        "Because the data $\\mathcal{D}$ is too large"
      ],
      correct: 1,
      explanation: "For a neural network with millions of parameters, $P(\\mathcal{D}) = \\int P(\\mathcal{D} \\mid \\theta) P(\\theta) d\\theta$ requires integrating over a million-dimensional space. This integral has no closed form (the likelihood is a complex nonlinear function of $\\theta$) and is too high-dimensional for numerical quadrature. This intractability motivates both MCMC (sampling) and variational inference (optimization) as approximation strategies."
    },
    {
      type: "mc",
      question: "In variational inference, we approximate the intractable posterior $P(\\theta \\mid \\mathcal{D})$ with a tractable distribution $q(\\theta)$ by minimizing $\\text{KL}(q(\\theta) \\| P(\\theta \\mid \\mathcal{D}))$. This is **reverse KL**. What consequence does this have?",
      options: [
        "$q$ will cover all modes of the posterior (mode-covering)",
        "$q$ will tend to concentrate on a single mode of the posterior (mode-seeking), potentially underestimating uncertainty",
        "$q$ will exactly match the posterior",
        "$q$ will be uniform over the parameter space"
      ],
      correct: 1,
      explanation: "Reverse KL is mode-seeking: $q$ will lock onto one mode of the posterior and fit it tightly, ignoring other modes. This means variational inference systematically **underestimates** posterior uncertainty. For multimodal posteriors, $q$ may miss important modes. This is the fundamental trade-off of variational inference vs. MCMC (which can in principle explore all modes)."
    },
    {
      type: "mc",
      question: "The **ELBO** (Evidence Lower Bound) is $\\mathcal{L}(q) = \\mathbb{E}_q[\\log P(\\mathcal{D} \\mid \\theta)] - \\text{KL}(q(\\theta) \\| P(\\theta))$. Why is it called a \"lower bound\"?",
      options: [
        "Because it lower-bounds the KL divergence",
        "Because $\\log P(\\mathcal{D}) = \\mathcal{L}(q) + \\text{KL}(q \\| P(\\theta \\mid \\mathcal{D})) \\geq \\mathcal{L}(q)$, since KL $\\geq 0$ — so the ELBO lower-bounds the log-evidence",
        "Because the expected log-likelihood is always negative",
        "Because the prior term is always a penalty"
      ],
      correct: 1,
      explanation: "We can decompose: $\\log P(\\mathcal{D}) = \\mathcal{L}(q) + \\text{KL}(q(\\theta) \\| P(\\theta \\mid \\mathcal{D}))$. Since KL is always $\\geq 0$, we get $\\log P(\\mathcal{D}) \\geq \\mathcal{L}(q)$. Maximizing the ELBO simultaneously (1) tightens the bound on the evidence, and (2) minimizes $\\text{KL}(q \\| P(\\theta \\mid \\mathcal{D}))$, making $q$ a better posterior approximation. The ELBO equals the evidence when $q$ equals the true posterior."
    },
    {
      type: "mc",
      question: "In the VAE (Variational Autoencoder), the ELBO for a single datapoint is $\\mathcal{L}(x) = \\mathbb{E}_{q(z|x)}[\\log p(x \\mid z)] - \\text{KL}(q(z \\mid x) \\| p(z))$. The two terms correspond to:",
      options: [
        "Accuracy and speed",
        "**Reconstruction** (how well can we decode $z$ back to $x$) and **regularization** (how close is the encoder's posterior to the prior)",
        "Training loss and validation loss",
        "Bias and variance of the model"
      ],
      correct: 1,
      explanation: "The first term $\\mathbb{E}_{q(z|x)}[\\log p(x|z)]$ encourages accurate reconstruction — $z$ must contain enough information to reproduce $x$. The second term $\\text{KL}(q(z|x) \\| p(z))$ regularizes the latent space, pushing $q(z|x)$ toward the prior $p(z) = \\mathcal{N}(0, I)$. The tension between these creates a learned compression: encode enough to reconstruct, but stay close to a simple prior."
    },
    {
      type: "mc",
      question: "The **reparameterization trick** in VAEs writes $z = \\mu(x) + \\sigma(x) \\odot \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, I)$. Why is this needed?",
      options: [
        "To make the latent space continuous",
        "To move the stochasticity outside the computational graph, enabling backpropagation through the sampling operation via the deterministic path $\\mu(x)$ and $\\sigma(x)$",
        "To ensure $z$ follows a Gaussian distribution",
        "To reduce the dimensionality of $z$"
      ],
      correct: 1,
      explanation: "You can't backpropagate through a random sampling operation $z \\sim q(z|x)$. The reparameterization trick rewrites this as a deterministic function of learned parameters ($\\mu, \\sigma$) plus fixed noise ($\\epsilon$). Now gradients flow through $\\mu$ and $\\sigma$ to the encoder. This same idea (making gradients flow through stochastic operations) appears in policy gradient methods and Gumbel-softmax for discrete distributions."
    },
    {
      type: "mc",
      question: "**Mean-field variational inference** assumes the variational distribution factorizes: $q(\\theta) = \\prod_i q_i(\\theta_i)$. What does this independence assumption sacrifice?",
      options: [
        "The ability to model the mean of each parameter",
        "All correlations between parameters — it cannot capture posterior dependencies, often leading to underestimated uncertainty",
        "The ability to compute the ELBO",
        "Convergence guarantees"
      ],
      correct: 1,
      explanation: "Mean-field assumes all parameters are independent in the posterior, which is almost never true. In neural networks, weight correlations are rampant (e.g., a large weight in one layer affects optimal weights in the next). This means mean-field posteriors are overconfident — they give tight marginals that ignore how parameters co-vary. Despite this limitation, mean-field VI is widely used because it's tractable."
    },
    {
      type: "mc",
      question: "The derivation of **DPO** (Direct Preference Optimization) starts from the KL-constrained RLHF objective and finds its closed-form optimal policy. The derivation critically uses:",
      options: [
        "The Lagrangian dual of the KL constraint, yielding $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$ — a Gibbs/Boltzmann distribution where the partition function acts as a value function",
        "Monte Carlo sampling of the reward function",
        "The gradient of the JS divergence",
        "The central limit theorem applied to token probabilities"
      ],
      correct: 0,
      explanation: "The RLHF objective $\\max_\\pi \\mathbb{E}[r] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ has the closed-form solution $\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$ where $Z(x) = \\sum_y \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. This is derived by taking the functional derivative and solving. The key insight is rearranging to express $r$ in terms of $\\pi^*/\\pi_{\\text{ref}}$, then substituting into the Bradley-Terry preference model to eliminate the reward entirely."
    },
    {
      type: "mc",
      question: "**Bayesian model comparison** uses the evidence $P(\\mathcal{D} \\mid M) = \\int P(\\mathcal{D} \\mid \\theta, M) P(\\theta \\mid M) d\\theta$ to compare models $M_1$ and $M_2$ via the Bayes factor $\\frac{P(\\mathcal{D} \\mid M_1)}{P(\\mathcal{D} \\mid M_2)}$. How does this naturally implement Occam's razor?",
      options: [
        "It penalizes models with more parameters through an explicit penalty term",
        "Complex models spread their prior mass over many possible datasets, so they assign lower prior probability to any specific dataset — the evidence automatically balances fit and complexity",
        "It uses cross-validation internally",
        "It selects the model with the highest likelihood on the training data"
      ],
      correct: 1,
      explanation: "A complex model can fit many possible datasets but must spread its prior predictive probability $P(\\mathcal{D} \\mid M) = \\int P(\\mathcal{D} \\mid \\theta) P(\\theta) d\\theta$ thinly over all of them. A simple model concentrates its probability on fewer datasets but assigns higher probability to those. The evidence thus naturally penalizes unnecessary complexity — no regularization parameter needed. This is called the \"Bayesian Occam's razor.\""
    },
    {
      type: "mc",
      question: "**Amortized inference** (as in VAEs) trains an encoder $q_\\phi(z \\mid x)$ to approximate the posterior for any $x$, rather than optimizing $q$ separately for each $x$. The key advantage is:",
      options: [
        "It guarantees exact posterior inference",
        "After training, inference for a new $x$ is a single forward pass through the encoder — no iterative optimization needed. This trades per-datapoint optimality for speed",
        "It eliminates the need for the ELBO",
        "It makes the posterior always Gaussian"
      ],
      correct: 1,
      explanation: "Traditional VI optimizes $q(z)$ separately per datapoint (expensive). Amortized inference learns a single function $q_\\phi(z|x)$ (the encoder) that maps any $x$ to an approximate posterior in one forward pass. The trade-off: the \"amortization gap\" — the shared encoder may not be as good as per-datapoint optimization. But the speedup (one pass vs. iterative optimization) makes it practical for large datasets."
    }
  ]
};
