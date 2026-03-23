// Assessment: Exponential Family & Maximum Likelihood Estimation
// Section 0.2: Diagnostic test — exponential families, sufficient statistics, MLE
// Pure assessment to gauge whether you need deeper study

export const exponentialFamilyAssessment = {
  id: "0.2-assess-exp-family",
  sectionId: "0.2",
  title: "Assessment: Exponential Families & MLE",
  difficulty: "medium",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Exponential Families & Maximum Likelihood",
      content: "This is a **diagnostic assessment** covering exponential family distributions, sufficient statistics, and maximum likelihood estimation.\n\nThese concepts underpin why cross-entropy loss works, how softmax relates to log-linear models, and why certain parameterizations are natural for gradient-based optimization.\n\nIf you score below 70%, consider reviewing these topics before proceeding."
    },
    {
      type: "mc",
      question: "An exponential family distribution has the form $p(x \\mid \\theta) = h(x) \\exp(\\eta(\\theta)^\\top T(x) - A(\\theta))$. What role does $A(\\theta)$ (the log-partition function) play?",
      options: ["It defines the prior distribution over $\\theta$ in the Bayesian conjugate update, controlling how much the posterior is influenced by the likelihood", "It computes the mode of the distribution by finding the value of $x$ that maximizes the unnormalized density $h(x)\\exp(\\eta^\\top T(x))$", "It is a normalization constant ensuring the distribution sums or integrates to 1, and its derivatives yield the moments of the distribution", "It measures the divergence between the model distribution and the true data-generating distribution, acting as an implicit regularizer"],
      correct: 2,
      explanation: "$A(\\theta) = \\log \\int h(x) \\exp(\\eta(\\theta)^\\top T(x))\\, dx$ is the log-partition function. It ensures normalization. Crucially, its derivatives give the moments: $\\nabla_\\eta A = \\mathbb{E}[T(x)]$ and $\\nabla^2_\\eta A = \\text{Cov}[T(x)]$. This means the log-partition function encodes everything about the distribution's moments."
    },
    {
      type: "mc",
      question: "The softmax output layer of an LLM defines a **categorical distribution** over tokens. In exponential family form, the natural parameters $\\eta_i$ correspond to:",
      options: ["The token embeddings produced by the final transformer layer, which encode semantic meaning before being converted to a probability distribution", "The attention weights computed across all heads in the last layer, which determine how much each token in the context influences the next-token prediction", "The softmax probabilities themselves, since the natural parameterization of the categorical distribution is in terms of the normalized output values", "The logits (pre-softmax scores), since the categorical distribution in exponential form is $P(w=i) = \\exp(\\eta_i - \\log\\sum_j \\exp(\\eta_j))$"],
      correct: 3,
      explanation: "The categorical distribution in exponential family form is $P(w = i) = \\exp(\\eta_i - A(\\eta))$ where $A(\\eta) = \\log \\sum_j \\exp(\\eta_j)$ (the log-sum-exp). The logits **are** the natural parameters. This is why working in logit space (rather than probability space) is natural for optimization — gradients in the natural parameterization have nice properties."
    },
    {
      type: "mc",
      question: "A **sufficient statistic** $T(x)$ for parameter $\\theta$ means that $T(x)$ captures all information about $\\theta$ in the data. For the Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$, the sufficient statistics of a sample $x_1, \\dots, x_n$ are:",
      options: [
        "The sample median and range, which together capture the location and spread of the data under the assumption of symmetry",
        "$\\sum_i x_i$ and $\\sum_i x_i^2$ (or equivalently, sample mean and sample variance) — these capture all information about $(\\mu, \\sigma^2)$",
        "The minimum and maximum values, which define the convex hull of the data and bound all possible parameter estimates",
        "All individual data points with no compression possible, because the Gaussian likelihood depends on every observation individually"
      ],
      correct: 1,
      explanation: "For the Gaussian, $T(X) = (\\sum x_i, \\sum x_i^2)$ are sufficient for $(\\mu, \\sigma^2)$. This means you can throw away the raw data and keep only these two numbers without losing any information about the parameters. For exponential families in general, the sufficient statistics are exactly the $T(x)$ appearing in the exponential family form."
    },
    {
      type: "mc",
      question: "The MLE for an exponential family distribution satisfies the **moment matching** condition. What does this mean?",
      options: [
        "The MLE parameters are the sample moments themselves, so the estimated natural parameters equal the empirical mean and variance directly",
        "The model's expected sufficient statistics equal the empirical sufficient statistics: $\\mathbb{E}_{\\hat{\\theta}}[T(x)] = \\frac{1}{n}\\sum_i T(x_i)$",
        "All moments of the model distribution must be finite for the MLE to exist, which restricts the family to light-tailed distributions only",
        "The distribution must have the same number of moments as free parameters, creating a square system of equations that uniquely determines the MLE"
      ],
      correct: 1,
      explanation: "Setting the gradient of the log-likelihood to zero gives $\\nabla_\\eta A(\\hat{\\eta}) = \\frac{1}{n} \\sum_i T(x_i)$. Since $\\nabla_\\eta A = \\mathbb{E}[T(x)]$, the MLE solution matches model moments to empirical moments. This is why cross-entropy training of a language model matches the model's predicted token frequencies to the training data's token frequencies."
    },
    {
      type: "mc",
      question: "Why is the **negative log-likelihood** loss function convex in the natural parameters $\\eta$ for exponential family models?",
      options: [
        "Because neural networks are convex models when parameterized in terms of their final-layer logits, making the entire optimization landscape globally convex",
        "Because $A(\\eta)$ is convex (its Hessian $\\nabla^2 A = \\text{Cov}[T(x)]$ is PSD), and the NLL is $A(\\eta) - \\eta^\\top T(x)$ which is convex in $\\eta$",
        "Because the data is always well-conditioned when drawn from an exponential family, ensuring the Hessian of the loss remains positive definite everywhere",
        "Convexity is not guaranteed in general — it depends on the specific dataset and whether the sufficient statistics span the natural parameter space"
      ],
      correct: 1,
      explanation: "The NLL per sample is $-\\log p(x \\mid \\eta) = A(\\eta) - \\eta^\\top T(x) + \\text{const}$. The Hessian w.r.t. $\\eta$ is $\\nabla^2 A(\\eta) = \\text{Cov}[T(x)] \\succeq 0$, which is PSD. So the NLL is convex in $\\eta$. Note: for neural networks, $\\eta$ is a nonlinear function of the weights, so the overall loss is non-convex in the weights — but the final layer (logits → loss) is convex."
    },
    {
      type: "mc",
      question: "The **Fisher information matrix** $\\mathcal{I}(\\theta) = \\mathbb{E}\\left[\\nabla \\log p(x \\mid \\theta) \\nabla \\log p(x \\mid \\theta)^\\top\\right]$ plays what role in MLE?",
      options: ["It gives the asymptotic covariance of the MLE: $\\hat{\\theta}_{\\text{MLE}} \\sim \\mathcal{N}(\\theta, \\mathcal{I}(\\theta)^{-1}/n)$ as $n \\to \\infty$, achieving the Cramér-Rao lower bound", "It determines the optimal learning rate schedule for SGD by measuring the local curvature of the loss surface at each point in parameter space", "It measures the divergence between the model's predicted distribution and the empirical data distribution, serving as an alternative to cross-entropy loss", "It equals the Hessian of the log-prior in Bayesian inference, connecting the Fisher matrix to the curvature of the regularization term in MAP estimation"],
      correct: 0,
      explanation: "The Cramér-Rao bound says no unbiased estimator can have variance lower than $\\mathcal{I}(\\theta)^{-1}/n$, and the MLE achieves this bound asymptotically. The Fisher matrix also defines the **natural gradient** $\\mathcal{I}^{-1} \\nabla \\ell$ — a direction that accounts for the geometry of the parameter space. This connects to Adam (which approximates diagonal Fisher) and to natural gradient methods in RL."
    },
    {
      type: "mc",
      question: "In practice, we train LLMs by minimizing cross-entropy on finite data. MLE is known to **overfit** without regularization. From a Bayesian perspective, L2 regularization ($\\lambda \\|\\theta\\|^2$) corresponds to:",
      options: ["A uniform prior over $\\theta$ that assigns equal probability to all parameter values, making the regularized estimate equivalent to constrained maximum likelihood", "A Laplace prior over $\\theta$ that places a sharp peak at zero, promoting sparsity by driving small weights exactly to zero during optimization", "A Gaussian prior $\\theta \\sim \\mathcal{N}(0, \\frac{1}{2\\lambda} I)$ — making the regularized loss equivalent to a MAP (maximum a posteriori) estimate under that prior", "A minimization of the Fisher information matrix trace, which reduces the model's sensitivity to small perturbations in the input data distribution"],
      correct: 2,
      explanation: "Adding $\\lambda \\|\\theta\\|^2$ to the NLL gives $-\\log p(x \\mid \\theta) + \\lambda \\|\\theta\\|^2 = -\\log p(x \\mid \\theta) - \\log p(\\theta) + \\text{const}$ where $p(\\theta) \\propto \\exp(-\\lambda \\|\\theta\\|^2)$ is a Gaussian. This makes the solution a **MAP** (maximum a posteriori) estimate rather than pure MLE. L1 regularization corresponds to a Laplace prior and promotes sparsity."
    },
    {
      type: "mc",
      question: "When computing the MLE for a mixture model $p(x) = \\sum_k \\pi_k p_k(x \\mid \\theta_k)$, the log-likelihood has a log-sum that makes direct optimization hard. What standard algorithm addresses this?",
      options: ["Gradient descent on the log-likelihood directly, using automatic differentiation to handle the log-sum-exp and its derivatives without any special treatment", "Grid search over all parameter values in a discretized space, evaluating the likelihood at each grid point to find the global maximum by exhaustive enumeration", "Gibbs sampling from the joint posterior over parameters and latent assignments, producing a Markov chain whose stationary distribution is the exact posterior", "The EM (Expectation-Maximization) algorithm, which alternates between computing posterior cluster assignments (E-step) and updating parameters to maximize expected complete-data log-likelihood (M-step)"],
      correct: 3,
      explanation: "EM handles the intractable log-sum by introducing latent variables $z$ (cluster assignments). The E-step computes $q(z) = P(z \\mid x, \\theta^{\\text{old}})$, and the M-step maximizes $\\mathbb{E}_q[\\log p(x, z \\mid \\theta)]$. Each iteration is guaranteed to increase the log-likelihood. EM is a special case of variational inference where the E-step is exact."
    }
  ]
};
