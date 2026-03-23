// Assessment: Probability Foundations
// Section 0.2: Diagnostic test — Bayes' theorem, conditional probability, distributions, expectations
// Pure assessment (no info steps) to gauge whether you need to study these topics

export const probabilityFoundationsAssessment = {
  id: "0.2-assess-prob-foundations",
  sectionId: "0.2",
  title: "Assessment: Probability Foundations",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Probability Foundations",
      content: "This is a **diagnostic assessment** — there are no teaching steps. Answer each question to gauge whether you need to study probability foundations in more depth.\n\nTopics covered: conditional probability, Bayes' theorem, common distributions, expectations, and the law of total probability.\n\nIf you score below 70%, consider reviewing these fundamentals before proceeding."
    },
    {
      type: "mc",
      question: "A language model assigns probability $P(w_t \\mid w_{<t})$ to each token. By the chain rule of probability, the joint probability of a sequence $w_1, \\dots, w_T$ is:",
      options: ["$\\sum_{t=1}^{T} P(w_t \\mid w_{<t})$", "$\\frac{1}{T} \\sum_{t=1}^{T} P(w_t \\mid w_{<t})$", "$\\prod_{t=1}^{T} P(w_t \\mid w_{<t})$", "$\\max_{t} P(w_t \\mid w_{<t})$"],
      correct: 2,
      explanation: "The chain rule of probability says $P(w_1, \\dots, w_T) = \\prod_{t=1}^{T} P(w_t \\mid w_1, \\dots, w_{t-1})$. This is the foundational factorization that autoregressive language models exploit — they model each conditional factor in the product."
    },
    {
      type: "mc",
      question: "You have a classifier with 95% true positive rate ($P(\\text{pos} \\mid \\text{spam}) = 0.95$) and 3% false positive rate ($P(\\text{pos} \\mid \\text{ham}) = 0.03$). If only 1% of emails are spam ($P(\\text{spam}) = 0.01$), what is $P(\\text{spam} \\mid \\text{pos})$ approximately?",
      options: ["About 95% — the high true positive rate directly translates into high posterior confidence about spam classification", "About 50% — the competing evidence from prior and likelihood roughly cancels out, making it effectively a coin flip", "About 1% — the extremely low base rate completely overwhelms any discriminative signal from the classifier's likelihood", "About 24% — the low base rate substantially dilutes the classifier's signal despite the high true positive rate"],
      correct: 3,
      explanation: "By Bayes' theorem: $P(\\text{spam} \\mid \\text{pos}) = \\frac{0.95 \\times 0.01}{0.95 \\times 0.01 + 0.03 \\times 0.99} = \\frac{0.0095}{0.0095 + 0.0297} \\approx 0.242$. This is the base rate fallacy — even with a good classifier, a low prior dramatically reduces the posterior. This is critical intuition for understanding how priors interact with likelihoods."
    },
    {
      type: "mc",
      question: "Which property distinguishes the **Gaussian distribution** from other distributions, making it appear so frequently in machine learning?",
      options: [
        "It is the only continuous distribution that has a well-defined and finite variance parameter",
        "Among all distributions with a given mean and variance, it maximizes entropy",
        "It is the only continuous distribution that belongs to the exponential family of distributions",
        "It is the only distribution that remains closed under the operation of multiplication"
      ],
      correct: 1,
      explanation: "The Gaussian is the **maximum entropy distribution** for a given mean and variance. This means it makes the fewest assumptions beyond those two constraints. This is why Gaussian assumptions appear everywhere — they're the most \"uninformative\" choice given first and second moments. The central limit theorem reinforces this: sums of many independent variables converge to Gaussian regardless of individual distributions."
    },
    {
      type: "mc",
      question: "In a language model's output layer, the **softmax** function converts logits $z_i$ to probabilities. Which distribution does this define over the vocabulary?",
      options: ["A multinomial (categorical) distribution where one token is sampled from the vocabulary according to the computed probabilities", "A Bernoulli distribution representing a binary choice between the most likely token and all remaining alternatives", "A Poisson distribution modeling the expected count of each token type appearing in the generated sequence", "A Dirichlet distribution parameterizing a distribution over probability vectors across the full vocabulary space"],
      correct: 0,
      explanation: "Softmax produces a **categorical distribution** over the vocabulary: $P(w = i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$. Each token gets a probability, probabilities sum to 1, and we sample one token. When we sample multiple tokens with replacement, the counts follow a multinomial. The Dirichlet would be a distribution *over* such probability vectors, not a single draw."
    },
    {
      type: "mc",
      question: "The **law of total expectation** states $\\mathbb{E}[X] = \\mathbb{E}[\\mathbb{E}[X \\mid Y]]$. In the context of language modeling, if $X$ is the log-probability of a sequence and $Y$ is the topic, this means:",
      options: ["The overall log-probability equals the best topic-conditional log-probability, computed by selecting the highest-scoring topic assignment", "The log-probability is statistically independent of the topic, so conditioning on topic provides no additional predictive information", "The overall expected log-probability is the topic-weighted average of per-topic expected log-probabilities", "You must first identify the exact topic before you can compute any meaningful estimate of the sequence log-probability"],
      correct: 2,
      explanation: "The law of total expectation says you can decompose an unconditional expectation by first conditioning on $Y$, computing the conditional expectation, then averaging over $Y$. Here: $\\mathbb{E}[\\log P] = \\sum_{\\text{topic}} P(\\text{topic}) \\cdot \\mathbb{E}[\\log P \\mid \\text{topic}]$. This is fundamental to understanding mixture models and how perplexity varies across data subsets."
    },
    {
      type: "mc",
      question: "Random variables $X$ and $Y$ are **independent** if and only if:",
      options: [
        "$\\mathbb{E}[XY] = 0$, meaning the expected value of the product of $X$ and $Y$ is exactly zero",
        "$P(X, Y) = P(X) \\cdot P(Y)$ for all values, meaning the joint distribution fully factorizes into marginals",
        "$\\text{Cov}(X, Y) = 0$, meaning the linear correlation between $X$ and $Y$ vanishes completely",
        "$P(X \\mid Y) > P(X)$ for all values, meaning observing $Y$ always increases the probability of $X$"
      ],
      correct: 1,
      explanation: "Independence means the joint factorizes: $P(X, Y) = P(X) P(Y)$. This is stronger than zero covariance — uncorrelated variables can still be dependent (e.g., $X \\sim \\text{Uniform}(-1,1)$ and $Y = X^2$ have $\\text{Cov}(X,Y) = 0$ but are clearly dependent). In LLMs, tokens are NOT independent — the whole point of the model is to capture dependencies."
    },
    {
      type: "mc",
      question: "The **law of total probability** says $P(A) = \\sum_i P(A \\mid B_i) P(B_i)$ for a partition $\\{B_i\\}$. When marginalizing over latent variable $z$ in a latent variable model, we compute:",
      options: ["$P(x) = \\max_z P(x \\mid z) P(z)$, selecting the single latent configuration that maximizes the joint probability", "$P(x) = \\sum_z P(z \\mid x) P(x)$, weighting by the posterior distribution over the latent variable given the observation", "$P(x) = P(x \\mid z^*)$ where $z^* = \\arg\\max P(z)$, conditioning on the most probable latent state from the prior", "$P(x) = \\sum_z P(x \\mid z) P(z)$ (or $\\int$ for continuous $z$), summing the likelihood weighted by the prior over all latent states"],
      correct: 3,
      explanation: "Marginalization is exactly the law of total probability applied to a latent variable: $P(x) = \\sum_z P(x, z) = \\sum_z P(x \\mid z) P(z)$. This integral is often intractable, which is why we need variational inference (ELBO) or sampling methods. The max version gives the MAP estimate, not the marginal."
    },
    {
      type: "mc",
      question: "For a discrete random variable $X$ with PMF $p(x)$, the **variance** $\\text{Var}(X)$ can be computed as:",
      options: [
        "$\\mathbb{E}[X]^2 - \\mathbb{E}[X^2]$",
        "$\\mathbb{E}[X^2] - \\mathbb{E}[X]^2$",
        "$\\mathbb{E}[|X - \\mathbb{E}[X]|]$",
        "$\\sqrt{\\mathbb{E}[(X - \\mathbb{E}[X])^2]}$"
      ],
      correct: 1,
      explanation: "$\\text{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$. This is the \"raw second moment minus squared first moment\" identity, essential for deriving variance of estimators, understanding gradient variance in SGD, and computing variance of importance weights in off-policy RL."
    },
    {
      type: "mc",
      question: "A token $w$ is drawn from a categorical distribution with probabilities $p_1 = 0.5, p_2 = 0.3, p_3 = 0.2$. What is $\\mathbb{E}[-\\log_2 p(w)]$, the expected number of bits needed to encode this draw?",
      options: [
        "$-\\log_2(0.5) = 1$ bit, using only the most probable token",
        "$0.5 \\cdot 1 + 0.3 \\cdot \\log_2(10/3) + 0.2 \\cdot \\log_2(5) \\approx 1.49$ bits",
        "$\\log_2(3) \\approx 1.58$ bits, the entropy of a uniform three-outcome distribution",
        "$3$ bits, since there are three tokens and each requires one bit of encoding"
      ],
      correct: 1,
      explanation: "This is the **entropy** $H(W) = -\\sum_i p_i \\log_2 p_i = 0.5(1) + 0.3(1.737) + 0.2(2.322) \\approx 1.49$ bits. It's less than $\\log_2(3) \\approx 1.58$ (the uniform case) because the distribution is non-uniform. Entropy is maximized when all outcomes are equally likely."
    }
  ]
};
