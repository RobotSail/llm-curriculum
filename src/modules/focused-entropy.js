// Focused learning module for ENTROPY as a single concept.
// Covers: Shannon entropy, conditional entropy, joint entropy,
// chain rule of entropy, maximum entropy principle, temperature control,
// and the entropy rate of English.

export const entropyLearning = {
  id: "0.2-entropy-learning-easy",
  sectionId: "0.2",
  title: "Entropy: Measuring Uncertainty",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — Entropy definition
    {
      type: "info",
      title: "Entropy: Expected Surprise",
      content: "**Entropy** measures the average surprise (or uncertainty) in a random variable:\n\n$$H(X) = -\\sum_x P(x) \\log P(x) = \\mathbb{E}_P[-\\log P(X)]$$\n\nEach outcome $x$ carries surprise $-\\log P(x)$: rare events are surprising, common ones are not. Entropy is the *expected* surprise across all outcomes.\n\n**High entropy** means the distribution is spread out (close to uniform) — you are very uncertain about the outcome. **Low entropy** means probability mass is concentrated on a few outcomes — you can predict well.\n\nFor a discrete distribution over $K$ outcomes:\n- **Maximum entropy** $= \\log K$, achieved by the uniform distribution $P(x) = 1/K$.\n- **Minimum entropy** $= 0$, achieved when all mass is on a single outcome (a delta distribution).\n\nEntropy answers a precise operational question via **Shannon's source coding theorem**: it is the minimum average number of bits needed to losslessly encode a sample from $P$. You cannot compress below $H(X)$ bits on average, and you can get arbitrarily close to it with an optimal code."
    },
    // Step 2: MC — Uniform vs concentrated entropy
    {
      type: "mc",
      question: "Distribution A: $P = (0.25, 0.25, 0.25, 0.25)$. Distribution B: $P = (0.97, 0.01, 0.01, 0.01)$. Which has higher entropy?",
      options: [
        "Distribution B — it has one dominant outcome, so more information content",
        "They have equal entropy because both are defined over 4 outcomes",
        "Distribution A — it is uniform, so maximum uncertainty",
        "Cannot determine without knowing what the outcomes represent"
      ],
      correct: 2,
      explanation: "Distribution A is uniform over 4 outcomes, giving $H(A) = \\log_2 4 = 2$ bits — the maximum possible for 4 outcomes. Distribution B concentrates almost all mass on one outcome: $H(B) \\approx 0.97 \\times 0.044 + 3 \\times 0.01 \\times 6.64 \\approx 0.24$ bits. High concentration means low uncertainty, hence low entropy. The number of outcomes sets the ceiling, but the shape of the distribution determines entropy."
    },
    // Step 3: MC — Coin flip entropy
    {
      type: "mc",
      question: "A coin flip has $P(\\text{heads}) = p$. At what value of $p$ is the entropy $H(X)$ maximized?",
      options: [
        "$p = 0.5$ — maximum uncertainty when both outcomes are equally likely",
        "$p = 0$ — certainty about the outcome maximizes information",
        "$p = 1$ — a guaranteed outcome carries the most information",
        "$p = 1/e \\approx 0.37$ — this minimizes $-p \\log p$"
      ],
      correct: 0,
      explanation: "For a Bernoulli variable, $H(X) = -p\\log p - (1-p)\\log(1-p)$. This is a concave function maximized at $p = 0.5$, where $H = 1$ bit. At $p = 0$ or $p = 1$, entropy is 0 — there is no uncertainty. The value $1/e$ minimizes the single-term function $-p\\log p$, but entropy involves two terms that together peak at $p = 0.5$."
    },
    // Step 4: Info — Conditional entropy
    {
      type: "info",
      title: "Conditional Entropy: Remaining Uncertainty",
      content: "**Conditional entropy** $H(X \\mid Y)$ measures how much uncertainty about $X$ remains after observing $Y$:\n\n$$H(X \\mid Y) = -\\sum_{x, y} P(x, y) \\log P(x \\mid y) = \\mathbb{E}_{(X,Y)}[-\\log P(X \\mid Y)]$$\n\nConditioning never increases entropy on average: $H(X \\mid Y) \\leq H(X)$, with equality only when $X$ and $Y$ are independent. Observing a related variable can only reduce (or maintain) your uncertainty.\n\nFor **language models**, the crucial quantity is $H(W_t \\mid W_{<t})$ — the conditional entropy of the next token given all preceding context. This represents the **intrinsic unpredictability of language** at position $t$.\n\nA perfect model $Q^* = P$ achieves training loss equal to this conditional entropy — the **Bayes-optimal loss**. No model can do better, no matter how large or well-trained, because language is genuinely stochastic: given a context, multiple valid continuations exist."
    },
    // Step 5: MC — Language model loss floor
    {
      type: "mc",
      question: "A language model's per-token training loss can never go below which quantity?",
      options: [
        "$\\log K$, the entropy of a uniform distribution over the full vocabulary of size $K$",
        "$H(W_t)$, the marginal entropy of the token distribution ignoring all context",
        "Zero — a sufficiently large model could memorize every sequence in the corpus",
        "$H(W_t \\mid W_{<t})$, the conditional entropy of the next token given full context"
      ],
      correct: 3,
      explanation: "The minimum achievable cross-entropy is the conditional entropy $H(W_t \\mid W_{<t})$ averaged over positions — the irreducible uncertainty given perfect context modeling. Even memorizing the training set doesn't help: the Bayes-optimal predictor must spread probability across all valid continuations, not just the one that appeared in the corpus. $\\log K$ is an upper bound (uniform model), $H(W_t)$ is also an upper bound (marginal entropy), and zero is unreachable because language is inherently stochastic."
    },
    // Step 6: Info — Joint entropy and the chain rule
    {
      type: "info",
      title: "Joint Entropy and the Chain Rule",
      content: "**Joint entropy** $H(X, Y)$ measures the total uncertainty in a pair of random variables:\n\n$$H(X, Y) = -\\sum_{x, y} P(x, y) \\log P(x, y)$$\n\nThe **chain rule of entropy** decomposes joint entropy into a marginal and a conditional term:\n\n$$H(X, Y) = H(X) + H(Y \\mid X)$$\n\nThis says: the total uncertainty of $(X, Y)$ equals the uncertainty in $X$ alone, plus the remaining uncertainty in $Y$ once $X$ is known. By symmetry, $H(X, Y) = H(Y) + H(X \\mid Y)$ as well.\n\nThe chain rule generalizes to sequences. For a sequence of tokens $W_1, W_2, \\ldots, W_T$:\n\n$$H(W_1, \\ldots, W_T) = \\sum_{t=1}^{T} H(W_t \\mid W_{<t})$$\n\nThis decomposition is exactly why **autoregressive language models** factor the joint probability as $P(W_1, \\ldots, W_T) = \\prod_t P(W_t \\mid W_{<t})$ — each factor corresponds to one conditional entropy term. The total sequence entropy is the sum of per-token conditional entropies."
    },
    // Step 7: MC — Chain rule application
    {
      type: "mc",
      question: "For two random variables $X$ and $Y$ that are statistically independent, which relationship holds?",
      options: [
        "$H(X, Y) = H(X) \\cdot H(Y)$ because independent probabilities multiply, and entropy follows suit",
        "$H(X, Y) = \\max(H(X), H(Y))$ because the joint uncertainty is dominated by the more uncertain variable",
        "$H(X, Y) = H(X) + H(Y) - H(X)H(Y)$ by the inclusion-exclusion principle for entropy",
        "$H(X, Y) = H(X) + H(Y)$ because conditioning on an independent variable does not reduce entropy"
      ],
      correct: 3,
      explanation: "By the chain rule, $H(X, Y) = H(X) + H(Y \\mid X)$. When $X$ and $Y$ are independent, knowing $X$ tells you nothing about $Y$, so $H(Y \\mid X) = H(Y)$. Therefore $H(X, Y) = H(X) + H(Y)$. Entropy is additive (not multiplicative) for independent variables — this parallels how log-probabilities add when probabilities multiply. The other options confuse entropy's additive structure with probability's multiplicative structure."
    },
    // Step 8: Info — The entropy rate of English
    {
      type: "info",
      title: "The Entropy Rate of English",
      content: "Shannon estimated the **entropy rate** of English at about **1.0 to 1.5 bits per character** through human prediction experiments. Compare this to the maximum if every letter were equally likely:\n\n$$H_{\\text{uniform}} = \\log_2 27 \\approx 4.75 \\text{ bits/char}$$\n\n(26 letters plus space). English is therefore roughly **70% redundant** — most of the information capacity of the character stream is consumed by spelling patterns, grammar, and semantic constraints.\n\nThis redundancy is exactly what language models exploit. A model that captures these patterns compresses language toward its entropy rate. Modern LLMs achieve cross-entropy below 1 bit per character on many benchmarks, approaching Shannon's estimates.\n\nAt the **token level** (with BPE tokenization averaging roughly 4 chars/token), the entropy rate scales up to approximately $4 \\times 1.3 \\approx 5.2$ bits/token. State-of-the-art models achieve perplexities around 6 to 10 on standard benchmarks, corresponding to about 3 bits/token — well below this character-level estimate because subword tokenization captures spelling redundancy directly in the tokenizer."
    },
    // Step 9: MC — Entropy rate of English
    {
      type: "mc",
      question: "Shannon estimated that English has an entropy rate of about 1.0 to 1.5 bits per character. If you built a language model that assigned equal probability to each of the 26 letters and space, what would its per-character cross-entropy be?",
      options: [
        "About 4.75 bits — $\\log_2(27)$, since the model treats all 27 symbols as equally likely",
        "About 1.3 bits — matching Shannon's entropy rate, since cross-entropy of any model is bounded above by the true source entropy rate",
        "About 8.0 bits — one full byte per character, because the model defaults to ASCII-level encoding of 8 bits per symbol",
        "About 2.7 bits — $\\log_2(\\sqrt{27})$, since a uniform model over $K$ symbols achieves half of $\\log_2(K)$ entropy"
      ],
      correct: 0,
      explanation: "A uniform model over 27 symbols (26 letters + space) assigns probability $1/27$ to every character, giving cross-entropy $-\\log_2(1/27) = \\log_2(27) \\approx 4.75$ bits per character. This is far above Shannon's estimate of 1.0 to 1.5 bits, showing that more than 3 bits per character of redundancy remain unexploited. This gap is what statistical and neural language models progressively close."
    },
    // Step 10: Info — Maximum entropy principle
    {
      type: "info",
      title: "The Maximum Entropy Principle",
      content: "The **maximum entropy principle** states: among all distributions consistent with your known constraints, choose the one with the **highest entropy**. This is the distribution that introduces the least additional assumptions beyond what you actually know.\n\nConcretely, if you know only the mean $\\mu$ and variance $\\sigma^2$ of a continuous random variable, the maximum entropy distribution is the **Gaussian** $\\mathcal{N}(\\mu, \\sigma^2)$. If you know only that the variable takes values in $\\{1, \\ldots, K\\}$ with no other constraints, the maximum entropy distribution is **uniform**.\n\nThis principle has direct applications in machine learning:\n- **Softmax outputs** in classification: the model outputs a distribution over classes, and the softmax with high temperature approaches maximum entropy (uniform), encoding maximum uncertainty.\n- **Prior selection**: when you lack strong prior knowledge, a maximum-entropy prior (e.g., uniform or Gaussian) is a principled default.\n- **Regularization**: entropy-based regularizers encourage models to maintain calibrated uncertainty rather than becoming overconfident."
    },
    // Step 11: MC — Maximum entropy principle
    {
      type: "mc",
      question: "You know that a random variable $X$ takes values in $\\{1, 2, 3, 4, 5, 6\\}$ and has mean $\\mathbb{E}[X] = 3.5$. According to the maximum entropy principle, which distribution should you choose?",
      options: [
        "A distribution concentrated equally on 3 and 4, since their average is 3.5 and a two-point distribution minimizes assumptions about spread",
        "The uniform distribution $P(x) = 1/6$ for all $x$, since it has the highest entropy among distributions with this mean",
        "A distribution placing most mass on 1 and 6 to maximize the variance, which is equivalent to maximizing entropy for discrete variables",
        "A triangular distribution peaking at 3.5 with symmetric linear falloff, since it best approximates a Gaussian on a bounded domain"
      ],
      correct: 1,
      explanation: "The uniform distribution over $\\{1, 2, 3, 4, 5, 6\\}$ has mean $(1+2+3+4+5+6)/6 = 3.5$, satisfying the constraint. Among all distributions on these six values with mean 3.5, the uniform distribution has maximum entropy $\\log_2 6 \\approx 2.58$ bits. Options A and D satisfy the mean constraint but have lower entropy because they concentrate mass. A triangular distribution also has lower entropy than uniform."
    },
    // Step 12: Info — Temperature and entropy control
    {
      type: "info",
      title: "Temperature and Entropy Control",
      content: "In language model decoding, **temperature** $\\tau$ directly controls the entropy of the output distribution. Given logits $z_i$, the temperature-scaled softmax is:\n\n$$P(x_i) = \\frac{e^{z_i / \\tau}}{\\sum_j e^{z_j / \\tau}}$$\n\nTemperature modulates entropy through a simple mechanism:\n\n- **$\\tau \\to 0$ (low temperature)**: The differences between logits are amplified ($z_i / \\tau$ diverges). The distribution collapses onto the highest-logit token. Entropy approaches **0** — greedy, deterministic decoding.\n- **$\\tau = 1$ (standard)**: The original model distribution is used unchanged.\n- **$\\tau \\to \\infty$ (high temperature)**: All scaled logits $z_i / \\tau \\to 0$, making every token equally likely. Entropy approaches $\\log K$ — the maximum possible — producing uniform random sampling.\n\nTemperature is thus a direct dial on the entropy of the sampling distribution. Higher temperature means higher entropy (more randomness), lower temperature means lower entropy (more determinism). The model's learned logits encode its beliefs; temperature decides how sharply or softly to act on them."
    },
    // Step 13: MC — Temperature and entropy
    {
      type: "mc",
      question: "A language model produces logits $z = [2.0, 1.0, 0.5, 0.5]$ over a 4-token vocabulary. What happens to the entropy of the output distribution as you increase the temperature $\\tau$ from 0.1 to 10?",
      options: [
        "Entropy decreases from $\\log 4$ toward 0, because high temperature sharpens the distribution",
        "Entropy increases from near 0 toward $\\log 4$, because high temperature flattens the distribution toward uniform",
        "Entropy stays constant because temperature only rescales probabilities without changing their ranking",
        "Entropy first increases then decreases, peaking at $\\tau = 1$ where the model is calibrated"
      ],
      correct: 1,
      explanation: "At low temperature ($\\tau = 0.1$), the logit differences are amplified by $1/\\tau = 10$, concentrating nearly all mass on the token with logit 2.0 — entropy is near 0. As temperature increases, the scaled logits $z_i / \\tau$ converge to each other, flattening the distribution. At very high temperature ($\\tau = 10$), all tokens are nearly equiprobable and entropy approaches $\\log 4 = 2$ bits. Entropy increases monotonically with temperature."
    },
    // Step 14: MC — Entropy properties
    {
      type: "mc",
      question: "Consider three random variables $X$, $Y$, and $Z$ forming a Markov chain $X \\to Y \\to Z$ (i.e., $Z$ is conditionally independent of $X$ given $Y$). Which statement about their conditional entropies is correct?",
      options: [
        "$H(Z \\mid X) \\leq H(Z \\mid Y)$ because $X$ is farther back in the chain, so it carries strictly more cumulative information about the final variable $Z$",
        "$H(Z \\mid X) = H(Z \\mid Y) + H(Y \\mid X)$ by applying the chain rule of entropy sequentially along each link of the Markov chain",
        "$H(Z \\mid X) \\geq H(Z \\mid Y)$ because $X$ can only inform $Z$ through $Y$, so knowing $X$ is never more useful than knowing $Y$ directly",
        "$H(Z \\mid X, Y) = H(Z \\mid X)$ because once you condition on the source $X$, the intermediate variable $Y$ adds no further information about $Z$"
      ],
      correct: 2,
      explanation: "In the Markov chain $X \\to Y \\to Z$, all information $X$ has about $Z$ passes through $Y$. Formally, $Z \\perp X \\mid Y$, so $H(Z \\mid X, Y) = H(Z \\mid Y)$. Since conditioning on more variables can only reduce entropy: $H(Z \\mid X, Y) \\leq H(Z \\mid X)$, giving us $H(Z \\mid Y) \\leq H(Z \\mid X)$. Knowing $Y$ directly is at least as useful as knowing $X$, because $X$ can only influence $Z$ indirectly through $Y$."
    }
  ]
};
