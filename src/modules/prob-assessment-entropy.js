// Assessment: Entropy, Cross-Entropy & Mutual Information
// Section 0.2: Diagnostic test — core information-theoretic quantities
// Pure assessment to gauge depth of understanding

export const entropyAssessment = {
  id: "0.2-assess-entropy",
  sectionId: "0.2",
  title: "Assessment: Entropy, Cross-Entropy & Mutual Information",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "info",
      title: "Diagnostic: Entropy, Cross-Entropy & Mutual Information",
      content: "This is a **diagnostic assessment** covering the core information-theoretic quantities that underpin LLM training and evaluation.\n\nEntropy, cross-entropy, and mutual information are not just abstract concepts — they are literally the loss functions, evaluation metrics, and theoretical tools used throughout deep learning.\n\nIf you score below 70%, these topics deserve serious review."
    },
    {
      type: "mc",
      question: "The **entropy** $H(X) = -\\sum_x p(x) \\log p(x)$ of a discrete random variable is maximized when:",
      options: [
        "The distribution is concentrated on a single outcome (deterministic)",
        "The distribution is uniform over all outcomes",
        "The distribution is Gaussian",
        "The distribution has the largest possible support"
      ],
      correct: 1,
      explanation: "For a discrete variable with $K$ outcomes, entropy is maximized at $H = \\log K$ when $p(x) = 1/K$ (uniform). Any deviation from uniform reduces entropy. Intuitively, entropy measures uncertainty — maximum uncertainty means every outcome is equally likely. This is why temperature scaling in LLMs (which pushes the distribution toward uniform) increases entropy and diversity of outputs."
    },
    {
      type: "mc",
      question: "The **cross-entropy** $H(P, Q) = -\\mathbb{E}_P[\\log Q(x)]$ between the data distribution $P$ and model $Q$ relates to KL divergence and entropy as:",
      options: [
        "$H(P, Q) = H(P) - \\text{KL}(P \\| Q)$",
        "$H(P, Q) = H(P) + \\text{KL}(P \\| Q)$",
        "$H(P, Q) = \\text{KL}(P \\| Q) - H(P)$",
        "$H(P, Q) = H(Q) + \\text{KL}(Q \\| P)$"
      ],
      correct: 1,
      explanation: "$H(P, Q) = H(P) + \\text{KL}(P \\| Q)$. Since $H(P)$ is fixed (it depends only on the data), minimizing cross-entropy is equivalent to minimizing $\\text{KL}(P \\| Q)$. This is why cross-entropy loss works — it drives the model $Q$ toward the data distribution $P$ in the KL sense."
    },
    {
      type: "mc",
      question: "**Perplexity** of a language model on data $P$ is defined as $\\text{PPL} = 2^{H(P,Q)}$ (or $e^{H(P,Q)}$ in nats). A model with perplexity 50 on a vocabulary of 50,000 tokens means:",
      options: [
        "The model gets 50% of predictions correct",
        "On average, the model is as uncertain as choosing uniformly among 50 equally likely tokens at each step",
        "The model uses 50 bits per token",
        "The vocabulary can be reduced to 50 tokens without loss"
      ],
      correct: 1,
      explanation: "Perplexity is the exponential of cross-entropy. A perplexity of 50 means the model's uncertainty is equivalent to choosing uniformly from 50 options. Lower perplexity = more confident = better model. A uniform model over 50K tokens would have perplexity 50,000. Getting from 50,000 to 50 represents learning enormous amounts of structure."
    },
    {
      type: "mc",
      question: "The **mutual information** $I(X; Y) = H(X) - H(X \\mid Y)$ measures the reduction in uncertainty about $X$ after observing $Y$. Which of the following is TRUE?",
      options: [
        "$I(X; Y)$ can be negative when $Y$ adds confusion about $X$",
        "$I(X; Y) = I(Y; X)$ — it is symmetric",
        "$I(X; Y) = H(X) + H(Y)$ always",
        "$I(X; Y) = 0$ implies $X$ and $Y$ are identically distributed"
      ],
      correct: 1,
      explanation: "$I(X; Y) = H(X) - H(X \\mid Y) = H(Y) - H(Y \\mid X) = I(Y; X)$. It's symmetric, non-negative (since conditioning cannot increase entropy on average), and equals zero if and only if $X$ and $Y$ are independent. Mutual information is the key quantity in the **information bottleneck** principle and in understanding what representations learn."
    },
    {
      type: "mc",
      question: "**Conditional entropy** $H(X \\mid Y) = \\mathbb{E}_Y[H(X \\mid Y = y)]$. In the context of language modeling, $H(W_t \\mid W_{<t})$ represents:",
      options: [
        "The entropy of the vocabulary distribution",
        "The average uncertainty about the next token given the context — the irreducible per-token loss for a perfect model",
        "The probability of the most likely next token",
        "The number of possible next tokens"
      ],
      correct: 1,
      explanation: "$H(W_t \\mid W_{<t})$ is the conditional entropy of the next token given all previous tokens. For a *perfect* model $Q = P$, the cross-entropy equals $H(P) = H(W_t \\mid W_{<t})$ — this is the irreducible uncertainty. No model can beat this bound. The gap between a model's cross-entropy and this quantity is exactly $\\text{KL}(P \\| Q)$."
    },
    {
      type: "mc",
      question: "The **data processing inequality** states that for a Markov chain $X \\to Y \\to Z$, we have $I(X; Z) \\leq I(X; Y)$. What does this imply for neural network representations?",
      options: [
        "Deeper layers always lose information about the input",
        "Each layer can only preserve or lose information about the input — never create new information about it",
        "The final layer has zero mutual information with the input",
        "All layers must have equal mutual information with the input"
      ],
      correct: 1,
      explanation: "The data processing inequality says information can only be lost, never gained, through processing. If $X \\to \\text{layer}_1 \\to \\text{layer}_2$, then $I(X; \\text{layer}_2) \\leq I(X; \\text{layer}_1)$. This is the theoretical foundation of the **information bottleneck** view: good representations compress away irrelevant information while retaining information relevant to the task."
    },
    {
      type: "mc",
      question: "The **chain rule of mutual information** states $I(X; Y, Z) = I(X; Y) + I(X; Z \\mid Y)$. If a context window includes both recent tokens $Y$ and distant tokens $Z$, and $I(X; Z \\mid Y) \\approx 0$, this means:",
      options: [
        "Distant tokens are more important than recent tokens",
        "Distant tokens provide no additional information about the next token beyond what recent tokens already provide",
        "The model should ignore all context",
        "Recent and distant tokens are independent of each other"
      ],
      correct: 1,
      explanation: "$I(X; Z \\mid Y) \\approx 0$ means that once you condition on recent context $Y$, distant context $Z$ adds negligible information about next token $X$. This does NOT mean $Z$ is independent of $Y$ — just that $Z$'s predictive value is already captured by $Y$. This relates to why many practical tasks work well with limited context, and why efficient attention methods that prioritize local context can work."
    },
    {
      type: "mc",
      question: "The **entropy rate** of a stochastic process is $h = \\lim_{n \\to \\infty} \\frac{1}{n} H(X_1, \\dots, X_n)$. For English text, the entropy rate is estimated at roughly 1-1.5 bits per character. This means:",
      options: [
        "English text is completely random",
        "Each character carries about 1-1.5 bits of information on average after accounting for all statistical patterns — English is highly redundant given its ~4.7-bit alphabet",
        "You need 1-1.5 characters to encode each bit",
        "A perfect language model would have perplexity between 1 and 1.5"
      ],
      correct: 1,
      explanation: "With 26+ characters, a uniform distribution would give ~4.7 bits/char. The entropy rate of ~1.3 bits/char means English is about 72% redundant — most characters are highly predictable from context. This massive redundancy is what LLMs exploit. Shannon estimated this in 1951 using human prediction experiments. Modern LLMs approach (and perhaps reach) this bound."
    },
    {
      type: "mc",
      question: "When fine-tuning with **label smoothing** (mixing the hard target with a uniform distribution), the effective target becomes $P'(y) = (1 - \\alpha) \\cdot \\mathbf{1}[y = y^*] + \\alpha / K$. How does this affect the cross-entropy loss landscape?",
      options: [
        "It has no effect on training dynamics",
        "It prevents the model from driving logits to $\\pm \\infty$ — the optimal logit gap becomes finite, acting as implicit regularization on confidence",
        "It makes the model converge to a uniform distribution",
        "It increases the entropy of the model to match the entropy of the smoothed labels"
      ],
      correct: 1,
      explanation: "Without label smoothing, the optimal logits are $+\\infty$ for the correct class and $-\\infty$ for others. Label smoothing caps the optimal logit gap at a finite value (proportional to $\\log((1-\\alpha)K/\\alpha)$), preventing overconfident predictions. This acts as entropy regularization — the model's output entropy stays bounded away from zero, improving calibration and generalization."
    }
  ]
};
