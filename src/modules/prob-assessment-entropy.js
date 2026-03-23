// Assessment: Entropy, Cross-Entropy & Mutual Information
// Section 0.2: Diagnostic test — core information-theoretic quantities
// Pure assessment to gauge depth of understanding

export const entropyAssessment = {
  id: "0.2-assess-entropy",
  sectionId: "0.2",
  title: "Assessment: Entropy, Cross-Entropy & Mutual Information",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Entropy, Cross-Entropy & Mutual Information",
      content: "This is a **diagnostic assessment** covering the core information-theoretic quantities that underpin LLM training and evaluation.\n\nEntropy, cross-entropy, and mutual information are not just abstract concepts — they are literally the loss functions, evaluation metrics, and theoretical tools used throughout deep learning.\n\nIf you score below 70%, these topics deserve serious review."
    },
    {
      type: "mc",
      question: "The **entropy** $H(X) = -\\sum_x p(x) \\log p(x)$ of a discrete random variable is maximized when:",
      options: ["The distribution is concentrated on a single outcome, making the variable fully deterministic with zero uncertainty", "The distribution has the largest possible support, regardless of how probability mass is allocated across outcomes", "The distribution is Gaussian with mean zero and unit variance, the maximum entropy continuous distribution", "The distribution is uniform over all outcomes, assigning equal probability $1/K$ to each of the $K$ possibilities"],
      correct: 3,
      explanation: "For a discrete variable with $K$ outcomes, entropy is maximized at $H = \\log K$ when $p(x) = 1/K$ (uniform). Any deviation from uniform reduces entropy. Intuitively, entropy measures uncertainty — maximum uncertainty means every outcome is equally likely. This is why temperature scaling in LLMs (which pushes the distribution toward uniform) increases entropy and diversity of outputs."
    },
    {
      type: "mc",
      question: "The **cross-entropy** $H(P, Q) = -\\mathbb{E}_P[\\log Q(x)]$ between the data distribution $P$ and model $Q$ relates to KL divergence and entropy as:",
      options: ["$H(P, Q) = H(P) - \\text{KL}(P \\| Q)$", "$H(P, Q) = \\text{KL}(P \\| Q) - H(P)$", "$H(P, Q) = H(P) + \\text{KL}(P \\| Q)$", "$H(P, Q) = H(Q) + \\text{KL}(Q \\| P)$"],
      correct: 2,
      explanation: "$H(P, Q) = H(P) + \\text{KL}(P \\| Q)$. Since $H(P)$ is fixed (it depends only on the data), minimizing cross-entropy is equivalent to minimizing $\\text{KL}(P \\| Q)$. This is why cross-entropy loss works — it drives the model $Q$ toward the data distribution $P$ in the KL sense."
    },
    {
      type: "mc",
      question: "**Perplexity** of a language model on data $P$ is defined as $\\text{PPL} = 2^{H(P,Q)}$ (or $e^{H(P,Q)}$ in nats). A model with perplexity 50 on a vocabulary of 50,000 tokens means:",
      options: ["On average, the model is as uncertain as choosing uniformly among 50 equally likely tokens at each position in the sequence", "The model correctly predicts exactly 50% of tokens in the sequence when evaluated using greedy argmax decoding", "The model requires approximately 50 bits of information to encode each token it encounters in the input sequence", "The effective vocabulary can be compressed down to just 50 tokens without any measurable loss of model performance"],
      correct: 0,
      explanation: "Perplexity is the exponential of cross-entropy. A perplexity of 50 means the model's uncertainty is equivalent to choosing uniformly from 50 options. Lower perplexity = more confident = better model. A uniform model over 50K tokens would have perplexity 50,000. Getting from 50,000 to 50 represents learning enormous amounts of structure."
    },
    {
      type: "mc",
      question: "The **mutual information** $I(X; Y) = H(X) - H(X \\mid Y)$ measures the reduction in uncertainty about $X$ after observing $Y$. Which of the following is TRUE?",
      options: [
        "$I(X; Y)$ can be negative when $Y$ adds confusion about $X$, increasing overall uncertainty beyond the marginal entropy",
        "$I(X; Y) = I(Y; X)$ — it is symmetric, meaning $X$ tells you as much about $Y$ as $Y$ tells you about $X$",
        "$I(X; Y) = H(X) + H(Y)$ always, since mutual information combines the total uncertainty of both variables",
        "$I(X; Y) = 0$ implies $X$ and $Y$ are identically distributed, meaning they share the same probability mass function"
      ],
      correct: 1,
      explanation: "$I(X; Y) = H(X) - H(X \\mid Y) = H(Y) - H(Y \\mid X) = I(Y; X)$. It's symmetric, non-negative (since conditioning cannot increase entropy on average), and equals zero if and only if $X$ and $Y$ are independent. Mutual information is the key quantity in the **information bottleneck** principle and in understanding what representations learn."
    },
    {
      type: "mc",
      question: "**Conditional entropy** $H(X \\mid Y) = \\mathbb{E}_Y[H(X \\mid Y = y)]$. In the context of language modeling, $H(W_t \\mid W_{<t})$ represents:",
      options: ["The total entropy of the full vocabulary distribution, computed without conditioning on any preceding context tokens", "The cardinality of the set of possible next tokens that have nonzero probability under the language model", "The log-probability of the single most likely next token, which determines the greedy decoding selection", "The average uncertainty about the next token given the context — the irreducible per-token loss for a perfect model"],
      correct: 3,
      explanation: "$H(W_t \\mid W_{<t})$ is the conditional entropy of the next token given all previous tokens. For a *perfect* model $Q = P$, the cross-entropy equals $H(P) = H(W_t \\mid W_{<t})$ — this is the irreducible uncertainty. No model can beat this bound. The gap between a model's cross-entropy and this quantity is exactly $\\text{KL}(P \\| Q)$."
    },
    {
      type: "mc",
      question: "The **data processing inequality** states that for a Markov chain $X \\to Y \\to Z$, we have $I(X; Z) \\leq I(X; Y)$. What does this imply for neural network representations?",
      options: ["Deeper layers always lose a strictly positive amount of information about the input at every processing step", "The final layer retains zero mutual information with the input, having discarded everything during processing", "Each layer can only preserve or lose information about the input — never create new information about it", "All layers in the network must maintain exactly equal mutual information with the original input signal"],
      correct: 2,
      explanation: "The data processing inequality says information can only be lost, never gained, through processing. If $X \\to \\text{layer}_1 \\to \\text{layer}_2$, then $I(X; \\text{layer}_2) \\leq I(X; \\text{layer}_1)$. This is the theoretical foundation of the **information bottleneck** view: good representations compress away irrelevant information while retaining information relevant to the task."
    },
    {
      type: "mc",
      question: "The **chain rule of mutual information** states $I(X; Y, Z) = I(X; Y) + I(X; Z \\mid Y)$. If a context window includes both recent tokens $Y$ and distant tokens $Z$, and $I(X; Z \\mid Y) \\approx 0$, this means:",
      options: ["Distant tokens provide no additional information about the next token beyond what the recent context tokens already capture", "Distant tokens carry substantially more predictive information about the next token than the recent tokens do", "The model should discard all contextual information and rely only on the unconditional token frequency distribution", "Recent tokens and distant tokens are statistically independent of each other with zero mutual information between them"],
      correct: 0,
      explanation: "$I(X; Z \\mid Y) \\approx 0$ means that once you condition on recent context $Y$, distant context $Z$ adds negligible information about next token $X$. This does NOT mean $Z$ is independent of $Y$ — just that $Z$'s predictive value is already captured by $Y$. This relates to why many practical tasks work well with limited context, and why efficient attention methods that prioritize local context can work."
    },
    {
      type: "mc",
      question: "The **entropy rate** of a stochastic process is $h = \\lim_{n \\to \\infty} \\frac{1}{n} H(X_1, \\dots, X_n)$. For English text, the entropy rate is estimated at roughly 1-1.5 bits per character. This means:",
      options: [
        "English text is effectively random with no exploitable statistical structure between consecutive characters",
        "Each character carries about 1-1.5 bits of information on average after accounting for all statistical patterns — English is highly redundant given its ~4.7-bit alphabet",
        "You need approximately 1-1.5 characters of input context to encode each single bit of information content",
        "A perfect language model operating at the entropy limit would achieve a perplexity score between 1.0 and 1.5"
      ],
      correct: 1,
      explanation: "With 26+ characters, a uniform distribution would give ~4.7 bits/char. The entropy rate of ~1.3 bits/char means English is about 72% redundant — most characters are highly predictable from context. This massive redundancy is what LLMs exploit. Shannon estimated this in 1951 using human prediction experiments. Modern LLMs approach (and perhaps reach) this bound."
    },
    {
      type: "mc",
      question: "When fine-tuning with **label smoothing** (mixing the hard target with a uniform distribution), the effective target becomes $P'(y) = (1 - \\alpha) \\cdot \\mathbf{1}[y = y^*] + \\alpha / K$. How does this affect the cross-entropy loss landscape?",
      options: ["It has no measurable effect on training dynamics because the smoothing parameter $\\alpha$ is typically too small to alter gradient magnitudes", "It raises the entropy floor of the model's output distribution to match the nonzero entropy of the smoothed target labels", "It forces the model to converge toward a uniform distribution over all classes, eliminating any discriminative signal", "It prevents the model from driving logits to $\\pm \\infty$ — the optimal logit gap becomes finite, acting as implicit regularization on confidence"],
      correct: 3,
      explanation: "Without label smoothing, the optimal logits are $+\\infty$ for the correct class and $-\\infty$ for others. Label smoothing caps the optimal logit gap at a finite value (proportional to $\\log((1-\\alpha)K/\\alpha)$), preventing overconfident predictions. This acts as entropy regularization — the model's output entropy stays bounded away from zero, improving calibration and generalization."
    },
    {
      type: "mc",
      question: "A model's training cross-entropy loss converges to 2.5 nats per token but validation loss plateaus at 3.2 nats. What is the most informative interpretation?",
      options: ["The model is underfitting and needs more parameters or layers to reduce the training loss further below its current plateau", "The gap of 0.7 nats between train and val represents overfitting: the model memorizes training patterns that don't generalize, wasting capacity on spurious correlations", "The true entropy rate of the underlying language distribution is exactly 2.5 nats, and the model has achieved the theoretical optimum", "The validation set comes from a substantially different language or domain, making the two loss values fundamentally incomparable"],
      correct: 1,
      explanation: "The train-val gap (0.7 nats) indicates the model has memorized training-specific patterns. Training loss reflects both genuine language structure and training-set-specific memorization; the difference (generalization gap) quantifies the latter. The true entropy rate $H(P)$ is at most the validation loss (3.2 nats), since $H(P, Q_{\\text{val}}) = H(P) + \\text{KL}(P \\| Q)$ and KL is non-negative. The training loss can go below $H(P)$ precisely because of memorization."
    },
    {
      type: "mc",
      question: "You apply temperature scaling with $T = 2.0$ to a model's logits during inference. If the original top-1 probability was 0.95, approximately what will it become?",
      options: ["Approximately $0.95 / 2 = 0.475$ — token probabilities scale linearly and inversely with temperature in the softmax function", "Still close to 0.95 — temperature scaling primarily redistributes mass among low-probability tokens and leaves the top token nearly unchanged", "Significantly reduced (roughly $0.87$ to $0.90$, depending on the logit structure) — dividing logits by 2 compresses them toward zero, spreading mass to other tokens", "Exactly $0.5$ regardless of the original distribution — doubling the temperature parameter always halves the top token's probability"],
      correct: 2,
      explanation: "Temperature scaling divides logits by $T$ before softmax. If the top logit was much larger than others (giving 0.95), dividing by 2 halves the logit gap, spreading probability to other tokens. The exact new probability depends on the logit distribution, but for a sharply peaked distribution going from $T=1$ to $T=2$ typically reduces the top probability noticeably. The relationship is NOT linear in $T$ — it goes through the softmax nonlinearity."
    },
    {
      type: "mc",
      question: "In contrastive learning (e.g., SimCLR), why does using a very small batch size (e.g., 32) limit performance even with longer training?",
      options: ["Small batches introduce excessive gradient noise that prevents the optimization from converging to a good minimum in the loss landscape", "The InfoNCE bound on MI is $\\log N$ where $N$ is batch size — a batch of 32 can estimate at most $\\log_2(32) = 5$ bits of MI, regardless of training duration", "Small batches inevitably cause complete mode collapse in the learned representation space, mapping all inputs to identical embeddings", "Small batches violate the i.i.d. sampling assumption required by the contrastive loss, introducing systematic bias into the gradient estimates"],
      correct: 1,
      explanation: "The InfoNCE loss provides a lower bound: $I(X;Y) \\geq \\log N - \\mathcal{L}$. Even if the loss reaches 0, you can only estimate $\\log N$ bits of MI. With batch size 32: $\\log_2(32) = 5$ bits maximum. If the true MI between views is much higher, this ceiling prevents the model from learning fine-grained distinctions. This is why CLIP uses batch sizes of 32K+ ($\\log_2(32768) \\approx 15$ bits) — it raises the MI ceiling, enabling richer representations."
    },
    {
      type: "mc",
      question: "For a Markov chain $X \\to Y \\to Z$, when does $I(X; Z) = I(X; Y)$ (equality in the data processing inequality)?",
      options: ["When $Z$ is a higher-dimensional version of $Y$ constructed by zero-padding, which increases the representation size without adding information", "When $Z$ is a sufficient statistic of $Y$ for $X$ — meaning $X \\to Z \\to Y$ is also a valid Markov chain, so no information about $X$ is lost", "Equality can never hold in practice — any deterministic or stochastic processing of $Y$ must strictly lose some information content", "Only when $Y = Z$ exactly (the identity mapping), since any nontrivial transformation necessarily destroys mutual information"],
      correct: 1,
      explanation: "DPI gives equality when $Z$ retains all information that $Y$ has about $X$, formally when $Z$ is a sufficient statistic of $Y$ for $X$. This means $P(X \\mid Y) = P(X \\mid Z)$ — knowing $Z$ is just as good as knowing $Y$ for predicting $X$. An invertible transformation $Z = g(Y)$ trivially satisfies this (since $Y$ can be recovered from $Z$), but sufficiency is the general condition. Identity is sufficient but not necessary."
    },
    {
      type: "mc",
      question: "A language model trained on English text achieves perplexity 8.0. Shannon estimated English entropy at ~1.3 bits/char, and BPE tokens average ~4 chars. How does this model compare to the theoretical limit?",
      options: ["The model has nearly reached the entropy limit — $\\log_2(8) = 3$ bits/token vs theoretical ~$4 \\times 1.3 = 5.2$ bits/token, actually beating it because the tokenizer absorbs spelling redundancy", "The model is far above the theoretical limit — a perplexity of 8 is very high and indicates the model has failed to capture most statistical structure", "The comparison is fundamentally invalid because Shannon's character-level entropy estimate cannot be meaningfully converted to a per-token entropy bound", "The model exactly matches the entropy limit — perplexity 8 on a 4-char-per-token vocabulary corresponds precisely to Shannon's 1.3 bits/char estimate"],
      correct: 0,
      explanation: "At perplexity 8: cross-entropy $= \\log_2(8) = 3$ bits/token. Shannon's character-level entropy rate (~1.3 bits/char) times ~4 chars/token gives ~5.2 bits/token as a crude token-level estimate. But BPE tokenization already encodes spelling patterns (absorbing ~1-2 bits/char of redundancy), so the effective per-token entropy limit is lower than 5.2 bits. The model at 3 bits/token is performing well, suggesting it captures most of the remaining (post-tokenization) statistical structure."
    }
  ]
};
