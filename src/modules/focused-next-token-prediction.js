// Focused learning module: Next-Token Prediction as a Training Objective
// Section 1.3: Pretraining Objectives & Dynamics
// Covers: autoregressive factorization, MLE connection, cross-entropy loss,
// teacher forcing, exposure bias, and why next-token prediction works so well.
// Grounded in Goodfellow et al. (2016) Ch. 3, 5, 10 — probability, MLE, sequence modeling.

export const nextTokenPredictionLearning = {
  id: "1.3-next-token-prediction-learning-easy",
  sectionId: "1.3",
  title: "Next-Token Prediction as a Training Objective",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: The core idea
    {
      type: "info",
      title: "Language Modeling: Assigning Probabilities to Text",
      content: "A **language model** assigns a probability to every possible sequence of tokens. Given a sequence $x_1, x_2, \\ldots, x_T$, the model estimates:\n\n$$P(x_1, x_2, \\ldots, x_T)$$\n\nThis is an extraordinarily high-dimensional distribution — with a vocabulary of $V = 50{,}000$ tokens and sequences of length $T = 1{,}000$, the space of possible sequences has $V^T$ elements. We cannot store or enumerate this distribution directly.\n\nThe key insight is the **chain rule of probability** (Goodfellow et al., 2016, §3.6). Any joint distribution can be factored as a product of conditionals:\n\n$$P(x_1, \\ldots, x_T) = P(x_1) \\cdot P(x_2 \\mid x_1) \\cdot P(x_3 \\mid x_1, x_2) \\cdots P(x_T \\mid x_1, \\ldots, x_{T-1})$$\n\n$$= \\prod_{t=1}^{T} P(x_t \\mid x_{<t})$$\n\nThis factorization is **exact** — no approximation is made. It reduces the problem of modeling one impossibly large joint distribution to modeling $T$ conditional distributions, each over just $V$ tokens."
    },
    // Step 2: MC — chain rule understanding
    {
      type: "mc",
      question: "The autoregressive factorization $P(x_1, \\ldots, x_T) = \\prod_{t=1}^{T} P(x_t \\mid x_{<t})$ makes an assumption about the ordering of tokens. Which statement is correct?",
      options: [
        "The factorization assumes tokens are generated left-to-right, which introduces an approximation error for languages that are read right-to-left (like Arabic or Hebrew)",
        "The factorization is a mathematical identity from the chain rule of probability — it holds for ANY ordering of variables and introduces no approximation whatsoever",
        "The factorization assumes that each token depends only on a fixed window of $k$ previous tokens, making it a $k$-th order Markov assumption",
        "The factorization requires that the conditional distributions $P(x_t \\mid x_{<t})$ are independent of each other, meaning each prediction is made without knowledge of how well the model predicted earlier tokens"
      ],
      correct: 1,
      explanation: "The chain rule of probability is an identity — it is algebraically true for any joint distribution, with no approximation. You could factorize in reverse order ($P(x_T) \\cdot P(x_{T-1} \\mid x_T) \\cdots$) and it would be equally valid. The left-to-right convention is a modeling choice, not a mathematical necessity. Importantly, the factorization does NOT impose a Markov assumption: each $P(x_t \\mid x_{<t})$ conditions on the ENTIRE prefix, not a fixed window."
    },
    // Step 3: MLE and cross-entropy
    {
      type: "info",
      title: "Training Objective: Maximum Likelihood Estimation",
      content: "Given a training corpus of sequences, we want to find model parameters $\\theta$ that make the observed data as likely as possible. This is **maximum likelihood estimation** (MLE) (Goodfellow et al., 2016, §5.5):\n\n$$\\theta^* = \\arg\\max_\\theta \\sum_{(x_1, \\ldots, x_T) \\in \\mathcal{D}} \\sum_{t=1}^{T} \\log P_\\theta(x_t \\mid x_{<t})$$\n\nEquivalently, we minimize the **negative log-likelihood**:\n\n$$\\mathcal{L}(\\theta) = -\\frac{1}{N} \\sum_{t=1}^{N} \\log P_\\theta(x_t \\mid x_{<t})$$\n\nwhere $N$ is the total number of tokens in the corpus. This is exactly the **cross-entropy** between the data distribution and the model (Goodfellow et al., 2016, §5.5.1):\n\n$$\\mathcal{L}(\\theta) = H(p_{\\text{data}}, p_\\theta) = H(p_{\\text{data}}) + D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$$\n\nSince $H(p_{\\text{data}})$ is a constant (the true entropy of language), minimizing cross-entropy is equivalent to minimizing $D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$ — driving the model distribution as close to the data distribution as possible."
    },
    // Step 4: MC — MLE and KL divergence
    {
      type: "mc",
      question: "A language model achieves a cross-entropy loss of 2.5 nats per token on a validation set. The true entropy of the language (an unknown constant) is approximately 1.5 nats per token. What is the KL divergence $D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$?",
      options: [
        "4.0 nats — the KL divergence is the sum of cross-entropy and entropy",
        "Cannot be determined — the KL divergence depends on the full distribution, not just the per-token cross-entropy",
        "0.6 nats — the KL divergence is the ratio $H(p_{\\text{data}}) / H(p_{\\text{data}}, p_\\theta) = 1.5 / 2.5$",
        "1.0 nat — since $H(p_{\\text{data}}, p_\\theta) = H(p_{\\text{data}}) + D_{\\text{KL}}$, we get $D_{\\text{KL}} = 2.5 - 1.5 = 1.0$"
      ],
      correct: 3,
      explanation: "From the decomposition $H(p_{\\text{data}}, p_\\theta) = H(p_{\\text{data}}) + D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$, the KL divergence is simply the gap between cross-entropy and entropy: $D_{\\text{KL}} = 2.5 - 1.5 = 1.0$ nat per token. This means the model wastes on average 1.0 nat per token due to its imperfect modeling. In practice we cannot measure $H(p_{\\text{data}})$ directly, but we know that as the model improves, its cross-entropy approaches the true entropy from above. The lower bound $H(p_{\\text{data}})$ is never reached because language has irreducible uncertainty."
    },
    // Step 5: What the model actually computes
    {
      type: "info",
      title: "From Logits to Probabilities",
      content: "At each position $t$, the transformer produces a **logit vector** $z_t \\in \\mathbb{R}^V$ — one raw score per vocabulary token. These logits are converted to a probability distribution using the **softmax** function:\n\n$$P_\\theta(x_t = w \\mid x_{<t}) = \\text{softmax}(z_t)_w = \\frac{\\exp(z_{t,w})}{\\sum_{w'=1}^{V} \\exp(z_{t,w'})}$$\n\nThe loss for a single position is the negative log-probability of the true token $x_t^*$:\n\n$$\\ell_t = -\\log P_\\theta(x_t^* \\mid x_{<t}) = -z_{t, x_t^*} + \\log \\sum_{w'=1}^{V} \\exp(z_{t,w'})$$\n\nThis is the **softmax cross-entropy** loss, the workhorse of neural language modeling. The gradient has an elegant form: $\\frac{\\partial \\ell_t}{\\partial z_{t,w}} = P_\\theta(w \\mid x_{<t}) - \\mathbb{1}[w = x_t^*]$.\n\nIn words: the gradient pushes down the probability of every token that isn't the ground truth, and pushes up the probability of the correct token. The magnitude of each push is proportional to the model's current confidence in that token."
    },
    // Step 6: MC — gradient interpretation
    {
      type: "mc",
      question: "At position $t$, the model assigns probability 0.8 to the correct next token and 0.15 to an incorrect token $w$. What is the gradient $\\frac{\\partial \\ell_t}{\\partial z_{t,w}}$ for the incorrect token?",
      options: [
        "$-0.15$ — the gradient is the negative of the probability, pushing the logit down proportionally",
        "$+0.15$ — the gradient equals the probability assigned to the wrong token, which will be used to decrease its logit during the weight update",
        "$+0.85$ — the gradient is $1 - P(w)$ for incorrect tokens, applying maximum correction to low-probability tokens",
        "$-0.65$ — the gradient is $P(\\text{correct}) - P(w) = 0.8 - 0.15$, reflecting the relative gap between the correct and incorrect token"
      ],
      correct: 1,
      explanation: "For any non-target token $w \\neq x_t^*$, the gradient is $\\frac{\\partial \\ell_t}{\\partial z_{t,w}} = P_\\theta(w \\mid x_{<t}) - 0 = P_\\theta(w \\mid x_{<t}) = 0.15$. A positive gradient means the optimizer (via gradient descent) will DECREASE $z_{t,w}$. Tokens the model confidently assigns high probability get large downward corrections; tokens already near zero probability get negligible corrections. This is efficient: the model focuses its learning on correcting its most confident mistakes."
    },
    // Step 7: Teacher forcing
    {
      type: "info",
      title: "Teacher Forcing: Parallel Training",
      content: "During training, the model could generate token-by-token, feeding each prediction back as input for the next step. But this would be sequential — we'd lose the transformer's ability to process all positions in parallel.\n\n**Teacher forcing** solves this: at every position $t$, we feed the **ground-truth** token $x_{t-1}$ as input, regardless of what the model would have predicted. Combined with the **causal mask** (which prevents position $t$ from seeing positions $t+1, t+2, \\ldots$), this lets us compute all $T$ predictions in a **single forward pass**.\n\nConcretely, a sequence of $T$ tokens produces $T - 1$ training signals in one forward pass:\n- Position 1 sees $x_1$, predicts $x_2$\n- Position 2 sees $x_1, x_2$, predicts $x_3$\n- $\\vdots$\n- Position $T-1$ sees $x_1, \\ldots, x_{T-1}$, predicts $x_T$\n\nThis is extraordinarily efficient. Processing a batch of sequences with 2,048 tokens each produces 2,047 gradient signals per sequence — the equivalent of 2,047 separate training examples."
    },
    // Step 8: MC — teacher forcing
    {
      type: "mc",
      question: "During training with teacher forcing, the model predicts $P(x_5 \\mid x_1, x_2, x_3, x_4)$ and gets it wrong — it assigns 0.01 to the correct $x_5$. When computing $P(x_6 \\mid x_1, \\ldots, x_5)$, which tokens does position 6 see as input?",
      options: [
        "The ground-truth tokens $x_1, \\ldots, x_5$ — teacher forcing always provides the correct prefix, even though the model's prediction of $x_5$ was wrong",
        "The tokens $x_1, x_2, x_3, x_4$ followed by the model's own prediction $\\hat{x}_5$ — the model must learn to handle its own errors during training",
        "The tokens $x_1, x_2, x_3, x_4$ only — position $x_5$ is masked out because the model's prediction was incorrect, preventing error propagation",
        "A probabilistic mixture: $0.01 \\cdot x_5 + 0.99 \\cdot \\hat{x}_5$, blending the ground truth and prediction weighted by the model's confidence"
      ],
      correct: 0,
      explanation: "Teacher forcing ALWAYS provides ground-truth tokens as input, regardless of the model's predictions. Position 6 sees the real $x_5$, not the model's (incorrect) prediction. This is what enables parallel computation: all inputs are known before the forward pass begins, so all positions can be computed simultaneously. The downside is **exposure bias** — during inference, the model must consume its OWN predictions (which may be wrong), creating a distribution shift from training conditions."
    },
    // Step 9: Exposure bias
    {
      type: "info",
      title: "Exposure Bias: The Train-Inference Gap",
      content: "Teacher forcing creates a mismatch between training and inference:\n\n**Training**: The model always sees the correct prefix — $P_\\theta(x_t \\mid x_1^*, x_2^*, \\ldots, x_{t-1}^*)$\n\n**Inference**: The model sees its own previous predictions — $P_\\theta(x_t \\mid \\hat{x}_1, \\hat{x}_2, \\ldots, \\hat{x}_{t-1})$\n\nThis is called **exposure bias** (Bengio et al., 2015): the model is never \"exposed\" to its own errors during training, so when an error occurs at inference time, the model enters unfamiliar territory and may generate increasingly incoherent text as errors compound.\n\nIn practice, exposure bias is less catastrophic for large language models than early work suggested, for two reasons:\n\n1. **At scale, per-token accuracy is very high** — if the model predicts the right token 98% of the time, error compounding is slow\n2. **Sampling strategies** like nucleus (top-$p$) sampling avoid low-probability regions where errors are most likely to cascade\n\nHowever, exposure bias remains relevant for structured generation tasks (code, math, logical reasoning) where a single wrong token can invalidate the entire output."
    },
    // Step 10: MC — exposure bias reasoning
    {
      type: "mc",
      question: "A language model generates a 100-token response during inference. Each token is predicted correctly with probability 0.95 (independent). What is the approximate probability that ALL 100 tokens are correct?",
      options: [
        "Approximately 95% — since each token is 95% correct, the sequence is also about 95% correct",
        "Approximately 5% — computed as $0.95^{100} \\approx 0.006$, but rounding up since some errors are recoverable",
        "Approximately 0.6% — computed as $0.95^{100} \\approx 0.006$, showing how small per-token errors compound rapidly over long sequences",
        "Exactly 0% — any model with less than 100% per-token accuracy will always produce at least one error in a 100-token sequence"
      ],
      correct: 2,
      explanation: "With independent 95% accuracy per token: $P(\\text{all correct}) = 0.95^{100} \\approx 0.006 = 0.6\\%$. This illustrates the severity of error compounding. Even very high per-token accuracy leads to poor sequence-level accuracy over long outputs. In practice, tokens are NOT independent (errors are correlated), and sampling strategies help avoid cascading failures. But this calculation shows why exposure bias matters: at training time, the model sees perfect prefixes 100% of the time, while at inference time, errors in the prefix can derail subsequent predictions."
    },
    // Step 11: Why next-token prediction works
    {
      type: "info",
      title: "Why Is Next-Token Prediction So Effective?",
      content: "Next-token prediction might seem too simple to produce capable AI systems. After all, we're just predicting the next word. But consider what accurate next-token prediction **requires** the model to learn.\n\nTo predict the next token well across a diverse training corpus, the model must implicitly learn:\n\n**Syntax**: Predicting \"The cat _\" requires knowing that a verb or adverb likely follows a noun phrase.\n\n**Semantics**: Predicting \"The capital of France is _\" requires knowing facts about the world.\n\n**Reasoning**: Predicting the next step in \"If $x > 5$ and $x < 10$, then $x$ _\" requires logical deduction.\n\n**Style and pragmatics**: Predicting the next token in a legal document vs. a tweet requires understanding register, formality, and conventions.\n\nThe cross-entropy objective places a cost on EVERY incorrectly predicted token. Across trillions of tokens from diverse sources, this pressure forces the model to build increasingly sophisticated internal representations of syntax, semantics, factual knowledge, and reasoning — not because we asked it to, but because these representations help minimize the loss.\n\nThis connects to the **information-theoretic view**: a model that achieves low cross-entropy is implicitly an efficient compressor of data (Shannon, 1948). Better compression requires deeper understanding of the data's structure."
    },
    // Step 12: MC — understanding implicit learning
    {
      type: "mc",
      question: "A language model is trained exclusively on next-token prediction over a corpus of mathematical proofs. Which capability would this training objective most naturally incentivize the model to develop?",
      options: [
        "The ability to verify whether a proof is correct, since the model needs to distinguish valid from invalid proof steps to maintain low loss on well-formed mathematical text",
        "The ability to generate novel theorems, since next-token prediction directly optimizes for producing statements that have never appeared in the training data",
        "Internal representations of logical dependencies between proof steps, since predicting the next line of a proof requires understanding what has been established and what can be concluded",
        "A formal understanding of mathematical axioms as explicit symbolic rules stored in a lookup table, since this is the most efficient way to minimize cross-entropy on mathematical text"
      ],
      correct: 2,
      explanation: "To predict the next step of a proof, the model must understand what has been proven so far and what logically follows. This creates pressure to build internal representations of logical dependency and deductive structure. The model doesn't need to \"verify\" proofs explicitly (option A) — it needs to predict what comes next, which requires implicit logical reasoning. It also doesn't need to generate novel theorems (option B) or use explicit symbolic rules (option D) — it needs flexible representations that capture the patterns of valid mathematical reasoning across many proof structures."
    },
    // Step 13: Practical considerations
    {
      type: "info",
      title: "Practical Considerations in LLM Pretraining",
      content: "Several practical details shape how next-token prediction is applied in modern LLM training:\n\n**Sequence packing**: Training sequences are typically fixed-length (e.g., 2,048 or 8,192 tokens). Multiple shorter documents are concatenated and separated by special tokens (like `<|endoftext|>`), so no computation is wasted on padding. The model learns to treat document boundaries as resets in context.\n\n**Loss masking**: In some settings, we don't want to compute loss on every token. For example, in instruction fine-tuning, we only compute loss on the **response** tokens, not the instruction tokens. The model still sees the instruction as context but isn't penalized for not predicting it token-by-token.\n\n**Token weighting**: Not all tokens are equally informative. Function words (\"the\", \"is\", \"a\") are easy to predict and contribute little gradient signal. Content words and rare tokens drive most of the learning. Some training approaches upweight harder tokens, though standard practice uses uniform weighting.\n\n**The role of data**: Since MLE drives $p_\\theta$ toward $p_{\\text{data}}$, the training data distribution directly determines what the model learns. Data quality, diversity, and deduplication are just as important as model architecture — the objective will faithfully learn whatever distribution the data defines, including any biases or errors."
    },
    // Step 14: MC — practical reasoning
    {
      type: "mc",
      question: "During instruction fine-tuning, loss is computed only on the response tokens, not the instruction tokens. If we instead computed loss on both instruction and response tokens, what would be the primary consequence?",
      options: [
        "The model would learn to generate instructions as well as responses, making it equally likely to produce user-style prompts as assistant-style answers in its output distribution",
        "Training would diverge because instruction tokens have a completely different distribution from response tokens, causing conflicting gradient signals that prevent convergence",
        "The model would become better at following instructions because it has additional gradient signal from the instruction tokens that helps it understand the task format",
        "The gradient budget would be diluted — easy-to-predict instruction formatting tokens would contribute noise that washes out the learning signal from the harder, more valuable response tokens"
      ],
      correct: 3,
      explanation: "Instruction tokens often follow formulaic patterns (\"You are a helpful assistant. Answer the following question:\") that are easy to predict and provide minimal learning signal. Including them in the loss means a significant fraction of gradients come from these low-information tokens, diluting the signal from response tokens where the real learning happens. Loss masking focuses the model's learning capacity on what matters: generating high-quality responses. Option A overstates the effect — the model wouldn't confuse its role just from seeing instruction loss. Option B is wrong because training can still converge; it's just less efficient."
    }
  ]
};
