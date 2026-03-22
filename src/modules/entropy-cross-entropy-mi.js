// Module: Entropy, Cross-Entropy, Mutual Information, Perplexity & Label Smoothing
// Section 0.2: Probability, statistics & information theory
// Three difficulty tracks targeting flagged gaps: perplexity interpretation,
// mutual information, conditional entropy in LM context, chain rule of MI,
// and label smoothing.

export const entropyEasy = {
  id: "0.2-entropy-easy",
  sectionId: "0.2",
  title: "Entropy & Cross-Entropy in Language Models",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 15,
  steps: [
    {
      type: "info",
      title: "Entropy: Expected Surprise",
      content: "**Entropy** measures the average surprise (or uncertainty) in a random variable:\n\n$$H(X) = -\\sum_x P(x) \\log P(x) = \\mathbb{E}_P[-\\log P(X)]$$\n\nEach outcome $x$ carries surprise $-\\log P(x)$: rare events are surprising, common ones are not. Entropy is the *expected* surprise.\n\n**High entropy** means the distribution is spread out (close to uniform) — you are very uncertain about the outcome. **Low entropy** means probability mass is concentrated on a few outcomes — you can predict well.\n\nFor a discrete distribution over $K$ outcomes:\n- **Maximum entropy** $= \\log K$, achieved by the uniform distribution $P(x) = 1/K$.\n- **Minimum entropy** $= 0$, achieved when all mass is on a single outcome.\n\nEntropy answers a precise operational question: it is the minimum average number of bits needed to encode a sample from $P$ (Shannon's source coding theorem)."
    },
    {
      type: "mc",
      question: "Distribution A: $P = (0.25, 0.25, 0.25, 0.25)$. Distribution B: $P = (0.97, 0.01, 0.01, 0.01)$. Which has higher entropy?",
      options: [
        "Distribution B — it has one dominant outcome, so more information content",
        "Distribution A — it is uniform, so maximum uncertainty",
        "They have equal entropy because both are over 4 outcomes",
        "Cannot determine without knowing what the outcomes represent"
      ],
      correct: 1,
      explanation: "Distribution A is uniform over 4 outcomes, giving $H(A) = \\log_2 4 = 2$ bits — the maximum possible for 4 outcomes. Distribution B concentrates almost all mass on one outcome: $H(B) \\approx 0.97 \\cdot 0.044 + 3 \\cdot 0.01 \\cdot 6.64 \\approx 0.24$ bits. High concentration means low uncertainty, hence low entropy."
    },
    {
      type: "info",
      title: "Cross-Entropy: When Your Model Is Wrong",
      content: "**Cross-entropy** measures the expected surprise when you use model $Q$ to encode data that actually follows distribution $P$:\n\n$$H(P, Q) = -\\sum_x P(x) \\log Q(x) = \\mathbb{E}_P[-\\log Q(X)]$$\n\nIf $Q = P$, this reduces to plain entropy $H(P)$. If $Q \\neq P$, you pay extra bits — and the excess is exactly the KL divergence:\n\n$$H(P, Q) = H(P) + \\text{KL}(P \\| Q)$$\n\nThis decomposition is the reason **language model training loss is cross-entropy**. When we train a model $Q_\\theta$ by minimizing $-\\sum_t \\log Q_\\theta(w_t \\mid w_{<t})$ over the training corpus, we are minimizing $H(P_{\\text{data}}, Q_\\theta)$. Since $H(P_{\\text{data}})$ is a constant (the intrinsic entropy of language), minimizing cross-entropy is equivalent to minimizing $\\text{KL}(P_{\\text{data}} \\| Q_\\theta)$."
    },
    {
      type: "mc",
      question: "If $P$ is the true next-token distribution and $Q$ is your language model, what does the gap $H(P, Q) - H(P)$ equal?",
      options: [
        "The mutual information $I(P; Q)$",
        "The KL divergence $\\text{KL}(P \\| Q)$",
        "The reverse KL $\\text{KL}(Q \\| P)$",
        "The Jensen-Shannon divergence $\\text{JS}(P \\| Q)$"
      ],
      correct: 1,
      explanation: "By definition: $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$, so $H(P, Q) - H(P) = \\text{KL}(P \\| Q)$. This is the *forward* KL — the penalty your model pays for deviating from the true distribution. It is always $\\geq 0$, with equality iff $Q = P$. This is why cross-entropy loss is a sound training objective: minimizing it drives $Q$ toward $P$."
    },
    {
      type: "info",
      title: "Conditional Entropy: The Irreducible Floor",
      content: "**Conditional entropy** $H(X \\mid Y)$ measures how uncertain $X$ remains after observing $Y$:\n\n$$H(X \\mid Y) = -\\sum_{x, y} P(x, y) \\log P(x \\mid y) = \\mathbb{E}_{(X,Y)}[-\\log P(X \\mid Y)]$$\n\nFor language models, the crucial quantity is $H(W_t \\mid W_{<t})$ — the entropy of the next token given all preceding context. This is the **intrinsic unpredictability of language** at position $t$.\n\nA perfect model $Q^* = P$ achieves training loss:\n\n$$H(P_{\\text{data}}, Q^*) = H(P_{\\text{data}}) = \\mathbb{E}[-\\log P(W_t \\mid W_{<t})]$$\n\nNo model can do better than this, no matter how large or well-trained. This quantity is the **Bayes-optimal loss** — the irreducible noise floor. It is nonzero because language is genuinely stochastic: given a context, multiple valid continuations exist.\n\nThe chain rule of entropy decomposes sequence entropy: $H(W_1, \\ldots, W_T) = \\sum_{t=1}^T H(W_t \\mid W_{<t})$. Each token's conditional entropy contributes to the total."
    },
    {
      type: "mc",
      question: "A language model's per-token training loss can never go below which quantity?",
      options: [
        "Zero — a sufficiently large model could memorize every sequence",
        "$H(W_t \\mid W_{<t})$, the conditional entropy of the next token given context",
        "$H(W_t)$, the marginal entropy of the token distribution",
        "$\\log K$ where $K$ is the vocabulary size"
      ],
      correct: 1,
      explanation: "The minimum achievable cross-entropy is $H(P_{\\text{data}}, Q^*) = H(P_{\\text{data}})$, which equals $\\mathbb{E}[H(W_t \\mid W_{<t})]$ averaged over positions. This is the conditional entropy of language itself — the irreducible uncertainty given perfect context modeling. Even memorizing the training set doesn't help: the Bayes-optimal predictor must spread probability across all valid continuations, not just the one that appeared in the corpus."
    },
    {
      type: "info",
      title: "Perplexity: Making Cross-Entropy Interpretable",
      content: "Cross-entropy in nats or bits is hard to interpret. **Perplexity** converts it to an intuitive quantity:\n\n$$\\text{PPL} = 2^{H(P, Q)} \\quad \\text{(if using } \\log_2\\text{)} \\qquad \\text{or} \\qquad \\text{PPL} = e^{H(P, Q)} \\quad \\text{(if using } \\ln\\text{)}$$\n\nPerplexity is the **effective vocabulary size** the model is choosing from. Concretely:\n\n- **PPL = 1**: The model is perfectly certain of every next token (entropy = 0). This only happens if language is fully deterministic.\n- **PPL = 50**: On average, the model's uncertainty is equivalent to choosing uniformly from 50 equally likely tokens.\n- **PPL = 50,000**: The model is as confused as if every token in a 50K vocabulary were equally likely — it has learned nothing.\n\nPerplexity is **exponential** in cross-entropy, so small improvements in loss translate to large perplexity reductions. Reducing cross-entropy from 5.0 to 4.5 bits cuts perplexity from $2^5 = 32$ to $2^{4.5} \\approx 22.6$ — a 30% reduction in effective uncertainty."
    },
    {
      type: "mc",
      question: "Model A has perplexity 30 and Model B has perplexity 90. How much better is A than B in cross-entropy (bits)?",
      options: [
        "60 bits — the difference in perplexity",
        "$\\log_2(90) - \\log_2(30) = \\log_2(3) \\approx 1.58$ bits",
        "$\\log_2(90/30) = \\log_2(3) \\approx 1.58$ bits per token, but the models might have different vocabularies",
        "Cannot compare without knowing the vocabulary size"
      ],
      correct: 1,
      explanation: "Cross-entropy $= \\log_2(\\text{PPL})$, so the difference is $\\log_2(90) - \\log_2(30) = \\log_2(90/30) = \\log_2(3) \\approx 1.58$ bits per token. This means Model B wastes 1.58 extra bits per token compared to A. Note that perplexity ratios correspond to cross-entropy *differences* (because $\\log$ turns ratios into differences) — a 3x perplexity improvement always equals $\\log_2 3 \\approx 1.58$ bits, regardless of the absolute values."
    },
    {
      type: "info",
      title: "The Entropy Rate of English",
      content: "Shannon estimated the **entropy rate** of English at about **1.0–1.5 bits per character** (through human prediction experiments). Compare this to the maximum:\n\n$$H_{\\text{uniform}} = \\log_2 26 \\approx 4.7 \\text{ bits/char}$$\n\nif every letter were equally likely. English is therefore roughly **70% redundant** — most of the \"information capacity\" of the character stream is consumed by spelling patterns, grammar, and semantic constraints.\n\nThis redundancy is exactly what language models exploit. A model that captures these patterns compresses language toward its entropy rate. Modern LLMs achieve cross-entropy below 1 bit per character on many benchmarks, approaching (and for some domains, roughly matching) Shannon's estimates.\n\nAt the token level (with BPE tokenization, average ~4 chars/token), the entropy rate is roughly $4 \\times 1.3 \\approx 5.2$ bits/token. GPT-4-class models achieve perplexities around 6–10 on standard benchmarks, corresponding to $\\log_2(8) = 3$ bits/token — well below the character-level entropy rate, because subword tokenization captures spelling redundancy in the tokenizer itself."
    },
    {
      type: "mc",
      question: "A uniform model over a 50,000-token vocabulary assigns $P(w) = 1/50000$ for every token regardless of context. What is its perplexity?",
      options: [
        "$\\log_2(50000) \\approx 15.6$",
        "$50000$",
        "$\\sqrt{50000} \\approx 224$",
        "$50000^2 = 2.5 \\times 10^9$"
      ],
      correct: 1,
      explanation: "Cross-entropy of a uniform model is $-\\sum_w P(w) \\log(1/50000) = \\log(50000)$. So $\\text{PPL} = e^{\\log 50000} = 50000$. The uniform model's perplexity equals the vocabulary size — it is equivalent to rolling a 50,000-sided die. This is the worst possible perplexity for this vocabulary, confirming the interpretation of perplexity as \"effective vocabulary size.\""
    },
    {
      type: "info",
      title: "Temperature and Entropy Control",
      content: "Given logits $z_1, \\ldots, z_K$, the temperature-scaled softmax is:\n\n$$P(i) = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}$$\n\nTemperature $\\tau$ directly controls the **entropy** of the output distribution:\n\n- $\\tau \\to 0$: the softmax sharpens to a one-hot vector on $\\arg\\max_i z_i$. Entropy $\\to 0$. This is **greedy decoding** — deterministic, repetitive, but high-confidence.\n- $\\tau = 1$: the standard softmax — the model's calibrated distribution.\n- $\\tau \\to \\infty$: all logit differences vanish, approaching the uniform distribution. Entropy $\\to \\log K$. Output is random noise.\n\nThe relationship is monotonic: increasing $\\tau$ strictly increases entropy (for non-degenerate logits). This makes temperature a direct entropy dial.\n\nIn practice, $\\tau \\in [0.7, 1.0]$ for focused text and $\\tau \\in [1.0, 1.5]$ for creative/diverse generation. The optimal temperature balances quality (low entropy, coherent) against diversity (high entropy, varied)."
    },
    {
      type: "mc",
      question: "During generation, you increase the temperature $\\tau$ from 0.7 to 1.5. What happens to the perplexity of the model's output distribution at each decoding step?",
      options: [
        "Perplexity decreases because the model becomes more creative",
        "Perplexity increases because the output distribution becomes more spread out (higher entropy)",
        "Perplexity stays the same — temperature only affects which token is sampled, not the distribution's entropy",
        "Perplexity first increases then decreases as temperature crosses 1.0"
      ],
      correct: 1,
      explanation: "Perplexity $= 2^{H}$ is monotonically increasing in entropy. Higher temperature → higher entropy → higher perplexity. At $\\tau = 0.7$, the distribution is sharper than the base model (lower PPL). At $\\tau = 1.5$, the distribution is flatter (higher PPL). The model is effectively \"choosing from more options\" at each step — which is exactly what the perplexity-as-effective-vocabulary-size interpretation says. Note: this is the *per-step* perplexity of the *generation* distribution, not the model's perplexity on held-out evaluation data."
    }
  ]
};

export const entropyMedium = {
  id: "0.2-entropy-medium",
  sectionId: "0.2",
  title: "Mutual Information & What Representations Learn",
  difficulty: "medium",
  moduleType: "learning",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Mutual Information: Shared Uncertainty",
      content: "**Mutual information** quantifies the information shared between two random variables:\n\n$$I(X; Y) = H(X) - H(X \\mid Y) = H(Y) - H(Y \\mid X)$$\n\nEquivalently, it is the KL divergence between the joint and the product of marginals:\n\n$$I(X; Y) = \\text{KL}\\big(P(X, Y) \\,\\|\\, P(X) P(Y)\\big) = \\sum_{x, y} P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}$$\n\nKey properties:\n- **Non-negative**: $I(X; Y) \\geq 0$, with equality iff $X \\perp Y$.\n- **Symmetric**: $I(X; Y) = I(Y; X)$. Knowing $X$ reduces uncertainty about $Y$ by the same amount that knowing $Y$ reduces uncertainty about $X$.\n- **Bounded**: $I(X; Y) \\leq \\min(H(X), H(Y))$. You can't learn more about $X$ from $Y$ than there is uncertainty in $X$ to begin with.\n\n$I(X; Y) = 0$ means the variables are independent — knowing one tells you nothing about the other. $I(X; Y) = H(X)$ means $X$ is fully determined by $Y$."
    },
    {
      type: "mc",
      question: "If $X$ and $Y$ are independent, what is $I(X; Y)$?",
      options: [
        "$H(X) + H(Y)$ — their total uncertainty",
        "$H(X) \\cdot H(Y)$",
        "$0$, because $H(X \\mid Y) = H(X)$ when $X \\perp Y$",
        "$-\\infty$, because the log-ratio in KL is undefined"
      ],
      correct: 2,
      explanation: "When $X \\perp Y$: $P(X, Y) = P(X)P(Y)$, so the KL divergence $\\text{KL}(P(X,Y) \\| P(X)P(Y)) = 0$. Equivalently, $H(X \\mid Y) = H(X)$ — observing $Y$ doesn't reduce uncertainty about $X$ — so $I(X;Y) = H(X) - H(X) = 0$. Independence means zero shared information."
    },
    {
      type: "info",
      title: "MI in Representation Learning",
      content: "In representation learning, an encoder maps input $X$ to representation $Z = f_\\theta(X)$. Two MI quantities matter:\n\n**$I(X; Z)$**: How much the representation remembers about the input. A lossless encoder has $I(X; Z) = H(X)$; a constant encoder has $I(X; Z) = 0$.\n\n**$I(Z; Y)$**: How useful the representation is for predicting target $Y$. This is what task performance depends on.\n\nThe **data processing inequality** (DPI) constrains what's possible: for any Markov chain $Y \\to X \\to Z$:\n\n$$I(Y; Z) \\leq I(Y; X)$$\n\nYou can never create information that wasn't in the input. The representation $Z$ can be *at most* as informative about $Y$ as the raw input $X$ is.\n\nThe **information bottleneck** objective formalizes the trade-off: minimize $I(X; Z)$ (compress) while maximizing $I(Z; Y)$ (preserve task signal). A good representation discards task-irrelevant information (low $I(X; Z)$ relative to $H(X)$) while retaining everything the task needs (high $I(Z; Y)$)."
    },
    {
      type: "mc",
      question: "A representation $Z = f(X)$ is a deterministic function of $X$. Can $I(X; Z)$ be zero?",
      options: [
        "No — a deterministic function always preserves some information",
        "Yes — if $Z$ is a constant function (maps all inputs to the same value), then $I(X; Z) = 0$",
        "Yes — if $f$ is a non-invertible hash function, MI is always zero",
        "It depends on the dimensionality of $Z$ relative to $X$"
      ],
      correct: 1,
      explanation: "$I(X; Z) = 0$ requires $X \\perp Z$, meaning $Z$ carries no information about $X$. For a deterministic $Z = f(X)$, this happens only if $f$ is constant — $Z$ takes the same value regardless of input. Any non-constant deterministic function has $I(X; Z) > 0$, because knowing $Z$ eliminates at least some uncertainty about $X$. A hash function that maps different inputs to different outputs would actually have high MI, even though it's not invertible."
    },
    {
      type: "info",
      title: "Data Processing Inequality: Information Only Flows Downhill",
      content: "The **data processing inequality** (DPI) is one of the most powerful results in information theory:\n\nIf $X \\to Y \\to Z$ is a Markov chain (i.e., $Z$ depends on $X$ only through $Y$), then:\n\n$$I(X; Z) \\leq I(X; Y)$$\n\nEach processing step can only **lose** information, never create it.\n\nIn a neural network, the layers form a Markov chain:\n\n$$X \\to h_1 \\to h_2 \\to \\cdots \\to h_L \\to \\hat{Y}$$\n\nSo DPI gives us: $I(X; h_1) \\geq I(X; h_2) \\geq \\cdots \\geq I(X; h_L)$. Deeper layers can only have *less* mutual information with the input.\n\nThis has a striking interpretation: as information flows through the network, each layer acts as an information bottleneck. Early layers retain most input information; deeper layers progressively discard task-irrelevant details. The **information plane** plots $I(X; h_l)$ vs. $I(Y; h_l)$ for each layer $l$ — a well-trained network should show layers moving toward the bottom-right: low input MI (compressed) but high task MI (useful).\n\n*Caveat*: For deterministic networks with invertible activations (like ReLU on distinct inputs), DPI is technically satisfied with equality. The \"information compression\" story is nuanced and debated."
    },
    {
      type: "mc",
      question: "Layer 3 of a network has $I(X; h_3) = 5$ bits. What can you say about $I(X; h_5)$ for layer 5?",
      options: [
        "$I(X; h_5) = 5$ bits — information is conserved across layers",
        "$I(X; h_5) \\leq 5$ bits — the data processing inequality prevents information from increasing",
        "$I(X; h_5) \\geq 5$ bits — deeper layers learn richer representations",
        "Nothing — MI between non-adjacent layers is not constrained by DPI"
      ],
      correct: 1,
      explanation: "Since $X \\to h_3 \\to h_4 \\to h_5$ is a Markov chain, DPI gives $I(X; h_5) \\leq I(X; h_3) = 5$ bits. Each additional layer can only preserve or lose information about the input. The *useful* information $I(Y; h_5)$ might be concentrated and refined, but the *total* input information $I(X; h_5)$ cannot exceed what layer 3 retained."
    },
    {
      type: "info",
      title: "Chain Rule of Mutual Information",
      content: "Just as entropy has a chain rule, so does MI:\n\n$$I(X; Y, Z) = I(X; Y) + I(X; Z \\mid Y)$$\n\nThe information $X$ shares with $(Y, Z)$ jointly equals the information from $Y$ alone plus the **additional** information $Z$ provides once $Y$ is already known.\n\nHere, the conditional MI is:\n\n$$I(X; Z \\mid Y) = H(X \\mid Y) - H(X \\mid Y, Z)$$\n\nThis decomposition is essential for understanding what different parts of the context contribute in language modeling.\n\nConsider a language model with recent tokens $Y$ and distant tokens $Z$ in the context window. The chain rule tells us:\n\n$$I(W_{\\text{next}}; Y, Z) = I(W_{\\text{next}}; Y) + I(W_{\\text{next}}; Z \\mid Y)$$\n\nThe second term $I(W_{\\text{next}}; Z \\mid Y)$ measures how much *additional* prediction signal the distant context provides beyond what local context already gives. If this is near zero, the distant tokens are redundant for prediction — they add nothing beyond what nearby context already captures."
    },
    {
      type: "mc",
      question: "In a language model's context window, $Y$ = recent tokens and $Z$ = distant tokens. If $I(W_{\\text{next}}; Z \\mid Y) \\approx 0$, what does this mean?",
      options: [
        "The distant tokens $Z$ are irrelevant to predicting $W_{\\text{next}}$ even without recent context",
        "The distant context $Z$ adds essentially no prediction signal beyond what the recent context $Y$ already provides",
        "The context window is too short to be useful",
        "The model should attend more strongly to distant tokens"
      ],
      correct: 1,
      explanation: "$I(W_{\\text{next}}; Z \\mid Y) \\approx 0$ means $H(W_{\\text{next}} \\mid Y) \\approx H(W_{\\text{next}} \\mid Y, Z)$ — the residual uncertainty about the next token is the same whether or not you have distant context, *given that you already have recent context*. The distant tokens are redundant for prediction, not necessarily irrelevant in isolation. This is why many sequences can be predicted well with short contexts — and why efficient architectures that limit long-range attention (e.g., sliding window attention) often lose little performance."
    },
    {
      type: "info",
      title: "Contrastive Learning Maximizes MI Bounds",
      content: "Contrastive methods like **CLIP** and **SimCLR** learn representations by maximizing mutual information between paired views.\n\nCLIP trains image encoder $f$ and text encoder $g$ so that matching (image, caption) pairs have high similarity while non-matching pairs have low similarity. The **InfoNCE** loss for a batch of $N$ pairs is:\n\n$$\\mathcal{L}_{\\text{InfoNCE}} = -\\frac{1}{N} \\sum_{i=1}^N \\log \\frac{\\exp(\\text{sim}(f(x_i), g(t_i)) / \\tau)}{\\sum_{j=1}^N \\exp(\\text{sim}(f(x_i), g(t_j)) / \\tau)}$$\n\nOord et al. (2018) showed this loss satisfies:\n\n$$I(X; Y) \\geq \\log N - \\mathcal{L}_{\\text{InfoNCE}}$$\n\nSo minimizing InfoNCE maximizes a **lower bound** on $I(X; Y)$. The bound's tightness depends on batch size $N$: the maximum estimable MI is $\\log N$. This is why contrastive methods use large batches — a batch of 32,768 allows estimating up to $\\log(32768) = 15$ bits of MI, while a batch of 64 caps you at $\\log(64) = 6$ bits.\n\nThis also explains why CLIP's performance scales with batch size: larger batches give a tighter bound on the true MI, enabling the model to capture more fine-grained image-text correspondences."
    },
    {
      type: "mc",
      question: "In CLIP training with batch size $N = 1024$, what is the maximum mutual information $I(\\text{image}; \\text{text})$ that the InfoNCE loss can estimate?",
      options: [
        "$1024$ bits",
        "$\\sqrt{1024} = 32$ bits",
        "$\\log_2(1024) = 10$ bits",
        "$1024 \\cdot \\log_2(1024) = 10240$ bits"
      ],
      correct: 2,
      explanation: "The InfoNCE bound is $I(X; Y) \\geq \\log N - \\mathcal{L}$. Even when the loss is driven to zero, the bound saturates at $\\log N$. With $N = 1024$: $\\log_2(1024) = 10$ bits. If the true MI exceeds 10 bits, the InfoNCE loss with this batch size simply cannot distinguish — the bound is too loose. This is why CLIP uses batch sizes of 32K+: $\\log_2(32768) = 15$ bits, allowing the model to capture finer-grained correspondences."
    },
    {
      type: "info",
      title: "The Challenge of MI Estimation",
      content: "Mutual information is conceptually clean but **computationally hard** to estimate in high dimensions.\n\nFor discrete distributions with small support, you can compute MI exactly from counts. But for continuous or high-dimensional variables (like neural network representations), MI estimation is a research problem in itself.\n\nThe core difficulty: MI involves a ratio of joint to marginal densities, $\\log \\frac{P(x,y)}{P(x)P(y)}$, and estimating densities in high dimensions is exponentially hard.\n\nVariational approaches provide bounds:\n- **InfoNCE** (lower bound): tractable but limited by batch size.\n- **MINE** (Mutual Information Neural Estimation): uses a learned critic network to estimate a tight lower bound, but has high variance.\n- **BA bound** (Barber-Agakov): provides an upper bound using a learned conditional $Q(X \\mid Z)$.\n\nA key result from McAllester & Statos (2020): **any MI estimator that provides a high-confidence lower bound on MI must have sample complexity exponential in the true MI value.** In other words, the harder the problem (higher true MI), the more data you need to confirm it — and the relationship is exponential, not polynomial."
    },
    {
      type: "mc",
      question: "Why is estimating MI between a 768-dimensional representation and its input fundamentally hard?",
      options: [
        "768 dimensions is too many for any neural network to process",
        "MI is not well-defined for continuous random variables",
        "Reliable MI estimation requires sample complexity that grows exponentially with the true MI value",
        "The representation must be discrete for MI to be computable"
      ],
      correct: 2,
      explanation: "MI is well-defined for continuous variables (as a KL divergence between densities), but *estimating* it reliably is the problem. The McAllester-Statos impossibility result shows that any estimator providing a high-confidence lower bound needs exponentially many samples in the true MI. A 768-dim representation can have very high MI with its input (potentially hundreds of bits for a deterministic encoder), making reliable estimation require astronomical sample sizes. This is why empirical \"information plane\" analyses of deep networks should be interpreted with caution."
    }
  ]
};

export const entropyHard = {
  id: "0.2-entropy-hard",
  sectionId: "0.2",
  title: "Label Smoothing, Calibration & the Entropy-Accuracy Tradeoff",
  difficulty: "hard",
  moduleType: "learning",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "Label Smoothing: Softening the Target",
      content: "Standard classification training uses **one-hot** targets: $P(y) = \\mathbf{1}[y = y^*]$. The cross-entropy loss drives the model to output $Q(y^*) \\to 1$, which requires the logit for the correct class to go to $+\\infty$ relative to all others.\n\n**Label smoothing** (Szegedy et al., 2016) replaces the one-hot target with a mixture of the one-hot and the uniform distribution:\n\n$$P'(y) = (1 - \\alpha) \\cdot \\mathbf{1}[y = y^*] + \\frac{\\alpha}{K}$$\n\nwhere $\\alpha \\in (0, 1)$ is the smoothing parameter and $K$ is the number of classes. This assigns probability $(1 - \\alpha + \\alpha/K)$ to the correct class and $\\alpha/K$ to each incorrect class.\n\nThe cross-entropy with smoothed targets decomposes as:\n\n$$H(P', Q) = (1 - \\alpha) \\cdot H(\\text{one-hot}, Q) + \\alpha \\cdot H(\\text{uniform}, Q)$$\n\n$$= -(1 - \\alpha) \\log Q(y^*) - \\frac{\\alpha}{K} \\sum_k \\log Q(k)$$\n\nThe second term $-\\frac{\\alpha}{K} \\sum_k \\log Q(k)$ penalizes the model for being too confident — it's an **entropy regularizer** that prevents the output distribution from collapsing to a point mass."
    },
    {
      type: "mc",
      question: "Without label smoothing ($\\alpha = 0$), what are the optimal logits for the correct class to minimize cross-entropy with a one-hot target?",
      options: [
        "The logit should equal 1.0",
        "The logit should equal $\\log K$ where $K$ is the number of classes",
        "The logit should go to $+\\infty$ (driving $Q(y^*) \\to 1$ requires unbounded logits)",
        "The logit should equal the log-prior $\\log P(y^*)$"
      ],
      correct: 2,
      explanation: "With a one-hot target, the loss is $-\\log Q(y^*)$, which is minimized by $Q(y^*) \\to 1$. Since $Q(y^*) = \\text{softmax}(z_{y^*})$, achieving $Q(y^*) = 1$ requires $z_{y^*} - z_k \\to \\infty$ for all $k \\neq y^*$. The optimal logits are unbounded — they grow without limit during training. This drives the model toward extreme confidence and encourages memorization, as the gradients never vanish regardless of how confident the model already is."
    },
    {
      type: "info",
      title: "The Finite Optimal Logit Gap",
      content: "With label smoothing, the optimal logits are **finite**. The smoothed target assigns:\n- Correct class: $p^* = 1 - \\alpha + \\alpha/K = 1 - \\alpha(K-1)/K$\n- Each incorrect class: $p_{\\text{wrong}} = \\alpha/K$\n\nThe optimal model output must match this target distribution. For a softmax with logits $z$, the optimal solution has all incorrect logits equal (by symmetry) and the gap between the correct logit $z^*$ and any incorrect logit $z_{\\text{wrong}}$ is:\n\n$$z^* - z_{\\text{wrong}} = \\log \\frac{p^*}{p_{\\text{wrong}}} = \\log \\frac{1 - \\alpha(K-1)/K}{\\alpha/K}$$\n\nFor small $\\alpha$ and large $K$, this simplifies to approximately:\n\n$$\\Delta z \\approx \\log \\frac{(1-\\alpha) \\cdot K}{\\alpha}$$\n\nThis is finite and well-defined. The model converges to a confident-but-not-infinitely-confident prediction. The output distribution retains nonzero entropy — the model maintains a \"soft\" probability over alternatives, which acts as a form of **built-in uncertainty quantification**."
    },
    {
      type: "mc",
      question: "With label smoothing $\\alpha = 0.1$ and vocabulary size $K = 50000$, what is the approximate optimal logit gap between the correct and incorrect classes?",
      options: [
        "$\\log(0.1 \\times 50000) \\approx 8.5$",
        "$\\log(0.9 \\times 50000 / 0.1) = \\log(450000) \\approx 13.0$",
        "$\\log(50000) \\approx 10.8$",
        "$\\log(0.9 / 0.1) = \\log(9) \\approx 2.2$"
      ],
      correct: 1,
      explanation: "The optimal gap is $\\log\\frac{(1-\\alpha)K}{\\alpha} = \\log\\frac{0.9 \\times 50000}{0.1} = \\log(450000) \\approx 13.0$ (using natural log). Without label smoothing, this gap would be $+\\infty$. With $\\alpha = 0.1$, it is a large but finite number. The logits stabilize rather than growing without bound, leading to better-conditioned gradients in the final layers."
    },
    {
      type: "info",
      title: "Calibration: When Confidence Matches Accuracy",
      content: "A model is **calibrated** if its confidence estimates match its actual accuracy:\n\n$$P(\\text{correct} \\mid \\text{confidence} = p) = p$$\n\nIf a calibrated model says \"90% confident\" on 1000 predictions, approximately 900 should be correct.\n\nCross-entropy is a **proper scoring rule**: the expected loss is uniquely minimized when $Q = P_{\\text{true}}$. This means a model trained to global optimum on infinite data with cross-entropy loss would be perfectly calibrated.\n\nBut in practice, three factors break calibration:\n\n**1. Finite data**: The model overfits, learning to be confident on training examples without proportional accuracy on new data.\n\n**2. Overparameterization**: Modern networks can fit the training data with room to spare, and the excess capacity drives predictions toward extreme confidence.\n\n**3. Batch normalization and weight decay**: These interact with softmax in subtle ways that shift the confidence distribution.\n\nThe result: modern neural networks are systematically **overconfident** — their predicted probabilities exceed their actual accuracy. Guo et al. (2017) demonstrated this across architectures: deeper and wider networks are more miscalibrated despite being more accurate."
    },
    {
      type: "mc",
      question: "A model outputs \"90% confident\" on 100 predictions. Of these, 72 are correct. Is the model well-calibrated, overconfident, or underconfident?",
      options: [
        "Well-calibrated — 72% is close enough to 90%",
        "Underconfident — it should have said higher confidence since some might be right by chance",
        "Overconfident — it claimed 90% confidence but only achieved 72% accuracy on those predictions",
        "Cannot determine — calibration requires looking at all confidence levels"
      ],
      correct: 2,
      explanation: "The model said \"90% confident\" but only 72/100 = 72% were correct. It is **overconfident**: its stated confidence (90%) significantly exceeds its actual accuracy (72%) for this confidence bin. The **expected calibration error** (ECE) would flag this as a large deviation. A well-calibrated model at 90% confidence should get approximately 90 out of 100 correct. This overconfidence is the norm for modern deep networks and motivates post-hoc calibration techniques."
    },
    {
      type: "info",
      title: "Temperature Scaling for Post-Hoc Calibration",
      content: "**Temperature scaling** (Guo et al., 2017) is a remarkably simple calibration fix: after training is complete, learn a single scalar $T > 0$ that rescales all logits:\n\n$$Q_{\\text{cal}}(y \\mid x) = \\text{softmax}(z(x) / T)$$\n\nThe temperature $T$ is chosen to minimize the negative log-likelihood on a **held-out validation set**.\n\nCritical insight: temperature scaling is a **monotonic transformation** of probabilities. It preserves the ranking of predictions — the $\\arg\\max$ class does not change. So:\n- **Accuracy is unchanged**: the top-1 prediction stays the same.\n- **Confidence is adjusted**: the softmax entropy increases ($T > 1$) or decreases ($T < 1$).\n\nFor overconfident models (the common case), the optimal $T > 1$: logits are divided by $T > 1$, compressing them toward zero, making the softmax distribution more uniform (higher entropy). This spreads probability from the top class to alternatives, bringing stated confidence in line with actual accuracy.\n\nRemarkably, this **single parameter** often suffices to calibrate even billion-parameter models. Platt scaling, isotonic regression, and other multi-parameter approaches rarely outperform it significantly."
    },
    {
      type: "mc",
      question: "Post-hoc temperature scaling with $T > 1$ does what to a trained model's predictions?",
      options: [
        "Changes which class is predicted (different $\\arg\\max$), improving accuracy",
        "Makes predictions less confident (higher entropy) without changing which class is predicted",
        "Makes predictions more confident (lower entropy), fixing underconfidence",
        "Retrains the final layer to improve calibration"
      ],
      correct: 1,
      explanation: "Dividing logits by $T > 1$ compresses them toward zero, making the softmax output more uniform — higher entropy, lower confidence. Since dividing all logits by the same positive constant preserves their ordering, the $\\arg\\max$ is unchanged: accuracy is identical. Only the *confidence* of the prediction changes. This is why temperature scaling is so appealing: it fixes calibration (confidence matches accuracy) with zero cost to accuracy and only one parameter to tune."
    },
    {
      type: "info",
      title: "The Entropy-Accuracy Tradeoff",
      content: "There is a fundamental tension between entropy and accuracy:\n\n**Maximum entropy** (uniform distribution): worst possible accuracy ($1/K$ for $K$ classes) but maximum coverage — every class gets equal probability. The model \"hedges\" completely, never committing to any prediction. Calibration is trivially perfect (it says $1/K$ confident and is right $1/K$ of the time).\n\n**Minimum entropy** (deterministic): best accuracy on seen patterns (always picks the most likely class with probability 1) but no uncertainty quantification and poor generalization. The model is maximally brittle — it provides no signal about when it might be wrong.\n\nGood models navigate between these extremes:\n- **High confidence** where training data is dense and patterns are clear.\n- **High entropy** where data is sparse, ambiguous, or out-of-distribution.\n\nLabel smoothing and temperature scaling push models away from the minimum-entropy extreme. The KL penalty in RLHF ($\\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$) does the same thing: it prevents the policy from collapsing to a low-entropy distribution that always produces the same high-reward response, maintaining the diversity of the reference model."
    },
    {
      type: "mc",
      question: "In RLHF, the KL penalty $\\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ prevents the policy from becoming too low-entropy. This is conceptually most similar to which training technique?",
      options: [
        "Dropout — both randomly zero out parts of the model",
        "Weight decay — both add a regularization term to the loss",
        "Label smoothing — both prevent the model from becoming overconfident by maintaining minimum entropy in the output distribution",
        "Gradient clipping — both bound the magnitude of updates"
      ],
      correct: 2,
      explanation: "Label smoothing and the RLHF KL penalty are both **entropy regularizers** that prevent distributional collapse:\n\n- **Label smoothing** mixes the target with a uniform distribution, ensuring the model's output retains nonzero entropy. The optimal output is confident but not infinitely so.\n- **RLHF KL penalty** penalizes deviation from $\\pi_{\\text{ref}}$, preventing the policy from collapsing to a low-entropy distribution that always generates the single highest-reward response.\n\nBoth mechanisms maintain diversity/uncertainty by penalizing overconfidence, and both are controlled by a scalar hyperparameter ($\\alpha$ or $\\beta$) that trades off task performance against distributional smoothness. Weight decay regularizes *parameters*; these regularize *output distributions* — a more targeted form of regularization."
    }
  ]
};
