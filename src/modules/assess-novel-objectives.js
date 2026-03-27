// Section B.5: Novel Pretraining Objectives Assessment

export const novelObjectivesAssessment = {
  id: "B.5-assess",
  sectionId: "B.5",
  title: "Assessment: Novel Pretraining Objectives",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Masked language modeling (MLM, as in BERT) and autoregressive language modeling (as in GPT) differ in a fundamental way regarding the joint distribution $P(x_1, \\dots, x_T)$. Which statement is correct?",
      options: [
        "Both model the exact same joint distribution but with different factorizations — the chain rule and the product of conditionals $P(x_t \\mid x_{\\setminus t})$ are mathematically equivalent decompositions of $P(x)$",
        "Autoregressive models factorize the exact joint via the chain rule $P(x) = \\prod_t P(x_t \\mid x_{<t})$, while MLM models conditional distributions $P(x_t \\mid x_{\\setminus t})$ that may not correspond to any valid joint",
        "MLM defines the joint distribution more efficiently because it processes all tokens in parallel, allowing bidirectional context to reduce the number of conditionals needed for the full joint",
        "Autoregressive models can only approximate the joint for left-to-right languages, while MLM captures the true joint by conditioning each token on all other tokens simultaneously"
      ],
      correct: 1,
      explanation: "The chain rule factorization used by autoregressive models is exact: $\\prod_t P(x_t \\mid x_{<t})$ is guaranteed to be a valid joint distribution. MLM trains conditional distributions $P(x_t \\mid x_{\\setminus t})$ (each token given all others). But a set of conditional distributions may be inconsistent — there may be no joint distribution that produces all of them. This makes MLM models unsuitable for generation without additional techniques (e.g., iterative refinement as in BERT-based generation)."
    },
    {
      type: "mc",
      question: "A key practical advantage of MLM over autoregressive LM during pretraining is:",
      options: ["MLM can use a smaller effective vocabulary because the masking procedure clusters rare tokens into shared prediction targets, reducing embedding table size and improving representation quality for low-frequency tokens", "MLM requires less training data because each sequence provides gradients for multiple masked positions simultaneously, extracting more supervision per example than next-token prediction provides per position", "MLM is faster at inference because it generates all tokens in parallel within a single forward pass, avoiding the sequential bottleneck of autoregressive decoding that limits throughput", "MLM processes all tokens bidirectionally — each masked position attends to both left and right context — producing richer representations for downstream understanding tasks like classification and NER"],
      correct: 3,
      explanation: "MLM's bidirectional context is its main strength for representation learning. When predicting a masked token, the model can use information from both sides, producing representations that capture the full context. Autoregressive models only see left context at each position. However, MLM only trains on the ~15% of tokens that are masked (the rest don't contribute to the loss), while autoregressive models get a gradient signal from every token. This makes autoregressive pretraining more compute-efficient per token."
    },
    {
      type: "mc",
      question: "UL2 (Unifying Language Learning Paradigms) proposes training a single model with multiple denoising objectives. Its core insight is:",
      options: ["UL2 eliminates the need for fine-tuning by training on every possible task format during pretraining, making the resulting model directly usable for any downstream task without additional gradient updates", "A single denoising objective is always optimal when its corruption rate is properly tuned, since mixing multiple objectives introduces conflicting gradient signals that degrade performance on each individual task", "Different downstream tasks benefit from different pretraining objectives (short spans for understanding, long spans for generation), so mixing denoising tasks with mode-switching tokens produces a model effective in both regimes", "UL2 replaces maximum likelihood with a reinforcement learning objective for the denoising task, enabling the model to learn adaptive reconstruction strategies that generalize beyond the token-level cross-entropy loss"],
      correct: 2,
      explanation: "UL2 defines three denoising modes: R-denoiser (short spans, like BERT), S-denoiser (sequential/prefix LM), and X-denoiser (extreme/long spans). A special sentinel token tells the model which mode is active. The key insight is that no single denoising objective dominates across all downstream tasks — short-span denoising helps classification and NLU, while long-span and prefix modes help generation. By mixing modes, UL2 produces a single model competitive on both understanding and generation benchmarks."
    },
    {
      type: "mc",
      question: "Diffusion models have been highly successful for continuous data (images, audio). Why is applying diffusion to discrete text fundamentally harder?",
      options: ["Discrete data cannot be interpolated smoothly — there is no natural continuous noise process for tokens, and discrete corruption processes lack the mathematical properties that make continuous diffusion tractable", "Text sequences are too short for the diffusion process to converge effectively, since diffusion models need long inputs to amortize the cost of the multi-step denoising chain across a sufficient number of tokens", "The vocabulary size is too large for the denoising network to predict accurately, since each denoising step must output a calibrated probability vector over the entire vocabulary at every position simultaneously", "Diffusion requires 2D spatial structure that text inherently lacks, since the denoising network architecture relies on spatial convolutions and pooling operations that cannot operate on one-dimensional token sequences"],
      correct: 0,
      explanation: "Continuous diffusion relies on gradually adding Gaussian noise and learning to reverse this process. For discrete tokens, there is no natural analog: you cannot \"slightly noise\" a token. Approaches include: (1) embedding tokens in continuous space and applying continuous diffusion (D3PM, Diffusion-LM), (2) using discrete corruption (token masking/replacement) as forward process (multinomial diffusion), or (3) score-matching on the simplex (MDLM). Each has trade-offs: continuous embeddings disconnect from the discrete structure; discrete corruption requires custom transition matrices."
    },
    {
      type: "mc",
      question: "Non-autoregressive generation (NAG) methods aim to generate all tokens in parallel rather than sequentially. The fundamental challenge they face is:",
      options: [
        "They require significantly more parameters than autoregressive models to achieve equivalent quality, since each position must independently reconstruct the full sequence context without sequential conditioning",
        "They must model the joint distribution without the chain rule's sequential factorization — tokens generated in parallel cannot condition on each other, leading to repetition and incoherence",
        "They cannot use the Transformer architecture because the causal attention mask is incompatible with simultaneous token generation, requiring entirely different architectures like diffusion-based decoders",
        "They are slower than autoregressive models in practice because the parallel decoding overhead and mandatory iterative refinement steps exceed the wall-clock cost of sequential token generation"
      ],
      correct: 1,
      explanation: "Autoregressive models factor $P(x_1, \\dots, x_T)$ into conditionals, each depending on all previous tokens. NAG must model $P(x_1, \\dots, x_T)$ without this sequential structure — often assuming conditional independence given some latent $z$: $P(x \\mid z) = \\prod_t P(x_t \\mid z)$. This \"conditional independence\" assumption is violated when strong dependencies exist between adjacent tokens (e.g., \"New York\" — generating \"New\" and \"York\" independently risks producing \"New London\" or duplicating tokens). Knowledge distillation from AR models, iterative refinement, and CTC losses are common mitigations."
    },
    {
      type: "mc",
      question: "Energy-based models (EBMs) for text define an unnormalized density $p_\\theta(x) \\propto \\exp(-E_\\theta(x))$ over sequences. The central computational challenge of EBMs is:",
      options: ["The energy function $E_\\theta(x)$ is difficult to parameterize for variable-length text because sequence pooling introduces information bottlenecks that prevent the model from capturing fine-grained token interactions", "The energy function must be non-negative by construction, which limits the expressiveness of the model class to distributions with bounded support and prevents modeling of heavy-tailed sequence distributions", "EBMs cannot assign meaningfully different probabilities to sequences of different lengths because the energy scale shifts with sequence length, collapsing probability ratios between short and long texts", "Computing the normalizing constant $Z_\\theta = \\sum_x \\exp(-E_\\theta(x))$ requires summing over all possible sequences (exponential in length and vocabulary), making exact likelihood and gradient computation intractable"],
      correct: 3,
      explanation: "The partition function $Z_\\theta$ sums over all possible token sequences — $|V|^T$ terms for vocabulary $V$ and length $T$. This is astronomically intractable. Training EBMs requires approximations: contrastive divergence (MCMC sampling for negative examples), noise contrastive estimation (NCE), or score matching. For text specifically, MCMC sampling is difficult because the discrete space makes gradient-based sampling (Langevin dynamics) inapplicable. These challenges are why EBMs remain niche for text despite their theoretical elegance."
    },
    {
      type: "mc",
      question: "The prefix language modeling objective (used in T5 and UL2) treats part of the input as a bidirectional prefix and the rest as an autoregressive target. Compared to pure causal LM, this means:",
      options: ["The model has fewer effective parameters because prefix and target tokens share a single attention mechanism, eliminating the overhead of separate encoder-decoder parameter sets used in standard seq2seq architectures", "The prefix must always be exactly half the sequence length to maintain a balanced ratio between bidirectional context encoding and autoregressive target generation, constraining the input-output split", "Tokens in the prefix attend to each other bidirectionally via full self-attention, while target tokens attend causally — unifying bidirectional encoding for context with autoregressive generation for output", "Prefix LM cannot perform zero-shot open-ended generation because it requires a non-empty prefix to condition on, limiting its use to conditional tasks where an input context is always provided"],
      correct: 2,
      explanation: "Prefix LM uses a single Transformer with a hybrid attention mask: prefix tokens see each other fully (bidirectional), target tokens see all prefix tokens plus previous target tokens (causal). This is strictly more expressive than causal LM for the prefix portion (which benefits from bidirectional context) while maintaining valid autoregressive generation for the target. It is a natural fit for conditional generation tasks (question$\\rightarrow$answer, document$\\rightarrow$summary) where the input benefits from bidirectional encoding."
    },
    {
      type: "mc",
      question: "Noise contrastive estimation (NCE) has been proposed as an alternative to maximum likelihood for training language models. NCE trains the model to distinguish real data from noise samples. Why has NCE not replaced cross-entropy for large-scale LM pretraining?",
      options: ["NCE requires a noise distribution close to the data distribution, but designing one for natural language is hard — and its statistical efficiency degrades with vocabulary size, needing many noise samples per data point", "NCE produces a discriminator rather than a generator, so the trained model can only score real-vs-fake token pairs and cannot be repurposed for autoregressive generation without a separate decoding procedure", "NCE cannot be combined with Transformer architectures because the binary contrastive objective requires a fundamentally different computational graph incompatible with causal self-attention masking", "NCE requires labeled data with explicit positive and negative sequence categories, which is unavailable in unsupervised pretraining settings where models learn from unlabeled raw text corpora"],
      correct: 0,
      explanation: "NCE converts density estimation into binary classification: real vs. noise. The quality of the noise distribution matters enormously — if noise is too different from data, the classification is trivial and uninformative; if too similar, training is slow. For LLMs with vocabulary sizes of 30K-100K, NCE needs $k$ noise samples per real token (where $k$ should ideally grow with $|V|$), making it less efficient than the softmax cross-entropy loss which processes the entire vocabulary in one shot via the log-sum-exp. Modern hardware makes full-vocabulary softmax feasible."
    },
    {
      type: "mc",
      question: "Discrete diffusion models like D3PM and MDLM define a forward corruption process that gradually replaces tokens with random tokens or a [MASK] symbol. The number of denoising steps $T$ at inference time presents a trade-off:",
      options: [
        "More steps always produce worse results due to error accumulation across the denoising chain, where small per-step prediction mistakes compound multiplicatively into large final sequence-level errors",
        "Fewer steps are faster but each must correct more corruption at once, requiring larger and less accurate jumps — more steps allow smaller, more accurate denoising increments but multiply inference latency",
        "The number of steps does not affect output quality and only impacts generation speed, since the denoising network converges to the same output distribution regardless of how many intermediate steps are used",
        "Discrete diffusion requires exactly 1000 steps to work correctly because the token transition matrices and noise schedules are jointly calibrated for that specific diffusion chain length"
      ],
      correct: 1,
      explanation: "This is the fundamental speed-quality trade-off in all diffusion models. With $T = 1$ step, the model must denoise from pure noise to clean text in one shot (essentially non-autoregressive generation with all its problems). With $T = 1000$ steps, each step only slightly adjusts the sequence, making each denoising prediction easier but inference very slow. Practical discrete diffusion models use 10-100 steps with techniques like stride scheduling to concentrate steps where they matter most. This is still slower than autoregressive generation for short sequences."
    },
    {
      type: "mc",
      question: "The \"exposure bias\" problem in autoregressive language models refers to the discrepancy between training and inference. Specifically:",
      options: ["The model is exposed to too much data during training, causing it to memorize surface-level statistical patterns rather than learning generalizable generation strategies that transfer to unseen domains", "Longer sequences receive disproportionately more gradient updates during training, biasing the model toward generating verbose outputs that maximize the number of tokens and inflate sequence-level log-likelihood", "The model is biased toward high-frequency tokens in the training corpus, causing it to systematically underrepresent rare tokens and produce repetitive outputs dominated by common n-gram patterns", "During training the model conditions on ground-truth previous tokens via teacher forcing, but at inference it conditions on its own predictions — errors compound because it never learns to recover from mistakes"],
      correct: 3,
      explanation: "Teacher forcing provides ground-truth context during training: $P(x_t \\mid x_1^*, \\dots, x_{t-1}^*)$. At inference, the model generates $P(x_t \\mid \\hat{x}_1, \\dots, \\hat{x}_{t-1})$ where $\\hat{x}$ are its own (potentially erroneous) predictions. The distribution of contexts at inference differs from training, causing errors to accumulate. Scheduled sampling (mixing ground-truth and model predictions during training) partially addresses this, and it is one motivation for non-autoregressive and diffusion-based alternatives. In practice, exposure bias is less damaging for very large LLMs because their per-token error rate is low."
    }
  ]
};
