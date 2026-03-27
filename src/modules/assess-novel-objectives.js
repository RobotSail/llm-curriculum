// B.5 Novel Pretraining Objectives — per-section test (split from assess-branch-b.js)

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
        "Both model the exact same joint distribution, just with different factorizations that are mathematically equivalent under the chain rule of probability",
        "Autoregressive models factorize the exact joint via the chain rule $P(x) = \\prod_t P(x_t \\mid x_{<t})$, while MLM does not define a consistent joint distribution — it models conditional distributions $P(x_t \\mid x_{\\setminus t})$ that may not correspond to any valid joint",
        "MLM defines the joint distribution more efficiently because it processes all tokens in parallel, capturing bidirectional dependencies in a single forward pass",
        "Autoregressive models can only generate text, while MLM can both generate and understand because bidirectional conditioning enables both directions of inference"
      ],
      correct: 1,
      explanation: "The chain rule factorization used by autoregressive models is exact: $\\prod_t P(x_t \\mid x_{<t})$ is guaranteed to be a valid joint distribution. MLM trains conditional distributions $P(x_t \\mid x_{\\setminus t})$ (each token given all others). But a set of conditional distributions may be inconsistent — there may be no joint distribution that produces all of them. This makes MLM models unsuitable for generation without additional techniques (e.g., iterative refinement as in BERT-based generation)."
    },
    {
      type: "mc",
      question: "A key practical advantage of MLM over autoregressive LM during pretraining is:",
      options: ["MLM can use a smaller vocabulary because the masking procedure naturally clusters rare tokens into shared prediction targets, reducing the effective vocabulary size and improving embedding quality", "MLM requires less training data because it uses each token more efficiently by training on multiple masked positions per sequence rather than a single next-token prediction per position", "MLM is faster at inference because it generates all tokens in parallel, avoiding the sequential bottleneck of autoregressive decoding that limits tokens per second during generation", "MLM processes all tokens bidirectionally — each masked position attends to both left and right context — producing richer representations for downstream understanding tasks like classification and NER"],
      correct: 3,
      explanation: "MLM's bidirectional context is its main strength for representation learning. When predicting a masked token, the model can use information from both sides, producing representations that capture the full context. Autoregressive models only see left context at each position. However, MLM only trains on the ~15% of tokens that are masked (the rest don't contribute to the loss), while autoregressive models get a gradient signal from every token. This makes autoregressive pretraining more compute-efficient per token."
    },
    {
      type: "mc",
      question: "UL2 (Unifying Language Learning Paradigms) proposes training a single model with multiple denoising objectives. Its core insight is:",
      options: ["UL2 eliminates the need for fine-tuning by training on all possible task formats, making the pretrained model directly usable for any downstream task", "A single denoising objective is always optimal if tuned properly, since mixing objectives introduces conflicting gradient signals that hurt performance", "Different downstream tasks benefit from different pretraining objectives (short spans for understanding, long spans for generation), so mixing multiple denoising tasks with mode-switching tokens creates a model that handles both regimes", "UL2 uses reinforcement learning instead of maximum likelihood to optimize the denoising objective, enabling the model to learn more complex reconstruction strategies"],
      correct: 2,
      explanation: "UL2 defines three denoising modes: R-denoiser (short spans, like BERT), S-denoiser (sequential/prefix LM), and X-denoiser (extreme/long spans). A special sentinel token tells the model which mode is active. The key insight is that no single denoising objective dominates across all downstream tasks — short-span denoising helps classification and NLU, while long-span and prefix modes help generation. By mixing modes, UL2 produces a single model competitive on both understanding and generation benchmarks."
    },
    {
      type: "mc",
      question: "Diffusion models have been highly successful for continuous data (images, audio). Why is applying diffusion to discrete text fundamentally harder?",
      options: ["Discrete data cannot be interpolated smoothly — there is no natural continuous noise process for tokens, and discrete corruption processes lack the mathematical properties that make continuous diffusion tractable", "Text sequences are too short for the diffusion process to work effectively, since diffusion models need long inputs to amortize the cost of the multi-step denoising procedure across enough tokens", "The vocabulary size is too large for the denoising network to predict over, since each denoising step must output a probability vector over the entire vocabulary at every position in the sequence", "Diffusion requires 2D spatial structure that text does not naturally have, since the denoising U-Net architecture relies on spatial convolutions that cannot operate on one-dimensional sequences"],
      correct: 0,
      explanation: "Continuous diffusion relies on gradually adding Gaussian noise and learning to reverse this process. For discrete tokens, there is no natural analog: you cannot \"slightly noise\" a token. Approaches include: (1) embedding tokens in continuous space and applying continuous diffusion (D3PM, Diffusion-LM), (2) using discrete corruption (token masking/replacement) as forward process (multinomial diffusion), or (3) score-matching on the simplex (MDLM). Each has trade-offs: continuous embeddings disconnect from the discrete structure; discrete corruption requires custom transition matrices."
    },
    {
      type: "mc",
      question: "Non-autoregressive generation (NAG) methods aim to generate all tokens in parallel rather than sequentially. The fundamental challenge they face is:",
      options: [
        "They require significantly more parameters than autoregressive models to achieve equivalent generation quality, making them impractical for deployment at the scale of modern LLMs",
        "They must model the joint distribution without the chain rule's sequential factorization — tokens generated in parallel cannot condition on each other, leading to repetition and incoherence",
        "They cannot use the Transformer architecture because the causal attention mask is incompatible with simultaneous token generation, requiring entirely different neural architectures",
        "They are slower than autoregressive models in practice because the parallel decoding overhead and iterative refinement steps exceed the cost of sequential generation"
      ],
      correct: 1,
      explanation: "Autoregressive models factor $P(x_1, \\dots, x_T)$ into conditionals, each depending on all previous tokens. NAG must model $P(x_1, \\dots, x_T)$ without this sequential structure — often assuming conditional independence given some latent $z$: $P(x \\mid z) = \\prod_t P(x_t \\mid z)$. This \"conditional independence\" assumption is violated when strong dependencies exist between adjacent tokens (e.g., \"New York\" — generating \"New\" and \"York\" independently risks producing \"New London\" or duplicating tokens). Knowledge distillation from AR models, iterative refinement, and CTC losses are common mitigations."
    },
    {
      type: "mc",
      question: "Energy-based models (EBMs) for text define an unnormalized density $p_\\theta(x) \\propto \\exp(-E_\\theta(x))$ over sequences. The central computational challenge of EBMs is:",
      options: ["The energy function $E_\\theta(x)$ is difficult to parameterize for text because variable-length sequences require architecture-specific pooling strategies", "The energy function must be non-negative by construction, which limits the expressiveness of the model class to distributions with bounded support", "EBMs cannot assign meaningfully different probabilities to different sequences because the softmax normalization collapses the energy differences", "Computing the normalizing constant $Z_\\theta = \\sum_x \\exp(-E_\\theta(x))$ requires summing over all possible sequences (exponential in length and vocabulary), making exact likelihood evaluation and gradient computation intractable"],
      correct: 3,
      explanation: "The partition function $Z_\\theta$ sums over all possible token sequences — $|V|^T$ terms for vocabulary $V$ and length $T$. This is astronomically intractable. Training EBMs requires approximations: contrastive divergence (MCMC sampling for negative examples), noise contrastive estimation (NCE), or score matching. For text specifically, MCMC sampling is difficult because the discrete space makes gradient-based sampling (Langevin dynamics) inapplicable. These challenges are why EBMs remain niche for text despite their theoretical elegance."
    },
    {
      type: "mc",
      question: "The prefix language modeling objective (used in T5 and UL2) treats part of the input as a bidirectional prefix and the rest as an autoregressive target. Compared to pure causal LM, this means:",
      options: ["The model has fewer parameters because the prefix encoder shares weights with the decoder, eliminating the need for separate encoder and decoder parameter sets", "The prefix must always be exactly half the sequence length to maintain a balanced ratio between bidirectional context encoding and autoregressive generation", "Tokens in the prefix attend to each other bidirectionally (full self-attention), while target tokens attend causally — this unifies the benefits of bidirectional encoding for the input context with autoregressive generation for the output", "Prefix LM cannot perform zero-shot generation because it requires a non-empty prefix to condition on, making it unsuitable for open-ended text generation tasks"],
      correct: 2,
      explanation: "Prefix LM uses a single Transformer with a hybrid attention mask: prefix tokens see each other fully (bidirectional), target tokens see all prefix tokens plus previous target tokens (causal). This is strictly more expressive than causal LM for the prefix portion (which benefits from bidirectional context) while maintaining valid autoregressive generation for the target. It is a natural fit for conditional generation tasks (question$\\rightarrow$answer, document$\\rightarrow$summary) where the input benefits from bidirectional encoding."
    },
    {
      type: "mc",
      question: "Noise contrastive estimation (NCE) has been proposed as an alternative to maximum likelihood for training language models. NCE trains the model to distinguish real data from noise samples. Why has NCE not replaced cross-entropy for large-scale LM pretraining?",
      options: ["NCE requires the noise distribution to be close to the data distribution for efficient learning, but designing such a noise distribution for natural language is itself a hard problem — and NCE's statistical efficiency degrades with vocabulary size, requiring many noise samples per data point", "NCE produces a discriminator rather than a generator, so it cannot be used for text generation since the model only learns to classify real versus fake tokens", "NCE cannot be combined with Transformer architectures because the contrastive objective requires a fundamentally different computational graph than autoregressive attention", "NCE requires labeled data with explicit positive and negative categories, which is unavailable in the unsupervised pretraining setting where models learn from raw text"],
      correct: 0,
      explanation: "NCE converts density estimation into binary classification: real vs. noise. The quality of the noise distribution matters enormously — if noise is too different from data, the classification is trivial and uninformative; if too similar, training is slow. For LLMs with vocabulary sizes of 30K-100K, NCE needs $k$ noise samples per real token (where $k$ should ideally grow with $|V|$), making it less efficient than the softmax cross-entropy loss which processes the entire vocabulary in one shot via the log-sum-exp. Modern hardware makes full-vocabulary softmax feasible."
    },
    {
      type: "mc",
      question: "Discrete diffusion models like D3PM and MDLM define a forward corruption process that gradually replaces tokens with random tokens or a [MASK] symbol. The number of denoising steps $T$ at inference time presents a trade-off:",
      options: [
        "More steps always produces worse results due to error accumulation across the denoising chain, where small per-step mistakes compound into large final errors",
        "Fewer steps are faster but each step must correct more corruption at once, requiring the model to make larger and less accurate jumps — more steps allow smaller, more accurate denoising increments but multiply the inference latency by $T$",
        "The number of steps does not affect output quality at all and only impacts generation speed, since the model converges to the same output regardless of stride",
        "Discrete diffusion requires exactly 1000 steps to work correctly because the token transition matrices are calibrated for that specific diffusion schedule length"
      ],
      correct: 1,
      explanation: "This is the fundamental speed-quality trade-off in all diffusion models. With $T = 1$ step, the model must denoise from pure noise to clean text in one shot (essentially non-autoregressive generation with all its problems). With $T = 1000$ steps, each step only slightly adjusts the sequence, making each denoising prediction easier but inference very slow. Practical discrete diffusion models use 10-100 steps with techniques like stride scheduling to concentrate steps where they matter most. This is still slower than autoregressive generation for short sequences."
    },
    {
      type: "mc",
      question: "The \"exposure bias\" problem in autoregressive language models refers to the discrepancy between training and inference. Specifically:",
      options: ["The model is exposed to too much data during training, causing it to memorize surface patterns rather than learning generalizable generation strategies", "Longer sequences receive more gradient updates during training, biasing the model toward generating verbose outputs that maximize the number of tokens produced", "The model is biased toward frequent tokens in the training data, causing it to underrepresent rare tokens and produce repetitive, high-frequency outputs", "During training the model conditions on ground-truth previous tokens (teacher forcing), but during inference it conditions on its own predictions — errors compound because the model never learns to recover from its own mistakes"],
      correct: 3,
      explanation: "Teacher forcing provides ground-truth context during training: $P(x_t \\mid x_1^*, \\dots, x_{t-1}^*)$. At inference, the model generates $P(x_t \\mid \\hat{x}_1, \\dots, \\hat{x}_{t-1})$ where $\\hat{x}$ are its own (potentially erroneous) predictions. The distribution of contexts at inference differs from training, causing errors to accumulate. Scheduled sampling (mixing ground-truth and model predictions during training) partially addresses this, and it is one motivation for non-autoregressive and diffusion-based alternatives. In practice, exposure bias is less damaging for very large LLMs because their per-token error rate is low."
    }
  ]
};
