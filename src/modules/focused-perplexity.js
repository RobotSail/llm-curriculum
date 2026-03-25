// Focused learning module for Perplexity.
// Covers interpretation, computation, pitfalls, and practical use.
// Assumes the student already knows entropy and cross-entropy.

export const perplexityLearning = {
  id: "0.2-perplexity-learning-easy",
  sectionId: "0.2",
  title: "Perplexity: Interpreting Language Model Loss",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 15,
  steps: [
    {
      type: "info",
      title: "Perplexity: Making Cross-Entropy Interpretable",
      content: "Cross-entropy in nats or bits is hard to interpret on its own. **Perplexity** converts it to an intuitive quantity:\n\n$$\\text{PPL} = 2^{H(P, Q)} \\quad \\text{(if using } \\log_2\\text{)} \\qquad \\text{or} \\qquad \\text{PPL} = e^{H(P, Q)} \\quad \\text{(if using } \\ln\\text{)}$$\n\nPerplexity is the **effective vocabulary size** the model is choosing from at each step. Think of it as: \"on average, this model is as uncertain as if it were picking uniformly from PPL equally likely tokens.\"\n\n- **PPL = 1**: The model is perfectly certain of every next token. This would require language to be fully deterministic.\n- **PPL = 50**: The model's average uncertainty is equivalent to choosing uniformly from 50 equally likely tokens.\n- **PPL = 50,000**: The model is as confused as if every token in a 50K vocabulary were equally likely — it has learned nothing.\n\nBecause perplexity is **exponential** in cross-entropy, small improvements in loss produce large perplexity reductions. Reducing cross-entropy from 5.0 to 4.5 bits cuts perplexity from $2^5 = 32$ to $2^{4.5} \\approx 22.6$ — a 30% reduction in effective uncertainty."
    },
    {
      type: "mc",
      question: "Model A has perplexity 30 and Model B has perplexity 90. How much better is A than B in cross-entropy (bits)?",
      options: [
        "$\\log_2(90) - \\log_2(30) = \\log_2(3) \\approx 1.58$ bits per token",
        "$90 - 30 = 60$ bits per token — perplexity differences equal cross-entropy differences",
        "$(90 - 30)/90 \\approx 0.67$ bits per token — the normalized perplexity gap",
        "$\\log_2(90/30) = \\log_2(3) \\approx 1.58$ bits per token, but only if both models use the same vocabulary"
      ],
      correct: 0,
      explanation: "Cross-entropy $= \\log_2(\\text{PPL})$, so the difference is $\\log_2(90) - \\log_2(30) = \\log_2(90/30) = \\log_2(3) \\approx 1.58$ bits per token. This means Model B wastes 1.58 extra bits per token compared to A. Note that perplexity ratios correspond to cross-entropy *differences* (because $\\log$ turns ratios into differences) — a 3x perplexity improvement always equals $\\log_2 3 \\approx 1.58$ bits, regardless of the absolute values. The vocabulary does not need to match for this relationship to hold — it is purely a property of the log-exp duality."
    },
    {
      type: "info",
      title: "Converting Between Loss Units and Perplexity",
      content: "In practice, most deep learning frameworks (PyTorch, JAX) compute cross-entropy loss using the natural logarithm, reporting loss in **nats**. Some information-theoretic papers use $\\log_2$, reporting in **bits**. The conversion to perplexity depends on which base was used:\n\n$$\\text{PPL} = e^{\\text{loss in nats}} = 2^{\\text{loss in bits}}$$\n\nBoth formulas give the **same perplexity** — the base of the exponent must match the base of the log used to compute the loss. To convert between nats and bits: $\\text{bits} = \\text{nats} / \\ln 2 \\approx \\text{nats} \\times 1.4427$.\n\nA quick sanity check: if your loss is 2.0 nats, perplexity is $e^2 \\approx 7.4$. If your loss is 2.0 bits, perplexity is $2^2 = 4$. These are different losses that happen to share the same number — always track your units."
    },
    {
      type: "mc",
      question: "A language model achieves a cross-entropy loss of 3.0 nats per token on a held-out set. What is the model's perplexity?",
      options: [
        "$3.0$ — the perplexity equals the raw loss value directly",
        "$2^3 = 8$ — exponentiate with base 2 since perplexity always uses bits",
        "$\\ln(3) \\approx 1.1$ — take the natural log to convert nats to perplexity",
        "$e^3 \\approx 20.1$ — exponentiate with base $e$ since the loss is in nats"
      ],
      correct: 3,
      explanation: "When cross-entropy is measured in nats (using $\\ln$), perplexity is $e^{\\text{loss}}$, so $e^{3.0} \\approx 20.1$. If the loss were in bits (using $\\log_2$), perplexity would be $2^{\\text{loss}}$. The choice of log base determines the exponentiation base. In most deep learning frameworks, the loss is in nats (natural log), so $e$ is the correct base."
    },
    {
      type: "info",
      title: "Perplexity as Compression Rate",
      content: "There is a deep connection between perplexity and **data compression**. A language model that assigns probability $Q$ to text is equivalent to a compression scheme: high-probability sequences get short codes, low-probability sequences get long codes (this is Shannon's source coding theorem).\n\nThe cross-entropy $H(P, Q)$ is the **average number of bits per token** needed to encode text drawn from $P$ using the code derived from $Q$. Perplexity is then $2^{\\text{bits per token}}$ — the effective number of equally-likely symbols at each coding step.\n\nA model with PPL = 20 on English text compresses each token to $\\log_2(20) \\approx 4.3$ bits on average. Since a 50K-token vocabulary has $\\log_2(50000) \\approx 15.6$ bits per token under uniform coding, the model achieves a **compression ratio** of roughly $15.6 / 4.3 \\approx 3.6\\times$.\n\nThis is why lower perplexity directly means better compression — and why organizations like DeepMind have framed language modeling as a compression benchmark."
    },
    {
      type: "mc",
      question: "A uniform model over a 50,000-token vocabulary assigns $P(w) = 1/50000$ for every token regardless of context. What is its perplexity?",
      options: [
        "$\\log_2(50000) \\approx 15.6$ — perplexity is the cross-entropy in bits",
        "$50000$ — perplexity equals the vocabulary size for a uniform distribution",
        "$\\sqrt{50000} \\approx 224$ — perplexity is the geometric mean of the vocabulary size",
        "$50000^2 = 2.5 \\times 10^9$ — perplexity squares the vocabulary for uniform models"
      ],
      correct: 1,
      explanation: "Cross-entropy of a uniform model is $-\\sum_w P(w) \\log(1/50000) = \\log(50000)$. So $\\text{PPL} = e^{\\log 50000} = 50000$. The uniform model's perplexity equals the vocabulary size — it is equivalent to rolling a 50,000-sided die. This is the worst possible perplexity for this vocabulary, confirming the interpretation of perplexity as \"effective vocabulary size.\""
    },
    {
      type: "info",
      title: "Bits-Per-Character (BPC) vs Perplexity",
      content: "Some benchmarks report **bits-per-character (BPC)** instead of token-level perplexity. BPC is the average cross-entropy in bits computed at the character level:\n\n$$\\text{BPC} = \\frac{\\text{total cross-entropy in bits}}{\\text{number of characters}}$$\n\nBPC and token-level perplexity measure the same underlying quantity — how well the model predicts — but at different granularities. You can convert between them if you know the average number of characters per token ($c$):\n\n$$\\text{BPC} = \\frac{\\log_2(\\text{PPL})}{c}$$\n\nFor BPE tokenizers on English, $c \\approx 4$. A model with PPL = 16 has $\\log_2(16) = 4$ bits/token, or roughly $4/4 = 1.0$ BPC.\n\n**Why use BPC?** It is **tokenizer-independent**. Two models with different vocabularies and subword schemes can be directly compared via BPC, because characters are a universal unit. This makes BPC the preferred metric on character-level benchmarks like enwik8 and text8."
    },
    {
      type: "mc",
      question: "Model X uses a 32K BPE vocabulary and reports PPL = 8.0 on a benchmark. Model Y uses a 64K BPE vocabulary and reports PPL = 7.5 on the same benchmark. Which model is definitively better at predicting text?",
      options: [
        "Model Y, because 7.5 < 8.0 and lower perplexity always means better prediction",
        "Model X, because its smaller vocabulary makes the task harder so PPL 8.0 is more impressive",
        "They are equally good because the perplexity ratio is within the margin caused by vocabulary size differences",
        "Neither can be concluded — token-level perplexities are not directly comparable across different tokenizers"
      ],
      correct: 3,
      explanation: "Token-level perplexity depends on the tokenization scheme. A model with a larger vocabulary may have lower perplexity simply because its tokens carry more information (e.g., encoding common multi-character sequences as single tokens). To compare models with different tokenizers, you need a tokenizer-independent metric like bits-per-character (BPC) or bits-per-byte (BPB). Without converting to a common unit, the PPL comparison is meaningless."
    },
    {
      type: "info",
      title: "The Entropy Rate of English and Modern LLMs",
      content: "Shannon estimated the **entropy rate** of English at about **1.0-1.5 bits per character** through human prediction experiments. Compare this to the maximum if every letter were equally likely:\n\n$$H_{\\text{uniform}} = \\log_2 26 \\approx 4.7 \\text{ bits/char}$$\n\nEnglish is therefore roughly **70% redundant** — most of the information capacity of the character stream is consumed by spelling patterns, grammar, and semantic constraints.\n\nModern LLMs achieve cross-entropy below 1 bit per character on many benchmarks, approaching Shannon's estimates. At the token level (with BPE tokenization, average ~4 chars/token), the entropy rate is roughly $4 \\times 1.3 \\approx 5.2$ bits/token. GPT-4-class models achieve perplexities around 6-10 on standard benchmarks, corresponding to $\\log_2(8) = 3$ bits/token — well below the character-level entropy rate, because subword tokenization captures spelling redundancy in the tokenizer itself.\n\nThis means that state-of-the-art models are **near the theoretical limit** of how well English can be predicted. Further perplexity improvements require capturing ever more subtle patterns in meaning, world knowledge, and reasoning."
    },
    {
      type: "mc",
      question: "A researcher evaluates two LLMs on a medical textbook corpus and a casual Reddit corpus. Model M gets PPL 45 on the medical text and PPL 12 on Reddit. Model N gets PPL 25 on both. Which interpretation is most accurate?",
      options: [
        "The perplexities reflect domain fit — Model M likely saw more casual internet text in training, while Model N is more balanced across domains",
        "Model N is strictly better because it has lower average perplexity across both domains",
        "Model M is better because PPL 12 on Reddit shows stronger language modeling overall",
        "The comparison is invalid because perplexity cannot be compared across different evaluation corpora"
      ],
      correct: 0,
      explanation: "Perplexity is highly sensitive to the evaluation domain. A model trained heavily on internet text will achieve low perplexity on Reddit-like data but struggle with specialized medical terminology. Model M's lopsided performance (PPL 12 vs 45) suggests domain imbalance in its training data, while Model N's uniform PPL 25 suggests more balanced training. Comparing perplexities across different corpora is valid — it reveals domain strengths and weaknesses — but the numbers reflect training distribution as much as model quality."
    },
    {
      type: "info",
      title: "Perplexity and Downstream Task Performance",
      content: "A natural question: does lower perplexity mean better performance on downstream tasks like question answering, summarization, or coding?\n\nThe relationship is **log-linear with diminishing returns**. Scaling law research (Kaplan et al., Hoffmann et al.) shows that as models get larger and cross-entropy decreases, downstream accuracy improves — but the improvements become smaller for each incremental drop in loss.\n\nThere are important caveats:\n\n**Perplexity is necessary but not sufficient.** A model can have low perplexity (it predicts common tokens well) yet still fail on tasks requiring reasoning, factual recall, or instruction-following. This is why RLHF-tuned models often have *higher* perplexity than their base models on raw text, yet perform better on benchmarks.\n\n**Domain mismatch matters.** Low perplexity on Wikipedia does not guarantee strong performance on code generation. The evaluation domain must match the task domain for perplexity to be predictive.\n\n**Perplexity improvements below a threshold can unlock capabilities.** Emergent abilities — tasks that go from near-random to high accuracy — often appear as perplexity crosses a critical level, not gradually."
    },
    {
      type: "mc",
      question: "An RLHF-tuned chat model has higher perplexity on a raw web text benchmark than its base pretrained model. A colleague argues this means RLHF made the model worse. What is the best response?",
      options: [
        "The colleague is correct — higher perplexity always indicates a worse language model regardless of the training objective",
        "The perplexity increase is a measurement artifact caused by RLHF changing the tokenizer, not the model's actual predictions",
        "RLHF trades raw next-token prediction accuracy for alignment with human preferences, so higher perplexity on generic text is expected and does not indicate worse task performance",
        "Higher perplexity after RLHF indicates the KL penalty was set too low, allowing the policy to drift too far from the base model"
      ],
      correct: 2,
      explanation: "RLHF optimizes for human preference rather than raw next-token prediction on generic text. The model learns to produce responses that humans rate highly, which may involve distributing probability differently than the base model would — for instance, upweighting helpful and harmless continuations at the expense of raw likelihood on arbitrary web text. This is a deliberate trade-off, not a defect. The KL penalty controls how far RLHF drifts from the base model, but some perplexity increase is inherent to the objective shift."
    },
    {
      type: "info",
      title: "Subword Perplexity Pitfalls: Why Comparisons Break",
      content: "A common mistake in LLM evaluation is comparing token-level perplexities across models that use **different tokenizers**. Here is why this fails:\n\nConsider the word \"unfortunately\". A tokenizer with a 32K vocabulary might split it into [\"un\", \"fortunate\", \"ly\"] (3 tokens), while a 64K tokenizer might encode it as [\"unfortunately\"] (1 token). The 64K model only needs to predict 1 token correctly; the 32K model needs 3.\n\nThe **total information** to predict the word may be similar, but it is distributed across a different number of tokens. The 64K model's single-token prediction carries more bits of information per token, while the 32K model spreads the same information across more tokens with fewer bits each.\n\nThis means a model with a larger vocabulary can report **lower perplexity** simply because each token carries more information — not because the model predicts better. To compare fairly:\n\n- Use **bits-per-byte (BPB)** or **bits-per-character (BPC)**, which normalize by a tokenizer-independent unit\n- Or evaluate on a **shared tokenization** by retokenizing outputs\n- Always report the tokenizer alongside perplexity numbers"
    },
    {
      type: "mc",
      question: "A research paper reports that their model achieves \"state-of-the-art perplexity of 3.1 on WikiText-103\" using a novel 128K-token vocabulary. Previous SOTA was 3.4 with a 32K vocabulary. What should a careful reviewer check first?",
      options: [
        "Whether the model was trained on more data, since data quantity directly determines perplexity independent of vocabulary size",
        "Whether the improvement holds when measured in bits-per-byte, since the larger vocabulary could artificially lower token-level perplexity",
        "Whether 3.1 is below the entropy rate of English, which would prove the result is erroneous",
        "Whether the model has more parameters, since perplexity is solely a function of parameter count according to scaling laws"
      ],
      correct: 1,
      explanation: "A 128K vocabulary encodes more information per token than a 32K vocabulary, which can mechanically lower token-level perplexity without improving the model's actual predictive ability. The reviewer should check whether the improvement persists in a tokenizer-independent metric like bits-per-byte. A PPL of 3.1 is not inherently below the entropy rate of English (that depends on the tokenization), and perplexity depends on many factors beyond parameter count."
    }
  ]
};
