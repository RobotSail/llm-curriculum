// Assessment modules for Tier 1, Part 1 (Sections 1.1–1.3)
// Pure MC assessment — no info steps. Feeds into warmup/daily quiz system.

// ─────────────────────────────────────────────────────────────────────────────
// Section 1.1: Transformer Architecture — Deep Understanding
// ─────────────────────────────────────────────────────────────────────────────
export const transformerAssessment = {
  id: "1.1-assess",
  sectionId: "1.1",
  title: "Assessment: Transformer Architecture",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "In scaled dot-product attention, queries and keys are divided by $\\sqrt{d_k}$ before the softmax. What failure mode does this scaling prevent?",
      options: [
        "It prevents the attention weights from summing to more than 1",
        "It prevents the dot products from growing large in magnitude, which would push softmax into saturated regions where gradients vanish",
        "It normalizes the queries and keys to unit length, ensuring cosine similarity",
        "It compensates for the fact that values have a different dimensionality than queries and keys"
      ],
      correct: 1,
      explanation: "When $d_k$ is large, dot products $q \\cdot k$ have variance proportional to $d_k$ (assuming unit-variance components). Without scaling, large dot products push softmax into regions where one logit dominates — the output approaches a one-hot vector, entropy collapses to near zero, and gradients through the softmax become vanishingly small. Dividing by $\\sqrt{d_k}$ keeps the variance of the logits at $O(1)$ regardless of dimension."
    },
    {
      type: "mc",
      question: "RoPE (Rotary Position Embeddings) encodes position by rotating query/key vectors in 2D subspaces of the embedding. Why does this enable better length generalization than learned absolute positional embeddings?",
      options: [
        "RoPE uses fewer parameters, so it is less prone to overfitting on short sequences",
        "RoPE makes the dot product $q_m^\\top k_n$ depend only on the relative distance $(m - n)$ via a rotation in the complex plane, so the model never sees an absolute position index it hasn't trained on",
        "RoPE projects positions into a Fourier basis, which is always periodic and can therefore handle any length",
        "RoPE embeddings decay to zero at long distances, acting as an implicit windowed attention"
      ],
      correct: 1,
      explanation: "RoPE applies a rotation $R_{\\theta, m}$ to position $m$, so $q_m^\\top k_n = (R_{\\theta,m} q)^\\top (R_{\\theta,n} k) = q^\\top R_{\\theta, n-m} k$. The attention logit depends only on the relative offset $(m - n)$, not the absolute positions. Learned absolute embeddings assign a fixed vector to each position index, so at inference time positions beyond the training length have never been seen. RoPE's relative formulation avoids this, though it still degrades at very long distances without further techniques (e.g., NTK-aware scaling, YaRN)."
    },
    {
      type: "mc",
      question: "Anthropic's \"circuits\" view describes the residual stream as a communication bus. Under this interpretation, what is the role of attention heads?",
      options: [
        "They compute nonlinear activation functions that gate information flow",
        "They read from and write to the residual stream, moving information between token positions — each head implements a specific information-routing operation",
        "They normalize the residual stream to prevent feature magnitudes from diverging",
        "They act as memory banks that store factual knowledge retrieved by the FFN layers"
      ],
      correct: 1,
      explanation: "In the residual stream view, the stream is a shared memory bus of dimension $d_{\\text{model}}$. Attention heads read from source positions (via $W_Q, W_K$ selecting what to attend to, and $W_V$ selecting what to read) and write to destination positions (via $W_O$). Each head moves specific types of information between positions — e.g., induction heads copy tokens that followed similar patterns. The FFN layers then read from the stream to process information locally at each position."
    },
    {
      type: "mc",
      question: "SwiGLU, used in LLaMA and PaLM, replaces the standard FFN with $\\text{SwiGLU}(x) = (\\text{Swish}(xW_1) \\odot xV) W_2$. What is the key structural difference from a vanilla two-layer FFN with ReLU?",
      options: [
        "SwiGLU uses three weight matrices instead of two, introducing a gating mechanism where one linear projection controls the information flow of another",
        "SwiGLU reduces the parameter count by using a single shared weight matrix",
        "SwiGLU eliminates the nonlinearity entirely, relying on the element-wise product for expressivity",
        "SwiGLU applies layer normalization between the two linear projections"
      ],
      correct: 0,
      explanation: "A standard FFN uses two matrices: $\\text{ReLU}(xW_1)W_2$. SwiGLU introduces a third matrix $V$ and uses a gating mechanism: one branch $\\text{Swish}(xW_1)$ produces gating values, and the other branch $xV$ produces candidate values, combined via element-wise product. This gated linear unit structure (from Dauphin et al., 2017) gives the network finer control over information flow. To keep parameter count comparable, the hidden dimension is typically reduced from $4d$ to $\\frac{8}{3}d$. The \"key-value memory\" interpretation views each hidden unit as storing a (key, value) pair where the key determines when the unit activates."
    },
    {
      type: "mc",
      question: "Pre-LN (LayerNorm before attention/FFN) is now standard over Post-LN (LayerNorm after). What training stability problem does Pre-LN solve?",
      options: [
        "Pre-LN prevents activation magnitudes from growing unboundedly across layers, allowing stable training without careful learning rate warmup — Post-LN amplifies gradients in early layers, causing divergence",
        "Pre-LN is mathematically equivalent to Post-LN but uses less memory during backpropagation",
        "Pre-LN eliminates the need for residual connections entirely",
        "Pre-LN ensures that the output distribution matches the input distribution at initialization, solving internal covariate shift"
      ],
      correct: 0,
      explanation: "In Post-LN, the residual path feeds raw (unnormalized) outputs back into the stream, and normalization happens after the addition. This causes gradient magnitudes to vary dramatically across layers — early layers receive amplified gradients, leading to instability. Pre-LN normalizes inputs before each sub-layer, ensuring the residual path carries well-behaved signals. Xiong et al. (2020) showed Pre-LN removes the need for careful warmup schedules. The tradeoff: some work suggests Post-LN produces slightly better final performance when it can be stabilized (e.g., with careful init or additional techniques)."
    },
    {
      type: "mc",
      question: "Grouped Query Attention (GQA) with $G$ groups uses $G$ key-value heads shared across $H$ query heads (where $G < H$). If a model has $H = 32$ query heads and uses GQA with $G = 8$, what is the KV cache memory reduction compared to standard MHA, and how does this compare to MQA?",
      options: [
        "KV cache is reduced by $32\\times$ vs MHA; MQA would reduce it by $8\\times$",
        "KV cache is reduced by $4\\times$ vs MHA ($8/32$); MQA ($G=1$) would reduce it by $32\\times$ but with greater quality degradation",
        "KV cache is the same size as MHA; GQA only reduces compute, not memory",
        "KV cache is reduced by $8\\times$ vs MHA; MQA and GQA have identical memory footprints"
      ],
      correct: 1,
      explanation: "In MHA, we store $H = 32$ distinct key-value pairs per layer per token. GQA with $G = 8$ stores only 8, so the KV cache shrinks by $32/8 = 4\\times$. MQA uses $G = 1$ (a single shared KV head), shrinking it by $32\\times$. GQA is a middle ground: it recovers most of MQA's memory savings while preserving quality closer to MHA. LLaMA 2 70B uses GQA with $G = 8$ for exactly this tradeoff. The quality gap between MQA and MHA is measurable on long-form generation tasks where KV diversity matters."
    },
    {
      type: "mc",
      question: "The concept of **superposition** in transformer residual streams refers to the hypothesis that:",
      options: [
        "Residual streams encode features in a one-to-one mapping, with each dimension representing exactly one feature",
        "The model represents more features than it has dimensions by encoding them as nearly orthogonal directions, tolerating small interference — analogous to compressed sensing",
        "Multiple attention heads attend to the same positions, creating redundant representations",
        "The residual stream superposes the outputs of all previous layers via simple addition"
      ],
      correct: 1,
      explanation: "Superposition (Elhage et al., 2022) is the hypothesis that transformers represent $m \\gg d$ features using $d$-dimensional residual streams by assigning nearly orthogonal directions to different features. This works because most features are sparse (rarely active simultaneously), so the interference from non-orthogonality rarely causes problems. This is closely related to compressed sensing and the Johnson-Lindenstrauss lemma — in high dimensions, random vectors are nearly orthogonal. Superposition makes interpretability hard because features don't align with individual neurons."
    },
    {
      type: "mc",
      question: "Consider a transformer with $L = 32$ layers, $d_{\\text{model}} = 4096$, $H = 32$ heads, and SwiGLU FFN with hidden dim $\\frac{8}{3} \\times 4096 \\approx 10923$. Approximately how many parameters are in the model (ignoring embeddings and final layer norm)?",
      options: [
        "About 1.5B — each layer has roughly 50M parameters",
        "About 7B — each layer has roughly 220M parameters from attention ($4d^2$) plus FFN ($3 \\times d \\times d_{\\text{ff}}$)",
        "About 13B — the FFN dominates with $8d^2$ per layer",
        "About 30B — attention and FFN contribute equally at $8d^2$ each"
      ],
      correct: 1,
      explanation: "Per layer: Attention has $W_Q, W_K, W_V, W_O$ each of size $d^2$, totaling $4d^2 \\approx 4 \\times 16.8M \\approx 67M$. SwiGLU FFN has three matrices: $W_1, V$ of size $d \\times d_{\\text{ff}}$ and $W_2$ of size $d_{\\text{ff}} \\times d$, totaling $3 \\times d \\times d_{\\text{ff}} \\approx 3 \\times 4096 \\times 10923 \\approx 134M$. Plus two LayerNorms at $2 \\times 2d \\approx 16K$. Per layer $\\approx 201M$. Multiply by 32 layers $\\approx 6.4B$. Adding embeddings ($V \\times d$) brings it near 7B. This matches LLaMA 7B's architecture."
    },
    {
      type: "mc",
      question: "An \"induction head\" is a two-head circuit that implements a specific attention pattern. What computation does it perform, and why is it considered a key mechanism for in-context learning?",
      options: [
        "It attends to the most frequent token in the context and copies it to the output, implementing a unigram frequency estimator",
        "It uses one head to identify previous occurrences of the current token and a second head to copy what followed those occurrences — implementing the rule 'if A B ... A, predict B'",
        "It computes the average of all key vectors to produce a summary representation of the context",
        "It detects syntactic patterns like subject-verb agreement by attending to structurally related positions"
      ],
      correct: 1,
      explanation: "Induction heads (Olsson et al., 2022) are a two-step circuit: (1) a \"previous-token head\" shifts information backward so each token's residual stream contains info about the preceding token; (2) an \"induction head\" uses this to find previous instances of the current token and attend to what came after them. This implements \"A B ... A $\\to$ B\" — a bigram copy operation that generalizes to longer patterns. Olsson et al. found that induction heads form during a phase transition early in training that coincides with a sharp drop in loss, and they account for the majority of in-context learning ability."
    },
    {
      type: "mc",
      question: "Why is the attention mechanism $O(n^2 d)$ in sequence length $n$, and which component specifically creates the quadratic bottleneck?",
      options: [
        "The $QK^\\top$ matrix multiplication produces an $n \\times n$ attention matrix, and both computing and storing this matrix scale quadratically in sequence length",
        "The value projection $W_V$ has $n^2$ parameters that must be learned",
        "The softmax normalization requires summing over $n^2$ elements",
        "The residual connections require adding $n^2$ vectors together"
      ],
      correct: 0,
      explanation: "The attention score matrix $A = \\text{softmax}(QK^\\top / \\sqrt{d_k})$ has shape $n \\times n$. Computing the $n \\times n$ dot products costs $O(n^2 d_k)$, storing the matrix costs $O(n^2)$ memory, and the matrix-vector product $AV$ costs $O(n^2 d_v)$. For long contexts (e.g., $n = 128K$), this $n^2$ term dominates everything. This is why methods like FlashAttention (which avoids materializing the full attention matrix via tiling), linear attention, and sparse attention exist — they aim to reduce either the compute or memory from $O(n^2)$."
    }
  ]
};


// ─────────────────────────────────────────────────────────────────────────────
// Section 1.2: Tokenization
// ─────────────────────────────────────────────────────────────────────────────
export const tokenizationAssessment = {
  id: "1.2-assess",
  sectionId: "1.2",
  title: "Assessment: Tokenization",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "Byte Pair Encoding (BPE) builds its vocabulary by iteratively merging the most frequent adjacent pair of tokens. If the training corpus contains the words \"low\" (5x), \"lower\" (2x), \"newest\" (6x), \"widest\" (3x), and the initial vocabulary is characters, which pair is merged first?",
      options: [
        "('e', 's') — appears 9 times across 'newest' and 'widest'",
        "('l', 'o') — appears 7 times across 'low' and 'lower'",
        "('n', 'e') — appears 6 times in 'newest'",
        "('w', 'i') — appears 3 times in 'widest'"
      ],
      correct: 0,
      explanation: "BPE counts pair frequencies across the entire corpus weighted by word frequency. ('e', 's') appears in 'newest' (6 times) and 'widest' (3 times) = 9 total occurrences. ('l', 'o') appears in 'low' (5) + 'lower' (2) = 7. ('n', 'e') appears 6 times. So ('e', 's') $\\to$ 'es' is the first merge. This greedy approach builds common subwords bottom-up. Note that BPE is deterministic given the corpus — the merge order defines the tokenizer completely."
    },
    {
      type: "mc",
      question: "A GPT-style tokenizer with vocabulary size 50,257 tokenizes the arithmetic expression \"123456 + 789\" into tokens [\"123\", \"456\", \" +\", \" 789\"]. Why does this tokenization make arithmetic difficult for the model?",
      options: [
        "The model cannot represent numbers larger than the vocabulary size",
        "The digit groupings are misaligned with place value — \"123456\" splits into \"123\" and \"456\", so the model must learn that the first token's digits occupy the hundred-thousands, ten-thousands, and thousands places, while the second token's digits are hundreds, tens, and ones — a positional encoding the tokenizer doesn't preserve",
        "The \"+\" operator is merged with a space, confusing the model about its semantic role",
        "The model has no embedding for the number 123456 since it exceeds the vocabulary"
      ],
      correct: 1,
      explanation: "Tokenization destroys digit-level positional structure. When \"123456\" becomes [\"123\", \"456\"], the model must implicitly learn that \"123\" means $123 \\times 1000$ (not just 123) based on the following token. Worse, the same token \"123\" might mean 123, 123000, or 123000000 depending on context. Each digit's place value is a function of total number length and token boundaries — information the model must infer rather than directly access. This is why digit-level tokenization (one token per digit) dramatically improves arithmetic performance."
    },
    {
      type: "mc",
      question: "The **fertility ratio** of a tokenizer for a language is the average number of tokens produced per word. A tokenizer trained predominantly on English text has a fertility ratio of 1.2 for English but 3.8 for Tamil. What concrete consequence does this have?",
      options: [
        "Tamil text will have higher perplexity because the model has seen less Tamil data",
        "Tamil text consumes roughly $3.2\\times$ more context window and costs $3.2\\times$ more per word in API pricing, and the model sees $3.2\\times$ fewer Tamil words in the same context length — systematically disadvantaging Tamil users",
        "The model will produce more fluent Tamil because it has more tokens to work with per word",
        "Tamil sentences will be more compressible because they use more tokens"
      ],
      correct: 1,
      explanation: "High fertility ratio means each Tamil word is fragmented into many tokens. With a 4096-token context window, English gets ~3400 words of context while Tamil gets only ~1080 words — a $3.2\\times$ reduction. API costs (priced per token) are proportionally higher per word. During training, each Tamil word also requires more forward-pass steps, reducing effective data efficiency. This is a systematic fairness issue: the tokenizer itself creates unequal capability across languages. Solutions include multilingual-aware BPE training, byte-level models, or language-specific tokenizers."
    },
    {
      type: "mc",
      question: "**MegaByte** (Yu et al., 2023) proposes a byte-level architecture that avoids subword tokenization entirely. What is the primary architectural innovation that makes byte-level modeling tractable despite sequences being ~4x longer than subword sequences?",
      options: [
        "It uses a convolutional backbone instead of attention, avoiding the quadratic cost entirely",
        "It uses a hierarchical architecture with a large \"global\" model operating on patches of bytes and a smaller \"local\" model predicting individual bytes within each patch, amortizing the cost of the large model",
        "It applies aggressive byte-level pruning to remove redundant bytes before processing",
        "It uses a fixed-size sliding window attention, limiting the effective context to 512 bytes"
      ],
      correct: 1,
      explanation: "MegaByte splits byte sequences into fixed-size patches (e.g., 8 bytes). A large transformer processes patch-level representations (sequence length divided by patch size, so $n/8$), and a smaller transformer predicts individual bytes within each patch conditioned on the global representation. The large model's $O((n/p)^2)$ cost is dramatically reduced, and the small model only operates on short sequences of length $p$. This \"patch-level then byte-level\" hierarchy avoids the $O(n^2)$ cost on raw byte sequences while eliminating the need for a fixed tokenizer vocabulary."
    },
    {
      type: "mc",
      question: "When generating Python code, a BPE tokenizer trained on mixed text/code often tokenizes 4-space indentation as a single token but may split unusual indentation inconsistently. Why is this problematic specifically for code generation?",
      options: [
        "It makes the model's code run slower at execution time",
        "Indentation is syntactically meaningful in Python — inconsistent tokenization of whitespace means the model must learn complex rules about how different token sequences map to the same indentation level, and errors produce silent semantic bugs (wrong block nesting) rather than obvious syntax errors",
        "The tokenizer will refuse to encode tab characters, preventing tab-indented code",
        "It increases the vocabulary size beyond what the embedding matrix can handle"
      ],
      correct: 1,
      explanation: "In Python, indentation defines block structure. If \"    \" (4 spaces) is one token at some depths but \"  \" + \"  \" (two tokens) at others, the model faces a many-to-one mapping problem: different token sequences produce identical indentation. Worse, getting whitespace wrong by even one space changes which block a line belongs to — a semantic error that's syntactically valid. Code-specific tokenizers (like Codex's) address this by ensuring consistent whitespace tokenization. This is one reason why code-specialized models often use different tokenizers than general-purpose LLMs."
    },
    {
      type: "mc",
      question: "Increasing BPE vocabulary size from 32K to 128K tokens has which set of tradeoffs?",
      options: [
        "Strictly better: shorter sequences, faster training, no downsides",
        "Shorter sequences and better compression, but larger embedding matrices ($V \\times d$ parameters), sparser token frequency distributions (many rare tokens with poor embeddings), and higher memory for the softmax output layer",
        "No effect on sequence length; it only changes the model's ability to represent rare words",
        "Longer sequences because more tokens are available, leading to finer-grained representations"
      ],
      correct: 1,
      explanation: "Larger vocab $\\Rightarrow$ more common strings get dedicated tokens $\\Rightarrow$ shorter sequences (better compression, faster inference). But: the embedding table grows from $32K \\times d$ to $128K \\times d$ (at $d = 4096$, that's 400M extra parameters). Many tokens in a 128K vocab are rare, so their embeddings are poorly trained. The final softmax over 128K classes is also more expensive. LLaMA uses 32K, GPT-4 uses ~100K, and Gemini uses 256K — the optimal size depends on the multilingual coverage needed and the model size (larger models can afford larger vocabs because the embedding table is a smaller fraction of total parameters)."
    },
    {
      type: "mc",
      question: "**WordPiece** (used in BERT) and **Unigram LM** (used in SentencePiece/T5) differ fundamentally in how they construct vocabularies. Which statement correctly describes the difference?",
      options: [
        "WordPiece builds bottom-up by merging pairs that maximize likelihood of the training corpus; Unigram LM starts with a large candidate vocabulary and prunes tokens whose removal least reduces the corpus likelihood — a top-down approach",
        "WordPiece operates on characters while Unigram LM operates on bytes",
        "WordPiece is deterministic while Unigram LM randomly samples tokenizations during training",
        "WordPiece produces fixed-length tokens while Unigram LM produces variable-length tokens"
      ],
      correct: 0,
      explanation: "WordPiece is bottom-up (like BPE) but selects merges that maximize corpus log-likelihood rather than raw frequency. Unigram LM is top-down: it starts with a large set of candidate subwords, assigns probabilities via an EM-like algorithm, and iteratively removes tokens that contribute least to corpus likelihood. A key advantage of Unigram LM is that it naturally supports multiple tokenizations of the same string with associated probabilities — enabling subword regularization (sampling different tokenizations during training as data augmentation). BPE/WordPiece produce a single deterministic segmentation."
    },
    {
      type: "mc",
      question: "A language model is asked \"How many r's are in 'strawberry'?\" and answers \"2\" (incorrect — the answer is 3). The tokenizer segments 'strawberry' as [\"str\", \"aw\", \"berry\"]. Why does the tokenization contribute to this failure?",
      options: [
        "The tokenizer removes duplicate characters during encoding",
        "The model never sees individual characters — it sees subword tokens, so it cannot directly count character occurrences within tokens. The 'r' in \"str\", the lack of 'r' in \"aw\", and the 'r' in \"berry\" are not represented at the character level in the model's input",
        "The tokenizer assigns the same embedding to 'r' regardless of context",
        "The model's vocabulary does not contain the character 'r' as a standalone token"
      ],
      correct: 1,
      explanation: "Subword tokenization is a lossy abstraction for character-level tasks. The model receives embeddings for [\"str\", \"aw\", \"berry\"] — there's no explicit representation of individual characters. To count 'r's, the model must have learned (from training data) the character composition of each token in its vocabulary — essentially memorizing a lookup table. It must recall that \"str\" contains one 'r', \"aw\" contains zero, and \"berry\" contains two 'r's, then sum to 3. This implicit character knowledge is unreliable, which is why character-level tasks (counting, spelling, anagrams) remain challenging for subword-tokenized models."
    },
    {
      type: "mc",
      question: "SentencePiece (used by LLaMA, T5, and others) treats the input as a raw byte stream and uses a special '\\u2581' (lower one eighth block) character to mark word boundaries. Why is this design choice important for multilingual models?",
      options: [
        "It allows the tokenizer to produce shorter sequences for all languages equally",
        "It removes the dependency on language-specific pre-tokenization rules (whitespace splitting, punctuation handling) — the tokenizer works identically on languages with spaces (English), without spaces (Chinese/Japanese), and with complex morphology (Turkish/Finnish)",
        "It enables the tokenizer to handle emojis and special characters",
        "It reduces the vocabulary size needed for CJK languages by sharing tokens with Latin scripts"
      ],
      correct: 1,
      explanation: "Traditional tokenizers (like GPT-2's) first split on whitespace and punctuation using regex rules designed for English, then apply BPE within each word. This fails for Chinese/Japanese (no spaces between words), agglutinative languages like Turkish (complex morphology), and many scripts. SentencePiece operates on raw text without any pre-tokenization assumptions — it learns word boundaries as part of the BPE/Unigram process, using '\\u2581' to mark where spaces were. This language-agnostic approach is essential for truly multilingual models."
    },
    {
      type: "mc",
      question: "A researcher notices that their 7B parameter model has a vocabulary of 256K tokens with $d_{\\text{model}} = 4096$. The embedding and language model head together account for what fraction of total model parameters, and is this a concern?",
      options: [
        "About 0.3% — negligible at any model scale",
        "About 30% ($2 \\times 256K \\times 4096 \\approx 2.1B$ out of ~7B) — a serious concern because these parameters are sparsely trained (each token's embedding updates only when that token appears) and the capacity is wasted on rare tokens",
        "Exactly 50% — embeddings always account for half of transformer parameters",
        "About 5% — noticeable but not worth addressing"
      ],
      correct: 1,
      explanation: "With untied embeddings: input embedding = $256K \\times 4096 \\approx 1.05B$ params. Output (LM head) = another $1.05B$. Total $\\approx 2.1B$ out of 7B = 30%. This is disproportionately large. Most of these parameters correspond to rare tokens that appear infrequently in training data, so their embeddings are poorly learned. Solutions include: tying input/output embeddings (halving the cost), using a smaller vocab, or factorizing the embedding matrix. LLaMA uses 32K vocab with untied embeddings — the embedding table is only $\\sim 260M$ params, a much healthier ratio at $\\sim 3.7\\%$."
    }
  ]
};


// ─────────────────────────────────────────────────────────────────────────────
// Section 1.3: Pretraining Objectives & Dynamics
// ─────────────────────────────────────────────────────────────────────────────
export const pretrainingAssessment = {
  id: "1.3-assess",
  sectionId: "1.3",
  title: "Assessment: Pretraining Objectives & Dynamics",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "The autoregressive language modeling objective maximizes $\\sum_{t=1}^{T} \\log P(x_t \\mid x_{<t}; \\theta)$. **Teacher forcing** means that during training:",
      options: [
        "The model's own predictions from the previous step are fed as input to the next step, allowing it to learn from its mistakes",
        "The ground-truth token $x_{t-1}$ is always provided as input when predicting $x_t$, regardless of what the model would have predicted — creating a train/inference mismatch called exposure bias",
        "A separate \"teacher\" model provides soft labels for knowledge distillation",
        "The learning rate is forced to follow a predetermined schedule rather than adapting"
      ],
      correct: 1,
      explanation: "During training, the model always sees the true previous tokens $x_{<t}$ (teacher forcing), but at inference it must consume its own potentially incorrect predictions. This creates **exposure bias**: the model never learns to recover from its own errors during training. In practice, LLMs are robust to this because (1) at scale, the model's predictions are usually correct, and (2) techniques like nucleus sampling avoid low-probability regions. Teacher forcing enables efficient parallelized training via causal masking — all positions can be computed in a single forward pass."
    },
    {
      type: "mc",
      question: "Chinchilla scaling laws (Hoffmann et al., 2022) suggest that for compute-optimal training, data tokens $D$ and parameters $N$ should scale roughly as $D \\approx 20N$. A 7B parameter model trained on 300B tokens — is this compute-optimal?",
      options: [
        "Yes — $300B / 7B \\approx 43$, which exceeds the 20:1 ratio, meaning the model is slightly over-trained on data but still compute-efficient",
        "No — Chinchilla says $D \\approx 20N = 140B$ tokens is optimal, so 300B tokens means this model is significantly over-trained relative to compute-optimal allocation; those FLOPs would be better spent on a larger model with fewer tokens",
        "No — the model needs far more data; the Chinchilla ratio suggests $D \\approx 20N = 140T$ tokens",
        "Yes — Chinchilla scaling only applies to models above 50B parameters"
      ],
      correct: 1,
      explanation: "Chinchilla-optimal for 7B params: $D \\approx 20 \\times 7B = 140B$ tokens. Training on 300B tokens means the compute budget spent on extra tokens ($300B - 140B = 160B$ extra tokens $\\times$ 6 $\\times$ 7B FLOPs/token) could instead have been used to train a ~13B model on 140B tokens for better performance. However, LLaMA (Touvron et al., 2023) deliberately over-trained on data because inference cost depends only on model size — a smaller, over-trained model is cheaper to deploy. This led to the \"LLaMA regime\" insight: Chinchilla-optimal isn't deployment-optimal."
    },
    {
      type: "mc",
      question: "**Grokking** (Power et al., 2022) is the phenomenon where a model first memorizes training data (achieving near-zero training loss while test loss remains high), then suddenly generalizes long after training loss has plateaued. What is the current best explanation?",
      options: [
        "The model runs out of capacity to memorize, forcing it to learn general patterns",
        "Weight decay gradually penalizes the high-norm memorization solution, eventually making the lower-norm generalizing solution more favorable — grokking is a competition between two loss basins mediated by regularization",
        "The learning rate schedule causes a sudden phase transition in the loss landscape",
        "Gradient noise from SGD randomly discovers the generalizing solution through diffusion"
      ],
      correct: 1,
      explanation: "The memorizing solution typically has higher weight norm than the generalizing solution. Without weight decay, the model stays in the memorization basin indefinitely. Weight decay slowly shrinks the weights, increasing the effective loss of the memorization solution until the generalizing basin becomes favorable. The \"delay\" in grokking corresponds to the time for weight decay to erode the memorization solution enough. Nanda et al. (2023) showed this mechanistically: models learn interpretable algorithms (e.g., modular arithmetic circuits) during grokking, and these circuits have lower weight norm than the memorization solution."
    },
    {
      type: "mc",
      question: "Learning rate warmup (linearly increasing LR from ~0 to peak over the first ~1-2% of training) is standard practice. What happens if you skip warmup and start at the peak learning rate?",
      options: [
        "Training is slightly slower but converges to the same final loss",
        "Early gradient updates are too large because the model's loss landscape is poorly conditioned at random initialization — Adam's variance estimates are also unreliable with few samples, producing oversized steps that can cause irreversible divergence",
        "The model immediately converges to a local minimum near initialization",
        "Warmup is only necessary for SGD; Adam-based optimizers don't need it"
      ],
      correct: 1,
      explanation: "Two factors make high initial LR dangerous: (1) At initialization, the loss landscape is highly curved and poorly conditioned — large steps overshoot. (2) Adam divides by $\\sqrt{\\hat{v}_t}$, the running estimate of squared gradients. With few steps, $\\hat{v}_t$ is a poor estimate, and the bias correction factor $1/(1-\\beta_2^t)$ amplifies early updates. Together, these cause enormous effective step sizes. Warmup gives Adam time to calibrate its moment estimates and lets the model find a smoother region of the loss landscape before taking full-sized steps."
    },
    {
      type: "mc",
      question: "The **cosine learning rate schedule** decays the LR as $\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})(1 + \\cos(\\pi t / T))$. Compared to linear decay, why has cosine become standard for LLM pretraining?",
      options: [
        "Cosine decay is mathematically optimal for convex optimization",
        "Cosine maintains a higher LR for longer during mid-training (slower initial decay), allowing continued exploration, then aggressively anneals at the end for fine convergence — the slow-fast decay profile empirically gives better final loss than linear",
        "Cosine decay uses less compute because it requires fewer gradient computations",
        "Cosine decay is equivalent to linear decay but is easier to implement in distributed settings"
      ],
      correct: 1,
      explanation: "Cosine decay's concave shape means the LR stays near $\\eta_{\\max}$ for roughly the first third of training, then drops steeply. This gives the model more time at high LR to escape poor basins and explore the loss landscape before committing to a solution. Linear decay reduces LR at a constant rate, which may reduce it too quickly in mid-training. Empirically, Loshchilov & Hutter (2016) showed cosine outperforms linear decay, and variants like cosine with warm restarts (SGDR) further improve performance by periodically resetting the LR."
    },
    {
      type: "mc",
      question: "The **linear scaling rule** states that when increasing batch size by a factor of $k$, the learning rate should also be scaled by $k$. What is the theoretical justification?",
      options: [
        "Larger batches have larger gradients, so a larger LR is needed to normalize them",
        "A $k\\times$ larger batch produces a gradient estimate with $k\\times$ less variance, which is equivalent to taking $k$ steps with the original batch — matching this requires $k\\times$ the LR per step to cover the same distance in weight space",
        "Larger batches require more epochs to converge, and higher LR compensates by covering more ground per step",
        "The linear scaling rule is a heuristic with no theoretical basis — it just works empirically"
      ],
      correct: 1,
      explanation: "With batch size $B$, one step moves weights by $\\eta \\cdot \\frac{1}{B}\\sum_{i=1}^{B} \\nabla_i$. With batch size $kB$, one step moves by $\\eta' \\cdot \\frac{1}{kB}\\sum_{i=1}^{kB} \\nabla_i$. If we want one large-batch step to equal $k$ small-batch steps (in expectation), we need $\\eta' = k\\eta$. This holds in the linear regime where the loss is approximately quadratic. Goyal et al. (2017) validated this up to batch sizes of 8K for ResNets. For very large batches, the linear regime breaks down and $\\sqrt{k}$ scaling or LARS/LAMB optimizers are needed."
    },
    {
      type: "mc",
      question: "During LLM pretraining, **loss spikes** — sudden sharp increases in training loss — are a common failure mode. Which of the following is the most common root cause and mitigation?",
      options: [
        "Data corruption in a specific batch; mitigated by data validation",
        "Gradient explosion due to outlier sequences or rare token combinations that produce extreme activations, amplified by the multiplicative nature of deep networks — mitigated by gradient clipping (typically to max norm 1.0) and sometimes by skipping the offending batch",
        "Learning rate being too low, causing the model to get stuck and then suddenly escape",
        "CPU-GPU synchronization errors in distributed training; mitigated by using synchronous SGD"
      ],
      correct: 1,
      explanation: "Loss spikes typically occur when a batch contains sequences that produce unusually large activations (e.g., repetitive tokens, unusual Unicode, or adversarial-like patterns). In a deep network, large activations cascade multiplicatively through layers, producing enormous gradients. Gradient clipping (capping $\\|g\\|$ at a threshold, typically 1.0) truncates these spikes. Some training runs also implement spike detection and batch skipping. The PaLM paper (Chowdhery et al., 2022) documents restarting training from earlier checkpoints after persistent spikes, sometimes re-ordering the data to skip problematic batches."
    },
    {
      type: "mc",
      question: "GPT-style models initialize the weights of residual-path projections (the output projections of attention and FFN) with a scaling factor of $\\frac{1}{\\sqrt{2L}}$ where $L$ is the number of layers. Why?",
      options: [
        "It makes the model output zero at initialization, which stabilizes the first gradient step",
        "Each of the $2L$ residual contributions (one from attention, one from FFN per layer) adds variance to the residual stream — scaling by $\\frac{1}{\\sqrt{2L}}$ keeps the total variance at initialization approximately constant regardless of depth, preventing signal explosion in deep networks",
        "It ensures all layers contribute equally to the final output at convergence",
        "It compensates for the variance reduction caused by LayerNorm at each layer"
      ],
      correct: 1,
      explanation: "The residual stream accumulates outputs from $2L$ sub-layers (attention + FFN per layer). If each sub-layer adds independent noise with variance $\\sigma^2$, the total variance after $2L$ additions is $2L\\sigma^2$, which grows with depth. Initializing output projections with std $\\propto 1/\\sqrt{2L}$ makes each contribution's variance $\\sigma^2/(2L)$, so the total remains $\\sigma^2$ regardless of $L$. This is critical for training stability: without it, activations in a 96-layer model would have $\\sim 14\\times$ the standard deviation of a 1-layer model at initialization. GPT-2 introduced this; it's sometimes called the \"residual scaling\" trick."
    },
    {
      type: "mc",
      question: "**Emergence** in LLMs refers to capabilities that appear abruptly as model scale increases — near-random performance below a threshold, then sharp improvement above it. Which recent finding has complicated the narrative around emergence?",
      options: [
        "Emerged capabilities are always predictable from smaller models using power laws",
        "Schaeffer et al. (2023) showed that many \"emergent\" abilities are artifacts of using discontinuous evaluation metrics (like exact-match accuracy) — when measured with continuous metrics (like token-level log-likelihood), performance improves smoothly and predictably with scale, not abruptly",
        "Emergence only occurs in models with more than 100B parameters",
        "Emergence is caused by phase transitions in the weight matrices analogous to physical phase transitions"
      ],
      correct: 1,
      explanation: "Schaeffer et al. (2023) demonstrated that apparent emergence is often a measurement artifact. Exact-match accuracy is a step function of model capability — a model that gets 90% of characters right in an answer still scores 0 on exact-match. As scale increases, per-token accuracy improves smoothly, and at some point crosses the threshold where entire answers are correct, causing exact-match to jump suddenly. Using continuous metrics like Brier score or per-token cross-entropy, the improvement is gradual and predictable from scaling laws. This doesn't mean all emergent behaviors are artifacts, but many reported cases are."
    },
    {
      type: "mc",
      question: "A pretraining run uses AdamW with $\\beta_1 = 0.9$, $\\beta_2 = 0.95$, weight decay $\\lambda = 0.1$, batch size 4M tokens, cosine LR schedule with peak $3 \\times 10^{-4}$, and gradient clipping at 1.0. The training loss suddenly becomes NaN at step 50,000. Which diagnostic would you check FIRST?",
      options: [
        "Whether the vocabulary size is correct",
        "The gradient norm history just before the NaN — a sudden spike above the clipping threshold, combined with checking for Inf/NaN in the most recent batch's activations and whether $\\beta_2 = 0.95$ (lower than typical 0.999) causes Adam's second moment to adapt too slowly to sudden gradient changes",
        "Whether the cosine schedule has decayed the LR too aggressively by step 50K",
        "Whether the tokenizer produced unknown tokens in the problematic batch"
      ],
      correct: 1,
      explanation: "NaN loss almost always traces back to numerical overflow in activations or gradients. The diagnostic chain: (1) Check gradient norm logs — was there a spike at step 49,999-50,000 that exceeded clip threshold? (2) Check for Inf/NaN in specific layers' activations (often the embedding or final logits). (3) Note that $\\beta_2 = 0.95$ (as used in LLaMA) makes Adam's second moment estimate more responsive but also more volatile — a sudden large gradient can cause the denominator $\\sqrt{v_t}$ to be small if previous gradients were small, leading to an enormous update. The fix: roll back to a checkpoint before the spike, optionally skip the offending batch, or add loss scaling."
    }
  ]
};
