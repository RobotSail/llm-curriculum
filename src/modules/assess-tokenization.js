// Assessment: Tokenization (Section 1.2)
// 10 MC questions, no info steps. Pure assessment module.

export const tokenizationAssessment = {
  id: "1.2-assess",
  sectionId: "1.2",
  title: "Assessment: Tokenization",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Byte Pair Encoding (BPE) builds its vocabulary by iteratively merging the most frequent adjacent pair of tokens. If the training corpus contains the words \"low\" (5x), \"lower\" (2x), \"newest\" (6x), \"widest\" (3x), and the initial vocabulary is characters, which pair is merged first?",
      options: ["('l', 'o') — appears 7 times across 'low' and 'lower'", "('e', 's') — appears 9 times across 'newest' and 'widest'", "('n', 'e') — appears 6 times in 'newest'", "('w', 'i') — appears 3 times in 'widest'"],
      correct: 1,
      explanation: "BPE counts pair frequencies across the entire corpus weighted by word frequency. ('e', 's') appears in 'newest' (6 times) and 'widest' (3 times) = 9 total occurrences. ('l', 'o') appears in 'low' (5) + 'lower' (2) = 7. ('n', 'e') appears 6 times. So ('e', 's') $\\to$ 'es' is the first merge. This greedy approach builds common subwords bottom-up. Note that BPE is deterministic given the corpus — the merge order defines the tokenizer completely."
    },
    {
      type: "mc",
      question: "A GPT-style tokenizer with vocabulary size 50,257 tokenizes the arithmetic expression \"123456 + 789\" into tokens [\"123\", \"456\", \" +\", \" 789\"]. Why does this tokenization make arithmetic difficult for the model?",
      options: ["The digit groupings are misaligned with place value — \"123456\" splits into \"123\" and \"456\", so the model must learn that the first token's digits occupy the hundred-thousands, ten-thousands, and thousands places, while the second token's digits are hundreds, tens, and ones — a positional encoding the tokenizer doesn't preserve", "The model cannot represent numbers larger than the vocabulary size of 50,257, so any operand or result exceeding this value falls outside the embedding table and is mapped to an unknown token that carries no numerical semantics", "The \"+\" operator is merged with a preceding space into a single token, confusing the model about its semantic role as an arithmetic operator and making it difficult to distinguish addition from other operations that share similar token patterns", "The model has no dedicated embedding for the composite number 123456 since it exceeds the vocabulary, forcing it to rely on a generic fallback representation that discards the magnitude and digit-level structure of the original number"],
      correct: 0,
      explanation: "Tokenization destroys digit-level positional structure. When \"123456\" becomes [\"123\", \"456\"], the model must implicitly learn that \"123\" means $123 \\times 1000$ (not just 123) based on the following token. Worse, the same token \"123\" might mean 123, 123000, or 123000000 depending on context. Each digit's place value is a function of total number length and token boundaries — information the model must infer rather than directly access. This is why digit-level tokenization (one token per digit) dramatically improves arithmetic performance."
    },
    {
      type: "mc",
      question: "The **fertility ratio** of a tokenizer for a language is the average number of tokens produced per word. A tokenizer trained predominantly on English text has a fertility ratio of 1.2 for English but 3.8 for Tamil. What concrete consequence does this have?",
      options: ["Tamil text will have higher perplexity because the model has seen less Tamil data during pretraining, but the tokenizer's fertility ratio has no direct effect on downstream task performance or cost", "The model will produce more fluent Tamil because it has more tokens per word to work with, giving the autoregressive decoder finer-grained control over each word's generation and more opportunities to correct errors", "Tamil text consumes roughly $3.2\\times$ more context window and costs $3.2\\times$ more per word in API pricing, and the model sees $3.2\\times$ fewer Tamil words in the same context length — systematically disadvantaging Tamil users", "Tamil sentences will be more compressible after tokenization because the higher token count provides more symbols for the model's internal entropy coding, enabling better downstream compression ratios in storage and transmission"],
      correct: 2,
      explanation: "High fertility ratio means each Tamil word is fragmented into many tokens. With a 4096-token context window, English gets ~3400 words of context while Tamil gets only ~1080 words — a $3.2\\times$ reduction. API costs (priced per token) are proportionally higher per word. During training, each Tamil word also requires more forward-pass steps, reducing effective data efficiency. This is a systematic fairness issue: the tokenizer itself creates unequal capability across languages. Solutions include multilingual-aware BPE training, byte-level models, or language-specific tokenizers."
    },
    {
      type: "mc",
      question: "**MegaByte** (Yu et al., 2023) proposes a byte-level architecture that avoids subword tokenization entirely. What is the primary architectural innovation that makes byte-level modeling tractable despite sequences being ~4x longer than subword sequences?",
      options: ["It uses a convolutional backbone instead of attention for the byte-level processing, applying dilated causal convolutions that provide a linearly-scaling receptive field while avoiding the quadratic memory and compute cost entirely", "It uses a fixed-size sliding window attention of 512 bytes at the byte level, limiting the effective local context but stacking multiple layers with increasing window sizes to approximate global attention through hierarchical receptive fields", "It applies aggressive byte-level pruning before processing, using a lightweight classifier to identify and remove redundant or uninformative bytes (whitespace, repeated characters, padding) to shorten the effective sequence length", "It uses a hierarchical architecture with a large \"global\" model operating on patches of bytes and a smaller \"local\" model predicting individual bytes within each patch, amortizing the cost of the large model"],
      correct: 3,
      explanation: "MegaByte splits byte sequences into fixed-size patches (e.g., 8 bytes). A large transformer processes patch-level representations (sequence length divided by patch size, so $n/8$), and a smaller transformer predicts individual bytes within each patch conditioned on the global representation. The large model's $O((n/p)^2)$ cost is dramatically reduced, and the small model only operates on short sequences of length $p$. This \"patch-level then byte-level\" hierarchy avoids the $O(n^2)$ cost on raw byte sequences while eliminating the need for a fixed tokenizer vocabulary."
    },
    {
      type: "mc",
      question: "When generating Python code, a BPE tokenizer trained on mixed text/code often tokenizes 4-space indentation as a single token but may split unusual indentation inconsistently. Why is this problematic specifically for code generation?",
      options: [
        "It makes the generated code slower at execution time because the token boundaries introduce implicit overhead, causing the Python interpreter to handle suboptimally chunked source strings during parsing and compilation",
        "Indentation is syntactically meaningful in Python — inconsistent tokenization of whitespace means the model must learn complex rules about how different token sequences map to the same indentation level, and errors produce silent semantic bugs (wrong block nesting) rather than obvious syntax errors",
        "The tokenizer cannot encode tab characters reliably because tabs fall outside the learned BPE merge table, preventing any tab-indented code from being generated and limiting the model to spaces-only output",
        "It forces the vocabulary to include many whitespace-only tokens of varying lengths, bloating the embedding matrix beyond what the model can efficiently learn and wasting capacity on non-semantic entries"
      ],
      correct: 1,
      explanation: "In Python, indentation defines block structure. If \"    \" (4 spaces) is one token at some depths but \"  \" + \"  \" (two tokens) at others, the model faces a many-to-one mapping problem: different token sequences produce identical indentation. Worse, getting whitespace wrong by even one space changes which block a line belongs to — a semantic error that's syntactically valid. Code-specific tokenizers (like Codex's) address this by ensuring consistent whitespace tokenization. This is one reason why code-specialized models often use different tokenizers than general-purpose LLMs."
    },
    {
      type: "mc",
      question: "Increasing BPE vocabulary size from 32K to 128K tokens has which set of tradeoffs?",
      options: ["Shorter sequences and better compression, but larger embedding matrices ($V \\times d$ parameters), sparser token frequency distributions (many rare tokens with poor embeddings), and higher memory for the softmax output layer", "Strictly better in all respects: shorter sequences reduce compute quadratically for attention, faster training per epoch due to fewer tokens, and no meaningful downsides at any model scale", "No effect on sequence length because the compression ratio is determined by the training data distribution, not vocabulary size; the only change is the model's ability to represent rare words as single tokens", "Longer sequences because the larger vocabulary fragments common words into more specialized subtokens, leading to finer-grained representations that increase sequence length but improve per-token information density"],
      correct: 0,
      explanation: "Larger vocab $\\Rightarrow$ more common strings get dedicated tokens $\\Rightarrow$ shorter sequences (better compression, faster inference). But: the embedding table grows from $32K \\times d$ to $128K \\times d$ (at $d = 4096$, that's 400M extra parameters). Many tokens in a 128K vocab are rare, so their embeddings are poorly trained. The final softmax over 128K classes is also more expensive. LLaMA uses 32K, GPT-4 uses ~100K, and Gemini uses 256K — the optimal size depends on the multilingual coverage needed and the model size (larger models can afford larger vocabs because the embedding table is a smaller fraction of total parameters)."
    },
    {
      type: "mc",
      question: "**WordPiece** (used in BERT) and **Unigram LM** (used in SentencePiece/T5) differ fundamentally in how they construct vocabularies. Which statement correctly describes the difference?",
      options: ["WordPiece produces a single deterministic segmentation while Unigram LM randomly samples from multiple possible tokenizations during training, but both use the same bottom-up merge strategy to build their vocabularies", "WordPiece operates on characters as its base units while Unigram LM operates on raw bytes, giving Unigram LM the ability to handle any input without unknown tokens but at the cost of longer sequences", "WordPiece builds bottom-up by merging pairs that maximize likelihood of the training corpus; Unigram LM starts with a large candidate vocabulary and prunes tokens whose removal least reduces the corpus likelihood — a top-down approach", "WordPiece produces fixed-length tokens of exactly $k$ characters while Unigram LM produces variable-length tokens, allowing Unigram LM to adapt its granularity to the morphological complexity of each word"],
      correct: 2,
      explanation: "WordPiece is bottom-up (like BPE) but selects merges that maximize corpus log-likelihood rather than raw frequency. Unigram LM is top-down: it starts with a large set of candidate subwords, assigns probabilities via an EM-like algorithm, and iteratively removes tokens that contribute least to corpus likelihood. A key advantage of Unigram LM is that it naturally supports multiple tokenizations of the same string with associated probabilities — enabling subword regularization (sampling different tokenizations during training as data augmentation). BPE/WordPiece produce a single deterministic segmentation."
    },
    {
      type: "mc",
      question: "A language model is asked \"How many r's are in 'strawberry'?\" and answers \"2\" (incorrect — the answer is 3). The tokenizer segments 'strawberry' as [\"str\", \"aw\", \"berry\"]. Why does the tokenization contribute to this failure?",
      options: ["The tokenizer removes duplicate consecutive characters during encoding, collapsing repeated letters like 'rr' in 'strawberry' into a single occurrence and losing count information in the process", "The model's vocabulary does not contain the character 'r' as a standalone token, so it has no embedding that directly represents the letter and must infer its presence from the subword tokens that contain it", "The tokenizer assigns the same fixed embedding to every occurrence of 'r' regardless of its position within a token, preventing the model from distinguishing the 'r' in 'str' from the 'r' in 'berry'", "The model never sees individual characters — it sees subword tokens, so it cannot directly count character occurrences within tokens. The 'r' in \"str\", the lack of 'r' in \"aw\", and the 'r' in \"berry\" are not represented at the character level in the model's input"],
      correct: 3,
      explanation: "Subword tokenization is a lossy abstraction for character-level tasks. The model receives embeddings for [\"str\", \"aw\", \"berry\"] — there's no explicit representation of individual characters. To count 'r's, the model must have learned (from training data) the character composition of each token in its vocabulary — essentially memorizing a lookup table. It must recall that \"str\" contains one 'r', \"aw\" contains zero, and \"berry\" contains two 'r's, then sum to 3. This implicit character knowledge is unreliable, which is why character-level tasks (counting, spelling, anagrams) remain challenging for subword-tokenized models."
    },
    {
      type: "mc",
      question: "SentencePiece (used by LLaMA, T5, and others) treats the input as a raw byte stream and uses a special '\\u2581' (lower one eighth block) character to mark word boundaries. Why is this design choice important for multilingual models?",
      options: [
        "It allows the tokenizer to produce equally short sequences for all languages by dynamically adjusting the merge table at inference time, ensuring uniform compression across scripts and morphologies",
        "It removes the dependency on language-specific pre-tokenization rules (whitespace splitting, punctuation handling) — the tokenizer works identically on languages with spaces (English), without spaces (Chinese/Japanese), and with complex morphology (Turkish/Finnish)",
        "It enables the tokenizer to handle emojis, special characters, and non-standard Unicode sequences that would otherwise be mapped to unknown tokens and lost during encoding",
        "It reduces the vocabulary size needed for CJK languages by automatically sharing subword tokens with Latin scripts, merging overlapping character patterns across writing systems"
      ],
      correct: 1,
      explanation: "Traditional tokenizers (like GPT-2's) first split on whitespace and punctuation using regex rules designed for English, then apply BPE within each word. This fails for Chinese/Japanese (no spaces between words), agglutinative languages like Turkish (complex morphology), and many scripts. SentencePiece operates on raw text without any pre-tokenization assumptions — it learns word boundaries as part of the BPE/Unigram process, using '\\u2581' to mark where spaces were. This language-agnostic approach is essential for truly multilingual models."
    },
    {
      type: "mc",
      question: "A researcher notices that their 7B parameter model has a vocabulary of 256K tokens with $d_{\\text{model}} = 4096$. The embedding and language model head together account for what fraction of total model parameters, and is this a concern?",
      options: ["About 30% ($2 \\times 256K \\times 4096 \\approx 2.1B$ out of ~7B) — a serious concern because these parameters are sparsely trained (each token's embedding updates only when that token appears) and the capacity is wasted on rare tokens", "About 0.3% — negligible at any model scale because the embedding dimensions are small relative to the cumulative parameter count of all transformer layers combined", "Exactly 50% — embeddings always account for half of transformer parameters by construction, since the input embedding and output projection mirror the total depth of the transformer stack", "About 5% — noticeable but not worth addressing, since the embedding parameters are dense and well-trained across the full vocabulary due to the uniform token distribution in web-scale corpora"],
      correct: 0,
      explanation: "With untied embeddings: input embedding = $256K \\times 4096 \\approx 1.05B$ params. Output (LM head) = another $1.05B$. Total $\\approx 2.1B$ out of 7B = 30%. This is disproportionately large. Most of these parameters correspond to rare tokens that appear infrequently in training data, so their embeddings are poorly learned. Solutions include: tying input/output embeddings (halving the cost), using a smaller vocab, or factorizing the embedding matrix. LLaMA uses 32K vocab with untied embeddings — the embedding table is only $\\sim 260M$ params, a much healthier ratio at $\\sim 3.7\\%$."
    }
  ]
};
