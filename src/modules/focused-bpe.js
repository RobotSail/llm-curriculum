// Focused learning module for BYTE-PAIR ENCODING as a single concept.
// Covers: the open-vocabulary problem, the BPE training algorithm,
// merge tables, greedy encoding at inference, compression interpretation,
// vocabulary size tradeoffs, and byte-level BPE.

export const bpeLearning = {
  id: "1.2-bpe-learning-easy",
  sectionId: "1.2",
  title: "Byte-Pair Encoding",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — The problem tokenization solves
    {
      type: "info",
      title: "Why Tokenization Matters",
      content: "A language model must convert raw text into a sequence of discrete symbols from a fixed **vocabulary** before it can process anything. The choice of vocabulary determines what the model \"sees.\"\n\nTwo extreme strategies fail in opposite ways:\n\n**Character-level**: Vocabulary = {a, b, c, ..., z, 0-9, punctuation}. Every word is representable, but sequences become extremely long. The word \"tokenization\" is 12 characters — and attention cost scales as $O(n^2)$. A 2000-word document might become 10,000+ tokens.\n\n**Word-level**: Vocabulary = {the, cat, tokenization, ...}. Sequences are short, but the vocabulary must be enormous to cover all words, and any word not in the vocabulary (a typo, a new name, a foreign word) maps to a useless $\\langle\\text{UNK}\\rangle$ token. This is the **open-vocabulary problem**: natural language has an unbounded number of distinct word forms.\n\n**Subword tokenization** solves both problems: it uses a vocabulary of common substrings — full words for frequent terms (\"the\", \"model\") and word pieces for rarer terms (\"token\" + \"ization\"). Sequences stay short for common text, and any string can be encoded by falling back to individual characters. **Byte-Pair Encoding (BPE)** is the most widely used algorithm for learning this subword vocabulary."
    },
    // Step 2: MC — Understanding the open-vocabulary problem
    {
      type: "mc",
      question: "A word-level tokenizer with a 50,000-word vocabulary encounters the input \"DeepSeek-R1 achieves SOTA on MATH-500.\" Which failure mode is most likely?",
      options: [
        "The sentence is too long for the model's context window because word-level tokens consume more positions than subword tokens would",
        "\"DeepSeek-R1\" and \"MATH-500\" are mapped to $\\langle\\text{UNK}\\rangle$ tokens, destroying the model's ability to reason about these entities — the core open-vocabulary problem",
        "The tokenizer splits every word into individual characters as a fallback, producing a sequence of 40+ tokens that overwhelms the attention mechanism",
        "The punctuation marks (hyphen, period) are dropped during tokenization because word-level vocabularies only contain alphabetic entries"
      ],
      correct: 1,
      explanation: "Word-level tokenizers have a fixed vocabulary learned from training data. Novel compound terms like \"DeepSeek-R1\" and \"MATH-500\" are almost certainly absent from the vocabulary and get replaced with $\\langle\\text{UNK}\\rangle$. The model receives no information about what these tokens actually are — it just sees \"unknown entity achieves SOTA on unknown entity.\" This is the open-vocabulary problem: language constantly produces new words, names, and combinations that a fixed word list cannot anticipate. Subword tokenizers avoid this by decomposing unknown words into known pieces."
    },
    // Step 3: Info — The BPE training algorithm
    {
      type: "info",
      title: "The BPE Training Algorithm",
      content: "BPE learns a subword vocabulary from a training corpus in a simple, greedy process:\n\n**Initialization**: Start with a base vocabulary of all individual characters (or bytes) that appear in the corpus. Represent each word as a sequence of these characters, with a special end-of-word marker.\n\n**Iterative merging**:\n1. Count the frequency of every adjacent pair of symbols across the entire corpus\n2. Find the most frequent pair\n3. Merge that pair into a new single symbol and add it to the vocabulary\n4. Replace all occurrences of the pair in the corpus with the new symbol\n5. Repeat from step 1\n\n**Example**: Given corpus words \"low\" (5×), \"lower\" (2×), \"newest\" (6×), \"widest\" (3×):\n- Character vocabulary: {l, o, w, e, r, n, s, t, i, d}\n- Count pairs: (e, s) appears 9× (6 from \"newest\" + 3 from \"widest\"), (l, o) appears 7×, ...\n- Merge (e, s) → \"es\". Now \"newest\" = [n, e, w, es, t] and \"widest\" = [w, i, d, es, t]\n- Next iteration: recount pairs on the updated corpus and merge the new most frequent pair\n\nThe process continues until the vocabulary reaches a target size (e.g., 32,000 or 50,257 tokens). The **merge table** — the ordered list of merges — fully defines the tokenizer."
    },
    // Step 4: MC — Tracing a BPE merge step
    {
      type: "mc",
      question: "After the first merge (e, s) → \"es\" in the example above, the corpus is: \"low\" (5×) = [l, o, w], \"lower\" (2×) = [l, o, w, e, r], \"newest\" (6×) = [n, e, w, es, t], \"widest\" (3×) = [w, i, d, es, t]. What is the next merge?",
      options: [
        "(es, t) → \"est\" — appears 9× (6 from \"newest\" + 3 from \"widest\")",
        "(l, o) → \"lo\" — appears 7× (5 from \"low\" + 2 from \"lower\")",
        "(n, e) → \"ne\" — appears 6× (all from \"newest\")",
        "(e, w) → \"ew\" — appears 6× (all from \"newest\")"
      ],
      correct: 0,
      explanation: "After the first merge, we recount all adjacent pairs: (es, t) appears in \"newest\" (6×) and \"widest\" (3×) = 9 total. (l, o) appears in \"low\" (5×) and \"lower\" (2×) = 7 total. (n, e) and (e, w) each appear 6 times. The most frequent pair is (es, t) with 9 occurrences, so it becomes the second merge. After this merge, \"newest\" = [n, e, w, est] and \"widest\" = [w, i, d, est]. Notice how BPE builds common suffixes like \"est\" bottom-up through successive merges."
    },
    // Step 5: Info — Encoding at inference time
    {
      type: "info",
      title: "Applying BPE: The Merge Table at Inference",
      content: "Training BPE produces an ordered **merge table** — a list of pairs in the order they were merged. At inference time, this table is applied deterministically to tokenize new text:\n\n1. Split the input word into individual characters\n2. Look through the merge table **in order** (earliest merges first)\n3. If the current merge pair exists anywhere in the sequence, apply it (combine the pair into the merged symbol)\n4. Continue through the merge table until no more merges apply\n\n**The order matters.** Consider a merge table where merge #47 is (e, s) → \"es\" and merge #102 is (es, t) → \"est\". When encoding the word \"test\":\n- Start: [t, e, s, t]\n- Apply merge #47: (e, s) → \"es\" gives [t, es, t]\n- Apply merge #102: (es, t) → \"est\" gives [t, est]\n- No more applicable merges → final tokens: [\"t\", \"est\"]\n\nIf the word \"test\" is common enough in training, a later merge might combine (t, est) → \"test\", yielding a single token. Otherwise, it stays as two tokens. This is how BPE gracefully handles the frequency spectrum: common words become single tokens, uncommon words decompose into familiar pieces, and truly novel strings fall back to characters."
    },
    // Step 6: MC — Merge order matters
    {
      type: "mc",
      question: "A BPE merge table contains these merges in order: #1: (t, h) → \"th\", #2: (th, e) → \"the\", #3: (i, n) → \"in\", #4: (in, g) → \"ing\". What is the tokenization of \"thing\"?",
      options: [
        "[\"thing\"] — the full word is a single token because all necessary merges are present",
        "[\"th\", \"ing\"] — merge #1 produces \"th\", then merges #3 and #4 produce \"ing\"",
        "[\"the\", \"ing\"] — merge #2 greedily consumes \"the\" first, but \"ing\" doesn't exist yet at that point",
        "[\"t\", \"h\", \"i\", \"n\", \"g\"] — none of the merges apply because \"thing\" wasn't in the training corpus"
      ],
      correct: 1,
      explanation: "Starting from [t, h, i, n, g]: Merge #1 (t, h) applies → [th, i, n, g]. Merge #2 (th, e) does NOT apply because the next character after \"th\" is \"i\", not \"e\". Merge #3 (i, n) applies → [th, in, g]. Merge #4 (in, g) applies → [th, ing]. No more merges apply. Result: [\"th\", \"ing\"]. Note that merge #2 tried to fire but couldn't — the merge table is applied in order, but each merge only fires when its exact pair is adjacent in the current sequence. BPE doesn't look ahead or backtrack."
    },
    // Step 7: Info — BPE as compression
    {
      type: "info",
      title: "BPE and Compression",
      content: "BPE has a deep connection to **data compression**. The algorithm was originally invented by Philip Gage (1994) as a compression scheme, not for NLP.\n\nEach merge replaces two symbols with one, shortening the encoded sequence. The most frequent pairs are merged first, so BPE preferentially compresses the most common patterns — exactly what a good compressor should do. After $k$ merges, the vocabulary has grown by $k$ symbols and the total corpus length (in tokens) has decreased.\n\nThe **compression ratio** — the ratio of character-level length to BPE token-level length — measures how efficiently the tokenizer represents text. For English with a 50K vocabulary, typical compression is about 3.5-4× (one token per ~3.7 characters on average). This means:\n- A 4096-token context window holds roughly 15,000 characters (~2,500 words)\n- Each token carries about $\\log_2(50{,}000) \\approx 15.6$ bits of vocabulary information\n\nLanguages with complex morphology or scripts not well-represented in training data compress less efficiently — a critical fairness consideration. A tokenizer trained mostly on English might achieve 4× compression on English but only 1.5× on Thai, meaning Thai users get less than half the effective context window."
    },
    // Step 8: MC — Compression understanding
    {
      type: "mc",
      question: "A BPE tokenizer with a 32K vocabulary achieves a compression ratio of 4.0× on English (1 token per 4 characters on average). A model with a 4096-token context window processes an English document. Approximately how many characters of text fit in the context?",
      options: [
        "4,096 characters — each token corresponds to exactly one character regardless of compression",
        "About 16,400 characters — $4096 \\times 4.0$ characters per token",
        "About 131,000 characters — $4096 \\times 32$ because the vocabulary size determines characters per token",
        "About 1,024 characters — the context window is divided by the compression ratio, not multiplied"
      ],
      correct: 1,
      explanation: "Compression ratio of 4.0× means each token represents ~4 characters on average. With 4096 tokens available, the model can process $4096 \\times 4.0 = 16{,}384$ characters. This is roughly 2,700 words of English text. The compression ratio is determined by how well the BPE vocabulary matches the input distribution — not by the vocabulary size directly (though larger vocabularies generally achieve better compression). This is why the same model effectively has different context lengths for different languages."
    },
    // Step 9: Info — Vocabulary size tradeoffs
    {
      type: "info",
      title: "Vocabulary Size: The Central Tradeoff",
      content: "The target vocabulary size $|V|$ is BPE's most important hyperparameter, and it controls a fundamental tradeoff:\n\n**Larger vocabulary** (e.g., 128K or 256K tokens):\n- Better compression → shorter sequences → faster inference\n- More words become single tokens → fewer fragmentation artifacts\n- But: the embedding matrix has $|V| \\times d$ parameters. At $|V| = 256K$ and $d = 4096$, that's ~1 billion parameters just for embeddings\n- Many tokens are rare and their embeddings are poorly trained\n- The softmax output layer over $|V|$ classes is more expensive\n\n**Smaller vocabulary** (e.g., 32K tokens):\n- Smaller embedding table → more parameter budget for transformer layers\n- Every token appears frequently → well-trained embeddings\n- But: longer sequences → higher attention cost ($O(n^2)$)\n- More words fragmented into pieces → harder for the model to learn word-level semantics\n\nIn practice: GPT-2 uses 50,257 tokens, LLaMA uses 32,000, GPT-4 uses ~100K, and Gemini uses 256K. The optimal size depends on model scale (larger models can afford larger vocabularies since embeddings are a smaller fraction of total parameters) and multilingual coverage needs."
    },
    // Step 10: MC — Vocabulary size effects
    {
      type: "mc",
      question: "A 7B parameter model uses a vocabulary of 256K tokens with $d_{\\text{model}} = 4096$. If the input and output embedding matrices are NOT tied (separate parameters), approximately what fraction of the model's parameters are in the embedding layers?",
      options: [
        "About 1% — embedding parameters are negligible at this model scale",
        "About 30% — input ($256K \\times 4096 \\approx 1.05B$) plus output ($\\approx 1.05B$) totals ~2.1B out of 7B",
        "About 7% — only the input embedding matters since the output head reuses input embeddings in modern architectures",
        "About 50% — embedding parameters always dominate in transformer models regardless of vocabulary size"
      ],
      correct: 1,
      explanation: "With untied embeddings: input embedding = $256{,}000 \\times 4{,}096 \\approx 1.05B$ parameters, output (LM head) = another $\\approx 1.05B$. Total embedding parameters $\\approx 2.1B$, which is $2.1/7.0 = 30\\%$ of the model. This is wasteful — most of those 256K tokens are rare, so their embeddings get few gradient updates. By contrast, LLaMA's 32K vocabulary with the same $d = 4096$ uses only $\\sim 260M$ embedding parameters ($\\sim 3.7\\%$). This illustrates why vocabulary size must be chosen relative to model size: a 256K vocabulary might be appropriate for a 70B model but is disproportionate for a 7B model."
    },
    // Step 11: Info — Handling unseen words
    {
      type: "info",
      title: "Graceful Degradation for Rare Words",
      content: "A key advantage of BPE over word-level tokenizers is **graceful degradation**: no input ever maps to $\\langle\\text{UNK}\\rangle$.\n\nWhen BPE encounters a word it hasn't seen during training, the merge table simply applies fewer merges. The word decomposes into smaller known pieces, ultimately falling back to individual characters (or bytes) if necessary.\n\n**Example**: The novel protein name \"CRISPR-Cas9\" might tokenize as [\"CR\", \"IS\", \"PR\", \"-\", \"Cas\", \"9\"]. None of these pieces are $\\langle\\text{UNK}\\rangle$ — they're all valid vocabulary entries that the model has seen in other contexts. The model can use its knowledge of these substrings to make reasonable inferences about the input.\n\nThis property means BPE's effective coverage is **100%** over any Unicode text (or any byte sequence, for byte-level BPE). The cost of a rare word is simply a longer token sequence, not a total loss of information. This is qualitatively different from word-level tokenizers, where an unknown word carries zero information.\n\nThe tradeoff is that heavily fragmented words are harder for the model to process: it must compose meaning from pieces across multiple positions, using attention to integrate them."
    },
    // Step 12: MC — Rare word behavior
    {
      type: "mc",
      question: "A BPE tokenizer trained primarily on English text encounters the Finnish word \"epäjärjestelmällisyydestäkään\" (a real word meaning \"not even from the lack of systematicity\"). What happens?",
      options: [
        "The word is mapped to $\\langle\\text{UNK}\\rangle$ because it was never seen during BPE training",
        "The tokenizer produces an error because the word exceeds the maximum token length supported by the BPE implementation",
        "The word is split into many small subword pieces — possibly down to individual characters for the Finnish-specific morphemes — producing a long but valid token sequence",
        "The word is transliterated into the closest English equivalent before tokenization, preserving meaning but losing the original orthography"
      ],
      correct: 2,
      explanation: "BPE never produces $\\langle\\text{UNK}\\rangle$ — it always has character-level (or byte-level) fallback. The Finnish word would be fragmented into many pieces: perhaps [\"ep\", \"ä\", \"j\", \"är\", \"jest\", \"el\", \"m\", \"äl\", \"lis\", \"yy\", \"dest\", \"ä\", \"kään\"]. Common English substrings like \"est\" and \"lis\" get merged; Finnish-specific characters like \"ä\" may remain as individual tokens. The word might become 10-15 tokens where an English word of similar length might be 2-3. This high **fertility ratio** is why multilingual models benefit from tokenizers trained on balanced multilingual corpora."
    },
    // Step 13: Info — Byte-level BPE
    {
      type: "info",
      title: "Byte-Level BPE: Eliminating the Character Set Problem",
      content: "Standard BPE starts from a character-level vocabulary — but what counts as a \"character\"? Unicode has over 150,000 characters across hundreds of scripts. Including all of them in the base vocabulary is wasteful; excluding some means those characters can't be encoded.\n\n**Byte-level BPE** (introduced by GPT-2) solves this elegantly: the base vocabulary is the 256 possible byte values (0x00 through 0xFF). Every possible input — any Unicode text, code, even binary data — is first encoded as a UTF-8 byte sequence, then BPE merges are applied to the bytes.\n\nAdvantages:\n- **Universal coverage**: any input that can be represented in UTF-8 can be tokenized. No $\\langle\\text{UNK}\\rangle$ ever, not even for rare scripts or emoji.\n- **Small base vocabulary**: only 256 base tokens vs. thousands of Unicode characters.\n- **Shared subword structure**: the byte representation lets BPE discover shared patterns across scripts (e.g., UTF-8 encodings of accented Latin characters share byte prefixes).\n\nThe downside is that non-ASCII characters require 2-4 bytes in UTF-8, so languages with non-Latin scripts start from a longer base sequence. GPT-2, GPT-3, GPT-4, and many other models use byte-level BPE. The merge table operates on byte sequences rather than character sequences, but the algorithm is otherwise identical."
    },
    // Step 14: MC — Byte-level BPE understanding
    {
      type: "mc",
      question: "In byte-level BPE, the Chinese character '中' is encoded in UTF-8 as three bytes: [0xE4, 0xB8, 0xAD]. If no BPE merges have been learned for this byte sequence, how is '中' tokenized?",
      options: [
        "As a single $\\langle\\text{UNK}\\rangle$ token — the character is outside the vocabulary",
        "As three separate byte tokens — each byte is a valid vocabulary entry in byte-level BPE",
        "As one token — individual Unicode characters are always atomic in byte-level BPE regardless of their UTF-8 length",
        "The tokenizer silently drops the character because non-ASCII bytes are filtered during preprocessing"
      ],
      correct: 1,
      explanation: "In byte-level BPE, the base vocabulary is all 256 byte values. Each byte — including 0xE4, 0xB8, and 0xAD — is a valid token. Without a learned merge for this sequence, '中' becomes three tokens. If Chinese text is common in the training corpus, BPE will eventually learn to merge these three bytes into a single token (or at least merge two of them). This is why tokenizers trained on English-heavy corpora have poor compression for CJK languages: the byte-level merges for CJK byte patterns were never learned (or learned late, at low priority), so each character consumes 2-3 tokens instead of 1."
    }
  ]
};
