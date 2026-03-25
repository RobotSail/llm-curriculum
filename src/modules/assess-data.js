// Assessment: Data (Section 1.4)
// 10 MC questions, no info steps. Pure assessment module.

export const dataAssessment = {
  id: "1.4-assess",
  sectionId: "1.4",
  title: "Assessment: Data",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In a typical web-scale data pipeline for LLM pre-training, which step comes **first**?",
      options: [
        "Deduplication using MinHash to remove near-duplicate documents from the raw crawl",
        "Language identification and URL-based filtering of raw crawl data",
        "Perplexity filtering with a trained language model to score document quality",
        "Data mixing and upsampling of underrepresented domains to balance the corpus"
      ],
      correct: 1,
      explanation: "The pipeline begins with coarse, cheap filters: language ID (fastText-based), URL blocklists, and removing boilerplate HTML. These steps discard the bulk of low-quality data before expensive operations like deduplication or model-based filtering. Running perplexity filtering on unfiltered Common Crawl would be prohibitively expensive and pointless on non-target-language text."
    },
    {
      type: "mc",
      question: "**MinHash** is used for near-duplicate detection in large corpora. It works by:",
      options: ["Applying multiple hash functions to token n-gram shingle sets and estimating Jaccard similarity from the fraction of matching minimum hash values", "Computing exact MD5 hashes of each document and performing pairwise comparison of the resulting fixed-length digests to identify identical documents", "Training a neural network to embed documents into a dense vector space and then clustering nearby embeddings to identify groups of similar content", "Sorting all documents alphabetically by their first 128 bytes and comparing adjacent pairs in the sorted order to detect near-duplicates within local neighborhoods"],
      correct: 0,
      explanation: "MinHash approximates the Jaccard similarity $J(A, B) = |A \\cap B| / |A \\cup B|$ between shingle sets. For each of $k$ hash functions, the minimum hash value is recorded. The fraction of matching minimums across two documents estimates their Jaccard similarity. This is efficient because it reduces each document to a fixed-size signature, enabling Locality-Sensitive Hashing (LSH) for scalable pairwise comparison without computing all $O(n^2)$ pairs."
    },
    {
      type: "mc",
      question: "What is the key difference between **exact deduplication** and **near-deduplication** in data pipelines?",
      options: ["Exact dedup is more computationally expensive because it requires comparing every pair of documents at the byte level, while near-dedup uses approximate hashing to avoid pairwise comparisons entirely", "Near-dedup only works on structured code where syntax trees provide a canonical representation for comparison, not on natural language where paraphrasing makes similarity harder to define", "Exact dedup removes byte-identical or hash-identical documents (or n-gram spans), while near-dedup removes documents above a similarity threshold, catching paraphrases, boilerplate variations, and templated content", "They produce identical results on web-scale corpora in practice but near-dedup is faster because it uses probabilistic data structures that trade a small false-positive rate for dramatically reduced computation"],
      correct: 2,
      explanation: "Exact dedup uses cryptographic hashes (SHA-256, MD5) on full documents or suffix arrays on n-gram spans to find verbatim matches. Near-dedup (MinHash + LSH, SimHash) catches documents that are substantially similar but not identical — e.g., the same news article republished with minor edits, or pages with shared templates. Both are necessary: exact dedup is cheap and catches copies; near-dedup catches the long tail of near-copies that dominate web crawls."
    },
    {
      type: "mc",
      question: "**Perplexity filtering** uses a language model (often a small KenLM n-gram model trained on curated text like Wikipedia) to score documents. Documents are kept if:",
      options: ["Their perplexity is as high as possible, indicating novel and diverse content that the language model has rarely encountered during training", "Their perplexity is below a fixed threshold of 100, ensuring the text closely resembles the style and fluency of curated reference corpora", "Their perplexity exactly matches Wikipedia's average perplexity, ensuring the text is neither too specialized nor too casual in its register", "Their perplexity falls within a middle range — not too low (repetitive/templated) and not too high (gibberish/non-fluent)"],
      correct: 3,
      explanation: "The key insight from CCNet and related work is that both extremes are bad. Very low perplexity indicates repetitive boilerplate (cookie notices, navigation menus) that the LM finds trivially predictable. Very high perplexity indicates garbled text, foreign-language fragments, or encoding artifacts. The \"Goldilocks zone\" in the middle captures fluent, diverse, informative text. The exact thresholds are tuned empirically per domain."
    },
    {
      type: "mc",
      question: "The **Phi-1** (\"Textbooks Are All You Need\") approach demonstrated that:",
      options: [
        "Models must be trained on at least 1 trillion tokens to achieve useful performance on coding benchmarks, because the statistical patterns needed for code generation require extremely large-scale data exposure",
        "A small model trained on high-quality synthetic \"textbook-style\" data and exercises can match or outperform much larger models trained on raw web data, showing that data quality can substitute for scale",
        "Synthetic data generated by a capable teacher model always outperforms real human-written data regardless of domain, because the teacher can produce more consistent and error-free examples at arbitrary volume",
        "GPT-4 can fully replace human data annotators at no cost for all downstream tasks, eliminating the need for human-curated datasets entirely and making data collection a solved problem"
      ],
      correct: 1,
      explanation: "Phi-1 (1.3B params) matched GPT-3.5 on HumanEval by training on ~7B tokens of GPT-4-generated textbook-quality code explanations and exercises. This challenged the prevailing \"more data = better\" paradigm by showing that carefully curated synthetic data with high information density per token can dramatically improve sample efficiency. The key was not just using synthetic data, but generating data specifically designed to teach concepts progressively."
    },
    {
      type: "mc",
      question: "**Model collapse** from training on self-generated data occurs because:",
      options: ["Each generation step loses information from the tails of the distribution — the model's approximation errors compound across generations, causing the distribution to narrow and eventually degenerate, losing minority patterns and rare knowledge", "The model runs out of GPU memory during iterative self-training because each generation stores additional KV cache entries that accumulate across rounds, eventually exceeding the available HBM capacity", "Synthetic data consistently contains more tokens per document than real data because language models tend to produce verbose outputs, causing the training set to grow unboundedly and diluting the signal-to-noise ratio", "The tokenizer cannot encode synthetic text reliably because model-generated outputs contain subtle Unicode artifacts and non-standard character sequences that fall outside the learned BPE merge table"],
      correct: 0,
      explanation: "Shumailov et al. (2023) showed that iteratively training on model-generated data causes progressive loss of distributional tails. The model slightly underweights low-probability events; when these outputs become training data, those events become even rarer in the next generation. Over multiple iterations, this positive feedback loop collapses the distribution to a narrow mode. Mathematically, each generation is like repeatedly applying an imperfect density estimator, and the composition amplifies errors — analogous to photocopying a photocopy."
    },
    {
      type: "mc",
      question: "A common **data mixing** strategy for pre-training assigns sampling weights to different domains (web, books, code, Wikipedia, etc.). The **DoReMi** approach determines these weights by:",
      options: ["Setting sampling weights proportional to each domain's raw byte count so that larger domains are represented more frequently, matching the natural data distribution", "Always upsampling code to 50% of the mixture regardless of its natural proportion, because code provides the strongest signal for reasoning and structured output generation", "Using a small proxy model to learn domain weights that minimize worst-case excess loss across domains, then applying those weights to train the large model", "Randomly shuffling all domains together with equal per-document sampling probability, ensuring uniform exposure across all sources without any domain-level weighting"],
      correct: 2,
      explanation: "DoReMi (Xie et al., 2023) trains a small proxy model with distributionally robust optimization (DRO) to find domain weights that minimize the worst-case excess loss over a reference model. These optimized weights are then transferred to train the full-scale model. This avoids expensive hyperparameter sweeps at scale and consistently outperforms heuristic mixtures. The key insight is that optimal mixing ratios are not proportional to data size — rare but high-quality domains (code, math) often deserve upsampling."
    },
    {
      type: "mc",
      question: "**Benchmark contamination** occurs when test data appears in the training corpus. Which detection method is most robust?",
      options: ["Checking if the model's accuracy exceeds 90% on the benchmark, since performance above this threshold is statistically unlikely without prior exposure to the test questions during training", "Running the benchmark with a different random seed for answer option ordering and comparing variance — contaminated models show low variance because they have memorized specific answer positions", "Asking the model directly if it has seen the benchmark before and analyzing the confidence of its response, since models trained on contaminated data tend to produce more certain affirmative replies", "Searching for exact n-gram overlaps (e.g., 8-gram or longer) between training data and benchmark instances, combined with analyzing performance gaps between contaminated and clean subsets of the benchmark"],
      correct: 3,
      explanation: "The standard approach (used in GPT-4, Llama, etc.) is n-gram overlap detection: if a training document shares a long n-gram (typically 8-13 grams) with a benchmark instance, that instance is flagged as contaminated. The complementary analysis compares performance on contaminated vs. clean subsets — a large gap indicates the model is benefiting from memorization rather than genuine capability. Neither method alone is sufficient: n-gram search has false negatives (paraphrased contamination), so the performance gap analysis provides a second signal."
    },
    {
      type: "mc",
      question: "When handling **Personally Identifiable Information (PII)** in training data, which approach best balances data utility and privacy?",
      options: [
        "Removing all documents that contain any proper names or identifiers, discarding them entirely from the training corpus to eliminate any risk of PII leakage at generation time",
        "Applying named entity recognition to detect and redact or replace PII (names, emails, phone numbers, addresses) with placeholder tokens, while preserving surrounding context for language learning",
        "Training the model to memorize PII patterns so it can generate realistic synthetic identities on demand, which replaces the need to scrub real data from the corpus",
        "Only training on data published before 2010, when privacy regulations were not yet in effect and web content could be freely used without PII concerns or legal constraints"
      ],
      correct: 1,
      explanation: "The practical approach is NER-based PII scrubbing: detect entities classified as personal information and either redact them (replace with [REDACTED]) or substitute them with synthetic replacements (fake names/emails). This preserves the linguistic structure and context around PII while removing the sensitive content. Removing entire documents is too aggressive and loses valuable training signal. This connects to broader concerns: GDPR right-to-be-forgotten creates legal obligations that static training data cannot easily satisfy, motivating techniques like machine unlearning."
    },
    {
      type: "mc",
      question: "A team deduplicates a 15TB web crawl and finds that 40% of documents are near-duplicates. After dedup, they train two models: one on the full 15TB (with duplicates) and one on the deduplicated 9TB. Based on empirical findings (e.g., from the Llama and Chinchilla papers), what is the most likely outcome?",
      options: ["The deduplicated 9TB model matches or outperforms the 15TB model on held-out evaluation, trains faster per epoch, and memorizes less — showing that unique tokens matter more than raw token count", "Both models perform identically on all downstream tasks because the information content is the same — duplicate documents add training steps but do not change the learned distribution", "The 15TB model is strictly better because it saw more tokens during training, and the scaling laws predict that performance improves monotonically with total token count regardless of uniqueness", "The deduplicated model catastrophically forgets the content that was present in the removed duplicates, losing coverage of topics that appeared frequently in the original crawl"],
      correct: 0,
      explanation: "Empirically (Lee et al., 2022; Llama papers), deduplication consistently improves or maintains downstream quality while reducing training cost. Duplicate data wastes compute on memorizing repeated patterns rather than learning new ones, inflates training loss estimates (the model appears to perform better because it has memorized training examples), and increases the risk of verbatim regurgitation. The deduplicated model achieves better generalization per FLOP because each gradient step carries more novel information. This is why modern pipelines invest heavily in dedup despite its computational cost."
    }
  ]
};
