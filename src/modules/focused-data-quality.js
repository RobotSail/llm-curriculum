// Focused learning module: Data Quality and Filtering for LLM Pretraining
// Section 1.4: Data
// Covers: why data quality matters, web data problems, perplexity filtering,
// heuristic filters, deduplication, data mixing, contamination.
// Single-concept module following CLAUDE.md conventions.

export const dataQualityLearning = {
  id: "1.4-data-quality-learning-easy",
  sectionId: "1.4",
  title: "Data Quality and Filtering for LLM Pretraining",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "Why Data Quality Matters More Than You Think",
      content: "The Chinchilla scaling laws (Hoffmann et al., 2022) showed that **data tokens and model parameters contribute roughly equally** to model performance. Doubling your data is as valuable as doubling your model size — and far cheaper in compute.\n\nBut this assumes your data is high-quality. In practice, most web-crawled text is noisy: duplicates, boilerplate, SEO spam, machine-generated filler, and toxic content. Training on low-quality data doesn't just waste compute — it actively degrades the model.\n\nThe empirical finding is striking: a 7B model trained on 1T high-quality tokens consistently outperforms a 7B model trained on 5T unfiltered tokens across benchmarks. The model doesn't just memorize noise — it **internalizes** the statistical patterns of whatever it trains on, including patterns of bad writing, factual errors, and repetitive structure.\n\nThis makes data curation arguably the highest-leverage activity in LLM development. Architecture and training recipe matter, but data quality sets the ceiling."
    },
    {
      type: "mc",
      question: "A team has a fixed compute budget sufficient to train a 7B model. They have 5T tokens of unfiltered web text and can invest effort to filter it down to 1T high-quality tokens. Based on empirical findings, which approach likely yields better downstream performance?",
      options: [
        "Training on all 5T tokens — more data always helps because the model can learn to ignore noise on its own through statistical averaging",
        "Training on 1T filtered tokens — data quality dominates quantity because the model internalizes statistical patterns of whatever it trains on, including noise",
        "Training on 5T tokens with curriculum learning — starting with noisy data and gradually introducing clean data gives the best of both approaches",
        "It depends entirely on the model architecture — transformer models are robust to noise while RNNs are not"
      ],
      correct: 1,
      explanation: "Empirically, models trained on smaller high-quality datasets consistently outperform those trained on larger noisy datasets at the same compute budget. The model cannot reliably distinguish noise from signal — it learns the distribution of its training data, including patterns of bad writing and factual errors. This is why data curation has become the highest-leverage activity in LLM development."
    },
    {
      type: "info",
      title: "Common Web Data Problems",
      content: "Web crawls (like Common Crawl) contain billions of pages, but most are not suitable for language model training. The main problems:\n\n**Duplicates**: The same content appears thousands of times across mirror sites, syndication networks, and scrapers. Duplicates cause the model to memorize specific text rather than learning general patterns, and waste compute on redundant gradient updates.\n\n**Boilerplate**: Navigation menus, cookie banners, copyright notices, and HTML artifacts. These are repetitive structures that teach the model nothing useful about language.\n\n**SEO spam and machine-generated text**: Content farms produce vast quantities of superficially coherent but uninformative text optimized for search engines. Training on this teaches the model to produce similarly vacuous output.\n\n**Personally identifiable information (PII)**: Names, addresses, phone numbers, and emails that create privacy risks if the model memorizes and regurgitates them.\n\n**Toxic content**: Hate speech, explicit material, and harassment that the model would learn to reproduce.\n\nEach problem requires different filtering strategies. No single method catches everything — effective data pipelines layer multiple complementary filters."
    },
    {
      type: "mc",
      question: "A data pipeline removes exact-duplicate documents but does NOT perform near-duplicate detection. Which problem is most likely to persist?",
      options: [
        "Boilerplate text from navigation menus, since boilerplate is always an exact copy across pages on the same site",
        "PII exposure, since personally identifiable information is duplicated across public records databases",
        "Syndicated content that is slightly reformatted across sites — same substance but different whitespace, headers, or minor word changes",
        "SEO spam, since search-optimized content is always generated from exact templates with no variation"
      ],
      correct: 2,
      explanation: "Syndicated content is republished across many sites with minor formatting changes — different headers, slightly reworded introductions, or reformatted paragraphs. Exact-duplicate detection (based on full-document hashing) misses these because they are not byte-identical. Near-duplicate detection (using techniques like MinHash/LSH) catches them by comparing approximate content similarity. Boilerplate and SEO spam vary substantially across sites, so they are not primarily a duplication problem."
    },
    {
      type: "info",
      title: "Perplexity-Based Quality Filtering",
      content: "One of the most effective filtering strategies uses a **reference language model** to score documents. The idea: train a small model on known high-quality text (Wikipedia, curated books), then use it to compute the perplexity of each web document.\n\nRecall that perplexity is $\\text{PPL} = \\exp(H(p_{\\text{data}}, p_\\theta))$ — the exponentiated cross-entropy. A document that looks like Wikipedia (well-structured, informative prose) will have **low perplexity** under the reference model. A document full of spam, gibberish, or boilerplate will have **high perplexity**.\n\nThe filtering rule: keep documents below a perplexity threshold, discard the rest. This is a soft quality signal — it doesn't detect specific problems but rather measures overall resemblance to the reference distribution.\n\nThe connection to information theory is precise: filtering by perplexity is equivalent to filtering by $D_{\\text{KL}}(p_{\\text{doc}} \\| p_{\\text{ref}}) + H(p_{\\text{doc}})$. Documents whose distribution is close to the reference (low KL) pass; documents whose distribution is far (high KL) are rejected.\n\n**Limitation**: Perplexity filtering biases toward Wikipedia-like text, potentially discarding valuable but stylistically different content (code, dialogue, technical writing). Most pipelines use it as one signal among many, not the sole filter."
    },
    {
      type: "mc",
      question: "A perplexity filter trained on Wikipedia text is applied to a web crawl. Which of the following documents is MOST likely to be incorrectly filtered out (false positive)?",
      options: [
        "A well-written news article about climate policy, since news prose uses a formal register that differs substantially from Wikipedia's encyclopedic conventions",
        "A forum thread where domain experts debate quantum computing, since multi-speaker dialogue with nested quotes deviates from single-author encyclopedia structure",
        "An auto-generated product listing with specifications and prices, since tabular commercial data has a very different token distribution from prose paragraphs",
        "A transcript of a casual podcast interview with informal speech, interruptions, and sentence fragments that look unlike encyclopedic prose to the reference model"
      ],
      correct: 3,
      explanation: "The podcast transcript is high-quality content but has high perplexity under a Wikipedia-trained model because its language distribution (informal, fragmented, conversational) is very different from encyclopedic prose. The reference model assigns low probability to sentence fragments, filler words, and conversational structures, inflating perplexity regardless of content quality. This is the fundamental limitation of perplexity filtering: it measures stylistic similarity to the reference, not intrinsic quality."
    },
    {
      type: "info",
      title: "Heuristic Quality Filters",
      content: "Complementing perplexity filtering, **heuristic rules** catch specific quality signals that statistical models miss:\n\n**Document length**: Very short documents (< 50 words) are often error pages, redirects, or incomplete scrapes. Very long documents may be auto-generated or data dumps.\n\n**Symbol-to-word ratio**: Documents with excessive punctuation, special characters, or HTML fragments ($> 10\\%$ non-alphanumeric characters) are likely boilerplate or poorly extracted.\n\n**Stop-word presence**: Natural language text contains common words (\"the\", \"is\", \"and\") at predictable rates. Documents with abnormally low stop-word frequency are often keyword lists, tables, or code rather than prose.\n\n**Language detection**: If training an English model, filter documents below a confidence threshold for English (e.g., using fastText language ID). Mixed-language documents or misidentified languages add noise.\n\n**\"Lorem ipsum\" and template detection**: Pattern-match for known placeholder text, template markers, and common boilerplate strings.\n\nThese rules are individually crude — each has false positives and false negatives. But in combination, they form a robust first-pass filter that removes the worst-quality content cheaply before more expensive statistical filtering."
    },
    {
      type: "mc",
      question: "A heuristic filter removes documents where fewer than 20% of words are common English stop words (\"the\", \"is\", \"of\", etc.). Which legitimate document type is MOST likely to be incorrectly caught by this filter?",
      options: [
        "A structured data table listing country populations and GDP figures, where most tokens are proper nouns and numbers rather than function words",
        "A literary essay using sophisticated vocabulary and complex sentence structures, since advanced English prose naturally reduces stop-word frequency below the 20% threshold",
        "A children's book with simplified language and short sentences, since a smaller working vocabulary produces higher stop-word density that easily clears the filter",
        "A scientific paper abstract discussing experimental methods and results, since academic writing follows standard English grammar patterns with typical stop-word rates"
      ],
      correct: 0,
      explanation: "Structured data tables consist primarily of proper nouns, numbers, and column headers — very few function words like \"the\" or \"is\". The stop-word percentage would be far below 20%, triggering the filter despite the data being potentially useful. Literary essays, children's books, and scientific abstracts all use standard English grammar where function words appear at roughly normal rates (typically 25-35% of tokens). This illustrates why heuristic filters need exceptions for specific content types."
    },
    {
      type: "info",
      title: "Deduplication: Exact and Near-Duplicate Removal",
      content: "Duplicates are one of the most damaging data quality problems. They cause:\n\n1. **Memorization**: The model learns to reproduce specific text verbatim rather than generalizing. With enough repetitions, the model will regurgitate memorized passages during generation.\n2. **Wasted compute**: Training on the same content $k$ times gives diminishing returns after the first few exposures but costs $k\\times$ the compute.\n3. **Train/test contamination**: If benchmark test data appears (even slightly modified) in training data, evaluation results are inflated and meaningless.\n\n**Exact deduplication** uses document-level hashing (e.g., SHA-256 of normalized text). It's fast ($O(n)$ time with a hash table) but only catches byte-identical copies.\n\n**Near-deduplication** catches reformatted copies using approximate methods:\n- **MinHash**: Compute a signature from $k$ random hash functions applied to the document's $n$-grams. Documents with similar signatures (high Jaccard similarity) are near-duplicates.\n- **Locality-Sensitive Hashing (LSH)**: Band MinHash signatures so that similar documents hash to the same bucket with high probability. This avoids comparing every pair ($O(n^2)$), giving roughly $O(n)$ amortized cost.\n\n**Paragraph-level deduplication** (used in C4 and RefinedWeb) removes individual paragraphs that appear verbatim in multiple documents, even if the surrounding text differs."
    },
    {
      type: "mc",
      question: "A training corpus has 10B documents. You want to remove near-duplicates using MinHash + LSH. Naively comparing all pairs would require $\\binom{10B}{2} \\approx 5 \\times 10^{19}$ comparisons. What makes MinHash + LSH practical at this scale?",
      options: [
        "MinHash compresses each document to a fixed-size signature, reducing memory but still requiring all pairwise comparisons between signatures",
        "LSH bands the MinHash signatures so similar documents hash to the same bucket with high probability, reducing the comparison space from $O(n^2)$ to approximately $O(n)$",
        "LSH randomly samples a small fraction of document pairs to compare, trading recall for speed through statistical approximation",
        "MinHash eliminates the need for pairwise comparisons entirely by computing a single global hash that clusters all duplicates together"
      ],
      correct: 1,
      explanation: "The key insight of LSH is that it uses banded hashing to ensure similar items (high Jaccard similarity) collide in at least one band with high probability, while dissimilar items rarely collide. This means you only compare documents within the same bucket — not all pairs. The expected number of comparisons drops from $O(n^2)$ to approximately $O(n)$ (each document is compared to a small number of bucket-mates). MinHash alone only provides compact signatures; LSH provides the efficient retrieval of similar pairs."
    },
    {
      type: "info",
      title: "Data Mixing: Balancing Domains",
      content: "A pretraining corpus combines text from multiple domains: web crawls, books, scientific papers, code, forums, and news. The **mixing proportions** — what fraction of each training batch comes from each domain — significantly affect downstream capabilities.\n\nKey findings:\n\n**Code improves reasoning**: Models trained with 10-20% code perform better on mathematical and logical reasoning benchmarks, even when evaluated only on natural language tasks. Code's structured, precise syntax appears to teach general logical patterns.\n\n**Books and long-form text improve coherence**: Extended prose teaches the model to maintain topic and argument structure over long sequences, improving generation quality.\n\n**Domain proportions are not equal to web proportions**: The web is overwhelmingly low-quality commercial content. Simply training on web data at natural proportions overweights noise. Most successful models drastically **upsample** high-quality domains (books, Wikipedia, code) relative to their natural web frequency.\n\nThe optimal mix is model-size dependent and task-dependent, making it one of the most important hyperparameters in LLM training. Teams like DeepSeek and LLaMA use extensive ablations on smaller models to find good mixing ratios before scaling up."
    },
    {
      type: "mc",
      question: "A pretraining corpus consists of 80% web text, 10% code, 5% books, and 5% scientific papers. The team finds that increasing the code fraction from 10% to 20% (reducing web to 70%) improves scores on math reasoning benchmarks by 8% while slightly improving natural language benchmarks too. What best explains this?",
      options: [
        "Code data is inherently more information-dense per token, so the model processes more useful bits per training step when the code fraction increases",
        "The web text fraction was too high and contained math-specific noise that interfered with learning — reducing it removed the interference",
        "The math benchmarks contain code-like formatting that the model recognizes better after seeing more code during training",
        "Code's structured syntax with explicit logic, variable tracking, and compositional operations trains general reasoning capabilities that transfer to mathematical and linguistic tasks"
      ],
      correct: 3,
      explanation: "Code requires tracking variable state, following explicit logical flow, and composing operations — skills that transfer to mathematical and general reasoning. This has been demonstrated empirically across multiple model families: code in pretraining improves performance even on purely natural-language reasoning tasks. The effect is not about formatting familiarity or noise reduction, but about learning transferable computational thinking patterns."
    },
    {
      type: "info",
      title: "Contamination: When Test Data Leaks Into Training",
      content: "**Benchmark contamination** occurs when evaluation benchmark data appears in the training corpus. This inflates scores and makes benchmarks unreliable.\n\nContamination can be:\n- **Direct**: The exact benchmark question-answer pair appears in the training data\n- **Indirect**: A paraphrased version, a webpage discussing the benchmark, or training data that shares the same source as the benchmark\n\nContamination is surprisingly common because:\n1. Web crawls are enormous and benchmarks are often publicly available online\n2. Many benchmarks are derived from sources (Wikipedia, textbooks) that also appear in training corpora\n3. Even \"held-out\" test sets can leak through blog posts, forum discussions, or solution manuals scraped from the web\n\n**Detection methods**:\n- **$n$-gram overlap**: Check if long $n$-grams (e.g., 13-grams) from the test set appear in the training data. Short $n$-grams produce too many false positives.\n- **Embedding similarity**: Flag training documents whose embeddings are very close to test examples.\n- **Canary strings**: Insert unique synthetic strings into test sets and check if the model can reproduce them (evidence of memorization).\n\nContamination detection is an active problem — no method catches all forms, and indirect contamination is especially hard to detect."
    },
    {
      type: "mc",
      question: "A model scores 92% on a popular benchmark. Investigation reveals that 15% of the benchmark's test questions appear verbatim in the training data. A colleague argues: \"85% of questions are uncontaminated, and the model still scores well on those, so the contamination doesn't matter.\" What is the flaw in this reasoning?",
      options: [
        "The 92% accuracy exceeds what contamination alone could produce, which means the contamination detection method has a high false-positive rate and the actual contamination level is much lower",
        "Contamination effects are all-or-nothing — if any questions are contaminated, the entire benchmark must be discarded because partial validity is statistically incoherent",
        "The contaminated 15% inflates the raw score, and the remaining 85% may suffer indirect contamination from benchmark discussions, solution walkthroughs, or paraphrased versions that exact matching misses",
        "The 15% contaminated questions are disproportionately the easiest ones, so the model would have answered them correctly anyway and removing them would barely change the score"
      ],
      correct: 2,
      explanation: "Two issues: (1) Direct contamination inflates the score — the model may get those 15% correct through memorization rather than understanding, boosting the raw score. (2) More critically, the 85% \"uncontaminated\" questions may have indirect contamination: discussions of the benchmark, solution walkthroughs, or paraphrased versions in the training data that exact-match detection doesn't catch. The absence of exact matches does NOT guarantee the model hasn't seen related content. This is why contamination undermines the entire benchmark, not just the directly affected questions."
    },
    {
      type: "info",
      title: "Putting It All Together: A Modern Data Pipeline",
      content: "A typical LLM data pipeline layers multiple stages:\n\n**Stage 1 — URL and format filtering**: Remove known-bad domains, extract text from HTML, discard non-text content. This is the cheapest filter and removes the most obvious junk.\n\n**Stage 2 — Heuristic quality filters**: Apply rule-based filters (document length, language ID, stop-word ratio, symbol ratio). Fast and effective for bulk noise removal.\n\n**Stage 3 — Deduplication**: Run exact dedup (document hashing) followed by near-dedup (MinHash + LSH). This is computationally expensive but critical.\n\n**Stage 4 — Model-based quality scoring**: Score remaining documents with a perplexity filter or a trained quality classifier. This is the most expensive per-document step.\n\n**Stage 5 — Safety filtering**: Remove PII, toxic content, and other harmful material using classifiers and pattern matching.\n\n**Stage 6 — Domain mixing and sampling**: Combine filtered data from different sources at target proportions. Upsample high-quality domains, downsample noisy ones.\n\nThe ordering matters: cheap filters first (stages 1-2) reduce the volume before expensive operations (stages 3-4). A typical pipeline reduces a raw web crawl by 80-95% — keeping only the highest-quality fraction for training."
    }
  ]
};
