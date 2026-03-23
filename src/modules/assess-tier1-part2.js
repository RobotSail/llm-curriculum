// Assessment modules for Tier 1, Part 2: Data, Evaluation, Distributed Training
// Sections 1.4, 1.5, 1.6 — pure assessment, no info steps
// 10 MC questions each, easy -> hard progression

// ─────────────────────────────────────────────────────────────
// Section 1.4: Data
// ─────────────────────────────────────────────────────────────

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
      options: ["Their perplexity is as high as possible, indicating novel content", "Their perplexity is below a fixed threshold of 100", "Their perplexity exactly matches Wikipedia's average perplexity", "Their perplexity falls within a middle range — not too low (repetitive/templated) and not too high (gibberish/non-fluent)"],
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


// ─────────────────────────────────────────────────────────────
// Section 1.5: Evaluation
// ─────────────────────────────────────────────────────────────

export const evaluationAssessment = {
  id: "1.5-assess",
  sectionId: "1.5",
  title: "Assessment: Evaluation",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Perplexity is the standard offline evaluation metric for language models. Its primary **limitation** as a measure of model quality is:",
      options: ["It is too expensive to compute for models above 10B parameters because evaluating perplexity requires a full forward pass on every token in the test corpus, making it impractical at scale", "It only works for models with fewer than 1 billion parameters because larger models produce distributions too peaked for the log-likelihood calculation to remain numerically stable in standard floating-point precision", "It measures next-token prediction accuracy on a fixed corpus but does not capture instruction-following ability, reasoning quality, factual accuracy, or user preference — a model can have excellent perplexity yet produce unhelpful or harmful outputs", "It requires labeled data with human annotations for each token position, making it dependent on the same costly annotation pipelines used for supervised fine-tuning and limiting its applicability to pre-annotated corpora"],
      correct: 2,
      explanation: "Perplexity measures $2^{H(P, Q)}$ — how well the model predicts the next token in a held-out corpus. A model can achieve low perplexity by being excellent at surface-level statistical patterns while failing at reasoning, following instructions, or producing factually correct content. This is why the field moved beyond perplexity to task-specific benchmarks and human evaluation. Perplexity remains useful for comparing base models on the same data distribution, but tells you little about downstream utility."
    },
    {
      type: "mc",
      question: "In **N-shot prompting** for evaluation, you prepend $N$ input-output examples to the test query. Increasing $N$ from 0 to 5 typically helps because:",
      options: ["It fine-tunes the model on those $N$ examples via implicit gradient descent in the forward pass, effectively performing a few steps of optimization on the demonstrated input-output pairs before generating the answer", "It reduces the model's perplexity on the training set by providing additional context that anchors the token predictions, making the evaluation metric artificially more favorable without improving actual capability", "It increases the model's effective parameter count by activating specialized attention patterns for each demonstrated example, temporarily expanding the model's capacity during inference for that specific task", "The examples specify the task format and output space, reducing ambiguity about what is expected — the model leverages in-context learning to condition its distribution on the demonstrated pattern without any weight updates"],
      correct: 3,
      explanation: "Few-shot prompting exploits in-context learning (ICL): the model conditions on the demonstrations to infer the task distribution. Crucially, no gradient update occurs — the model uses its pre-trained ability to recognize patterns in context. The demonstrations serve as a soft specification of the task: they disambiguate output format, label space, and style. However, results are sensitive to example selection, ordering, and format, which is why evaluation protocols must standardize these choices."
    },
    {
      type: "mc",
      question: "**MMLU** (Massive Multitask Language Understanding) evaluates a model across 57 academic subjects. What does strong MMLU performance primarily indicate?",
      options: [
        "The model can write fluent, coherent prose across a wide range of topics and registers, demonstrating strong generative language ability rather than factual recall or structured reasoning",
        "The model has broad factual knowledge and can perform multiple-choice reasoning across diverse academic domains — it tests breadth of world knowledge absorbed during pre-training more than deep reasoning ability",
        "The model has been explicitly fine-tuned on those 57 subjects with supervised examples, and the high score reflects domain-specific optimization rather than general pre-training knowledge",
        "The model has a large context window that allows it to process lengthy academic passages, which is the primary bottleneck for answering knowledge-intensive multiple-choice questions"
      ],
      correct: 1,
      explanation: "MMLU tests breadth of knowledge across STEM, humanities, social sciences, and professional domains using 4-choice MC questions. High performance requires the model to have absorbed factual knowledge during pre-training and to apply basic reasoning. However, the MC format means it tests recognition over generation, and many questions can be answered with surface-level pattern matching rather than deep understanding. It is a useful but incomplete signal — complementary to benchmarks like GSM8K (math reasoning) or HumanEval (code generation)."
    },
    {
      type: "mc",
      question: "**HumanEval** and **GSM8K** measure fundamentally different capabilities. HumanEval tests code generation (writing Python functions from docstrings), while GSM8K tests grade-school math word problems. A model scoring 90% on HumanEval but 40% on GSM8K most likely:",
      options: ["Was trained with heavy code emphasis but lacks robust multi-step arithmetic and reasoning chains — code generation relies more on pattern completion from the docstring, while GSM8K requires chaining 2-8 arithmetic operations with intermediate reasoning", "Has a bug in its math tokenizer that causes numerical digits to be split across subword boundaries, preventing the model from performing accurate arithmetic on multi-digit operands in word problems", "Is overfitting to GSM8K's test set through data contamination, memorizing specific answers without learning the underlying reasoning — the high HumanEval score reflects genuine capability while GSM8K is artificially deflated", "Needs a larger context window for math problems because GSM8K word problems average 150-200 tokens and the intermediate chain-of-thought reasoning can extend to 500+ tokens, exceeding the model's effective context length"],
      correct: 0,
      explanation: "This capability gap is common and informative. HumanEval functions can often be solved by recognizing the pattern from the docstring and generating idiomatic Python — a strong pattern-matching ability. GSM8K requires constructing multi-step reasoning chains: parsing the word problem, identifying relevant quantities, chaining 2-8 arithmetic operations correctly, and arriving at a numerical answer. Errors compound at each step. This dissociation reveals that code completion and mathematical reasoning are partially independent capabilities."
    },
    {
      type: "mc",
      question: "**Goodhart's Law** — \"when a measure becomes a target, it ceases to be a good measure\" — manifests in LLM benchmarks when:",
      options: ["Benchmarks become too easy for all models as capabilities improve, causing score saturation where every model achieves near-perfect accuracy and the benchmark loses its ability to discriminate between model quality levels", "Too many benchmarks exist for any single model to reasonably complete, fragmenting the evaluation landscape and making it impossible to form a coherent picture of model capabilities across the full benchmark ecosystem", "Models or training pipelines are optimized specifically to maximize benchmark scores (e.g., training on benchmark-similar data, prompt engineering for the exact format), inflating scores without proportional improvement in genuine capability", "The benchmark questions are poorly written with ambiguous answer choices and unclear evaluation criteria, making scores unreliable indicators of capability due to measurement noise rather than systematic optimization pressure"],
      correct: 2,
      explanation: "Goodhart's Law is pervasive in LLM evaluation. Examples include: training data contamination (benchmark questions leak into pre-training corpora), benchmark-specific prompt optimization, generating synthetic training data that mimics benchmark formats, and cherry-picking evaluation configurations. The result is that leaderboard rankings may reflect optimization pressure on the benchmark rather than genuine capability differences. This motivates held-out evaluation (Chatbot Arena), dynamic benchmarks that refresh questions, and multi-benchmark evaluation suites."
    },
    {
      type: "mc",
      question: "**Chatbot Arena** uses ELO ratings derived from pairwise human preferences. Its key methodological advantage over static benchmarks is:",
      options: ["It is cheaper to run than traditional benchmarks because pairwise comparisons require fewer total evaluations than absolute scoring, reducing the number of human judgments needed per model by an order of magnitude", "It only tests English language ability, which provides a more focused and reliable signal than multilingual benchmarks where translation quality confounds the measurement of underlying model reasoning capability", "It uses fully automated scoring with a calibrated LLM judge instead of human evaluation, eliminating annotator disagreement and enabling rapid evaluation of new models within hours of their release", "It continuously collects fresh human judgments on diverse, user-generated prompts — making it resistant to contamination, Goodhart's Law, and benchmark saturation, while directly measuring what users actually care about"],
      correct: 3,
      explanation: "Chatbot Arena addresses core limitations of static benchmarks: (1) prompts come from real users, not a fixed test set, so they cannot be trained on; (2) evaluation is continuous, so the benchmark evolves with model capabilities; (3) pairwise comparison (\"which response is better?\") is a more natural and reliable judgment than absolute scoring; (4) ELO ratings handle transitive preferences and provide a single ranking. The main limitations are cost, speed (thousands of comparisons needed per model), potential demographic bias in the user population, and sensitivity to response length/formatting."
    },
    {
      type: "mc",
      question: "When using an **LLM-as-judge** to evaluate other models' outputs, three well-documented biases are verbosity bias, position bias, and self-preference bias. **Verbosity bias** means:",
      options: [
        "The judge penalizes short responses by assigning lower quality scores regardless of content, because its training data associated brevity with low-effort answers",
        "The judge systematically prefers longer, more detailed responses even when the additional content is redundant or irrelevant — length serves as a proxy for quality in the judge's learned heuristics",
        "The judge produces verbose explanations of its own ratings, inflating the token cost of evaluation without improving the accuracy or reliability of the quality assessments",
        "The judge can only meaningfully evaluate responses above a minimum token length, because shorter outputs lack sufficient signal for the judge model to form a quality judgment"
      ],
      correct: 1,
      explanation: "LLM judges trained on human preference data inherit the bias that longer responses tend to be rated higher by human annotators (since length correlates with effort and completeness). This creates a systematic bias where a 500-word mediocre response may be preferred over a 100-word excellent one. Mitigation strategies include: controlling for length in the evaluation prompt, swapping response positions to average out position bias, using multiple judges, and explicitly instructing the judge to evaluate conciseness. Position bias (preferring the first or second response) and self-preference (GPT-4 favoring GPT-4-style outputs) are orthogonal but equally problematic."
    },
    {
      type: "mc",
      question: "The **elicitation gap** refers to the difference between:",
      options: ["A model's latent capability and the performance actually measured — the same model can appear much stronger or weaker depending on the prompting strategy, number of shots, chain-of-thought usage, and other evaluation choices that affect how well the model's knowledge is elicited", "The model's training loss and test loss — the difference between how well the model fits its training data and how well it performs on unseen examples, which is the standard measure of generalization in machine learning", "The gap between open-source and closed-source model performance on the same benchmarks, reflecting the resource disparity between academic labs and industry research organizations in terms of data and compute access", "The time between model training completion and model deployment to production, during which the model's knowledge becomes increasingly stale relative to the evolving state of the world"],
      correct: 0,
      explanation: "The elicitation gap is critical for interpreting benchmarks. The same model might score 40% on a math benchmark with zero-shot prompting but 75% with chain-of-thought and 5-shot prompting. The model's knowledge did not change — only how well the evaluation protocol extracted it. This means benchmark comparisons are only valid when using identical elicitation methods, and low scores may reflect poor elicitation rather than missing capability. It also implies that current benchmarks likely underestimate model capabilities, especially for models that are sensitive to prompt formatting."
    },
    {
      type: "mc",
      question: "A **contamination-robust** evaluation strategy should include:",
      options: ["Running benchmarks only once per model to prevent iterative optimization against the test set, and requiring all evaluation results to be reported from the first run without any prompt tuning or configuration changes", "Using only perplexity as the evaluation metric because it is computed on raw text without relying on specific question-answer formats, making it inherently immune to the format-specific contamination that affects structured benchmarks", "Multiple layers of defense: n-gram overlap detection between training data and test sets, canary string insertion in benchmarks, performance comparison between contaminated and clean subsets, rephrased/perturbed versions of benchmark questions to test whether performance drops (indicating memorization vs. understanding), and temporal holdouts using data created after the training cutoff", "Keeping all benchmark questions permanently secret and never publishing them, distributing only encrypted evaluation binaries that compute scores without exposing the underlying test instances to model developers or training pipelines"],
      correct: 2,
      explanation: "No single method catches all contamination. N-gram detection misses paraphrased contamination. Canary strings (unique identifiers embedded in test sets) detect if the test set was ingested but do not catch reformulated questions. Rephrased variants distinguish memorization from understanding: if performance drops sharply on semantically equivalent but syntactically different questions, the model likely memorized rather than learned. Temporal holdouts (benchmarks created after training cutoff) are the gold standard but require continuous benchmark creation. A robust evaluation combines all these approaches."
    },
    {
      type: "mc",
      question: "You evaluate two models on a new reasoning benchmark. Model A scores 82% with standard prompting. Model B scores 71% with standard prompting but 88% with chain-of-thought. A reviewer claims Model A is superior. The most accurate response is:",
      options: ["The reviewer is correct — Model A's 82% under standard prompting is the fair comparison point, since standard prompting is the canonical evaluation protocol and chain-of-thought gives Model B an unfair advantage by providing extra compute at inference time", "Model B is definitively better because 88% exceeds 82%, and the chain-of-thought result reveals Model B's true capability regardless of the prompting method used — the higher score under any elicitation strategy is the only number that matters for ranking", "Neither model can be meaningfully evaluated because the benchmark is new and has not yet been validated against human expert performance, so neither score can be interpreted as a reliable measure of reasoning ability", "The comparison is invalid without controlling for elicitation: Model B demonstrates higher latent capability when properly elicited (88% vs. 82%), but the models should be compared using the best-known elicitation strategy for each, with clear reporting of which strategies were used — raw scores under different elicitation regimes are not directly comparable"],
      correct: 3,
      explanation: "This question tests understanding of the elicitation gap and evaluation methodology. Fair comparison requires either: (1) fixing the elicitation method and comparing (in which case report both standard and CoT for each model), or (2) reporting peak performance under best elicitation for each model. Model B's 88% under CoT suggests stronger underlying reasoning capability than Model A's 82%. However, the full picture requires testing Model A with CoT as well. The fundamental point is that benchmark scores are a function of both model capability AND evaluation protocol — reporting scores without specifying the protocol is incomplete."
    }
  ]
};


// ─────────────────────────────────────────────────────────────
// Section 1.6: Distributed Training Infrastructure
// ─────────────────────────────────────────────────────────────

export const distributedTrainingAssessment = {
  id: "1.6-assess",
  sectionId: "1.6",
  title: "Assessment: Distributed Training Infrastructure",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In **Distributed Data Parallel (DDP)** training, each GPU holds a full copy of the model. After the backward pass, gradients are synchronized across GPUs using:",
      options: [
        "A parameter server architecture that collects gradients from every GPU, computes the global average on a dedicated node, and then redistributes the averaged result back to each GPU",
        "An **all-reduce** operation that efficiently computes the sum (or average) of gradients across all GPUs so every replica ends up with identical gradients — typically implemented as a ring all-reduce to minimize communication overhead",
        "Each GPU sends its full gradient tensor to GPU 0, which performs the averaging computation and then broadcasts the result back to all other GPUs in a hub-and-spoke pattern",
        "Gradients are not synchronized at all — each GPU trains on its own data shard independently, and the parameter divergence is reconciled only at periodic checkpoint intervals"
      ],
      correct: 1,
      explanation: "DDP uses all-reduce (typically ring all-reduce or tree all-reduce via NCCL) to synchronize gradients. In ring all-reduce, each GPU sends a chunk of its gradient to its neighbor, and after $2(N-1)$ steps (N = number of GPUs), all GPUs have the complete averaged gradient. The communication volume per GPU is $2 \\cdot (N-1)/N \\cdot |\\text{params}|$, which approaches $2|\\text{params}|$ as $N$ grows — nearly independent of GPU count. This is far more efficient than the naive reduce-broadcast approach via a parameter server, which creates a bottleneck at the central node."
    },
    {
      type: "mc",
      question: "**Tensor parallelism** and **pipeline parallelism** split the model across GPUs in fundamentally different ways. Tensor parallelism:",
      options: ["Splits individual layers (e.g., partitioning weight matrices column-wise or row-wise) across GPUs so each GPU computes a portion of every layer's output, requiring intra-layer communication at each forward and backward step", "Assigns different training examples to different GPUs", "Assigns entire layers to different GPUs in sequence", "Replicates the model on every GPU"],
      correct: 0,
      explanation: "Tensor parallelism (Megatron-LM style) partitions weight matrices within a layer. For example, a linear layer $Y = XW$ can be split column-wise: $W = [W_1 | W_2]$, with each GPU computing $XW_i$. This requires an all-reduce after each layer to combine partial results. Pipeline parallelism, by contrast, assigns whole layers to different GPUs — GPU 0 runs layers 1-10, GPU 1 runs layers 11-20, etc. Tensor parallelism has higher communication frequency (every layer) but lower latency per communication; pipeline parallelism has lower communication frequency but suffers from the bubble problem."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 1** shards the **optimizer states** across GPUs while each GPU still holds a full copy of parameters and gradients. For a model with $\\Psi$ parameters using Adam in mixed precision, Stage 1 reduces per-GPU optimizer memory from $12\\Psi$ bytes to approximately:",
      options: ["$12\\Psi$ bytes — no savings", "$4\\Psi$ bytes — only the FP16 parameters", "$12\\Psi / N$ bytes, where $N$ is the number of GPUs — each GPU stores only $1/N$ of Adam's first moment ($m$), second moment ($v$), and FP32 master weights", "$2\\Psi$ bytes — only the FP16 gradients"],
      correct: 2,
      explanation: "Adam requires per-parameter state: FP32 master weights (4 bytes), FP32 first moment $m$ (4 bytes), and FP32 second moment $v$ (4 bytes) = 12 bytes per parameter. ZeRO Stage 1 partitions these 12$\\Psi$ bytes across $N$ GPUs, so each GPU stores $12\\Psi/N$ bytes of optimizer state. The FP16 parameters ($2\\Psi$) and FP16 gradients ($2\\Psi$) remain fully replicated. For a 7B model on 8 GPUs: optimizer memory drops from 84 GB to ~10.5 GB per GPU, while parameter and gradient memory remain at 14 GB + 14 GB."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 3** (or equivalently, **FSDP** — Fully Sharded Data Parallel) shards parameters, gradients, AND optimizer states. The key runtime overhead compared to DDP is:",
      options: ["No additional overhead compared to DDP — it is strictly better in all respects because the sharded communication patterns have the same total bandwidth cost as the replicated all-reduce", "It cannot overlap communication with computation because each layer's full parameters must be reconstructed and verified before any forward-pass arithmetic can begin on that layer's inputs", "It requires twice as many GPUs as DDP to achieve the same training throughput, because half the GPUs are dedicated to managing the sharded parameter storage and communication scheduling", "All-gather operations to reconstruct full parameter tensors before each forward/backward computation, and reduce-scatter operations to distribute gradients — trading communication volume for memory savings"],
      correct: 3,
      explanation: "In ZeRO-3/FSDP, each GPU stores only a $1/N$ shard of every parameter tensor. Before computing a layer's forward pass, an all-gather reconstructs the full parameters from all shards. After the backward pass, a reduce-scatter distributes gradient shards. The total communication volume per step is $3 \\times 2\\Psi$ (vs. $2\\Psi$ for DDP), a 3x increase. However, this communication can be overlapped with computation by prefetching the next layer's parameters during the current layer's computation. The memory savings are dramatic: total per-GPU memory approaches $(12\\Psi + 2\\Psi + 2\\Psi) / N = 16\\Psi/N$."
    },
    {
      type: "mc",
      question: "The **pipeline bubble problem** in pipeline parallelism arises because:",
      options: [
        "Data cannot be split into micro-batches when using pipeline parallelism because the sequential layer dependencies prevent any form of batch-level decomposition across the pipeline stages",
        "At the start and end of each training step, some pipeline stages are idle waiting for activations from upstream or gradients from downstream — with $p$ pipeline stages and $m$ micro-batches, the bubble fraction is $(p - 1) / m$, wasting compute proportional to the number of stages",
        "GPUs cannot communicate activation tensors across pipeline stages fast enough over PCIe or NVLink, creating a communication bottleneck that serializes the entire forward pass regardless of the pipeline schedule",
        "The model's loss function becomes non-differentiable across pipeline boundaries because the activation tensors are quantized to reduce inter-stage communication, introducing discontinuities that prevent gradient flow"
      ],
      correct: 1,
      explanation: "With naive scheduling, GPU $k$ must wait for GPUs $0, \\dots, k-1$ to complete before starting, creating a \"bubble\" of idle time. Splitting the batch into $m$ micro-batches and interleaving them reduces the bubble fraction to $(p-1)/m$. For example, with 8 pipeline stages and 32 micro-batches, the bubble is $7/32 \\approx 22\\%$ — meaning 22% of compute is wasted. The **1F1B** (one-forward-one-backward) schedule further optimizes memory by limiting the number of in-flight micro-batches, reducing peak activation memory from $O(m)$ to $O(p)$."
    },
    {
      type: "mc",
      question: "The **1F1B** (one-forward-one-backward) pipeline schedule works by:",
      options: ["After an initial warmup phase, alternating between one forward micro-batch and one backward micro-batch on each pipeline stage — this limits the number of in-flight micro-batches per stage to at most $p$ (the pipeline depth), bounding peak activation memory", "Running all $m$ forward passes first across the full pipeline before starting any backward passes, maximizing GPU utilization by keeping all stages active during the forward phase and then active again during the backward phase", "Running the forward and backward passes simultaneously on the same micro-batch by pipelining the gradient computation within each layer so that early layers begin their backward pass while later layers are still completing the forward pass", "Eliminating the pipeline bubble entirely by dynamically reassigning idle pipeline stages to data-parallel replicas, converting wasted bubble time into useful gradient computation on additional data samples"],
      correct: 0,
      explanation: "In 1F1B, each stage goes through a warmup phase (receiving and forwarding micro-batches), then enters a steady state where it performs one forward pass followed by one backward pass in alternation. This means each stage holds activations for at most $p$ micro-batches at any time (rather than all $m$ micro-batches in the naive all-forward-then-all-backward schedule). The bubble fraction remains $(p-1)/m$, but peak memory is dramatically reduced. Interleaved scheduling (where virtual pipeline stages are assigned cyclically) can further reduce the bubble to $(p-1)/(m \\cdot v)$ where $v$ is the number of virtual stages."
    },
    {
      type: "mc",
      question: "**BF16** (bfloat16) is preferred over **FP16** for LLM training because:",
      options: ["BF16 has higher precision for small numbers due to its 10-bit mantissa, making it more accurate than FP16 for the small gradient values that are critical during the early stages of fine-tuning and the final convergence phase", "BF16 uses less memory than FP16 by storing values in 12 bits instead of 16, achieving a 25% reduction in memory footprint per tensor while maintaining sufficient precision for most training workloads through adaptive rounding", "BF16 uses the same 8-bit exponent as FP32 (range $\\pm 3.4 \\times 10^{38}$), avoiding the overflow/underflow issues that plague FP16 (5-bit exponent, range $\\pm 65504$) — this eliminates the need for loss scaling even though BF16 has less mantissa precision (7 bits vs FP16's 10 bits)", "BF16 is the only reduced-precision format supported by modern GPU tensor cores for training workloads — FP16 tensor core support was removed starting with the A100 architecture to simplify the hardware and encourage BF16 adoption"],
      correct: 2,
      explanation: "FP16 has 5 exponent bits (range $\\sim 6 \\times 10^{-8}$ to $6.5 \\times 10^4$) and 10 mantissa bits. BF16 has 8 exponent bits (same range as FP32: $\\sim 10^{-38}$ to $\\sim 10^{38}$) and 7 mantissa bits. In LLM training, gradients and activations span a wide dynamic range — FP16's limited range causes underflow (small gradients become zero) or overflow (large activations become inf), requiring careful loss scaling. BF16's FP32-matching range avoids these issues entirely at the cost of slightly reduced precision. Both use 16 bits (2 bytes per value). The practical result: BF16 training is nearly as stable as FP32 with half the memory."
    },
    {
      type: "mc",
      question: "**Activation checkpointing** (gradient checkpointing) trades compute for memory by:",
      options: ["Compressing activations using quantization from FP16 to INT8 during the forward pass, halving activation memory at the cost of small numerical errors that are corrected during the backward pass via stochastic dequantization", "Reducing the number of active layers in the model by dynamically skipping layers whose gradient contribution falls below a learned threshold, trading model capacity for memory savings during the backward pass", "Storing all intermediate activations on CPU system RAM instead of GPU HBM during the forward pass, then transferring them back to the GPU on demand during the backward pass via PCIe or NVLink-to-host transfers", "Discarding intermediate activations during the forward pass and recomputing them from saved checkpoints during the backward pass — this reduces activation memory from $O(L)$ to $O(\\sqrt{L})$ (with optimal checkpoint placement) at the cost of one additional forward pass, roughly 33% more compute"],
      correct: 3,
      explanation: "During the forward pass, only activations at checkpoint boundaries are saved; intermediate activations are discarded. During the backward pass, when intermediate activations are needed for gradient computation, the forward pass is re-run from the nearest checkpoint. With checkpoints every $\\sqrt{L}$ layers (for $L$ total layers), memory is $O(\\sqrt{L})$ and compute increases by ~33% (one extra forward pass). This is often the single most impactful memory optimization: for a 70B model, it can reduce activation memory from hundreds of GB to a manageable level. The tradeoff is almost always worthwhile — memory is the binding constraint, not compute."
    },
    {
      type: "mc",
      question: "A **70B parameter model** trained with Adam in mixed precision requires approximately how much **optimizer state memory** (across all GPUs combined)?",
      options: ["140 GB (2 bytes per parameter for the FP16 working weights that are used in the forward and backward passes)", "840 GB (12 bytes per parameter: FP32 master weights + FP32 first moment + FP32 second moment)", "280 GB (4 bytes per parameter for the FP32 master copy only, which is the dominant optimizer cost)", "70 GB (1 byte per parameter when using 8-bit Adam with quantized optimizer states)"],
      correct: 1,
      explanation: "Adam maintains three FP32 buffers per parameter: (1) master copy of weights — 4 bytes, (2) first moment estimate $m$ — 4 bytes, (3) second moment estimate $v$ — 4 bytes. Total: $12 \\times 70 \\times 10^9 = 840 \\times 10^9$ bytes $= 840$ GB. This is the dominant memory cost and the primary motivation for ZeRO/FSDP. On 8 GPUs with ZeRO Stage 1, this drops to ~105 GB/GPU. Adding the FP16 model parameters (140 GB) and FP16 gradients (140 GB), total memory is ~1120 GB, or ~140 GB/GPU with 8-way ZeRO-1 for optimizer states alone (parameters and gradients still replicated at Stage 1)."
    },
    {
      type: "mc",
      question: "**Sequence parallelism** addresses a specific limitation of tensor parallelism. In standard Megatron-style tensor parallelism, operations like LayerNorm and dropout are **replicated** on every GPU. Sequence parallelism fixes this by:",
      options: ["Partitioning the sequence dimension across GPUs for these replicated operations (LayerNorm, dropout, activation functions), so each GPU processes a portion of the sequence — then transitioning back to tensor-parallel partitioning for the attention and MLP computations", "Splitting the vocabulary across GPUs so that each GPU computes the embedding lookup and final softmax for a subset of tokens, reducing the per-GPU memory footprint of these large vocabulary-dependent layers", "Using a longer context window by distributing the extended sequence across GPUs, with each GPU responsible for a contiguous chunk of the full context and cross-GPU attention computed via ring communication", "Replacing LayerNorm with a parallelizable alternative such as RMSNorm that decomposes into independent per-GPU computations without requiring the cross-GPU all-reduce needed for computing global mean and variance statistics"],
      correct: 0,
      explanation: "In tensor parallelism, matrix multiplications (attention projections, MLP layers) are split across GPUs, but LayerNorm, dropout, and activation functions operate on the full hidden dimension and are redundantly computed on every GPU. Sequence parallelism (Korthikanti et al., 2022) partitions these operations along the sequence dimension instead: each GPU handles $\\text{seq\\_len}/N$ tokens for LayerNorm/dropout, then the layout transitions to tensor-parallel for the split matrix multiplications. This eliminates the redundant computation and memory for these operations, saving ~30-40% of activation memory that would otherwise be wasted on replicated non-tensor-parallel regions."
    }
  ]
};
