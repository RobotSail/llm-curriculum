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
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "In a typical web-scale data pipeline for LLM pre-training, which step comes **first**?",
      options: [
        "Deduplication",
        "Language identification and URL-based filtering of raw crawl data",
        "Perplexity filtering with a trained language model",
        "Data mixing and upsampling"
      ],
      correct: 1,
      explanation: "The pipeline begins with coarse, cheap filters: language ID (fastText-based), URL blocklists, and removing boilerplate HTML. These steps discard the bulk of low-quality data before expensive operations like deduplication or model-based filtering. Running perplexity filtering on unfiltered Common Crawl would be prohibitively expensive and pointless on non-target-language text."
    },
    {
      type: "mc",
      question: "**MinHash** is used for near-duplicate detection in large corpora. It works by:",
      options: [
        "Computing exact MD5 hashes of each document and comparing them",
        "Applying multiple hash functions to token n-gram shingle sets and estimating Jaccard similarity from the fraction of matching minimum hash values",
        "Training a neural network to embed documents and clustering them",
        "Sorting all documents alphabetically and comparing adjacent pairs"
      ],
      correct: 1,
      explanation: "MinHash approximates the Jaccard similarity $J(A, B) = |A \\cap B| / |A \\cup B|$ between shingle sets. For each of $k$ hash functions, the minimum hash value is recorded. The fraction of matching minimums across two documents estimates their Jaccard similarity. This is efficient because it reduces each document to a fixed-size signature, enabling Locality-Sensitive Hashing (LSH) for scalable pairwise comparison without computing all $O(n^2)$ pairs."
    },
    {
      type: "mc",
      question: "What is the key difference between **exact deduplication** and **near-deduplication** in data pipelines?",
      options: [
        "Exact dedup is more computationally expensive",
        "Exact dedup removes byte-identical or hash-identical documents (or n-gram spans), while near-dedup removes documents above a similarity threshold, catching paraphrases, boilerplate variations, and templated content",
        "Near-dedup only works on code, not natural language",
        "They produce identical results but near-dedup is faster"
      ],
      correct: 1,
      explanation: "Exact dedup uses cryptographic hashes (SHA-256, MD5) on full documents or suffix arrays on n-gram spans to find verbatim matches. Near-dedup (MinHash + LSH, SimHash) catches documents that are substantially similar but not identical — e.g., the same news article republished with minor edits, or pages with shared templates. Both are necessary: exact dedup is cheap and catches copies; near-dedup catches the long tail of near-copies that dominate web crawls."
    },
    {
      type: "mc",
      question: "**Perplexity filtering** uses a language model (often a small KenLM n-gram model trained on curated text like Wikipedia) to score documents. Documents are kept if:",
      options: [
        "Their perplexity is as high as possible, indicating novel content",
        "Their perplexity falls within a middle range — not too low (repetitive/templated) and not too high (gibberish/non-fluent)",
        "Their perplexity exactly matches Wikipedia's average perplexity",
        "Their perplexity is below a fixed threshold of 100"
      ],
      correct: 1,
      explanation: "The key insight from CCNet and related work is that both extremes are bad. Very low perplexity indicates repetitive boilerplate (cookie notices, navigation menus) that the LM finds trivially predictable. Very high perplexity indicates garbled text, foreign-language fragments, or encoding artifacts. The \"Goldilocks zone\" in the middle captures fluent, diverse, informative text. The exact thresholds are tuned empirically per domain."
    },
    {
      type: "mc",
      question: "The **Phi-1** (\"Textbooks Are All You Need\") approach demonstrated that:",
      options: [
        "Models must be trained on at least 1 trillion tokens to be useful",
        "A small model trained on high-quality synthetic \"textbook-style\" data and exercises can match or outperform much larger models trained on raw web data, showing that data quality can substitute for scale",
        "Synthetic data always outperforms real data",
        "GPT-4 can fully replace human data annotators at no cost"
      ],
      correct: 1,
      explanation: "Phi-1 (1.3B params) matched GPT-3.5 on HumanEval by training on ~7B tokens of GPT-4-generated textbook-quality code explanations and exercises. This challenged the prevailing \"more data = better\" paradigm by showing that carefully curated synthetic data with high information density per token can dramatically improve sample efficiency. The key was not just using synthetic data, but generating data specifically designed to teach concepts progressively."
    },
    {
      type: "mc",
      question: "**Model collapse** from training on self-generated data occurs because:",
      options: [
        "The model runs out of GPU memory",
        "Each generation step loses information from the tails of the distribution — the model's approximation errors compound across generations, causing the distribution to narrow and eventually degenerate, losing minority patterns and rare knowledge",
        "Synthetic data has more tokens than real data",
        "The tokenizer cannot encode synthetic text"
      ],
      correct: 1,
      explanation: "Shumailov et al. (2023) showed that iteratively training on model-generated data causes progressive loss of distributional tails. The model slightly underweights low-probability events; when these outputs become training data, those events become even rarer in the next generation. Over multiple iterations, this positive feedback loop collapses the distribution to a narrow mode. Mathematically, each generation is like repeatedly applying an imperfect density estimator, and the composition amplifies errors — analogous to photocopying a photocopy."
    },
    {
      type: "mc",
      question: "A common **data mixing** strategy for pre-training assigns sampling weights to different domains (web, books, code, Wikipedia, etc.). The **DoReMi** approach determines these weights by:",
      options: [
        "Setting weights proportional to each domain's raw byte count",
        "Using a small proxy model to learn domain weights that minimize worst-case excess loss across domains, then applying those weights to train the large model",
        "Always upsampling code to 50% of the mixture",
        "Randomly shuffling all domains together with equal probability"
      ],
      correct: 1,
      explanation: "DoReMi (Xie et al., 2023) trains a small proxy model with distributionally robust optimization (DRO) to find domain weights that minimize the worst-case excess loss over a reference model. These optimized weights are then transferred to train the full-scale model. This avoids expensive hyperparameter sweeps at scale and consistently outperforms heuristic mixtures. The key insight is that optimal mixing ratios are not proportional to data size — rare but high-quality domains (code, math) often deserve upsampling."
    },
    {
      type: "mc",
      question: "**Benchmark contamination** occurs when test data appears in the training corpus. Which detection method is most robust?",
      options: [
        "Checking if the model's accuracy exceeds 90% on the benchmark",
        "Searching for exact n-gram overlaps (e.g., 8-gram or longer) between training data and benchmark instances, combined with analyzing performance gaps between contaminated and clean subsets of the benchmark",
        "Asking the model if it has seen the benchmark before",
        "Running the benchmark with a different random seed"
      ],
      correct: 1,
      explanation: "The standard approach (used in GPT-4, Llama, etc.) is n-gram overlap detection: if a training document shares a long n-gram (typically 8-13 grams) with a benchmark instance, that instance is flagged as contaminated. The complementary analysis compares performance on contaminated vs. clean subsets — a large gap indicates the model is benefiting from memorization rather than genuine capability. Neither method alone is sufficient: n-gram search has false negatives (paraphrased contamination), so the performance gap analysis provides a second signal."
    },
    {
      type: "mc",
      question: "When handling **Personally Identifiable Information (PII)** in training data, which approach best balances data utility and privacy?",
      options: [
        "Removing all documents that contain any names",
        "Applying named entity recognition to detect and redact or replace PII (names, emails, phone numbers, addresses) with placeholder tokens, while preserving surrounding context for language learning",
        "Training the model to memorize PII so it can generate realistic data",
        "Only training on data from before 2010 when PII concerns did not exist"
      ],
      correct: 1,
      explanation: "The practical approach is NER-based PII scrubbing: detect entities classified as personal information and either redact them (replace with [REDACTED]) or substitute them with synthetic replacements (fake names/emails). This preserves the linguistic structure and context around PII while removing the sensitive content. Removing entire documents is too aggressive and loses valuable training signal. This connects to broader concerns: GDPR right-to-be-forgotten creates legal obligations that static training data cannot easily satisfy, motivating techniques like machine unlearning."
    },
    {
      type: "mc",
      question: "A team deduplicates a 15TB web crawl and finds that 40% of documents are near-duplicates. After dedup, they train two models: one on the full 15TB (with duplicates) and one on the deduplicated 9TB. Based on empirical findings (e.g., from the Llama and Chinchilla papers), what is the most likely outcome?",
      options: [
        "The 15TB model is strictly better because it saw more tokens",
        "Both models perform identically because the information content is the same",
        "The deduplicated 9TB model matches or outperforms the 15TB model on held-out evaluation, trains faster per epoch, and memorizes less — showing that unique tokens matter more than raw token count",
        "The deduplicated model catastrophically forgets the duplicate content"
      ],
      correct: 2,
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
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "Perplexity is the standard offline evaluation metric for language models. Its primary **limitation** as a measure of model quality is:",
      options: [
        "It is too expensive to compute",
        "It measures next-token prediction accuracy on a fixed corpus but does not capture instruction-following ability, reasoning quality, factual accuracy, or user preference — a model can have excellent perplexity yet produce unhelpful or harmful outputs",
        "It only works for models with fewer than 1 billion parameters",
        "It requires labeled data with human annotations"
      ],
      correct: 1,
      explanation: "Perplexity measures $2^{H(P, Q)}$ — how well the model predicts the next token in a held-out corpus. A model can achieve low perplexity by being excellent at surface-level statistical patterns while failing at reasoning, following instructions, or producing factually correct content. This is why the field moved beyond perplexity to task-specific benchmarks and human evaluation. Perplexity remains useful for comparing base models on the same data distribution, but tells you little about downstream utility."
    },
    {
      type: "mc",
      question: "In **N-shot prompting** for evaluation, you prepend $N$ input-output examples to the test query. Increasing $N$ from 0 to 5 typically helps because:",
      options: [
        "It fine-tunes the model on those examples",
        "The examples specify the task format and output space, reducing ambiguity about what is expected — the model leverages in-context learning to condition its distribution on the demonstrated pattern without any weight updates",
        "It increases the model's parameter count",
        "It reduces the model's perplexity on the training set"
      ],
      correct: 1,
      explanation: "Few-shot prompting exploits in-context learning (ICL): the model conditions on the demonstrations to infer the task distribution. Crucially, no gradient update occurs — the model uses its pre-trained ability to recognize patterns in context. The demonstrations serve as a soft specification of the task: they disambiguate output format, label space, and style. However, results are sensitive to example selection, ordering, and format, which is why evaluation protocols must standardize these choices."
    },
    {
      type: "mc",
      question: "**MMLU** (Massive Multitask Language Understanding) evaluates a model across 57 academic subjects. What does strong MMLU performance primarily indicate?",
      options: [
        "The model can write fluent prose",
        "The model has broad factual knowledge and can perform multiple-choice reasoning across diverse academic domains — it tests breadth of world knowledge absorbed during pre-training more than deep reasoning ability",
        "The model has been fine-tuned on those 57 subjects",
        "The model has a large context window"
      ],
      correct: 1,
      explanation: "MMLU tests breadth of knowledge across STEM, humanities, social sciences, and professional domains using 4-choice MC questions. High performance requires the model to have absorbed factual knowledge during pre-training and to apply basic reasoning. However, the MC format means it tests recognition over generation, and many questions can be answered with surface-level pattern matching rather than deep understanding. It is a useful but incomplete signal — complementary to benchmarks like GSM8K (math reasoning) or HumanEval (code generation)."
    },
    {
      type: "mc",
      question: "**HumanEval** and **GSM8K** measure fundamentally different capabilities. HumanEval tests code generation (writing Python functions from docstrings), while GSM8K tests grade-school math word problems. A model scoring 90% on HumanEval but 40% on GSM8K most likely:",
      options: [
        "Has a bug in its math tokenizer",
        "Was trained with heavy code emphasis but lacks robust multi-step arithmetic and reasoning chains — code generation relies more on pattern completion from the docstring, while GSM8K requires chaining 2-8 arithmetic operations with intermediate reasoning",
        "Is overfitting to GSM8K's test set",
        "Needs a larger context window for math problems"
      ],
      correct: 1,
      explanation: "This capability gap is common and informative. HumanEval functions can often be solved by recognizing the pattern from the docstring and generating idiomatic Python — a strong pattern-matching ability. GSM8K requires constructing multi-step reasoning chains: parsing the word problem, identifying relevant quantities, chaining 2-8 arithmetic operations correctly, and arriving at a numerical answer. Errors compound at each step. This dissociation reveals that code completion and mathematical reasoning are partially independent capabilities."
    },
    {
      type: "mc",
      question: "**Goodhart's Law** — \"when a measure becomes a target, it ceases to be a good measure\" — manifests in LLM benchmarks when:",
      options: [
        "Benchmarks become too easy for all models",
        "Models or training pipelines are optimized specifically to maximize benchmark scores (e.g., training on benchmark-similar data, prompt engineering for the exact format), inflating scores without proportional improvement in genuine capability",
        "Too many benchmarks exist for any single model to complete",
        "The benchmark questions are poorly written"
      ],
      correct: 1,
      explanation: "Goodhart's Law is pervasive in LLM evaluation. Examples include: training data contamination (benchmark questions leak into pre-training corpora), benchmark-specific prompt optimization, generating synthetic training data that mimics benchmark formats, and cherry-picking evaluation configurations. The result is that leaderboard rankings may reflect optimization pressure on the benchmark rather than genuine capability differences. This motivates held-out evaluation (Chatbot Arena), dynamic benchmarks that refresh questions, and multi-benchmark evaluation suites."
    },
    {
      type: "mc",
      question: "**Chatbot Arena** uses ELO ratings derived from pairwise human preferences. Its key methodological advantage over static benchmarks is:",
      options: [
        "It is cheaper to run",
        "It continuously collects fresh human judgments on diverse, user-generated prompts — making it resistant to contamination, Goodhart's Law, and benchmark saturation, while directly measuring what users actually care about",
        "It uses automated scoring instead of human evaluation",
        "It only tests English language ability"
      ],
      correct: 1,
      explanation: "Chatbot Arena addresses core limitations of static benchmarks: (1) prompts come from real users, not a fixed test set, so they cannot be trained on; (2) evaluation is continuous, so the benchmark evolves with model capabilities; (3) pairwise comparison (\"which response is better?\") is a more natural and reliable judgment than absolute scoring; (4) ELO ratings handle transitive preferences and provide a single ranking. The main limitations are cost, speed (thousands of comparisons needed per model), potential demographic bias in the user population, and sensitivity to response length/formatting."
    },
    {
      type: "mc",
      question: "When using an **LLM-as-judge** to evaluate other models' outputs, three well-documented biases are verbosity bias, position bias, and self-preference bias. **Verbosity bias** means:",
      options: [
        "The judge penalizes short responses",
        "The judge systematically prefers longer, more detailed responses even when the additional content is redundant or irrelevant — length serves as a proxy for quality in the judge's learned heuristics",
        "The judge produces verbose explanations of its ratings",
        "The judge can only evaluate verbose prompts"
      ],
      correct: 1,
      explanation: "LLM judges trained on human preference data inherit the bias that longer responses tend to be rated higher by human annotators (since length correlates with effort and completeness). This creates a systematic bias where a 500-word mediocre response may be preferred over a 100-word excellent one. Mitigation strategies include: controlling for length in the evaluation prompt, swapping response positions to average out position bias, using multiple judges, and explicitly instructing the judge to evaluate conciseness. Position bias (preferring the first or second response) and self-preference (GPT-4 favoring GPT-4-style outputs) are orthogonal but equally problematic."
    },
    {
      type: "mc",
      question: "The **elicitation gap** refers to the difference between:",
      options: [
        "The model's training loss and test loss",
        "A model's latent capability and the performance actually measured — the same model can appear much stronger or weaker depending on the prompting strategy, number of shots, chain-of-thought usage, and other evaluation choices that affect how well the model's knowledge is elicited",
        "The gap between open-source and closed-source model performance",
        "The time between model training and model deployment"
      ],
      correct: 1,
      explanation: "The elicitation gap is critical for interpreting benchmarks. The same model might score 40% on a math benchmark with zero-shot prompting but 75% with chain-of-thought and 5-shot prompting. The model's knowledge did not change — only how well the evaluation protocol extracted it. This means benchmark comparisons are only valid when using identical elicitation methods, and low scores may reflect poor elicitation rather than missing capability. It also implies that current benchmarks likely underestimate model capabilities, especially for models that are sensitive to prompt formatting."
    },
    {
      type: "mc",
      question: "A **contamination-robust** evaluation strategy should include:",
      options: [
        "Running benchmarks only once per model",
        "Multiple layers of defense: n-gram overlap detection between training data and test sets, canary string insertion in benchmarks, performance comparison between contaminated and clean subsets, rephrased/perturbed versions of benchmark questions to test whether performance drops (indicating memorization vs. understanding), and temporal holdouts using data created after the training cutoff",
        "Using only perplexity as the evaluation metric",
        "Keeping all benchmark questions secret forever"
      ],
      correct: 1,
      explanation: "No single method catches all contamination. N-gram detection misses paraphrased contamination. Canary strings (unique identifiers embedded in test sets) detect if the test set was ingested but do not catch reformulated questions. Rephrased variants distinguish memorization from understanding: if performance drops sharply on semantically equivalent but syntactically different questions, the model likely memorized rather than learned. Temporal holdouts (benchmarks created after training cutoff) are the gold standard but require continuous benchmark creation. A robust evaluation combines all these approaches."
    },
    {
      type: "mc",
      question: "You evaluate two models on a new reasoning benchmark. Model A scores 82% with standard prompting. Model B scores 71% with standard prompting but 88% with chain-of-thought. A reviewer claims Model A is superior. The most accurate response is:",
      options: [
        "The reviewer is correct — 82% > 71%",
        "Model B is better because 88% > 82%",
        "The comparison is invalid without controlling for elicitation: Model B demonstrates higher latent capability when properly elicited (88% vs. 82%), but the models should be compared using the best-known elicitation strategy for each, with clear reporting of which strategies were used — raw scores under different elicitation regimes are not directly comparable",
        "Neither model can be evaluated because the benchmark is new"
      ],
      correct: 2,
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
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "In **Distributed Data Parallel (DDP)** training, each GPU holds a full copy of the model. After the backward pass, gradients are synchronized across GPUs using:",
      options: [
        "A parameter server that collects and redistributes all gradients",
        "An **all-reduce** operation that efficiently computes the sum (or average) of gradients across all GPUs so every replica ends up with identical gradients — typically implemented as a ring all-reduce to minimize communication overhead",
        "Each GPU sends its gradients to GPU 0, which broadcasts the averaged result",
        "Gradients are not synchronized — each GPU trains independently"
      ],
      correct: 1,
      explanation: "DDP uses all-reduce (typically ring all-reduce or tree all-reduce via NCCL) to synchronize gradients. In ring all-reduce, each GPU sends a chunk of its gradient to its neighbor, and after $2(N-1)$ steps (N = number of GPUs), all GPUs have the complete averaged gradient. The communication volume per GPU is $2 \\cdot (N-1)/N \\cdot |\\text{params}|$, which approaches $2|\\text{params}|$ as $N$ grows — nearly independent of GPU count. This is far more efficient than the naive reduce-broadcast approach via a parameter server, which creates a bottleneck at the central node."
    },
    {
      type: "mc",
      question: "**Tensor parallelism** and **pipeline parallelism** split the model across GPUs in fundamentally different ways. Tensor parallelism:",
      options: [
        "Assigns different training examples to different GPUs",
        "Splits individual layers (e.g., partitioning weight matrices column-wise or row-wise) across GPUs so each GPU computes a portion of every layer's output, requiring intra-layer communication at each forward and backward step",
        "Assigns entire layers to different GPUs in sequence",
        "Replicates the model on every GPU"
      ],
      correct: 1,
      explanation: "Tensor parallelism (Megatron-LM style) partitions weight matrices within a layer. For example, a linear layer $Y = XW$ can be split column-wise: $W = [W_1 | W_2]$, with each GPU computing $XW_i$. This requires an all-reduce after each layer to combine partial results. Pipeline parallelism, by contrast, assigns whole layers to different GPUs — GPU 0 runs layers 1-10, GPU 1 runs layers 11-20, etc. Tensor parallelism has higher communication frequency (every layer) but lower latency per communication; pipeline parallelism has lower communication frequency but suffers from the bubble problem."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 1** shards the **optimizer states** across GPUs while each GPU still holds a full copy of parameters and gradients. For a model with $\\Psi$ parameters using Adam in mixed precision, Stage 1 reduces per-GPU optimizer memory from $12\\Psi$ bytes to approximately:",
      options: [
        "$12\\Psi$ bytes — no savings",
        "$12\\Psi / N$ bytes, where $N$ is the number of GPUs — each GPU stores only $1/N$ of Adam's first moment ($m$), second moment ($v$), and FP32 master weights",
        "$4\\Psi$ bytes — only the FP16 parameters",
        "$2\\Psi$ bytes — only the FP16 gradients"
      ],
      correct: 1,
      explanation: "Adam requires per-parameter state: FP32 master weights (4 bytes), FP32 first moment $m$ (4 bytes), and FP32 second moment $v$ (4 bytes) = 12 bytes per parameter. ZeRO Stage 1 partitions these 12$\\Psi$ bytes across $N$ GPUs, so each GPU stores $12\\Psi/N$ bytes of optimizer state. The FP16 parameters ($2\\Psi$) and FP16 gradients ($2\\Psi$) remain fully replicated. For a 7B model on 8 GPUs: optimizer memory drops from 84 GB to ~10.5 GB per GPU, while parameter and gradient memory remain at 14 GB + 14 GB."
    },
    {
      type: "mc",
      question: "**ZeRO Stage 3** (or equivalently, **FSDP** — Fully Sharded Data Parallel) shards parameters, gradients, AND optimizer states. The key runtime overhead compared to DDP is:",
      options: [
        "No additional overhead — it is strictly better",
        "All-gather operations to reconstruct full parameter tensors before each forward/backward computation, and reduce-scatter operations to distribute gradients — trading communication volume for memory savings",
        "It requires twice as many GPUs",
        "It cannot overlap communication with computation"
      ],
      correct: 1,
      explanation: "In ZeRO-3/FSDP, each GPU stores only a $1/N$ shard of every parameter tensor. Before computing a layer's forward pass, an all-gather reconstructs the full parameters from all shards. After the backward pass, a reduce-scatter distributes gradient shards. The total communication volume per step is $3 \\times 2\\Psi$ (vs. $2\\Psi$ for DDP), a 3x increase. However, this communication can be overlapped with computation by prefetching the next layer's parameters during the current layer's computation. The memory savings are dramatic: total per-GPU memory approaches $(12\\Psi + 2\\Psi + 2\\Psi) / N = 16\\Psi/N$."
    },
    {
      type: "mc",
      question: "The **pipeline bubble problem** in pipeline parallelism arises because:",
      options: [
        "Data cannot be split into micro-batches",
        "At the start and end of each training step, some pipeline stages are idle waiting for activations from upstream or gradients from downstream — with $p$ pipeline stages and $m$ micro-batches, the bubble fraction is $(p - 1) / m$, wasting compute proportional to the number of stages",
        "GPUs cannot communicate across pipeline stages",
        "The model's loss function is non-differentiable across pipeline boundaries"
      ],
      correct: 1,
      explanation: "With naive scheduling, GPU $k$ must wait for GPUs $0, \\dots, k-1$ to complete before starting, creating a \"bubble\" of idle time. Splitting the batch into $m$ micro-batches and interleaving them reduces the bubble fraction to $(p-1)/m$. For example, with 8 pipeline stages and 32 micro-batches, the bubble is $7/32 \\approx 22\\%$ — meaning 22% of compute is wasted. The **1F1B** (one-forward-one-backward) schedule further optimizes memory by limiting the number of in-flight micro-batches, reducing peak activation memory from $O(m)$ to $O(p)$."
    },
    {
      type: "mc",
      question: "The **1F1B** (one-forward-one-backward) pipeline schedule works by:",
      options: [
        "Running all forward passes first, then all backward passes",
        "After an initial warmup phase, alternating between one forward micro-batch and one backward micro-batch on each pipeline stage — this limits the number of in-flight micro-batches per stage to at most $p$ (the pipeline depth), bounding peak activation memory",
        "Running forward and backward passes simultaneously on the same micro-batch",
        "Eliminating the pipeline bubble entirely"
      ],
      correct: 1,
      explanation: "In 1F1B, each stage goes through a warmup phase (receiving and forwarding micro-batches), then enters a steady state where it performs one forward pass followed by one backward pass in alternation. This means each stage holds activations for at most $p$ micro-batches at any time (rather than all $m$ micro-batches in the naive all-forward-then-all-backward schedule). The bubble fraction remains $(p-1)/m$, but peak memory is dramatically reduced. Interleaved scheduling (where virtual pipeline stages are assigned cyclically) can further reduce the bubble to $(p-1)/(m \\cdot v)$ where $v$ is the number of virtual stages."
    },
    {
      type: "mc",
      question: "**BF16** (bfloat16) is preferred over **FP16** for LLM training because:",
      options: [
        "BF16 has higher precision for small numbers",
        "BF16 uses the same 8-bit exponent as FP32 (range $\\pm 3.4 \\times 10^{38}$), avoiding the overflow/underflow issues that plague FP16 (5-bit exponent, range $\\pm 65504$) — this eliminates the need for loss scaling even though BF16 has less mantissa precision (7 bits vs FP16's 10 bits)",
        "BF16 uses less memory than FP16",
        "BF16 is the only format supported by modern GPUs"
      ],
      correct: 1,
      explanation: "FP16 has 5 exponent bits (range $\\sim 6 \\times 10^{-8}$ to $6.5 \\times 10^4$) and 10 mantissa bits. BF16 has 8 exponent bits (same range as FP32: $\\sim 10^{-38}$ to $\\sim 10^{38}$) and 7 mantissa bits. In LLM training, gradients and activations span a wide dynamic range — FP16's limited range causes underflow (small gradients become zero) or overflow (large activations become inf), requiring careful loss scaling. BF16's FP32-matching range avoids these issues entirely at the cost of slightly reduced precision. Both use 16 bits (2 bytes per value). The practical result: BF16 training is nearly as stable as FP32 with half the memory."
    },
    {
      type: "mc",
      question: "**Activation checkpointing** (gradient checkpointing) trades compute for memory by:",
      options: [
        "Compressing activations using quantization",
        "Discarding intermediate activations during the forward pass and recomputing them from saved checkpoints during the backward pass — this reduces activation memory from $O(L)$ to $O(\\sqrt{L})$ (with optimal checkpoint placement) at the cost of one additional forward pass, roughly 33% more compute",
        "Storing activations on CPU instead of GPU",
        "Reducing the number of layers in the model"
      ],
      correct: 1,
      explanation: "During the forward pass, only activations at checkpoint boundaries are saved; intermediate activations are discarded. During the backward pass, when intermediate activations are needed for gradient computation, the forward pass is re-run from the nearest checkpoint. With checkpoints every $\\sqrt{L}$ layers (for $L$ total layers), memory is $O(\\sqrt{L})$ and compute increases by ~33% (one extra forward pass). This is often the single most impactful memory optimization: for a 70B model, it can reduce activation memory from hundreds of GB to a manageable level. The tradeoff is almost always worthwhile — memory is the binding constraint, not compute."
    },
    {
      type: "mc",
      question: "A **70B parameter model** trained with Adam in mixed precision requires approximately how much **optimizer state memory** (across all GPUs combined)?",
      options: [
        "140 GB (2 bytes per parameter for FP16 weights)",
        "280 GB (4 bytes per parameter for FP32 master copy only)",
        "840 GB (12 bytes per parameter: FP32 master weights + FP32 first moment + FP32 second moment)",
        "70 GB (1 byte per parameter)"
      ],
      correct: 2,
      explanation: "Adam maintains three FP32 buffers per parameter: (1) master copy of weights — 4 bytes, (2) first moment estimate $m$ — 4 bytes, (3) second moment estimate $v$ — 4 bytes. Total: $12 \\times 70 \\times 10^9 = 840 \\times 10^9$ bytes $= 840$ GB. This is the dominant memory cost and the primary motivation for ZeRO/FSDP. On 8 GPUs with ZeRO Stage 1, this drops to ~105 GB/GPU. Adding the FP16 model parameters (140 GB) and FP16 gradients (140 GB), total memory is ~1120 GB, or ~140 GB/GPU with 8-way ZeRO-1 for optimizer states alone (parameters and gradients still replicated at Stage 1)."
    },
    {
      type: "mc",
      question: "**Sequence parallelism** addresses a specific limitation of tensor parallelism. In standard Megatron-style tensor parallelism, operations like LayerNorm and dropout are **replicated** on every GPU. Sequence parallelism fixes this by:",
      options: [
        "Splitting the vocabulary across GPUs",
        "Partitioning the sequence dimension across GPUs for these replicated operations (LayerNorm, dropout, activation functions), so each GPU processes a portion of the sequence — then transitioning back to tensor-parallel partitioning for the attention and MLP computations",
        "Using a longer context window",
        "Replacing LayerNorm with a parallelizable alternative"
      ],
      correct: 1,
      explanation: "In tensor parallelism, matrix multiplications (attention projections, MLP layers) are split across GPUs, but LayerNorm, dropout, and activation functions operate on the full hidden dimension and are redundantly computed on every GPU. Sequence parallelism (Korthikanti et al., 2022) partitions these operations along the sequence dimension instead: each GPU handles $\\text{seq\\_len}/N$ tokens for LayerNorm/dropout, then the layout transitions to tensor-parallel for the split matrix multiplications. This eliminates the redundant computation and memory for these operations, saving ~30-40% of activation memory that would otherwise be wasted on replicated non-tensor-parallel regions."
    }
  ]
};
