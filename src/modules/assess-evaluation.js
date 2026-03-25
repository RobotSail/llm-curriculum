// Assessment: Evaluation (Section 1.5)
// 10 MC questions, no info steps. Pure assessment module.

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
