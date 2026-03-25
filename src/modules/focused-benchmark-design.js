// Focused learning module: Benchmark Design and Evaluation Pitfalls
// Section 1.5: Evaluation
// Covers: why LLM evaluation is hard, perplexity as eval, MC benchmarks,
// elicitation gap, Goodhart's Law, LLM-as-judge, calibration in deployment.
// Single-concept module following CLAUDE.md conventions.

export const benchmarkDesignLearning = {
  id: "1.5-benchmark-design-learning-easy",
  sectionId: "1.5",
  title: "Benchmark Design and Evaluation Pitfalls",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "Why Evaluating LLMs Is Fundamentally Hard",
      content: "Classification models have clean evaluation: compare predicted labels to ground truth, compute accuracy. Language models generate **open-ended text**, and there is no single correct answer to most prompts.\n\nConsider: \"Explain quantum entanglement.\" There are thousands of valid responses varying in depth, formality, accuracy, and structure. An evaluation system must somehow distinguish a brilliant explanation from a mediocre one from a confidently wrong one — and human judges often disagree on the ranking.\n\nThis creates a fundamental tension:\n- **Automated metrics** (perplexity, BLEU, ROUGE) are cheap and reproducible but correlate weakly with human judgment of quality.\n- **Human evaluation** is the gold standard but expensive, slow, subjective, and hard to reproduce.\n- **Benchmark suites** (MMLU, HellaSwag, etc.) convert open-ended evaluation into structured tasks — but every structured task is a **lossy proxy** for real-world capability.\n\nThe evaluation challenge is not just technical — it is epistemological. We are trying to measure intelligence-like capabilities with finite test sets, and every measurement introduces distortions."
    },
    {
      type: "mc",
      question: "A team develops a new LLM and reports state-of-the-art perplexity on a held-out web corpus. Users report that its generated text is less coherent and useful than a competitor with higher perplexity. What best explains this discrepancy?",
      options: [
        "Perplexity measures how well the model predicts the held-out data distribution, which may not reflect generation quality — a model can achieve low perplexity by being conservative (assigning moderate probability to many tokens) without producing coherent long-form text",
        "The held-out corpus is contaminated with training data, artificially lowering the measured perplexity without improving true capability",
        "Perplexity is computed on random token-level predictions and has no mathematical relationship to text generation quality",
        "The users are biased toward the competitor's brand and are not evaluating output quality objectively"
      ],
      correct: 0,
      explanation: "Perplexity measures next-token prediction quality on held-out text, which is necessary but not sufficient for good generation. A model can achieve low perplexity by hedging — assigning reasonable probability to many plausible continuations without committing to coherent long-form structure. Generation quality depends on maintaining topic consistency, avoiding repetition, and producing informative content over hundreds of tokens — properties that token-level perplexity does not directly measure."
    },
    {
      type: "info",
      title: "Perplexity as Evaluation: Strengths and Limitations",
      content: "Perplexity ($\\text{PPL} = \\exp(H(p_{\\text{data}}, p_\\theta))$) is the oldest and most principled LLM metric. It directly measures the model's predictive quality in information-theoretic terms.\n\n**Strengths**:\n- Requires no task-specific labels — any text corpus serves as a test set\n- Mathematically grounded: lower perplexity = closer to the true data distribution in KL divergence\n- Cheap to compute: one forward pass over the test corpus\n- Sensitive to improvements: small architecture or data changes show up clearly\n\n**Limitations**:\n- **Tokenizer-dependent**: Different tokenizers split the same text differently. A model using BPE with 32K vocabulary and one with 50K vocabulary cannot be compared by raw perplexity because they predict different units.\n- **Favors hedging**: A model that assigns probability 0.01 to 100 plausible next tokens has lower perplexity than one that assigns 0.5 to the right token and 0.5 to a wrong one — but the latter is more useful for generation.\n- **Doesn't measure generation**: Perplexity evaluates $P(x_t | x_{<t})$ with ground-truth prefixes. It says nothing about what happens when the model generates its own prefixes.\n- **Domain-sensitive**: Perplexity on Wikipedia ≠ perplexity on code ≠ perplexity on dialogue. A single number hides per-domain variation."
    },
    {
      type: "mc",
      question: "Model A uses a 32K-token BPE vocabulary and achieves perplexity 15.2 on a test set. Model B uses a 64K-token vocabulary and achieves perplexity 18.7 on the same test set. Can we conclude Model A is better?",
      options: [
        "Yes — perplexity is a universal metric that allows direct comparison regardless of tokenizer differences",
        "Yes — but only if both models were trained on the same data, since training data affects perplexity more than tokenizer choice",
        "No — different tokenizers produce different token sequences from the same text, making raw perplexity values incomparable; per-character or per-byte perplexity would be needed for fair comparison",
        "No — the model with the larger vocabulary always has higher perplexity because it must predict from a larger set, so Model B is actually better after normalization"
      ],
      correct: 2,
      explanation: "Different tokenizers decompose the same text into different numbers of tokens. A 64K vocabulary may tokenize \"unfortunately\" as one token while a 32K vocabulary splits it into \"un\" + \"fortunate\" + \"ly\". Perplexity is per-token, so the models are predicting different quantities. To compare fairly, normalize by character or byte count: compute bits-per-character (BPC) or bits-per-byte (BPB). The larger vocabulary does NOT automatically have higher perplexity — it depends on the model quality."
    },
    {
      type: "info",
      title: "Multiple-Choice Benchmarks: How They Work",
      content: "Benchmarks like MMLU, HellaSwag, and ARC convert evaluation into multiple-choice format. The standard evaluation protocol uses **log-probability scoring**:\n\n1. Present the question as a prompt: \"Question: What is the capital of France?\\nAnswer:\"\n2. Compute the model's log-probability of each answer option: $\\log P_\\theta(\\text{\"Paris\"} | \\text{prompt})$, $\\log P_\\theta(\\text{\"London\"} | \\text{prompt})$, etc.\n3. Select the option with the highest log-probability as the model's answer.\n\nThis avoids parsing the model's free-form generation and gives a clean accuracy number. But it introduces subtle biases:\n\n**Length bias**: Longer answer options get lower log-probability (more tokens to predict). Some benchmarks normalize by answer length; others don't, creating systematic bias toward shorter answers.\n\n**Surface form competition**: If two options start with the same tokens, the model's probability mass is split between them early in the sequence, artificially lowering both.\n\n**Prompt sensitivity**: The exact wording of the prompt, the number and order of few-shot examples, and even whitespace can change scores by 5-15%. This means benchmark numbers are only reproducible if the exact evaluation protocol is specified."
    },
    {
      type: "mc",
      question: "On a multiple-choice benchmark, a model consistently selects shorter answer options over longer ones, even when the longer option is correct. This bias is most likely caused by:",
      options: [
        "The model has learned a shortcut from training data where correct answers tend to be concise",
        "The softmax function penalizes longer sequences by dividing probability mass across more tokens",
        "Log-probability scoring without length normalization: longer options accumulate more negative log-probability terms, making them systematically less likely to be selected",
        "The model's attention mechanism cannot process answer options beyond a certain token length"
      ],
      correct: 2,
      explanation: "Without length normalization, the log-probability of a multi-token answer is $\\sum_t \\log P(w_t | w_{<t}, \\text{prompt})$ — a sum of negative terms. More tokens means more negative terms, so longer answers get lower (more negative) total log-probability regardless of quality. Length normalization (dividing by token count) corrects this, but not all evaluation harnesses apply it consistently. This is a property of the evaluation protocol, not the model itself."
    },
    {
      type: "info",
      title: "The Elicitation Gap",
      content: "A model may \"know\" something but fail to demonstrate it under a specific evaluation protocol. The **elicitation gap** is the difference between what the model is capable of and what we manage to measure.\n\nSources of the elicitation gap:\n\n**Prompt format**: A model might answer correctly with \"Q: ... A:\" format but fail with \"Question: ... Answer:\" — not because it lacks knowledge, but because the format doesn't match its training distribution well.\n\n**Few-shot vs zero-shot**: Many models perform dramatically better with 5-shot examples than zero-shot, not because the examples teach new knowledge, but because they **demonstrate the expected response format**.\n\n**Chain-of-thought**: Asking the model to \"think step by step\" before answering can improve accuracy by 10-30% on reasoning tasks. The model had the reasoning capability all along — it just needed to be prompted to use it.\n\n**This means benchmark scores are lower bounds on capability.** A model scoring 60% on a benchmark with zero-shot evaluation might score 80% with optimal prompting. The gap between these numbers is real and practically important — a user who prompts well gets a fundamentally different experience than one who doesn't.\n\nThe elicitation gap also means that benchmark comparisons are only valid when models are evaluated with comparable prompting effort."
    },
    {
      type: "mc",
      question: "Model A scores 65% on a reasoning benchmark with zero-shot prompting and 82% with chain-of-thought prompting. Model B scores 75% zero-shot and 78% with chain-of-thought. A leaderboard ranks Model B higher based on zero-shot scores. What should we conclude?",
      options: [
        "Model B is genuinely better — zero-shot evaluation is the fairest comparison because it requires no prompt engineering tricks",
        "Model A is genuinely better — chain-of-thought scores reflect true capability, and the 82% vs 78% gap shows Model A has stronger reasoning that zero-shot evaluation fails to elicit",
        "The ranking depends on the use case — if users will prompt carefully, Model A is better; the zero-shot ranking reflects out-of-the-box usability while chain-of-thought reflects peak capability",
        "Neither comparison is valid — the 17-point gap for Model A suggests its chain-of-thought scores are inflated by prompt overfitting rather than genuine reasoning"
      ],
      correct: 2,
      explanation: "Both rankings are valid but measure different things. Zero-shot measures out-of-the-box usability — how well the model performs with minimal user effort. Chain-of-thought measures peak elicited capability. Model A has higher ceiling but lower floor. The right comparison depends on the deployment context: an API serving expert prompters should use chain-of-thought rankings; a consumer product should weight zero-shot performance. A 17-point CoT improvement is within normal range and doesn't indicate overfitting."
    },
    {
      type: "info",
      title: "Goodhart's Law: When Benchmarks Become Targets",
      content: "\"When a measure becomes a target, it ceases to be a good measure.\" — Charles Goodhart\n\nThis principle applies forcefully to LLM evaluation. Once a benchmark is widely used for model comparison, incentives emerge to optimize for it specifically:\n\n**Benchmark contamination**: Benchmark data leaks into training corpora through web scraping. Models memorize answers rather than learning to reason. A model that has seen MMLU questions during training will score well on MMLU without necessarily being smarter.\n\n**Teaching to the test**: Training pipelines can include benchmark-like data, fine-tune on similar formats, or adjust data mixing to favor domains covered by benchmarks — all of which inflate scores without proportional improvement in real-world utility.\n\n**RLHF reward hacking**: When a reward model is used as a proxy for human preference, the policy can learn to exploit patterns in the reward model rather than genuinely improving. The reward model score goes up while actual quality plateaus or degrades. This is Goodhart's Law applied to the training objective itself.\n\n**Benchmark saturation**: As models approach ceiling performance on a benchmark, differences in scores reflect noise and evaluation artifacts rather than meaningful capability differences. A benchmark where top models score 95-98% is no longer discriminating."
    },
    {
      type: "mc",
      question: "Over 3 years, the top score on a popular LLM benchmark rises from 45% to 94%. The benchmark creators note that the questions haven't changed and contamination checks show no direct leakage. Does this prove that LLMs have dramatically improved in the capability the benchmark measures?",
      options: [
        "Yes — if contamination checks pass and the questions are unchanged, score improvements must reflect genuine capability gains across all models tested",
        "Not necessarily — indirect contamination (training on similar content, format-specific optimization, data mixing tuned to benchmark domains) can inflate scores without direct leakage, and Goodhart's Law suggests the metric has become less reliable as a target",
        "No — benchmark scores are entirely unreliable once models exceed 80% accuracy, so the 94% score is meaningless and should be discarded",
        "Yes — but only if the benchmark uses randomized question ordering and answer positions to prevent positional memorization"
      ],
      correct: 1,
      explanation: "Score improvements likely reflect a mix of genuine capability gains AND benchmark-specific optimization. Contamination checks only catch direct leakage — they miss indirect contamination (training on text discussing the benchmark, optimizing data mix for benchmark-relevant domains, format-specific fine-tuning). As the benchmark becomes a target, an increasing fraction of score gains come from optimization rather than capability. The scores are not meaningless, but they overestimate true improvement. This is why the community continuously develops new benchmarks."
    },
    {
      type: "info",
      title: "LLM-as-Judge: Using Models to Evaluate Models",
      content: "With human evaluation expensive and benchmarks gameable, a pragmatic middle ground has emerged: using a **strong LLM as an automated judge** to evaluate weaker models.\n\nThe protocol: present the judge with a prompt and two candidate responses, ask it to pick the better one (or rate each on a scale). This is cheaper than human evaluation and more flexible than fixed benchmarks.\n\n**Known biases**:\n- **Position bias**: The judge tends to prefer the response presented first (or second, depending on the judge model). Mitigated by evaluating both orderings and averaging.\n- **Verbosity bias**: Longer, more detailed responses are rated higher even when the extra detail is redundant or incorrect. The judge conflates thoroughness with quality.\n- **Self-preference**: Models tend to rate outputs from their own model family higher than outputs from other families, even when humans disagree.\n- **Style over substance**: Judges can prefer confidently wrong answers over correctly hedged ones, especially when the judge model itself doesn't know the correct answer.\n\nDespite these biases, LLM judges correlate 80-90% with human rankings in controlled studies — good enough for rapid iteration, though not a replacement for human evaluation on high-stakes decisions."
    },
    {
      type: "mc",
      question: "An LLM judge evaluates pairs of responses. In 70% of cases where it disagrees with humans, the judge preferred the longer response while humans preferred the shorter one. How should the evaluation protocol be modified?",
      options: [
        "Replace the LLM judge with a smaller, faster model that has less bias toward verbosity due to simpler internal representations",
        "Add explicit instructions to the judge to penalize length, with calibration: evaluate pairs of known-quality responses at different lengths to tune the penalty",
        "Discard all judgments where the responses differ in length by more than 20%, since the judge cannot reliably compare responses of different lengths",
        "Switch to human evaluation entirely, since LLM judges are fundamentally unreliable when response lengths differ"
      ],
      correct: 1,
      explanation: "The most practical fix is prompt engineering: instruct the judge to focus on accuracy and relevance rather than completeness, and calibrate using pairs where the shorter response is known to be better. This addresses the verbosity bias directly without discarding data or abandoning automated evaluation. Smaller models often have worse judgment overall. Discarding length-mismatched pairs wastes most of the data. And human evaluation, while ideal, is too expensive for routine iteration."
    },
    {
      type: "info",
      title: "From Benchmarks to Deployment: The Calibration Gap",
      content: "A model scoring 85% on MMLU does **not** mean it will be correct 85% of the time in production. The gap between benchmark performance and deployment reliability has several sources:\n\n**Distribution shift**: Benchmarks are curated, clean, and well-formatted. Real user queries are noisy, ambiguous, multi-step, and often underspecified. A model calibrated on benchmark distributions is not calibrated on user distributions.\n\n**Task mismatch**: MMLU tests factual recall and multiple-choice reasoning. Deployment tasks include summarization, instruction following, code generation, creative writing, and multi-turn dialogue — capabilities that MMLU doesn't measure.\n\n**Adversarial inputs**: Benchmarks don't include deliberately misleading, adversarial, or edge-case inputs. Real users (intentionally or accidentally) probe failure modes that benchmarks miss.\n\n**The aggregation problem**: A single benchmark score averages across thousands of questions spanning dozens of domains. A model at 85% overall might be 99% on geography and 40% on abstract algebra. The average hides dangerous domain-specific weaknesses.\n\nThe practical implication: benchmark scores are useful for **model selection** (comparing candidates) but not for **reliability estimation** (predicting production failure rates). Deployment requires domain-specific evaluation on data that matches the actual use case."
    },
    {
      type: "mc",
      question: "A team deploys an LLM that scores 90% on a medical knowledge benchmark. In production, doctors report that the model gives dangerously wrong advice roughly 30% of the time on their specific queries. What is the most likely explanation?",
      options: [
        "The benchmark is flawed and its scores are meaningless — a 90% score should not be interpreted as any indication of medical capability",
        "The model has degraded since the benchmark was run due to concept drift in the model weights over time",
        "Distribution shift between benchmark questions (clean, textbook-style, multiple-choice) and real clinical queries (complex, multi-condition, requiring nuanced judgment and caveats) makes the 90% score a poor predictor of deployment reliability",
        "The doctors are using the model incorrectly — with proper prompting, the model would achieve close to 90% accuracy on their queries as well"
      ],
      correct: 2,
      explanation: "Medical benchmarks test textbook knowledge in clean multiple-choice format. Real clinical queries involve complex patient histories, interacting conditions, ambiguous symptoms, and the need for appropriate hedging and referral recommendations. The model may have the factual knowledge (reflected in the 90% score) but fail at the reasoning and judgment required for real clinical scenarios. This is a classic distribution shift problem — the benchmark measures a necessary but insufficient component of the actual deployment task."
    }
  ]
};
