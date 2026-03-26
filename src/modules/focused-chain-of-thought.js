// Focused learning module: Chain-of-Thought Reasoning
// Section D.1: Chain-of-Thought & Reasoning
// Covers: why CoT works (external working memory / computational depth extension),
// few-shot vs zero-shot CoT, self-consistency, when CoT helps vs hurts,
// and the faithfulness problem.
// Single-concept module: chain-of-thought as a mechanism for extending LLM reasoning.

export const chainOfThoughtLearning = {
  id: "D.1-cot-learning-easy",
  sectionId: "D.1",
  title: "Chain-of-Thought Reasoning",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: The computational depth problem
    {
      type: "info",
      title: "The Bounded Computation Problem",
      content: "A transformer processes input through a fixed number of layers $L$. Each token passes through every layer once during a forward pass, giving the model $O(L)$ sequential computation steps to produce its next-token prediction.\n\nFor many tasks this is enough — classifying sentiment, translating a short phrase, or recalling a fact requires only shallow processing. But consider multi-step reasoning:\n\n- \"If Alice is taller than Bob, and Bob is taller than Carol, who is shortest?\"\n- \"What is $23 \\times 47$?\"\n- \"A train leaves at 3 PM going 60 mph. Another leaves at 4 PM going 80 mph. When do they meet?\"\n\nEach of these requires **sequential computation steps** — you must resolve one relationship or operation before proceeding to the next. If the number of required steps exceeds the model's depth $L$, the model cannot solve the problem in a single forward pass, no matter how many parameters it has.\n\nThis is the core limitation that chain-of-thought reasoning addresses: by generating intermediate tokens, the model converts a single $O(L)$-depth computation into $O(T \\times L)$ depth across $T$ generated tokens."
    },
    // Step 2: MC
    {
      type: "mc",
      question: "A 32-layer transformer is asked to solve a problem requiring 5 sequential reasoning steps. Without chain-of-thought, the model must fit all 5 steps into one forward pass. With CoT, the model writes each step as a token sequence before producing the answer. Why does this help?",
      options: [
        "Each generated token triggers a full forward pass through all 32 layers, and the result is fed back as input — so 5 generated reasoning steps give the model $5 \\times 32 = 160$ effective layers of sequential computation",
        "CoT tokens activate specialized reasoning circuits in the attention heads that are dormant during standard generation, enabling deeper processing within the same number of layers",
        "The additional tokens increase the sequence length, which expands the attention window and allows the model to attend to more context simultaneously",
        "CoT works by increasing the model's parameter count at inference time through dynamic layer duplication for each reasoning step"
      ],
      correct: 0,
      explanation: "Each generated token requires a full autoregressive forward pass through all $L$ layers. The output token is appended to the input for the next generation step. So $T$ reasoning tokens give $T \\times L$ total sequential computation, with each step conditioned on all previous results. This is the **scratchpad hypothesis**: intermediate tokens serve as external working memory, allowing the model to decompose multi-step problems into a sequence of single-step computations, each within its $O(L)$ budget."
    },
    // Step 3: Few-shot CoT
    {
      type: "info",
      title: "Few-Shot Chain-of-Thought Prompting",
      content: "Chain-of-thought prompting (Wei et al., 2022) demonstrated that including **exemplar reasoning chains** in the prompt dramatically improves multi-step reasoning.\n\nStandard few-shot prompting:\n```\nQ: Roger has 5 tennis balls. He buys 2 cans of 3. How many does he have now?\nA: 11\n\nQ: [new question]\nA:\n```\n\nFew-shot CoT prompting:\n```\nQ: Roger has 5 tennis balls. He buys 2 cans of 3. How many does he have now?\nA: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.\n\nQ: [new question]\nA:\n```\n\nThe only difference is showing the **reasoning process** in the exemplar answers. The model learns to mimic this format, producing its own intermediate steps for new questions.\n\nKey findings from Wei et al.:\n- CoT provides large gains on math (GSM8K), logic, and commonsense reasoning\n- Gains scale with model size: CoT helps large models (100B+) much more than small ones\n- On GSM8K: PaLM 540B went from 18% (standard) to 57% (CoT) — a 39-point jump\n- Small models (<10B) often produce incoherent reasoning chains that hurt rather than help"
    },
    // Step 4: MC
    {
      type: "mc",
      question: "Few-shot CoT dramatically improves large model reasoning but provides little benefit to small models (under ~10B parameters). What is the most likely explanation for this scale dependence?",
      options: [
        "Small models lack the representational capacity to maintain coherent multi-step reasoning — they produce surface-level imitations of reasoning format without executing the underlying logical operations, so errors compound across steps",
        "Small models have not seen enough reasoning examples during pretraining to learn the CoT format, and few-shot examples cannot overcome this gap due to limited in-context learning capacity",
        "The attention mechanism in small models cannot track dependencies across the longer sequence lengths created by CoT tokens, causing information loss between reasoning steps",
        "Small models have fewer layers, so each generated reasoning token adds less computation than in a large model — the computational depth extension is proportionally smaller"
      ],
      correct: 0,
      explanation: "Small models can imitate the format (\"Let me break this down...\") but lack the capacity to perform correct computation at each reasoning step. When a small model writes \"5 + 6 = 12\" as an intermediate step, the error propagates to subsequent steps. CoT helps only when the model can reliably execute each individual step — which requires sufficient model capacity. This is why CoT shows an **emergent** pattern: it provides essentially zero benefit below a model-size threshold, then rapidly becomes useful above it."
    },
    // Step 5: Zero-shot CoT
    {
      type: "info",
      title: "Zero-Shot CoT: \"Let's Think Step by Step\"",
      content: "Kojima et al. (2022) discovered that adding a simple trigger phrase — **\"Let's think step by step\"** — to the prompt (with no exemplars) elicits reasoning chains that improve accuracy.\n\nThis works because:\n1. **Pretraining data** contains millions of step-by-step explanations (math textbooks, tutorials, Stack Overflow answers)\n2. **Instruction tuning** explicitly trains the model to produce detailed, structured explanations when prompted\n3. **RLHF** rewards thorough reasoning — human annotators prefer responses that show their work over bare answers\n\nThe trigger phrase shifts the model's generation distribution from the \"answer directly\" mode to the \"explain step by step\" mode. These modes were learned from training data — the model has internalized the concept of \"showing work\" and can activate it on demand.\n\nZero-shot CoT is weaker than few-shot CoT (fewer concrete examples to calibrate the reasoning format) but much more convenient — it requires no task-specific exemplars. It works best on instruction-tuned models that have been heavily trained to follow natural-language directives."
    },
    // Step 6: MC
    {
      type: "mc",
      question: "A base model (pretrained only, no instruction tuning) is prompted with \"Let's think step by step\" on a math problem. An instruction-tuned version of the same base model is prompted identically. The instruction-tuned model produces a structured reasoning chain; the base model produces rambling, off-topic text. What explains the difference?",
      options: [
        "Instruction tuning modifies the transformer architecture to add specialized reasoning pathways that the base model lacks, enabling systematic step-by-step processing",
        "The base model has the same latent reasoning ability but generates in a different format — extracting the answer from its rambling output would yield equivalent accuracy",
        "Instruction tuning trained the model to interpret natural-language directives as task specifications and to respond with structured outputs, so it recognizes \"step by step\" as a request for sequential reasoning format",
        "The base model's tokenizer encodes \"step by step\" differently than the instruction-tuned model's tokenizer, causing the trigger phrase to be misinterpreted"
      ],
      correct: 2,
      explanation: "Instruction tuning teaches the model to follow instructions — it learns that phrases like \"step by step\" are directives requesting structured output. The base model has seen similar text in pretraining but hasn't been trained to treat it as a command. Both models have similar knowledge, but the instruction-tuned model has learned the mapping from instruction format to response format. This is why zero-shot CoT is much more effective on instruction-tuned or RLHF models than on base models."
    },
    // Step 7: Self-consistency
    {
      type: "info",
      title: "Self-Consistency: Majority Voting Over Reasoning Paths",
      content: "A single CoT chain can contain errors — an arithmetic mistake or a wrong inference in step 3 that derails the rest. **Self-consistency** (Wang et al., 2022) mitigates this by sampling multiple independent reasoning chains and taking a majority vote on the final answer.\n\nThe procedure:\n1. Sample $N$ reasoning chains using temperature $T > 0$ (e.g., $N = 40$, $T = 0.7$)\n2. Extract the final answer from each chain\n3. Return the most frequent answer (majority vote)\n\nThe key insight: **correct answers are reachable via many valid reasoning paths, while incorrect answers typically result from specific errors that vary across samples.** If the model gets the right answer 60% of the time in individual chains, a majority vote over 40 chains will be correct with very high probability.\n\nFormally, self-consistency approximates:\n$$\\hat{a} = \\arg\\max_a \\sum_{i=1}^{N} \\mathbf{1}[a_i = a]$$\n\nwhich is an empirical estimate of $\\arg\\max_a P(a | \\text{question})$ — marginalizing over all possible reasoning paths.\n\nThe cost is $N\\times$ generation — significant for large models. Diminishing returns set in quickly: the jump from $N=1$ to $N=5$ is large, but from $N=20$ to $N=40$ is small. Hard problems where the model's individual accuracy is below 50% cannot be fixed by majority voting, since the wrong answer is the majority."
    },
    // Step 8: MC
    {
      type: "mc",
      question: "A model answers a math problem correctly 40% of the time with single-chain CoT. Self-consistency with $N = 100$ samples still fails to produce the correct answer. Why does majority voting fail here?",
      options: [
        "100 samples is too few for the central limit theorem to apply — with 1,000 samples, the majority vote would converge to the correct answer",
        "The 40% correct rate means the wrong answer (or answers) collectively hold 60% probability mass — majority voting amplifies the most frequent answer, which is wrong",
        "Self-consistency requires all $N$ chains to use the same reasoning strategy — diverse strategies split the correct-answer vote among different phrasings",
        "Temperature sampling at $T > 0$ introduces noise that degrades accuracy below the greedy baseline, and this noise outweighs any benefit from majority voting"
      ],
      correct: 1,
      explanation: "Majority voting returns the most frequent answer. If the correct answer appears 40% of the time and a single wrong answer appears 45% of the time, the wrong answer wins regardless of $N$. More precisely, if the wrong answers don't split (e.g., there's one dominant error mode), majority voting converges to the wrong answer with more samples. Self-consistency helps when the correct answer is the plurality (most common) among individual chains. When it isn't, you need a better model or better prompting, not more samples."
    },
    // Step 9: When CoT helps and hurts
    {
      type: "info",
      title: "When Chain-of-Thought Helps vs. Hurts",
      content: "CoT is not universally beneficial. Understanding when it helps reveals what it actually does.\n\n**CoT helps on:**\n- **Multi-step arithmetic**: Decomposing $23 \\times 47$ into $(23 \\times 40) + (23 \\times 7)$\n- **Multi-hop reasoning**: Connecting facts across a chain of deductions\n- **Word problems**: Translating natural language into a sequence of operations\n- **Planning**: Breaking a complex task into sub-goals\n\nThe common thread: these tasks require **more sequential computation than a single forward pass provides**.\n\n**CoT hurts on:**\n- **Simple factual recall**: \"What is the capital of France?\" — reasoning tokens add noise without adding value\n- **Pattern matching / classification**: Sentiment analysis, named entity recognition — the model's first instinct is usually correct, and reasoning can rationalize wrong answers\n- **Tasks below model capacity**: If the model can already solve it in one pass, CoT adds latency without improving accuracy\n\nA useful heuristic: **if a human would need scratch paper to solve it, CoT will likely help.** If a human would answer instantly, CoT is unnecessary or harmful.\n\nEmpirical evidence: on simple QA benchmarks like TriviaQA, CoT sometimes decreases accuracy by 2-5% — the model \"overthinks\" and introduces doubt about answers it would have gotten right."
    },
    // Step 10: MC
    {
      type: "mc",
      question: "An LLM is tested on two tasks: (1) multi-digit multiplication ($347 \\times 829$) and (2) translating common English phrases to French. On task 1, CoT improves accuracy from 12% to 68%. On task 2, CoT decreases accuracy from 94% to 91%. What explains this asymmetry?",
      options: [
        "Multi-digit multiplication requires sequential carry operations that exceed single-pass depth, while translation is a parallel mapping that the model already handles within its layer budget",
        "CoT is specifically designed for mathematical operations and has no benefit for language tasks — the prompting format is incompatible with translation",
        "The translation task has a larger training data representation, so the model memorized more translation pairs and doesn't need additional computation",
        "The multiplication improvement comes from the model using a different algorithm (long multiplication) that it can only access through CoT prompting"
      ],
      correct: 0,
      explanation: "Multiplication requires a chain of sequential operations (multiply digits, carry, accumulate) that exceed what 32-100 layers can compute in one pass. CoT externalizes these steps. Translation of common phrases is a well-learned mapping that the model executes comfortably within its layer budget — each input-output pair is a parallel lookup rather than a sequential computation. Adding reasoning steps to translation introduces unnecessary processing that can perturb the model's confident direct mapping, slightly degrading accuracy."
    },
    // Step 11: Faithfulness
    {
      type: "info",
      title: "The Faithfulness Problem",
      content: "A critical question about CoT: **does the written reasoning actually reflect the model's computation, or is it a post-hoc rationalization?**\n\nEvidence that CoT is sometimes unfaithful:\n\n1. **Biased coin experiments** (Turpin et al., 2024): When irrelevant features are added to CoT prompts (e.g., a hint suggesting the wrong answer), models often produce reasoning chains that arrive at the hinted answer — writing plausible-sounding but incorrect justifications. The model's final answer was influenced by the bias, but its reasoning chain doesn't acknowledge this.\n\n2. **Consistent conclusions from inconsistent reasoning**: Models sometimes produce reasoning chains with logical errors in intermediate steps but still arrive at the correct answer — suggesting the answer was determined independently of the written reasoning.\n\n3. **Steganographic encoding**: In principle, a model could encode information in subtle token choices that are invisible to human readers but influence subsequent generation steps. The visible reasoning would be a cover story.\n\nFaithfulness matters because if CoT reasoning is a rationalization rather than a computation, we cannot trust it for:\n- **Safety**: Reasoning-based monitoring assumes the model's stated reasoning reflects its actual decision process\n- **Interpretability**: CoT is only useful for understanding model behavior if it's honest\n- **Verification**: Checking reasoning chains for errors only catches real errors if the chain is faithful"
    },
    // Step 12: MC
    {
      type: "mc",
      question: "A model is given a multiple-choice question with a misleading hint (\"A professor says the answer is B\"). The model writes a reasoning chain that arrives at B with plausible-looking justification, but the correct answer is C. When the same question is asked without the hint, the model correctly answers C with valid reasoning. What does this demonstrate about CoT faithfulness?",
      options: [
        "The model's reasoning chain is unfaithful — its final answer was influenced by the social bias (authority hint) but the written reasoning fabricated a domain-based justification rather than acknowledging the hint's influence",
        "The model correctly updated its beliefs based on expert testimony (the professor's opinion), which is rational Bayesian behavior — the hint genuinely changed the right answer",
        "This is expected behavior from few-shot learning — the hint acts as an additional training example that legitimately shifts the model's posterior over answers",
        "The model has two separate reasoning systems: a fast intuitive system influenced by hints and a slow deliberate system used for CoT — the hint bypassed the deliberate system"
      ],
      correct: 0,
      explanation: "This is a classic faithfulness failure from Turpin et al. (2024). The model's answer was causally influenced by the irrelevant hint, but the reasoning chain fabricated a content-based justification. The model didn't write \"I'm choosing B because the professor said so\" — it constructed a plausible-looking domain argument for B. This means the visible reasoning chain did not faithfully represent the model's actual decision process. If we relied on the reasoning chain for safety monitoring (\"does the model's reasoning contain flaws?\"), we would miss the actual cause of the error."
    },
    // Step 13: Practical guidelines
    {
      type: "info",
      title: "Practical Guidelines for CoT Deployment",
      content: "Based on the research evidence, here are practical guidelines for using CoT effectively:\n\n**When to use CoT:**\n- Problems requiring multi-step reasoning (math, logic, planning)\n- Tasks where the model's direct-answer accuracy is mediocre but improvable\n- Settings where you can afford 2-10x inference cost (for self-consistency)\n\n**When to skip CoT:**\n- Simple factual retrieval or classification\n- Latency-critical applications where extra tokens are costly\n- Small models (<10B parameters) where CoT is unreliable\n\n**Self-consistency best practices:**\n- Start with $N = 5$-$10$ — captures most of the accuracy gain\n- Use temperature $T = 0.5$-$0.8$ for diverse but coherent chains\n- Monitor individual-chain accuracy: if below 30%, self-consistency won't help\n\n**Faithfulness considerations:**\n- Don't assume the reasoning chain reveals the model's actual decision process\n- For safety-critical applications, verify conclusions independently rather than trusting reasoning\n- Be aware that biased prompts can produce biased conclusions wrapped in plausible reasoning"
    },
    // Step 14: MC
    {
      type: "mc",
      question: "A team deploys an LLM with CoT for a medical diagnosis support tool. They plan to have doctors review the model's reasoning chains to catch errors before acting on recommendations. A colleague argues this safety strategy is insufficient. What is the strongest version of their argument?",
      options: [
        "Doctors lack the technical expertise to evaluate LLM reasoning chains, since the model uses mathematical notation and probabilistic language that medical professionals are not trained to interpret",
        "CoT reasoning chains may be unfaithful — the model could arrive at a conclusion through biases or pattern matching while generating a plausible-looking medical justification, meaning errors in the actual decision process are invisible in the written reasoning",
        "The model's reasoning chains are too long for doctors to review efficiently in a clinical setting, making the safety check impractical regardless of its theoretical soundness",
        "Reviewing reasoning chains creates automation bias — doctors will tend to agree with plausible-sounding reasoning rather than independently evaluating the conclusion against their own clinical judgment"
      ],
      correct: 1,
      explanation: "The faithfulness problem is the strongest objection. If the model's written reasoning doesn't reflect its actual computation, then reviewing the reasoning only catches errors in the cover story, not in the actual decision process. A model might recommend a treatment based on pattern-matching to similar cases in training data, but write a reasoning chain citing relevant clinical guidelines. The reasoning chain would look medically sound to a reviewer, but the actual decision was made on different grounds entirely. Option D (automation bias) is also a valid concern, but it's a human factor issue, not a fundamental limitation of the CoT approach itself."
    }
  ]
};
