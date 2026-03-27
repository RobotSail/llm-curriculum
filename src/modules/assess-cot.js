// Assessment module for D.1: Chain-of-Thought & Reasoning
// Split from assess-branch-cd.js — per-section test (10 questions)

export const cotAssessment = {
  id: "D.1-assess",
  sectionId: "D.1",
  title: "Assessment: Chain-of-Thought & Reasoning",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The **scratchpad hypothesis** for why chain-of-thought (CoT) improves LLM reasoning posits that:",
      options: ["CoT activates special reasoning circuits in the transformer that are dormant during standard next-token prediction and require explicit prompting to engage them", "CoT prompts contain the answer implicitly within their exemplar structure, and the model merely extracts it through surface-level pattern matching on the format", "Writing more tokens gives the model more time to think in wall-clock time, and the additional GPU processing cycles during generation accumulate into deeper reasoning", "The intermediate tokens serve as **external working memory** — they allow the model to decompose multi-step problems into sequential single-step computations, extending its bounded computational depth"],
      correct: 3,
      explanation: "A transformer's forward pass has fixed computational depth (number of layers). For problems requiring more sequential computation steps than layers, the model cannot solve them in a single pass. CoT externalizes intermediate computations into generated tokens, which are then fed back as input. Each token generation step adds another forward pass worth of computation, conditioned on previous results. This effectively gives the model $O(T \\times L)$ sequential computation for $T$ output tokens and $L$ layers, rather than just $O(L)$."
    },
    {
      type: "mc",
      question: "Self-consistency (Wang et al., 2022) improves CoT reasoning by:",
      options: ["Sampling **multiple independent reasoning chains** (with temperature > 0), extracting the final answer from each, and returning the answer with the highest frequency — a majority voting scheme that marginalizes over diverse reasoning paths", "Using a verifier model to score each reasoning chain and selecting the single chain with the highest verification score as the final output", "Asking the model to check its own work by appending a verification prompt after the initial answer and regenerating if errors are detected", "Training the model to be internally consistent by adding an auxiliary loss that penalizes contradictions between different parts of its reasoning"],
      correct: 0,
      explanation: "Self-consistency samples $N$ reasoning chains (e.g., $N = 40$) with temperature sampling, then takes a majority vote on the final answers. The intuition: correct answers tend to be reachable via multiple valid reasoning paths, while incorrect answers typically result from specific errors that vary across samples. Formally, it approximates $\\arg\\max_a P(a \\mid \\text{question})$ by marginalizing over reasoning paths. This is a simple but powerful technique — it improved GSM8K accuracy from 56% (single CoT) to 74% on PaLM 540B."
    },
    {
      type: "mc",
      question: "Tree-of-Thought (ToT) extends chain-of-thought by:",
      options: [
        "Using tree-structured attention patterns that allow each token to attend to multiple branching context paths simultaneously during a single forward pass",
        "Generating answers in a tree-structured format where each branch represents a different aspect of the solution, merged into a final unified response",
        "Exploring a **tree of possible reasoning steps**, where the model evaluates and selects among multiple candidate next steps at each node — using search algorithms like BFS or DFS with self-evaluation to prune unpromising branches",
        "Training on tree-structured data that explicitly encodes branching reasoning paths, teaching the model to produce multi-path solutions during inference"
      ],
      correct: 2,
      explanation: "ToT structures reasoning as a search problem over a tree: each node represents a partial solution, each edge is a reasoning step, and the model both generates candidate steps and evaluates their promise. This enables backtracking (abandoning bad reasoning paths) and look-ahead (evaluating partial solutions before committing). For problems like creative writing or planning, ToT significantly outperforms linear CoT because it avoids the irrecoverable commitment problem — a bad step in linear CoT permanently derails the solution."
    },
    {
      type: "mc",
      question: "Process reward models (PRMs) provide feedback on each **step** of a reasoning chain, as opposed to outcome reward models (ORMs) that only score the final answer. The advantage of PRMs for math reasoning is:",
      options: ["PRMs are cheaper to train because they use shorter input sequences, with each step evaluated independently rather than requiring full-chain processing through the entire verifier model", "PRMs provide **dense, step-level credit assignment** — they identify exactly where a reasoning chain went wrong, enabling more efficient search and better training signal than a single reward for the whole chain", "PRMs always produce more accurate final answers because step-level verification guarantees that every intermediate result is mathematically correct before allowing the next step to proceed", "PRMs don't require human annotations since the correctness of individual reasoning steps can be verified automatically through symbolic execution of each mathematical operation"],
      correct: 1,
      explanation: "ORMs suffer from the sparse reward problem: a 10-step derivation gets a single correct/incorrect label, giving no signal about which step caused a failure. PRMs score each step (e.g., step 3 introduced an algebraic error), enabling: (1) best-first search that prunes reasoning chains at the first erroneous step, (2) more efficient training since each step provides a supervision signal, and (3) interpretable failure analysis. Let's Verify Step by Step (Lightman et al., 2023) showed PRMs substantially outperform ORMs for math reasoning on MATH benchmark."
    },
    {
      type: "mc",
      question: "When does chain-of-thought reasoning **hurt** performance compared to direct answering?",
      options: ["Never — CoT always helps regardless of the task type, since more reasoning tokens always improve the quality of the final output produced by the model", "On tasks requiring **fast pattern matching or factual recall** — CoT can introduce errors by overthinking simple retrievals, leading the model astray on questions where direct answering is confident", "Only on mathematical problems, since CoT was designed specifically for step-by-step arithmetic and has no benefit on verbal, logical, or commonsense reasoning tasks", "When the model is very large, since models above 100B parameters already have sufficient internal computation depth to solve problems without externalizing reasoning steps"],
      correct: 1,
      explanation: "CoT helps most on multi-step reasoning tasks (math, logic, multi-hop QA) where the problem genuinely requires sequential computation. It hurts on: (1) simple factual recall (\"What is the capital of France?\") where reasoning steps are unnecessary noise, (2) tasks where the model's first instinct is correct but overthinking introduces doubt, (3) pattern-matching tasks like sentiment analysis where reasoning can rationalize wrong answers. Empirically, CoT provides little benefit or even degrades performance for models below ~100B parameters on many tasks."
    },
    {
      type: "mc",
      question: "The debate over whether LLMs are \"truly reasoning\" versus \"pattern matching\" centers on:",
      options: ["Whether LLMs use symbolic or subsymbolic representations internally, since only symbolic representations can support true logical inference and formal deduction over structured inputs", "Whether LLMs were trained on sufficient reasoning data to acquire genuine reasoning capabilities, or whether the gap requires fundamentally different training objectives beyond scaling", "Whether LLMs exhibit **systematic generalization** — solving novel problem compositions they haven't seen in training — or merely interpolate between memorized training examples and surface patterns", "Whether LLMs use more parameters than the human brain has synapses, since raw computational capacity determines the boundary between shallow pattern matching and real reasoning"],
      correct: 2,
      explanation: "The critical test is out-of-distribution generalization: can models solve problems requiring novel combinations of learned operations? Evidence for pattern matching: performance degrades on math problems with unusual number ranges, models are fooled by irrelevant information that shouldn't affect logical reasoning, and models struggle with problems structurally identical to training examples but with different surface forms. Evidence for reasoning: models can solve some novel compositions, show consistent performance on well-structured problems, and benefit from CoT in ways consistent with genuine computation. The truth likely involves both capabilities in different proportions."
    },
    {
      type: "mc",
      question: "A model is asked to solve $23 \\times 47$ using chain-of-thought. It writes: \"$23 \\times 47 = 23 \\times 40 + 23 \\times 7 = 920 + 161 = 1081$.\" This is a correct decomposition. The model likely learned this strategy because:",
      options: [
        "It has a built-in calculator module that activates when arithmetic expressions are detected in the token stream during generation, offloading the computation to a symbolic subsystem that returns exact results without relying on learned patterns",
        "Its training data contains many examples of **distributive property decomposition** for multiplication — the model learned the pattern $a \\times (b + c) = ab + ac$ from seeing similar worked examples, but may fail on numbers requiring different decomposition strategies or carrying patterns it hasn't seen",
        "Transformers can natively perform multiplication through the interaction of attention patterns and feed-forward layers at sufficient depth, implementing the algorithm implicitly in the residual stream without needing explicit step-by-step reasoning tokens",
        "The chain-of-thought prompt forced the correct algorithm by constraining the generation to follow the specific decomposition pattern shown in the examples, leaving the model no room to choose an alternative factoring strategy or numerical approach"
      ],
      correct: 1,
      explanation: "The model applies a learned strategy (distributive property) that it has seen in training data. It can execute $23 \\times 40 = 920$ and $23 \\times 7 = 161$ as separate, simpler retrievals/computations. This illustrates both the power and limitation of CoT: it enables multi-step computation by chaining learned operations, but the model's \"arithmetic\" is fundamentally pattern-matching on trained examples. It may fail on numbers that trigger different carrying patterns or require strategies not well-represented in training data."
    },
    {
      type: "mc",
      question: "Few-shot CoT prompting provides exemplar reasoning chains in the prompt. Zero-shot CoT instead uses a trigger phrase like \"Let's think step by step.\" The fact that zero-shot CoT works at all suggests:",
      options: ["Instruction-tuned models have **internalized a general reasoning mode** that can be activated by trigger phrases — this mode was learned from the many step-by-step explanations in training data and RLHF, making the model predisposed to produce structured reasoning when prompted appropriately", "The model doesn't actually need examples because the chain-of-thought format is hardcoded into the model's architecture through special attention patterns", "Zero-shot CoT works only on easy problems where the model already has high confidence, and fails on problems that genuinely require multi-step reasoning", "The trigger phrase is mathematically optimal for maximizing the log-probability of correct reasoning traces under the model's learned output distribution"],
      correct: 0,
      explanation: "Zero-shot CoT (Kojima et al., 2022) showed that \"Let's think step by step\" improves accuracy across diverse reasoning tasks without any exemplars. This works because: (1) pre-training on web data includes millions of step-by-step explanations, (2) instruction tuning explicitly rewards detailed reasoning, (3) RLHF preferences favor thorough explanations. The trigger phrase shifts the model's generation distribution toward the \"explanation\" mode it learned during training. This is evidence that reasoning capabilities are latent in the model and can be elicited by appropriate prompting."
    },
    {
      type: "mc",
      question: "When using self-consistency with $N = 40$ samples, the computational cost is 40x a single generation. Which statement about the cost-accuracy trade-off is correct?",
      options: ["Accuracy scales linearly with $N$ — doubling from 40 to 80 samples doubles the accuracy gain, so more samples always yield proportional improvements", "All $N$ samples must use the same temperature — varying temperature across samples invalidates the majority voting assumption by mixing incompatible distributions", "Self-consistency with $N=2$ is always worse than single CoT — the majority vote between two samples degenerates to random selection when they disagree", "Accuracy improvements follow **diminishing returns** — gains from $N=1$ to $N=5$ are much larger than from $N=20$ to $N=40$, and the optimal $N$ depends on the task difficulty"],
      correct: 3,
      explanation: "Majority voting accuracy follows a law of diminishing returns. For an easy problem where the model gets the right answer 80% of the time, even $N=5$ gives >95% accuracy (binomial probability). For a hard problem where the correct rate is 30%, even $N=100$ won't produce a majority-correct result. The marginal improvement from each additional sample decreases because the majority vote converges. The compute-optimal strategy is to allocate more samples to harder problems (adaptive compute) rather than using a fixed $N$ across all inputs."
    },
    {
      type: "mc",
      question: "A model answers a math problem correctly with CoT, but when the same problem is rephrased with a misleading \"common sense\" cue (e.g., adding irrelevant context suggesting a different answer), the model changes its answer. This demonstrates:",
      options: ["**Reasoning fragility** — the model's chain-of-thought is influenced by surface-level features and social priors rather than following purely logical computation that is invariant to irrelevant context", "CoT doesn't work for math problems because the step-by-step format introduces opportunities for error at each reasoning step that direct answering would avoid", "The rephrased problem was actually harder because adding context increases the input complexity, requiring the model to parse and filter more information before solving", "The model needs more training data on adversarially rephrased problems, and scaling the dataset with such examples would eliminate the sensitivity to irrelevant cues"],
      correct: 0,
      explanation: "This is a key piece of evidence in the reasoning-vs-pattern-matching debate. Studies show that adding irrelevant information (e.g., \"Alice has 5 apples\" in a problem that doesn't involve Alice) or framing a problem to suggest a common-but-wrong answer can cause models to produce incorrect reasoning chains that rationalize the wrong answer. True compositional reasoning should be invariant to such perturbations. This suggests the model's \"reasoning\" is partly a post-hoc rationalization process guided by distributional cues rather than a faithful logical computation."
    }
  ]
};
