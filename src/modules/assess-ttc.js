// Assessment module for D.2: Test-Time Compute
// Split from assess-branch-cd.js — per-section test (10 questions)

export const testTimeComputeAssessment = {
  id: "D.2-assess",
  sectionId: "D.2",
  title: "Assessment: Test-Time Compute",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The o1/o3 family of models use \"extended thinking\" at inference time. The core mechanism is:",
      options: [
        "Running the model on multiple GPUs simultaneously to perform parallel beam search over the top-$k$ most likely token sequences, selecting the highest-scoring completion",
        "Searching over a database of pre-computed answers indexed by question embeddings and returning the closest semantic match from the cached response store",
        "Fine-tuning the model on each new question using a few gradient steps of in-context adaptation before generating the response, customizing weights per query",
        "Generating a **long internal chain-of-thought** (potentially thousands of tokens) before producing the final answer — trained via RL to explore, backtrack, and verify"
      ],
      correct: 3,
      explanation: "o1-style models generate an extended hidden reasoning trace before answering. The model was trained with reinforcement learning to use this trace productively: exploring multiple approaches, catching and correcting errors, and verifying intermediate results. This trades inference compute (generating many tokens) for accuracy. The reasoning trace can be thousands of tokens long for hard math/coding problems, but the model learns when to think briefly vs. extensively based on problem difficulty."
    },
    {
      type: "mc",
      question: "Best-of-N sampling generates $N$ complete responses and selects the best one using a reward model or verifier. Compared to self-consistency (majority voting), best-of-N:",
      options: ["Uses less compute since it only generates one response and applies the reward model as a cheap post-hoc filter on that single output to verify correctness", "Always produces better results because the reward model provides a strictly more informative selection signal than simple answer frequency counting across samples", "Can select based on **response quality rather than just answer frequency** — it can prefer a well-reasoned minority answer, but its effectiveness is bounded by verifier accuracy", "Is identical to self-consistency in both mechanism and outcome, since the reward model's ranking always agrees with the majority vote result in practice"],
      correct: 2,
      explanation: "Self-consistency selects by answer frequency, implicitly assuming that correct reasoning paths outnumber incorrect ones. Best-of-N uses a learned verifier to evaluate response quality, which can capture reasoning quality beyond just the final answer. However, best-of-N is limited by: (1) verifier accuracy — a miscalibrated verifier may prefer confident-sounding wrong answers, (2) sample coverage — $N$ samples may not include a correct response for very hard problems, and (3) reward hacking if the verifier has exploitable biases."
    },
    {
      type: "mc",
      question: "Applying Monte Carlo Tree Search (MCTS) to LLM reasoning faces a fundamental challenge that doesn't exist in game-playing (e.g., AlphaGo):",
      options: ["The **branching factor is enormous** — at each token or reasoning step there are thousands of possible continuations, making exhaustive tree search intractable compared to Go's ~250 moves", "MCTS requires a board representation with a fixed set of discrete legal moves, and natural language reasoning cannot be decomposed into a comparably finite action space", "LLMs can't play games because they lack the spatial reasoning capabilities that MCTS requires for effective evaluation of intermediate positions and states", "LLMs are too slow for real-time search because each node evaluation requires a full forward pass through billions of parameters, making deep tree exploration impractical"],
      correct: 0,
      explanation: "MCTS succeeds in Go because: (1) the branching factor (~250) is manageable, (2) a value network provides reliable position evaluations, and (3) the game has a clear terminal reward. For LLM reasoning: (1) the branching factor at the token level is ~32K-128K, requiring either very aggressive pruning or operating at the \"reasoning step\" level, (2) evaluating partial reasoning chains is much harder than evaluating board positions, and (3) defining what constitutes a \"move\" in reasoning is ambiguous. Approaches like LATS and RAP address these by chunking reasoning into coarse steps and using the LLM itself as a value function."
    },
    {
      type: "mc",
      question: "Compute-optimal inference allocation suggests that the optimal amount of test-time compute should depend on:",
      options: ["Only the model size — larger models should always think longer because they have more representational capacity to utilize during extended reasoning and search", "The time of day and current server load, since compute allocation should be dynamically adjusted based on available infrastructure capacity and demand patterns", "The **difficulty of the specific input** — easy questions should use minimal test-time compute while hard questions benefit from extended reasoning, search, or multiple samples", "The user's subscription tier, since higher-paying users should receive more inference compute regardless of the difficulty or complexity of their specific query"],
      correct: 2,
      explanation: "Inference scaling research shows that the value of additional test-time compute follows a difficulty-dependent curve. For easy problems, the model's first answer is usually correct — additional compute is wasted. For hard problems, extended thinking, multiple samples, or search can dramatically improve accuracy. The optimal strategy adaptively allocates compute: a difficulty classifier or confidence estimator decides how much inference-time computation to invest per query. This mirrors the human intuition of \"thinking harder\" about harder problems."
    },
    {
      type: "mc",
      question: "Inference scaling laws describe how performance improves with test-time compute. Compared to training scaling laws (Chinchilla), inference scaling:",
      options: [
        "Can be **more cost-efficient for hard problems** — a smaller model with 100x inference compute can sometimes match a 10x larger model with 1x compute on reasoning tasks",
        "Shows the same power-law exponent as training scaling, meaning identical returns per FLOP whether spent during pre-training or at inference time",
        "Never matches training-time scaling in practice — no amount of inference-time search can compensate for insufficient model capacity or training data",
        "Follows an exponential rather than power law, meaning each additional unit of inference compute yields exponentially larger accuracy improvements"
      ],
      correct: 0,
      explanation: "Research from DeepMind and OpenAI shows that for certain problem types (particularly verifiable reasoning), investing in test-time compute can substitute for model size. A 7B model with best-of-256 sampling can match a 70B model's accuracy on math problems. However, the trade-off depends heavily on task type: for knowledge-intensive tasks, the information must be in the model's parameters, and no amount of inference compute can compensate. The optimal compute allocation between training and inference depends on deployment volume — high-volume tasks favor larger trained models, while rare hard queries favor inference-time scaling."
    },
    {
      type: "mc",
      question: "A system uses best-of-N with a reward model to solve math problems. With $N = 1$, accuracy is 40%. With $N = 100$, accuracy reaches 75%. Doubling to $N = 200$ would most likely yield:",
      options: ["~50% — additional samples beyond 100 introduce so many near-miss candidates that the reward model's selection accuracy degrades substantially", "~100% accuracy — doubling the sample count from 100 to 200 doubles the improvement, continuing the linear trend from earlier gains", "~75% — the accuracy curve plateaus exactly at $N = 100$, with additional samples providing zero marginal benefit beyond that threshold", "~78-80% — improvements follow **logarithmic scaling** with $N$; each doubling provides diminishing marginal gains as coverage approaches the model's ceiling"],
      correct: 3,
      explanation: "Best-of-N accuracy scales as $1 - (1 - p)^N$ where $p$ is the per-sample probability of a correct-and-top-ranked response. For large $N$, this saturates. If the model has a 1% chance of producing the correct answer per sample, $N=100$ gives $\\sim 63\\%$ coverage, and $N=200$ gives $\\sim 87\\%$. But with a reward model selector, the binding constraint is often the reward model's ability to distinguish correct from incorrect solutions, not just coverage. Empirically, accuracy scales roughly as $a + b \\log(N)$ in the relevant range."
    },
    {
      type: "mc",
      question: "Process reward models (PRMs) can be used for step-level beam search during test-time reasoning. In this approach, the search:",
      options: ["Generates all tokens simultaneously using the PRM scores as a non-autoregressive selection criterion over the full vocabulary at each position", "Only evaluates the final answer and uses the PRM score to decide whether to accept or reject the complete reasoning chain before presenting it", "Uses the PRM to generate tokens directly by sampling from the PRM's output distribution rather than from the language model's next-token predictions", "Maintains a beam of $B$ partial reasoning chains, scoring each at every reasoning step using the PRM and pruning low-scoring branches — this focuses compute on the most promising reasoning paths rather than committing to a single chain or running independent samples"],
      correct: 3,
      explanation: "Step-level beam search with PRMs is a structured test-time compute strategy: at each reasoning step, $B$ candidate continuations are generated, the PRM scores each partial chain, and only the top-$B$ survive. This is more efficient than best-of-N (which runs $N$ independent chains to completion) because compute is concentrated on promising paths early. The PRM acts as a value function estimating the probability of reaching a correct answer from each partial state. This is the approach advocated by Let's Verify Step by Step and subsequent work on verifier-guided search."
    },
    {
      type: "mc",
      question: "A key distinction between o1-style extended thinking and standard chain-of-thought prompting is:",
      options: ["Extended thinking models are trained with **RL to optimize the reasoning process itself** — the model learns when to explore, backtrack, and verify rather than relying on pre-trained generation", "Extended thinking only works in English because the RL training data for reasoning optimization was collected exclusively from English-language mathematical proofs and solutions", "Standard CoT produces longer reasoning traces because it includes more verbose natural language explanations, while extended thinking uses compressed internal token representations", "Extended thinking uses a different model architecture with specialized reasoning layers that are activated only during the thinking phase and bypassed for direct answers"],
      correct: 0,
      explanation: "Standard CoT relies on the model's pre-trained distribution over explanation-like text — it produces plausible-looking reasoning but has no explicit incentive to reason correctly or to self-correct. o1-style models are trained with RL where the reward signal is answer correctness, so the model learns to use its reasoning trace strategically: trying multiple approaches, identifying and recovering from errors, and allocating reasoning effort proportional to problem difficulty. The RL training fundamentally changes the model's relationship to its own reasoning output from \"generating plausible text\" to \"using text as a computational medium.\""
    },
    {
      type: "mc",
      question: "Parallel test-time compute (generating $N$ independent samples) vs sequential test-time compute (one long reasoning chain with $N$ times the tokens) differ in that:",
      options: [
        "They always produce identical results since both approaches use the same total number of generated tokens and therefore explore equivalent solution spaces",
        "Parallel sampling provides **diversity** (exploring independent starting points) but no depth, while sequential provides **depth** but risks compounding errors over the extended chain",
        "Sequential is always better because generating more tokens in a single chain provides strictly more information than the same tokens spread across independent samples",
        "Parallel is always better because it uses more GPUs in parallel, maximizing hardware utilization and achieving higher throughput than a single sequential chain"
      ],
      correct: 1,
      explanation: "This is a fundamental axis in test-time compute allocation. Parallel sampling (best-of-N, self-consistency) explores breadth: independent samples cover more of the solution space but each sample is limited in depth. Sequential reasoning (extended CoT, iterative refinement) explores depth: building elaborate reasoning chains but committed to a single trajectory. For problems requiring a creative insight (exploration), parallel sampling is better. For problems requiring long derivations (depth), sequential is better. Optimal strategies often combine both: parallel samples of extended reasoning chains."
    },
    {
      type: "mc",
      question: "A company deploys a reasoning model and observes that 80% of user queries are simple (answered correctly with 1 inference step) while 20% are hard (requiring 50 inference steps). If they allocate the same 50-step budget to all queries, the wasted compute fraction is approximately:",
      options: ["0% — more compute never hurts model quality, so allocating extra reasoning steps to easy queries only improves confidence without wasting any resources", "About 78% — the 80% of easy queries waste 49 of their 50 allocated steps: $\\frac{0.8 \\times 49}{0.8 \\times 50 + 0.2 \\times 50} = \\frac{39.2}{50}$ of total compute goes to unnecessary reasoning", "About 20% — only the hard queries contribute waste, since easy queries complete early and release their allocated compute back to the shared resource pool", "About 50% — half the total compute is wasted because the 50-step budget is optimal for exactly half the queries (the hard ones) and excessive for the other half"],
      correct: 1,
      explanation: "With uniform 50-step allocation: easy queries use 1 useful step + 49 wasted = 0.8 * 49 = 39.2 wasted step-equivalents. Hard queries use all 50 steps productively. Total compute = 50 steps * 100% of queries = 50 units. Wasted = 39.2 units. Waste fraction = 39.2/50 = 78.4%. Adaptive allocation (1 step for easy, 50 for hard) uses 0.8*1 + 0.2*50 = 10.8 units — a 4.6x compute reduction. This illustrates why difficulty-adaptive inference allocation is crucial for cost-effective deployment of reasoning models."
    }
  ]
};
