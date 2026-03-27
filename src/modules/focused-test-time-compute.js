// Focused module: Test-Time Compute Scaling
// Section D.2: The systematic relationship between inference-time computation and performance
// ONE concept: spending more compute at inference time improves outputs predictably

export const testTimeComputeLearning = {
  id: "D.2-learning-easy",
  sectionId: "D.2",
  title: "Test-Time Compute Scaling",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "The Core Insight: Compute at Inference Time",
      content: "Scaling laws for training are well-established: more data, more parameters, more FLOPs → better models. But there is a second dimension of scaling that has transformed LLM capabilities: **test-time compute**.\n\nThe key insight is simple: instead of only investing compute during training, you can invest additional compute **when generating each response**. A model that \"thinks harder\" on a difficult question can outperform a much larger model that answers immediately.\n\nThis creates a fundamental trade-off in system design:\n\n$$\\text{Total quality} = f(\\text{training compute}) + g(\\text{inference compute})$$\n\nFor a fixed compute budget, you can choose to train a larger model and run it once per query, or train a smaller model and run it many times (or for many more tokens) per query. The optimal split depends on the task difficulty and deployment volume."
    },
    {
      type: "mc",
      question: "A company has a fixed compute budget for serving. They can either deploy a 70B model answering each query in one pass, or a 7B model using 10x inference compute per query. When is the smaller model + more inference compute likely to win?",
      options: [
        "On knowledge-intensive factual recall tasks where the answer must be stored in the model's parameters during pretraining",
        "On creative writing tasks where a single coherent narrative voice is more important than exploring multiple alternatives",
        "On verifiable reasoning tasks like math and coding where generating multiple attempts and checking answers is possible",
        "On classification tasks where the model needs to output a single label with calibrated confidence probabilities"
      ],
      correct: 2,
      explanation: "Test-time compute is most valuable when: (1) the task has a verifier — you can check if a solution is correct, and (2) the search space is rich — multiple reasoning paths exist. Math and coding satisfy both: solutions can be verified by execution or proof-checking, and there are many valid approaches. For factual recall, the knowledge must be in the parameters — no amount of reasoning compensates for missing facts. For classification, the model's first pass is usually sufficient."
    },
    {
      type: "info",
      title: "Two Axes: Parallel vs Sequential Compute",
      content: "Test-time compute can be spent along two fundamentally different axes:\n\n**Parallel (breadth)**: Generate $N$ independent responses and select the best one. Each sample explores a different region of the solution space. Methods include:\n- **Best-of-N sampling**: Generate $N$ candidates, score with a reward model, return the highest-scoring one\n- **Self-consistency / majority voting**: Generate $N$ candidates, extract final answers, return the most common answer\n\n**Sequential (depth)**: Generate a single longer response with more reasoning steps. The model builds on its own intermediate results. Methods include:\n- **Extended chain-of-thought**: Trained via RL to produce long reasoning traces (o1-style)\n- **Iterative refinement**: Generate a draft, critique it, revise — repeat\n\nThe crucial difference: parallel sampling provides **exploration diversity** (independent starting points), while sequential reasoning provides **reasoning depth** (building multi-step arguments). Neither dominates the other — problems requiring a creative insight benefit from parallel diversity, while problems requiring long derivations benefit from sequential depth."
    },
    {
      type: "mc",
      question: "A math competition problem requires discovering a non-obvious algebraic substitution that simplifies the entire problem. Once found, the solution is straightforward. Which test-time compute strategy is most effective?",
      options: [
        "Sequential depth — one very long chain of reasoning that methodically explores every algebraic manipulation in order",
        "A combination of both — parallel sampling to discover the key substitution, then sequential reasoning to complete the proof",
        "Neither strategy helps — the model either knows the substitution from training or it doesn't, and no amount of search changes this",
        "Parallel sampling with many independent attempts — each sample might stumble upon the key insight from a different starting approach"
      ],
      correct: 3,
      explanation: "When the bottleneck is discovering a single key insight, parallel sampling excels: each independent attempt starts from a different angle, and any one of them might find the substitution. Sequential reasoning from a single starting point may never encounter the right approach. This is analogous to the exploration-exploitation trade-off: parallel sampling is pure exploration (many independent trials), while sequential reasoning is exploitation (deepening a single line of thought). For insight-dependent problems, breadth beats depth."
    },
    {
      type: "info",
      title: "Scaling Behavior: Logarithmic Returns",
      content: "How does performance improve as we increase test-time compute? Empirically, for best-of-N sampling:\n\n$$\\text{accuracy}(N) \\approx a + b \\cdot \\log(N)$$\n\nThis means **doubling $N$ adds a constant improvement** — diminishing returns, but the improvement never fully stops (until hitting the model's coverage ceiling).\n\nWhy logarithmic? Consider a model with per-sample success probability $p$ on a problem. The probability of **at least one** correct sample in $N$ independent tries is:\n\n$$P(\\text{success}) = 1 - (1-p)^N$$\n\nFor small $p$, this grows roughly as $1 - e^{-pN}$, which saturates. The log-scaling of accuracy reflects the fact that each additional sample has a decreasing marginal probability of being the first correct one.\n\nCritically, the value of extra samples is **difficulty-dependent**: if $p = 0.9$ (easy problem), even $N = 1$ usually works. If $p = 0.001$ (hard problem), you need $N \\approx 1000$ for a 63% chance of success. This motivates **adaptive compute allocation** — spending more inference compute on harder queries."
    },
    {
      type: "mc",
      question: "A model solves a particular problem correctly 5% of the time ($p = 0.05$). Using best-of-N with a perfect verifier, approximately how many samples are needed for a 95% chance of at least one correct solution?",
      options: [
        "About 60 samples — solving $1 - (0.95)^N = 0.95$ gives $N = \\log(0.05)/\\log(0.95) \\approx 58.4$",
        "About 20 samples — since $p = 0.05$, we need $1/p = 20$ tries to expect one success on average",
        "About 200 samples — we need $N$ such that $Np = 10$ for the Poisson approximation to give $> 95\\%$ coverage",
        "About 1000 samples — the logarithmic scaling law means we need $20 \\times 50 = 1000$ for near-certainty"
      ],
      correct: 0,
      explanation: "We need $1 - (1-p)^N \\geq 0.95$, so $(0.95)^N \\leq 0.05$. Taking logs: $N \\geq \\log(0.05)/\\log(0.95) \\approx 58.4$, so about 60 samples. Note this is ~3x the naive $1/p = 20$ estimate (which gives only ~64% success probability). The gap between \"expected one success\" ($N = 1/p$) and \"high-confidence success\" ($N \\approx 3/p$) is important for deployment reliability."
    },
    {
      type: "info",
      title: "The Verifier Bottleneck",
      content: "Test-time compute strategies that generate multiple candidates need a way to **select the best one**. This is the verifier bottleneck — the quality of selection bounds the value of additional samples.\n\n**Outcome Reward Models (ORMs)**: Score complete responses. Trained on preference data or correctness labels. Limitation: they can only evaluate the final output, not intermediate reasoning quality.\n\n**Process Reward Models (PRMs)**: Score individual reasoning steps. Trained on step-level correctness annotations (e.g., \"Let's Verify Step by Step\"). Advantage: they enable **step-level beam search**, pruning bad reasoning paths early rather than wasting compute on doomed chains.\n\nWith a PRM, you can run a search that maintains a beam of $B$ partial reasoning chains, scoring and pruning at each step:\n\n$$\\text{beam search cost} \\approx B \\times L \\times c_{\\text{step}}$$\n\nwhere $L$ is the number of reasoning steps and $c_{\\text{step}}$ is the per-step cost. This is more efficient than best-of-N (which runs $N$ full chains) because compute is concentrated on promising paths.\n\nThe key limitation: **the verifier's accuracy caps the benefit of more samples**. If the verifier can't reliably distinguish correct from incorrect solutions, adding more samples adds noise, not signal."
    },
    {
      type: "mc",
      question: "A process reward model (PRM) is used for step-level beam search with beam width $B = 4$ over 10 reasoning steps. Compared to best-of-16 with an outcome reward model (ORM), which uses roughly similar total compute, the PRM approach:",
      options: [
        "Is strictly worse because it constrains exploration by pruning early, potentially discarding chains that would have self-corrected in later steps",
        "Produces identical results because both methods explore the same number of total reasoning tokens and therefore cover the same solution space",
        "Focuses compute on the most promising reasoning paths by pruning low-quality branches early, but requires reliable step-level scoring to avoid premature elimination of correct chains",
        "Is strictly better because process-level feedback always provides more information than outcome-level feedback regardless of the verifier's calibration quality"
      ],
      correct: 2,
      explanation: "PRM-guided beam search concentrates compute: at each step, only the top-$B$ partial chains survive. This avoids wasting tokens on chains that went wrong early. But it's a double-edged sword: if the PRM incorrectly scores a promising chain low at step 3, that chain is pruned and never recovers. Best-of-N with an ORM is more robust to verifier errors at intermediate steps (since each chain runs to completion), but wastes compute on chains that went wrong early. The choice depends on PRM calibration quality — well-calibrated PRMs make beam search very efficient."
    },
    {
      type: "info",
      title: "Extended Thinking: RL-Trained Reasoning",
      content: "The most dramatic test-time compute results come from models trained via **reinforcement learning to reason** (o1, o3, R1, QwQ). Unlike prompted chain-of-thought, these models have been optimized to use their reasoning trace as a productive computational medium.\n\nThe training process:\n1. Start from a capable base/SFT model\n2. Give it problems with verifiable answers (math, code, logic)\n3. Let it generate long reasoning traces\n4. Reward correct final answers via RL (e.g., GRPO, PPO)\n5. The model learns to use its token budget effectively: exploring approaches, catching errors, backtracking, and verifying\n\nThe critical difference from prompted CoT: **the model learns *when and how* to reason**, not just to produce reasoning-like text. It learns to:\n- Allocate more tokens to harder sub-problems\n- Backtrack when a line of reasoning fails\n- Verify intermediate results before building on them\n- Try alternative approaches when stuck\n\nThis transforms the reasoning trace from \"plausible-sounding text\" into genuine computation. The model's effective intelligence scales with the token budget it's allowed to use."
    },
    {
      type: "mc",
      question: "Why does RL training (as in o1) produce fundamentally different reasoning behavior than supervised fine-tuning on human-written chain-of-thought solutions?",
      options: [
        "RL training uses a larger model architecture with dedicated reasoning layers that are absent in the SFT model's transformer blocks",
        "SFT on CoT teaches the model to imitate the *surface form* of reasoning, while RL with outcome reward teaches the model to produce reasoning that *actually leads to correct answers* — including strategies humans wouldn't write",
        "RL models generate shorter reasoning traces because the optimization pressure encourages efficiency, producing more compressed but equally effective reasoning",
        "SFT and RL produce identical reasoning behaviors in practice; the performance difference comes entirely from RL models being trained on harder problems"
      ],
      correct: 1,
      explanation: "SFT trains the model to match the distribution of human-written solutions — it learns to produce text that *looks like* reasoning. But humans write solutions for communication, not for computation: they skip \"obvious\" steps, don't show failed attempts, and present reasoning linearly. RL with correctness reward optimizes for a different objective: produce token sequences that lead to correct answers. The model discovers strategies humans wouldn't write: trying multiple approaches in sequence, performing explicit arithmetic verification, and adaptively allocating reasoning depth. The reasoning trace becomes a computational tool, not a communication artifact."
    },
    {
      type: "info",
      title: "Adaptive Compute Allocation",
      content: "Perhaps the most important practical insight from test-time compute research: **not all queries deserve the same inference budget**.\n\nConsider a deployment where 80% of queries are easy (solved correctly with 1 inference step) and 20% are hard (requiring 50 steps). With uniform allocation of 50 steps per query:\n\n$$\\text{Waste} = 0.8 \\times 49 = 39.2 \\text{ wasted steps out of } 50 \\text{ total} = 78\\%$$\n\nAdaptive allocation (1 step for easy, 50 for hard) uses only $0.8 \\times 1 + 0.2 \\times 50 = 10.8$ steps on average — a **4.6× reduction** in compute with no quality loss.\n\nImplementing adaptive allocation requires a **difficulty estimator** — a mechanism to decide how much compute each query deserves. Options include:\n- **Confidence-based routing**: If the model's first-pass confidence is high, return immediately; otherwise, extend reasoning\n- **Learned routers**: A small classifier predicts problem difficulty and routes to appropriate compute tiers\n- **Speculative decoding variant**: Start cheap, escalate if the initial response fails a verifier check\n\nThis connects to a deep principle: the optimal inference strategy is not a property of the model alone, but of the **(model, query) pair**."
    },
    {
      type: "mc",
      question: "A system routes queries to either a fast path (1 forward pass, cost $c$) or a slow path (best-of-32 sampling, cost $32c$). The router misclassifies 10% of hard queries as easy and 5% of easy queries as hard. Compared to always using the slow path, this system:",
      options: [
        "Saves substantial compute overall but sacrifices accuracy on 10% of hard queries that receive insufficient reasoning — the trade-off depends on the relative cost of errors vs compute savings",
        "Wastes more compute than the uniform strategy because the 5% of easy queries routed to the slow path consume more resources than the savings from correctly routing easy queries",
        "Achieves identical accuracy to the uniform strategy because the fast path and slow path produce the same outputs on easy queries regardless of compute allocation",
        "Fails completely because any misrouted hard query produces a confidently wrong answer that is worse than not answering, making the system less reliable overall"
      ],
      correct: 0,
      explanation: "The router creates two types of errors: (1) hard queries on the fast path lose accuracy (10% of hard queries get insufficient compute), and (2) easy queries on the slow path waste compute but don't hurt accuracy (5% of easy queries get unnecessary extra computation). The system saves ~(0.95 × fraction_easy × 31c) compute while losing accuracy on (0.10 × fraction_hard) queries. Whether this trade-off is worthwhile depends on the cost of errors vs. compute savings — a medical diagnosis system might prefer uniform heavy compute, while a coding assistant might accept occasional misses for 3-4x throughput improvement."
    },
    {
      type: "mc",
      question: "Research shows that inference scaling laws for extended thinking (o1-style) exhibit different behavior from best-of-N scaling. Specifically, the relationship between thinking tokens and accuracy is:",
      options: [
        "Strictly linear — each additional thinking token contributes a fixed marginal improvement regardless of problem difficulty, with no diminishing returns observed empirically",
        "Exponentially decreasing — the first few thinking tokens provide nearly all the benefit, and extending beyond ~100 tokens yields negligible gains across all task types",
        "Unpredictable and task-independent — the relationship between token count and accuracy varies randomly, defying any systematic characterization across benchmarks",
        "S-shaped — minimal improvement during initial setup tokens, rapid gains during productive reasoning, then saturation at the model's capability ceiling for that problem"
      ],
      correct: 3,
      explanation: "Extended thinking shows S-shaped (sigmoidal) scaling: the model needs some minimum number of tokens to \"understand\" the problem and set up its approach (low initial returns), then enters a productive phase where each additional step makes real progress (rapid improvement), then saturates as it either solves the problem or exhausts its effective strategies (diminishing returns). This differs from best-of-N's logarithmic scaling because sequential reasoning has dependencies between steps — early tokens enable later tokens to be more productive, creating the initial acceleration phase."
    }
  ]
};
