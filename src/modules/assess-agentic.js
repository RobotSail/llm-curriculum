// Assessment module for D.4: Agentic Systems
// Split from assess-branch-cd.js — per-section test (10 questions)

export const agenticAssessment = {
  id: "D.4-assess",
  sectionId: "D.4",
  title: "Assessment: Agentic Systems",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The ReAct (Reasoning + Acting) framework structures agent behavior as an interleaved sequence of:",
      options: [
        "Training and inference steps, where the agent alternates between updating its weights on new observations and generating predictions based on the updated model parameters",
        "Encoding and decoding steps, where the encoder processes the environment state into a latent representation and the decoder generates the appropriate action sequence",
        "**Thought -> Action -> Observation** cycles — the agent reasons about what to do, executes a tool call, observes the result, and repeats until the task is complete",
        "Forward and backward passes through the model, where the forward pass generates a candidate action and the backward pass updates the policy based on observed reward"
      ],
      correct: 2,
      explanation: "ReAct (Yao et al., 2022) combines chain-of-thought reasoning with tool use in a structured loop. The Thought step allows the agent to plan, interpret observations, and decide what to do next. The Action step interfaces with external tools (search, calculator, API calls). The Observation step grounds the agent in real-world feedback. This structure outperforms both pure reasoning (no grounding) and pure acting (no planning) because reasoning guides action selection while observations correct reasoning errors."
    },
    {
      type: "mc",
      question: "If an agent makes correct decisions with 95% accuracy at each step, its probability of completing a 10-step task correctly (assuming independent steps) is approximately:",
      options: ["~60% — computed as $0.95^{10} \\approx 0.599$; this **compounding error problem** means even highly reliable per-step performance degrades rapidly over multi-step tasks", "95% — the per-step accuracy carries through to the overall task because each step is evaluated independently of previous outcomes", "~90% — nearly as good as per-step accuracy because errors at individual steps are unlikely to cascade into downstream step failures", "50% — it's essentially random over 10 steps because the accumulated uncertainty from each 5% error rate overwhelms the base accuracy"],
      correct: 0,
      explanation: "$0.95^{10} = 0.5987$. This is the compounding error problem: each step's small error probability multiplies. At 20 steps: $0.95^{20} = 0.358$. At 50 steps: $0.95^{50} = 0.077$. This has profound implications for agent design: (1) error recovery mechanisms are essential — agents must detect and recover from mistakes, not just avoid them, (2) reducing task horizon (fewer steps) is as valuable as improving per-step accuracy, (3) verification at intermediate checkpoints can reset the error accumulation by catching and correcting mistakes before they compound."
    },
    {
      type: "mc",
      question: "To achieve 90% task success rate on a 10-step task, the required per-step accuracy is:",
      options: ["90% — the task success rate directly equals the per-step accuracy when errors are independent across steps", "95% — a small margin above the target task success rate is sufficient to absorb the compounding effect over ten steps", "99.9% — each step must be nearly infallible because even a 0.1% error rate compounds to significant task failure over ten sequential steps", "~99% — since $x^{10} = 0.9$ gives $x = 0.9^{1/10} \\approx 0.9895$; this illustrates why agentic systems need near-perfect per-step reliability for multi-step tasks"],
      correct: 3,
      explanation: "Solving $x^{10} = 0.9$: $x = 0.9^{0.1} = e^{0.1 \\ln(0.9)} = e^{-0.01053} \\approx 0.9895$. So you need ~98.95% per-step accuracy for 90% task completion. For a 20-step task: $x = 0.9^{0.05} \\approx 0.9947$ (99.47% per step). This quantifies why long-horizon agentic tasks are so challenging: the reliability requirement per step grows exponentially with task length. It also explains why current agents work best on short, well-defined tasks and struggle with open-ended, multi-step workflows."
    },
    {
      type: "mc",
      question: "Agent memory architectures typically distinguish between short-term and long-term memory. In LLM-based agents, these correspond to:",
      options: ["GPU memory vs CPU memory, with high-priority recent observations stored on the faster GPU and historical data offloaded to the CPU for retrieval when needed", "The **context window** (short-term: recent observations and current plan) and **external storage** (long-term: vector databases of past experiences and learned procedures)", "Training data vs inference data, where the model's parametric memory from training serves as long-term storage and the inference-time context provides short-term task information", "Attention keys vs attention values within the transformer, with keys serving as indexable short-term memory and values storing the long-term content retrieved through attention"],
      correct: 1,
      explanation: "Short-term memory is the context window: it contains the current task description, recent Thought-Action-Observation steps, and immediate working state. It's limited by the model's context length and is lost between sessions. Long-term memory uses external storage: (1) episodic memory — past task executions stored as retrievable summaries, (2) semantic memory — facts and procedures in a vector database, (3) procedural memory — learned tool-use patterns. The retrieval system selectively loads relevant long-term memories into the context window, analogous to human memory retrieval."
    },
    {
      type: "mc",
      question: "Multi-agent debate systems use multiple LLM instances that argue different positions and critique each other's reasoning. The primary benefit over a single model is:",
      options: [
        "Adversarial interaction can surface **errors and unstated assumptions** that a single model would not catch — each agent's critique forces others to justify their reasoning, improving the quality of the final consensus through dialectical refinement",
        "Multi-agent systems are cheaper to run because each agent handles a smaller portion of the task, reducing the per-agent context window size and inference cost",
        "Multiple models always agree on the correct answer because they share the same training data and therefore converge to identical reasoning patterns",
        "Each agent can use a different programming language, enabling the system to leverage the strengths of multiple language ecosystems for code generation"
      ],
      correct: 0,
      explanation: "A single model generating a response has no external check on its reasoning. In multi-agent debate: (1) a critic agent identifies logical gaps, unsupported claims, or errors, (2) the original agent must defend or revise its reasoning, (3) this iterative process often converges to higher-quality outputs. However, limitations exist: LLM agents may be too deferential (agreeing with critiques even when the original reasoning was correct), or they may share systematic biases (multiple instances of the same model make the same mistakes). Diversity of model or prompting strategy helps mitigate this."
    },
    {
      type: "mc",
      question: "Evaluating agentic systems is fundamentally harder than evaluating standard LLM outputs because:",
      options: ["Agent outputs cannot be compared to ground truth because the space of valid solutions is so large that no reference set can cover all acceptable outcomes", "Agents produce longer outputs that are more expensive to evaluate because human reviewers must read through the entire multi-step trace to assess quality", "Agents only work in production environments with real-world tools, making controlled benchmark evaluation infeasible without expensive infrastructure simulation", "Agent evaluation requires assessing **multi-step trajectories** in environments with stochastic outcomes — the same agent may take different valid paths to the same goal, intermediate states are hard to evaluate, success criteria can be ambiguous, and the environment may change between evaluation runs"],
      correct: 3,
      explanation: "Standard LLM evaluation compares a single output to a reference. Agent evaluation must handle: (1) multiple valid solution paths (did the agent take a suboptimal but correct path?), (2) partial credit for partially completed tasks, (3) environment stochasticity (web pages change, APIs have different latencies), (4) difficulty of attributing failure to specific steps vs. the overall strategy, (5) cost of evaluation (each agent run may take minutes and cost dollars in API calls). Benchmarks like SWE-bench, WebArena, and GAIA attempt standardized agent evaluation but each captures only a narrow slice of real-world agent capabilities."
    },
    {
      type: "mc",
      question: "An agent attempts to book a flight: it searches for flights (step 1), selects one (step 2), fills in passenger details (step 3), but enters the wrong date (step 4), and proceeds to payment (step 5). The most robust agent architecture would:",
      options: ["Complete the booking and hope the date is correct, since backtracking would waste the compute already invested in the preceding steps of the workflow", "Start over from step 1 after every step to ensure a clean state, accepting the substantial computational overhead as the necessary cost of reliable execution", "Include a **verification step before irreversible actions** — verify entered information against the original request before payment, catching accumulated errors at checkpoints", "Use a separate specialized model for each step, since task-specific models fine-tuned for booking make fewer errors than a single general-purpose agent"],
      correct: 2,
      explanation: "Robust agent design distinguishes between reversible and irreversible actions. Browsing and form-filling are reversible (can go back and correct). Payment is irreversible. The agent should: (1) maintain an explicit plan with checkpoints, (2) verify accumulated state against the original goal before irreversible actions, (3) use self-reflection (\"Does this match what the user asked for?\") at critical junctions. This verification overhead reduces throughput but dramatically reduces costly errors. The principle mirrors software engineering: validate inputs before committing transactions."
    },
    {
      type: "mc",
      question: "The \"inner monologue\" approach to agent reasoning, where the agent maintains an explicit running commentary of its state and plans, helps primarily by:",
      options: ["Making the agent's responses more verbose for users, providing transparency into the decision-making process and building trust through detailed explanations", "Keeping the agent's **goals, current state, and plan in the active context window** — without explicit tracking, earlier context scrolls out of the attention window over long trajectories", "Improving the agent's language fluency by generating practice text that refines the model's output distribution toward more coherent and well-structured responses", "Reducing the number of API calls by allowing the agent to reason through multiple steps internally before committing to an external action or tool invocation"],
      correct: 1,
      explanation: "Over a 50-step trajectory, the early steps (including the original task description) may be hundreds of tokens back in the context, receiving diminished attention. Inner monologue explicitly restates: \"My goal is X. I have completed A, B, C. Current state is Y. Next I need to do Z.\" This keeps critical information in the recent context window where the model attends most strongly. It also serves as a form of self-verification — stating the current plan explicitly can reveal inconsistencies. The cost is additional token generation, but this is usually worthwhile for complex tasks."
    },
    {
      type: "mc",
      question: "When deploying an LLM agent with access to real-world tools (email, file system, web browsing), the primary safety concern is:",
      options: [
        "**Unintended or adversarial actions with real-world consequences** — the agent might delete files, send unintended emails, or be manipulated via indirect prompt injection",
        "The agent might generate offensive text in its reasoning traces, which could be exposed to users through logging or debugging interfaces during development",
        "The agent will use too much compute by entering infinite reasoning loops, consuming expensive GPU resources without producing useful output for the user",
        "The agent's actions are too slow for real-time applications, since each tool call adds network latency that compounds across the multi-step execution"
      ],
      correct: 0,
      explanation: "Unlike chat-only LLMs where the worst case is bad text, tool-using agents can cause real harm: deleting data, sending emails as the user, making purchases, or modifying code in production. Key risks: (1) misinterpreted instructions amplified by tool capabilities, (2) indirect prompt injection — malicious content in web pages or documents instructing the agent to take harmful actions, (3) compounding errors leading to unintended state changes. Mitigations include: sandboxing, requiring human approval for irreversible actions, input/output filtering, and principle of least privilege for tool access."
    },
    {
      type: "mc",
      question: "An agent system decomposes a complex task into subtasks handled by specialized sub-agents (e.g., a \"researcher\" agent, a \"coder\" agent, a \"reviewer\" agent). The orchestration challenge is:",
      options: ["Sub-agents are more expensive than a single agent because each sub-agent requires its own full model instance, multiplying the total inference cost by the number of agents", "Ensuring correct **information flow and task dependency management** — tracking sub-task completion, passing the right context to each sub-agent, and merging conflicting outputs", "Sub-agents can't communicate with each other directly and must route all information through the user, creating a bottleneck that limits collaborative problem-solving", "Each sub-agent needs a separate GPU to run concurrently, making multi-agent systems impractical without access to large-scale GPU clusters for parallel execution"],
      correct: 1,
      explanation: "Multi-agent orchestration is essentially a workflow management problem with LLM-specific challenges: (1) context management — each sub-agent needs enough context to do its job but not so much that it's distracted, (2) dependency tracking — the coder agent can't start until the researcher agent provides specifications, (3) failure handling — if the researcher returns low-quality results, the orchestrator must detect this and retry or adjust, (4) output merging — when the reviewer critiques the coder's output, the orchestrator must route the feedback and manage the revision loop. Frameworks like LangGraph and AutoGen provide abstractions for this but the fundamental complexity remains."
    }
  ]
};
