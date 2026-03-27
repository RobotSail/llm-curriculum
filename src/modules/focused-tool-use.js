// Focused module: Tool Use and Function Calling in LLMs
// Section D.3: Tool Use & Function Calling
// ONE concept: How LLMs learn to invoke external tools through structured
// function calls, and the reasoning-acting loop that orchestrates tool use at inference time.

export const toolUseLearning = {
  id: "D.3-tool-use-learning",
  sectionId: "D.3",
  title: "Tool Use and Function Calling",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "Why LLMs Need Tools",
      content: "LLMs store vast knowledge in their parameters, but this **parametric knowledge** has fundamental limits:\n\n1. **Staleness**: Parameters are frozen after training. A model trained in January cannot know February's stock prices.\n2. **Imprecise computation**: LLMs approximate arithmetic. Ask for $3{,}847 \\times 2{,}914$ and you get a plausible-sounding but often wrong answer.\n3. **No grounding**: The model cannot check a database, read a live webpage, or verify facts against an authoritative source.\n\nTool use solves these problems by letting the model **delegate** specific subtasks to external systems. Instead of hallucinating the weather, the model calls a weather API. Instead of guessing at multiplication, it calls a calculator.\n\nThe key insight: the LLM's role shifts from **answering directly** to **orchestrating** — it understands the user's intent, decides which tool to invoke, generates the correct arguments, interprets the result, and composes a final response. The model contributes language understanding and reasoning; the tools contribute precise execution."
    },
    {
      type: "mc",
      question: "A user asks a model \"What was NVIDIA's closing stock price yesterday?\" The model has no tool access and was trained 6 months ago. What failure mode is most likely?",
      options: [
        "The model refuses to answer entirely, correctly identifying the question as outside its knowledge boundary and training data cutoff",
        "The model correctly answers by extrapolating from price trends it learned during training using its internal time-series prediction capabilities",
        "The model outputs a random number because transformer attention cannot represent numerical financial data in its embedding space",
        "The model generates a confident, specific price that is fabricated — parametric knowledge cannot contain post-training real-time information"
      ],
      correct: 3,
      explanation: "This is the classic hallucination failure mode for factual queries about post-training events. The model has no mechanism to know yesterday's price — it was not in the training data. But models are trained to be helpful and generate confident-sounding responses, so they typically produce a specific, plausible number (perhaps near historical prices they saw in training) rather than admitting ignorance. Tool use solves this: the model should recognize it needs a stock price API and delegate the lookup."
    },
    {
      type: "info",
      title: "Function Calling: The Interface",
      content: "Modern LLMs implement tool use through **function calling** — the model generates structured output (typically JSON) that specifies which function to invoke and with what arguments. The serving system intercepts this structured output, executes the function, and returns the result to the model.\n\nA typical function call looks like:\n\n$$\\text{User query} \\to \\text{Model generates} \\to \\begin{cases} \\texttt{tool: \"get\\_weather\"} \\\\ \\texttt{args: \\{\"city\": \"London\"\\}} \\end{cases}$$\n\nThe model is provided with **tool definitions** — JSON schemas describing each available tool's name, purpose, and parameter types. For example:\n\n- **Name**: get_weather\n- **Description**: Returns current weather for a city\n- **Parameters**: city (string, required), units (string, optional)\n\nDuring generation, the model decides whether to emit a regular text response or a structured function call. This decision is learned during training — the model is fine-tuned on datasets of (query, function call, function result, final response) sequences. The model learns three capabilities simultaneously: **when** to call a tool (intent recognition), **which** tool to call (selection), and **what arguments** to pass (parameter generation)."
    },
    {
      type: "mc",
      question: "A model is provided with 5 tool definitions and a user query: \"Translate 'hello' to French and check if it's raining in Paris.\" What should the model do?",
      options: [
        "Call both the translation tool and weather tool in sequence, using each result to construct a unified response addressing both parts of the query",
        "Answer the translation from parametric knowledge (\"bonjour\") and only call the weather tool, since translation of common words doesn't require a tool",
        "Refuse to process the request because it contains two distinct intents, which violates the single-function-call-per-turn constraint",
        "Call only the weather tool because it requires real-time data, and append a disclaimer that translation accuracy cannot be verified"
      ],
      correct: 1,
      explanation: "Good tool use involves knowing when NOT to use tools. The model almost certainly knows that \"hello\" in French is \"bonjour\" — calling a translation API adds latency without improving accuracy. But the weather in Paris is real-time information the model cannot know from its parameters. The optimal strategy is to answer the easy part directly and delegate only the part that requires external information. This selective routing between parametric knowledge and tools is a key skill learned during function-calling fine-tuning."
    },
    {
      type: "info",
      title: "Training for Tool Use",
      content: "How do models learn to use tools? There are two main approaches:\n\n**Supervised fine-tuning on tool-use datasets**: Curate datasets of conversations where the correct response involves function calls. Each example contains: the user query, available tool definitions, the correct function call (with arguments), the function's return value, and the model's final response incorporating the result. The model is trained with standard next-token prediction on these sequences, learning to emit structured function calls as part of its generation.\n\n**Self-supervised tool learning (Toolformer)**: Schick et al. (2023) showed that models can teach themselves tool use. The approach: (1) sample candidate positions in text where a tool call might help, (2) generate candidate API calls at those positions, (3) execute each call and compare perplexity with vs. without the tool result — keep only calls that **reduce perplexity** (i.e., the tool result genuinely helps predict subsequent tokens), (4) fine-tune on the filtered dataset.\n\nThe Toolformer insight is powerful: the model discovers when tools are useful by measuring whether tool results improve its own predictions. No human annotation of \"this is where you should call a tool\" is needed — the filtering criterion is purely self-supervised.\n\nIn practice, most production systems use supervised fine-tuning on curated datasets, often built on top of instruction-tuned models. LoRA is typically sufficient — tool use is a **format skill** (learning when and how to emit structured calls) rather than a knowledge-intensive capability."
    },
    {
      type: "mc",
      question: "Toolformer's self-supervised filtering criterion keeps a candidate tool call at position $i$ only if inserting the API call and its result reduces perplexity on subsequent tokens. Why is this filtering step critical?",
      options: [
        "Without filtering, the model would insert tool calls at every position in the text, making generation impossibly slow due to excessive API invocations",
        "The filter prevents the model from learning tool calls that are syntactically malformed, since only parseable API calls can return valid results that affect perplexity",
        "It ensures the model only learns to call tools when the result provides genuinely useful information — calls that don't improve prediction are noise that would degrade performance",
        "The filter is needed because the language modeling loss cannot backpropagate through external API calls, so filtering substitutes for gradient-based optimization"
      ],
      correct: 2,
      explanation: "The perplexity filter is what makes Toolformer's self-supervision work. Without it, the training data would contain many useless tool calls — e.g., calling a calculator for \"2 + 2\" when the model already knows the answer, or calling a search engine when the next tokens don't depend on the result. These noisy examples would teach the model to over-rely on tools. By keeping only calls where the result measurably improves prediction, the model learns the decision boundary between \"I can handle this\" and \"I need external help.\""
    },
    {
      type: "info",
      title: "The ReAct Loop: Reasoning and Acting",
      content: "Single-turn function calling handles simple queries, but complex tasks require **multi-step reasoning with tool use**. The **ReAct** framework (Yao et al., 2023) formalizes this as an interleaved loop of reasoning and acting:\n\n**Thought**: The model reasons about what it knows, what it needs, and what tool would help. This reasoning trace is generated as text (visible and interpretable).\n\n**Action**: Based on the reasoning, the model emits a structured tool call.\n\n**Observation**: The tool's result is appended to the context.\n\nThe loop repeats until the model has enough information to answer.\n\n**Example** — \"Who is older, the CEO of Apple or the CEO of Microsoft?\"\n\n1. **Thought**: I need to find the CEOs and their birth dates. Let me start with Apple's CEO.\n2. **Action**: search(\"current CEO of Apple\")\n3. **Observation**: Tim Cook, born November 1, 1960\n4. **Thought**: Now I need Microsoft's CEO.\n5. **Action**: search(\"current CEO of Microsoft\")\n6. **Observation**: Satya Nadella, born August 19, 1967\n7. **Thought**: Tim Cook (1960) is older than Satya Nadella (1967).\n8. **Final answer**: Tim Cook is older.\n\nThe key advantage over pure reasoning: each intermediate fact is **grounded** in a tool result, not hallucinated. And the key advantage over pure acting (calling tools without reasoning): the thought traces allow the model to plan, decompose problems, and course-correct."
    },
    {
      type: "mc",
      question: "A ReAct agent is answering: \"What is the population of the country where the Eiffel Tower is located?\" After Step 1 (search for Eiffel Tower location → France), it encounters an API error on Step 2 (search for France population). What should the agent do?",
      options: [
        "Generate a reasoning trace analyzing the error, reformulate the query or try an alternative tool, and retry with a bounded attempt limit before gracefully degrading",
        "Return the partial answer \"The Eiffel Tower is in France\" and inform the user that population data is unavailable due to a system error",
        "Hallucinate a plausible population figure for France based on parametric knowledge, since the user expects a complete numerical answer",
        "Restart the entire reasoning chain from scratch, since the error may have corrupted the agent's internal state and prior observations"
      ],
      correct: 0,
      explanation: "Robust tool use requires error recovery as a core capability. The agent should: (1) reason about the error (API timeout? malformed query? rate limit?), (2) attempt recovery (reformulate query, try alternative search terms, use a different data source), (3) retry with a bounded limit (e.g., 3 attempts) to prevent infinite loops. Only after recovery fails should it gracefully degrade — perhaps using parametric knowledge with an explicit caveat about uncertainty, or informing the user. Restarting from scratch wastes the valid information already obtained."
    },
    {
      type: "info",
      title: "When to Use Tools vs. Parametric Knowledge",
      content: "A well-trained tool-use model must solve the **routing problem**: for each query, decide whether to answer from parametric knowledge or delegate to a tool. This requires a form of **epistemic self-awareness** — knowing what you know and what you don't.\n\nThe routing decision depends on several factors:\n\n**Use parametric knowledge when:**\n- The answer is well-established and unlikely to have changed (\"What is the capital of France?\")\n- The task is reasoning-heavy with no factual uncertainty (\"Explain why water boils at lower temperatures at high altitude\")\n- Tool invocation would add latency without improving accuracy\n\n**Use tools when:**\n- The query requires real-time or post-training information (\"What's the weather today?\")\n- Precise computation is needed (\"Calculate $\\int_0^1 e^{-x^2} dx$ to 6 decimal places\")\n- The answer requires grounding in an authoritative source (\"What does section 3.2 of the contract say?\")\n- The model's confidence is low and verification would help\n\nTwo failure modes exist: **overthinking** (using parametric reasoning when a tool call would be faster and more reliable) and **overacting** (calling tools for simple questions the model already knows). Training on datasets with both tool-assisted and direct-answer examples for similar queries teaches the model this decision boundary."
    },
    {
      type: "mc",
      question: "A tool-equipped model receives: \"What is 15% of 847?\" It has access to a calculator tool. Should it use the tool?",
      options: [
        "Yes — arithmetic is a fundamental weakness of autoregressive models, and even simple percentage calculations benefit from exact tool-based computation",
        "No — the model should always attempt parametric reasoning first and only fall back to tools after detecting an error in its own output",
        "Yes — but only because the numbers are large; for single-digit arithmetic like 15% of 10, tools are unnecessary overhead",
        "No — percentage calculations are a special case where LLMs are reliable because the operation maps directly to token-level pattern matching"
      ],
      correct: 0,
      explanation: "Arithmetic is a well-known failure mode of LLMs. The model might compute 15% of 847 as 127.05 (correct) or 126.70 or 128.05 (common errors from imprecise internal computation). A calculator tool returns the exact answer instantly. The general principle: for tasks where tools provide exact results and the model's parametric computation is unreliable, always prefer the tool regardless of apparent simplicity. The latency cost of a calculator call is negligible compared to the cost of returning wrong arithmetic."
    },
    {
      type: "info",
      title: "Code Execution as Tool Use",
      content: "A particularly powerful form of tool use is **code execution** — the model generates code (typically Python), a sandboxed interpreter runs it, and the result is returned to the model. This generalizes the calculator concept to arbitrary computation.\n\nCode execution is valuable because:\n\n1. **Exact computation**: Not just arithmetic but statistical analysis, data transformation, symbolic math — anything expressible as code\n2. **Verifiability**: The code is inspectable. Unlike a text-based reasoning chain, you can check whether the computation is correct by reading the code.\n3. **Iteration**: The model can inspect intermediate results, detect errors, and fix its code — forming a **generate-execute-debug loop**\n\nThe pattern works like this:\n- User: \"What is the standard deviation of [3, 7, 2, 9, 4, 6, 8, 1, 5]?\"\n- Model generates: `import numpy as np; print(np.std([3, 7, 2, 9, 4, 6, 8, 1, 5]))`\n- Interpreter returns: `2.581988897471611`\n- Model: \"The standard deviation is approximately 2.58.\"\n\nSecurity is critical: code execution must happen in a **sandboxed environment** with restricted filesystem access, network isolation, memory limits, and execution timeouts. The model should never be able to execute code that affects the host system."
    },
    {
      type: "mc",
      question: "A user asks: \"Analyze this CSV of 10,000 sales records and find the top 5 products by revenue.\" The model has access to both a code interpreter and a search tool. What is the optimal approach?",
      options: [
        "Use the search tool to find a pre-existing analysis of common sales datasets and adapt the findings to match the user's data format",
        "Refuse the request because analyzing 10,000 records exceeds the model's reliable computation capacity for any available tool pathway",
        "Reason through the CSV data row-by-row in the model's context window, mentally computing running revenue totals for each product category",
        "Generate Python code that loads the CSV, computes per-product revenue, sorts, and returns the top 5 — then execute it in the sandboxed interpreter"
      ],
      correct: 3,
      explanation: "This is a textbook case for code execution as tool use. The task requires precise computation over a large dataset — exactly what code interpreters excel at. The model generates a few lines of pandas code (read CSV, groupby product, sum revenue, sort, head), the interpreter executes it on the actual data, and the model reports the exact results. Attempting to reason through 10,000 rows in context would be impossibly slow and error-prone. The search tool is irrelevant — the user has specific data to analyze, not a general knowledge question."
    },
    {
      type: "info",
      title: "Multi-Step Tool Use and Failure Modes",
      content: "Real-world tool use often involves **chains of tool calls** where each step depends on previous results. This introduces failure modes beyond single-call errors:\n\n**Cascading failures**: An error in step 1 propagates through all subsequent steps. If the model retrieves the wrong person in a biographical lookup, every subsequent question about \"that person\" will produce wrong answers built on the initial error.\n\n**Context exhaustion**: Each tool call and its result consume context tokens. Long chains can exhaust the context window, causing the model to \"forget\" earlier observations. Mitigation: summarize completed sub-tasks to compress the history.\n\n**Infinite loops**: The model retries the same failing tool call repeatedly. Mitigation: implement explicit retry limits (typically 2-3 attempts) and require the model to try a different approach after repeated failures.\n\n**State inconsistency**: The model assumes something about the world that was true 3 steps ago but has since changed (e.g., an API response that returns different results each call). Mitigation: re-verify critical assumptions before acting on them.\n\nRobust tool-use systems combine the model's reasoning with explicit guardrails: retry budgets, output validation, state verification checkpoints, and graceful degradation paths when tools fail."
    },
    {
      type: "mc",
      question: "A ReAct agent is executing a 5-step plan. In step 3, a tool returns an unexpected result that contradicts the assumption made in step 1. What should the agent do?",
      options: [
        "Ignore the contradiction and continue with the original plan, since earlier steps were already validated when they were executed",
        "Immediately restart the entire 5-step plan from step 1, discarding all intermediate results to ensure a clean state",
        "Generate a reasoning trace acknowledging the contradiction, re-evaluate the step-1 assumption using the new evidence, and adjust the remaining plan accordingly",
        "Report the contradiction to the user as a system error, since well-designed tool chains should never produce contradictory intermediate observations"
      ],
      correct: 2,
      explanation: "This is where the \"Thought\" component of ReAct is essential. The model should: (1) explicitly reason about the contradiction (\"Step 3's result conflicts with my step 1 assumption because...\"), (2) determine which information is more reliable (is the step 3 result more current? was the step 1 assumption based on weak evidence?), (3) revise the plan. This adaptive replanning is what distinguishes robust agents from brittle scripts. Ignoring contradictions leads to wrong answers; full restarts waste valid work; treating contradictions as system errors misattributes a normal property of dynamic information gathering."
    }
  ]
};
