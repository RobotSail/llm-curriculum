// Assessment module for D.3: Tool Use & Function Calling
// Split from assess-branch-cd.js — per-section test (10 questions)

export const toolUseAssessment = {
  id: "D.3-assess",
  sectionId: "D.3",
  title: "Assessment: Tool Use & Function Calling",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Function calling in LLMs is typically trained by:",
      options: ["Fine-tuning on datasets of (user query, function call, function result, final response) sequences — the model learns to emit **structured function call tokens** that the serving system intercepts", "Giving the model direct access to execute code during training so it learns the input-output behavior of each function through repeated trial and error interactions", "Hard-coding specific API calls into the model's architecture through specialized attention heads that route queries to external function endpoints during inference", "Training a separate classifier that maps user queries to the appropriate function based on intent detection, then invoking the function through an independent pipeline"],
      correct: 0,
      explanation: "Function calling training involves: (1) curating datasets where the correct response involves calling specific functions with appropriate arguments, (2) defining a structured output format (often JSON) that the model generates inline, (3) teaching the model when to call functions vs. answer directly. The serving system parses these structured outputs, executes the function, and appends the result to the context for the model to incorporate. This is how models like GPT-4 and Claude handle tool use — the function call is part of the model's text generation, not a separate system."
    },
    {
      type: "mc",
      question: "In a RAG (Retrieval-Augmented Generation) system, the chunking strategy — how documents are split into retrievable units — critically affects quality. Which statement is correct?",
      options: ["Larger chunks are always better because they provide more surrounding context for the LLM to synthesize its answer from each retrieved passage without fragmentation", "All documents should be stored as single chunks to preserve the full semantic coherence of each source and avoid fragmenting arguments across retrieval boundaries", "Chunk size involves a **precision-recall trade-off**: smaller chunks yield precise retrieval but miss context, while larger ones capture context but dilute relevance with irrelevant text", "Chunking doesn't matter if the embedding model is good enough, since a sufficiently powerful embedder can represent any chunk size with equal fidelity and precision"],
      correct: 2,
      explanation: "Too-small chunks: high retrieval precision (the chunk matches the query well) but insufficient context for the LLM to synthesize an answer. Too-large chunks: the relevant passage is buried in irrelevant text, diluting the embedding representation and the LLM's ability to extract the answer. Strategies to mitigate: (1) overlapping windows (e.g., 512 tokens with 128 overlap), (2) hierarchical chunking (retrieve small chunks but expand to parent chunks for context), (3) semantic chunking (split at topic boundaries rather than fixed token counts). The optimal strategy is domain-dependent."
    },
    {
      type: "mc",
      question: "A RAG pipeline uses embedding similarity for initial retrieval (top-100), then a cross-encoder reranker to select the final top-5 passages. The reranker improves results because:",
      options: [
        "It uses a larger embedding dimension that captures more nuanced semantic distinctions than the compact vectors used by the initial bi-encoder retriever, enabling finer-grained similarity scoring at the cost of higher memory usage per document",
        "Cross-encoders process the **query and passage jointly** through full attention, capturing fine-grained semantic interactions — unlike bi-encoders that independently embed query and passage into separate vectors and compare with cosine similarity, missing token-level cross-attention between query and document",
        "It has access to more training data because reranker models are trained on the union of all retrieval datasets rather than a single domain-specific corpus, giving them broader coverage of query-document relevance patterns across domains",
        "It runs faster than the initial retriever because it only processes the top-100 candidates rather than searching over the entire document collection, and this speed advantage allows deeper per-candidate analysis within the same latency budget"
      ],
      correct: 1,
      explanation: "Bi-encoders (used for initial retrieval) embed query $q$ and passage $p$ independently: $\\text{sim}(e(q), e(p))$. This enables fast ANN search over millions of passages but misses cross-attention between query and passage tokens. Cross-encoder rerankers process $[q; p]$ jointly through a transformer, enabling rich token-level interactions: the model can attend from query tokens to passage tokens and vice versa. This is dramatically more expressive but expensive ($O(N)$ forward passes for $N$ candidates), hence the two-stage retrieve-then-rerank pipeline."
    },
    {
      type: "mc",
      question: "Multi-step retrieval (also called iterative or multi-hop RAG) is needed when:",
      options: ["The document collection is very large, exceeding the capacity of the embedding index to return relevant results in a single retrieval step over the full corpus", "The answer requires **synthesizing information from multiple documents** via a chain of lookups — each retrieval step uses prior results to formulate the next query", "The embedding model has limited context length and cannot encode queries that are longer than a few hundred tokens into a single vector representation effectively", "The user asks multiple questions in one turn, requiring the system to retrieve separate sets of documents for each independent sub-question in the input"],
      correct: 1,
      explanation: "Single-shot retrieval fails on multi-hop questions because the initial query doesn't contain the intermediate information needed to formulate the final retrieval. Multi-step RAG decomposes the question: (1) retrieve \"transformer architecture inventor\" -> Vaswani et al., (2) retrieve birthplace information -> identify country, (3) retrieve GDP data. Each retrieval step uses information from prior steps to formulate the next query. Architectures like IRCoT (Interleaving Retrieval with CoT) and Self-RAG automate this decomposition."
    },
    {
      type: "mc",
      question: "When an LLM generates a function call that fails (e.g., API returns an error), the ideal error recovery behavior is:",
      options: ["Analyzing the error, **reformulating the function call** (e.g., correcting parameter types, trying alternative APIs, or decomposing into simpler sub-calls), and retrying — this error recovery loop should be bounded to prevent infinite retries, with graceful degradation when recovery fails", "Ignoring the error and generating an answer without the tool result, relying on the model's parametric knowledge to fill in the missing information", "Immediately returning the raw error message to the user without interpretation, since the user can diagnose the tool failure more accurately than the model", "Calling the same function again with identical parameters, since transient errors like rate limits or timeouts will resolve on subsequent attempts"],
      correct: 0,
      explanation: "Robust tool use requires error handling as a core capability, not an afterthought. The model should: (1) parse the error message to diagnose the failure (auth error? malformed input? rate limit?), (2) determine if the call can be retried with corrections or if an alternative approach is needed, (3) attempt recovery with a bounded retry count, (4) gracefully inform the user if recovery fails, explaining what was attempted. Training for error recovery involves including error scenarios in the fine-tuning data — models that only see successful tool calls perform poorly when tools fail."
    },
    {
      type: "mc",
      question: "Code generation can be viewed as a form of tool use where the \"tool\" is a code interpreter. The key advantage of code execution over pure text reasoning for tasks like data analysis is:",
      options: ["Code is always shorter than natural language for expressing the same computation, making it more token-efficient for the model to generate and process during inference", "Code generation doesn't require pre-training on code-specific data, since the model can derive programming syntax from its understanding of natural language grammar", "Code runs faster than the LLM can reason through the same computation in natural language, so execution provides a pure speed advantage over text-based reasoning", "Code execution provides **exact computation and verification** — the interpreter executes on real data and returns exact results, eliminating the hallucinated computations that plague text reasoning"],
      correct: 3,
      explanation: "When asked \"What is the mean of these 1000 data points?\", an LLM reasoning in natural language will hallucinate a plausible-sounding number. Code generation + execution computes the exact answer. This is the core insight behind tools like Code Interpreter and open-source equivalents: the LLM's strength is understanding the user's intent and translating it to code; the interpreter's strength is exact, verifiable computation. The model can also inspect intermediate results, detect errors, and iterate — forming a generate-execute-debug loop."
    },
    {
      type: "mc",
      question: "Embedding models used for RAG retrieval are typically trained with **contrastive learning**. The training objective is:",
      options: [
        "Predicting the next word in a document, using the autoregressive language modeling objective to learn contextual representations that capture document-level semantics",
        "Classifying documents into predefined categories using cross-entropy loss, then extracting the penultimate layer's activations as general-purpose retrieval embeddings",
        "Minimizing the reconstruction error of documents by training an autoencoder that compresses and reconstructs each passage through a fixed-dimension bottleneck vector",
        "Learning representations where **semantically similar query-passage pairs have high cosine similarity** while dissimilar pairs are pushed apart using hard negatives"
      ],
      correct: 3,
      explanation: "Contrastive training (e.g., DPR, E5, GTE) uses pairs $(q_i, p_i^+)$ of queries and relevant passages. The loss pulls together $e(q_i)$ and $e(p_i^+)$ while pushing apart $e(q_i)$ and $e(p_j^+)$ for $j \\neq i$ (in-batch negatives). Hard negatives — passages that are superficially similar to the query but not actually relevant (e.g., same topic but different answer) — are critical for learning fine-grained distinctions. Modern embedding models (GTE, E5-Mistral) build on decoder LLMs fine-tuned with contrastive objectives, achieving strong retrieval by leveraging the LLM's pre-trained language understanding."
    },
    {
      type: "mc",
      question: "A naive RAG system retrieves the top-5 most similar passages and concatenates them into the LLM's context. A more sophisticated approach uses the **lost-in-the-middle** finding to:",
      options: ["Place the most relevant passages at the **beginning and end** of the context rather than ranked order — LLMs attend more strongly to boundaries, so middle positions are underutilized", "Only retrieve 1 passage to avoid confusion — the lost-in-the-middle effect shows that multiple passages interfere with each other and degrade answer quality", "Randomize the passage order on each query — stochastic ordering prevents the model from developing positional biases that favor early or late passages", "Retrieve from the middle of each document only — the lost-in-the-middle finding shows that document boundaries contain less informative content than central sections"],
      correct: 0,
      explanation: "Liu et al. (2023) showed that LLMs' ability to use information from retrieved passages follows a U-shaped curve: performance is highest when the relevant passage is at the beginning or end of the context, and lowest when it's in the middle. This means naive \"rank by relevance\" ordering (most relevant first, then decreasing) places important passages in the problematic middle positions. Solutions include: reordering passages to place the most relevant at the boundaries, using reciprocal rank fusion, or citing specific passages by reference to force the model's attention."
    },
    {
      type: "mc",
      question: "When deciding whether to call a tool or answer directly, the model must assess its own **epistemic uncertainty**. Which approach to this calibration challenge is most effective?",
      options: ["Always calling tools to be safe — routing every query through tool execution ensures no question relies on potentially outdated or incorrect parametric knowledge", "Using a hardcoded keyword list to trigger tool calls — mapping specific terms like dates, numbers, or entity names to automatic tool invocations without model judgment", "Training on data that includes both tool-calling and direct-answering examples, with the decision based on **whether the model's knowledge is sufficient and current for the query**", "Letting the user decide when to use tools — exposing the tool selection as an explicit option so the human can route queries based on their own assessment of need"],
      correct: 2,
      explanation: "Effective tool-use routing requires the model to know what it doesn't know. Training strategies include: (1) self-knowledge probing — including examples where the model should say \"I need to look this up\" for recent events or precise numbers, (2) confidence calibration — training the model to express appropriate uncertainty, (3) curriculum with both tool-assisted and direct answers for similar queries, so the model learns the decision boundary. Models like Toolformer automate this by learning to insert API calls only when they improve prediction quality, using a self-supervised filtering criterion."
    },
    {
      type: "mc",
      question: "A RAG system's retrieval component returns passages with relevance scores. At what point does adding more retrieved passages to the LLM context typically **hurt** performance?",
      options: ["Never — more context is always better because the model can simply ignore irrelevant passages and focus on the useful information within the retrieved set", "When additional passages have **low relevance scores** — they introduce noise that can distract the LLM and cause hallucinated synthesis of contradictory information sources", "After exactly 3 passages in all cases, since empirical studies have converged on this as the universal optimal number for retrieval-augmented generation tasks", "Only when the context window is physically exceeded, since the model benefits from all available information up to the maximum sequence length of the architecture"],
      correct: 1,
      explanation: "Adding passages with relevance scores below a task-specific threshold introduces noise. The LLM may: (1) hallucinate a synthesis of contradictory information from relevant and irrelevant passages, (2) get distracted by plausible-sounding but incorrect passages (especially problematic for factual QA), (3) suffer from the lost-in-the-middle effect as the context grows. Empirically, accuracy often peaks at 3-5 highly relevant passages and degrades with more. Adaptive retrieval strategies set a relevance threshold or use the LLM's own confidence to decide when enough context has been retrieved."
    }
  ]
};
