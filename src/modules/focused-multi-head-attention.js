// Focused learning module: Multi-Head Attention
// Section 1.1: Transformer Architecture
// Covers: why multiple heads, head dimensionality, concatenation + output projection,
// attention head specialization, and practical variants (MQA, GQA).
// Prerequisite: self-attention mechanism.

export const multiHeadAttentionLearning = {
  id: "1.1-multi-head-attention-learning-easy",
  sectionId: "1.1",
  title: "Multi-Head Attention",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Why multiple heads?
    {
      type: "info",
      title: "One Head Is Not Enough",
      content: "A single attention head computes one set of attention weights per token — one pattern of \"who attends to whom.\" But language requires attending to multiple things simultaneously.\n\nConsider the word \"bank\" in: \"The bank by the river raised interest rates.\" To understand this sentence, the model needs to simultaneously:\n- Attend to \"river\" for disambiguation (which kind of bank?)\n- Attend to \"raised\" for predicate-argument structure (the bank did what?)\n- Attend to \"rates\" for object completion (raised what?)\n\nA single attention head produces a single weighted average of values, which creates a compromise: it cannot fully attend to \"river\" and \"raised\" at the same time.\n\n**Multi-head attention** solves this by running $h$ independent attention heads in parallel, each with its own $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ projections. Each head can learn a different attention pattern — one for syntactic dependencies, another for semantic similarity, another for positional proximity, and so on."
    },
    // Step 2: MC
    {
      type: "mc",
      question: "A single attention head produces output $o_i = \\sum_j \\alpha_{ij} v_j$ — a weighted average of value vectors. Why does this weighted average limit the expressivity of a single head?",
      options: [
        "A weighted average is a convex combination, meaning the output must lie within the convex hull of the value vectors — the head cannot produce outputs outside the span of existing values",
        "A weighted average forces all attention weights to be positive after softmax, so the head cannot implement inhibitory (negative) attention patterns between tokens",
        "A single set of attention weights must trade off between different types of relevance — attending more to syntactically relevant tokens necessarily means attending less to semantically relevant ones",
        "The weighted average loses ordering information because addition is commutative — the head cannot distinguish whether the attended tokens appeared before or after the current position"
      ],
      correct: 2,
      explanation: "The fundamental limitation is that one set of weights $\\alpha_{ij}$ must serve all purposes simultaneously. If a token needs syntactic information from position 3 and semantic information from position 7, a single head must split its attention weight budget between them, diluting both signals. Multiple heads avoid this by letting each head specialize in one type of attention pattern. Options A and B describe real properties of softmax attention but are not the primary motivation for multiple heads."
    },
    // Step 3: Multi-head mechanics
    {
      type: "info",
      title: "Multi-Head Attention Mechanics",
      content: "Multi-head attention runs $h$ independent attention heads and combines their outputs:\n\n$$\\text{MultiHead}(X) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h) W_O$$\n\nwhere each head is:\n$$\\text{head}_i = \\text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})$$\n\n**Key design choice**: the per-head dimension is $d_k = d_v = d_{\\text{model}} / h$. With $d_{\\text{model}} = 768$ and $h = 12$ heads (GPT-2), each head operates in a 64-dimensional subspace.\n\nThe total parameter count stays the same as a single large head:\n- Single head: $W_Q, W_K, W_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}$, plus $W_O$ — total $\\approx 4 d_{\\text{model}}^2$\n- $h$ heads: each has $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}$ — total $h \\times 3 \\times d_{\\text{model}} \\times d_k = 3 d_{\\text{model}}^2$, plus $W_O \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}$ — total $\\approx 4 d_{\\text{model}}^2$.\n\nSame parameter budget, but much more expressive: $h$ independent attention patterns instead of one."
    },
    // Step 4: MC — parameter calculation
    {
      type: "mc",
      question: "A model has $d_{\\text{model}} = 1024$ and $h = 16$ heads. What is the per-head dimension $d_k$, and how many total parameters are in the multi-head attention layer (Q, K, V, and O projections)?",
      options: [
        "$d_k = 64$; total parameters $= 4 \\times 1024^2 = 4{,}194{,}304$ — the same as a single-head attention with $d_k = 1024$",
        "$d_k = 16$; total parameters $= 16 \\times 3 \\times 1024 \\times 16 = 786{,}432$ — much fewer than single-head because each head is small",
        "$d_k = 64$; total parameters $= 16 \\times 4 \\times 1024 \\times 64 = 4{,}194{,}304$ — each head has its own independent parameters with no sharing",
        "$d_k = 1024$; total parameters $= 16 \\times 4 \\times 1024^2 = 67{,}108{,}864$ — each of the 16 heads has full-dimension Q, K, V, O matrices"
      ],
      correct: 0,
      explanation: "Per-head dimension: $d_k = d_{\\text{model}} / h = 1024 / 16 = 64$. Total parameters: $W_Q, W_K, W_V$ each map from $d_{\\text{model}}$ to all heads combined ($h \\times d_k = d_{\\text{model}}$), so each is $1024 \\times 1024$. Plus $W_O$ is $1024 \\times 1024$. Total: $4 \\times 1024^2 \\approx 4.2$M. This equals the single-head parameter count — multi-head attention doesn't add parameters, it redistributes them across $h$ independent subspaces."
    },
    // Step 5: Head specialization
    {
      type: "info",
      title: "What Do Different Heads Learn?",
      content: "Research on trained transformers reveals that attention heads specialize into distinct roles:\n\n**Positional heads**: Attend to fixed relative positions (e.g., always attend to the previous token, or to positions at a fixed offset). These implement a form of n-gram processing.\n\n**Syntactic heads**: Track grammatical structure — subject-verb agreement, dependency parsing, coreference. Clark et al. (2019) found BERT heads that closely match human-annotated dependency parses.\n\n**Induction heads** (Olsson et al., 2022): A key discovery. These two-head circuits implement pattern completion: if the model has seen the sequence \"...AB...A\" it predicts \"B\" will follow. Head 1 (previous-token head) copies positional information backward. Head 2 (the induction head) uses this to find matching prior contexts. These circuits are believed to be the primary mechanism for in-context learning.\n\n**Rare-token heads**: Attend disproportionately to rare or informative tokens, implementing a form of TF-IDF-like salience weighting.\n\nNot all heads are equally important. Pruning studies (Michel et al., 2019; Voita et al., 2019) show that many heads can be removed with minimal quality degradation — typically only a handful per layer are critical."
    },
    // Step 6: MC — head specialization
    {
      type: "mc",
      question: "An induction head implements the pattern: if \"...AB...A\" has appeared, predict \"B\" next. This requires composing two operations across layers. Which decomposition is correct?",
      options: [
        "A single head in one layer performs both operations simultaneously by using the query to match \"A\" tokens and the value to retrieve \"B\" tokens in a single attention step",
        "Head 1 attends to all previous occurrences of the current token; Head 2 attends to the token following each match and averages their embeddings to predict the most likely continuation",
        "The induction pattern is hardcoded in the positional embeddings rather than learned by attention heads, which is why it generalizes to unseen positions",
        "Head 1 in an early layer copies the token embedding of each position to the next position; Head 2 in a later layer uses this copied information to find where the current token appeared before and retrieve what followed it"
      ],
      correct: 3,
      explanation: "Induction heads require cross-layer composition (Olsson et al., 2022). In layer $l$, a 'previous token head' copies each token's identity to the next position's residual stream. In layer $l' > l$, the induction head at position $t$ queries for tokens matching $x_t$ (finding position $s$ where \"A\" previously appeared), but reads from the residual stream which now contains information shifted by one position, effectively retrieving the token that followed \"A\" (i.e., \"B\"). This two-step composition is why induction requires depth."
    },
    // Step 7: KV cache and inference
    {
      type: "info",
      title: "The KV Cache: Efficient Autoregressive Generation",
      content: "During autoregressive generation, the model produces one token at a time. At step $t$, it needs to attend to all previous tokens $1, \\ldots, t-1$.\n\nNaively, this would require recomputing keys and values for all previous positions at every step — an $O(t^2)$ cost for generating $t$ tokens total.\n\nThe **KV cache** eliminates this redundancy: since the keys and values for positions $1, \\ldots, t-1$ don't change (they depend only on the input at those positions), we compute them once and cache them.\n\nAt each generation step:\n1. Compute $q_t, k_t, v_t$ for the new token only — cost $O(d_{\\text{model}} \\cdot d_k)$\n2. Append $k_t, v_t$ to the cache\n3. Compute attention of $q_t$ against all cached keys — cost $O(t \\cdot d_k)$\n4. Weighted sum over all cached values — cost $O(t \\cdot d_v)$\n\nThe KV cache memory per layer per head is $2 \\times t \\times d_k$ (storing both keys and values). For a 70B model with 80 layers, 64 heads, $d_k = 128$, at sequence length 8192:\n\n$$\\text{KV cache} = 80 \\times 64 \\times 2 \\times 8192 \\times 128 \\times 2 \\text{ bytes} \\approx 20 \\text{ GB (in fp16)}$$\n\nThis is often the **bottleneck** for serving long-context models."
    },
    // Step 8: MC — KV cache
    {
      type: "mc",
      question: "A serving system runs a model with 40 layers, 32 heads, $d_k = 128$, and must handle a batch of 8 concurrent users each with context length 4096. How much KV cache memory is needed in fp16 (2 bytes per element)?",
      options: [
        "$40 \\times 32 \\times 2 \\times 4096 \\times 128 \\times 2 = 2.7$ GB total — the batch dimension doesn't multiply the cache because keys and values are shared across users",
        "$8 \\times 40 \\times 32 \\times 2 \\times 4096 \\times 128 \\times 2 \\approx 21$ GB — each user has an independent KV cache because their sequences are different",
        "$40 \\times 32 \\times 4096 \\times 128 \\times 2 \\approx 1.3$ GB — only keys are cached, values are recomputed from the cached keys at each step",
        "$8 \\times 40 \\times 2 \\times 4096 \\times 128 \\times 2 \\approx 0.67$ GB — the 32 heads share a single set of keys and values in multi-head attention"
      ],
      correct: 1,
      explanation: "Each user has a unique conversation, so they each need their own KV cache. Per user: $40 \\text{ layers} \\times 32 \\text{ heads} \\times 2 \\text{ (K+V)} \\times 4096 \\text{ tokens} \\times 128 \\text{ dim} \\times 2 \\text{ bytes} \\approx 2.7$ GB. For 8 users: $\\approx 21$ GB. This is why KV cache memory often dominates GPU memory during inference and why techniques like GQA (which reduces the number of KV heads) and PagedAttention (which manages cache memory efficiently) are critical for production serving."
    },
    // Step 9: MQA and GQA
    {
      type: "info",
      title: "Reducing KV Cache: MQA and GQA",
      content: "The KV cache scales with the number of **key-value heads**. Two architectural variants reduce this:\n\n**Multi-Query Attention (MQA)** (Shazeer, 2019): Use a single shared key-value head across all $h$ query heads. Each query head still has its own $W_Q^{(i)}$, but all share the same $W_K$ and $W_V$.\n\n- KV cache reduction: $h \\times$ (e.g., $32\\times$ for 32-head models)\n- Tradeoff: measurable quality loss on tasks requiring diverse attention patterns\n\n**Grouped Query Attention (GQA)** (Ainslie et al., 2023): A middle ground. Use $G$ key-value head groups, where each group serves $h/G$ query heads.\n\n- $G = 1$: equivalent to MQA\n- $G = h$: equivalent to standard MHA\n- Typical: $G = 8$ with $h = 32$ gives $4\\times$ KV cache reduction\n\nLLaMA 2 70B uses GQA with $G = 8$. The intuition: nearby query heads often learn similar attention patterns, so forcing them to share keys and values loses little while saving substantial memory at inference. Training cost is unchanged — the savings are purely at inference time."
    },
    // Step 10: MC — GQA tradeoff
    {
      type: "mc",
      question: "A team is choosing between MHA ($G = 32$), GQA ($G = 8$), and MQA ($G = 1$) for a 32-head model deployed at scale. Which factor should most strongly influence their choice?",
      options: [
        "Training speed — MQA trains significantly faster because it has fewer parameters in the attention layer, reducing both compute and memory during training",
        "Inference memory and throughput — KV cache size directly limits batch size and therefore serving throughput, and the reduction scales as $32/G$",
        "Model quality on short sequences — GQA and MQA degrade significantly on sequences under 512 tokens where the KV cache is small anyway",
        "Compatibility with FlashAttention — MQA and GQA require specialized attention kernels that are not supported by standard FlashAttention implementations"
      ],
      correct: 1,
      explanation: "The primary motivation for GQA/MQA is inference efficiency. The KV cache often dominates GPU memory during serving — with MHA, a 70B model might use 20+ GB just for KV cache at long contexts. GQA with $G=8$ reduces this to $\\sim$5 GB, allowing larger batch sizes and higher throughput. Training cost is nearly identical across all three variants (the KV projection parameters are a tiny fraction of total parameters). Quality differences are measurable but small for GQA, more noticeable for MQA. FlashAttention supports all variants."
    }
  ]
};
