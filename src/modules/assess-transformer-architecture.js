// Assessment: Transformer Architecture (Section 1.1)
// 10 MC questions, no info steps. Pure assessment module.

export const transformerAssessment = {
  id: "1.1-assess",
  sectionId: "1.1",
  title: "Assessment: Transformer Architecture",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In scaled dot-product attention, queries and keys are divided by $\\sqrt{d_k}$ before the softmax. What failure mode does this scaling prevent?",
      options: ["It prevents the attention weights from summing to more than 1, which would violate the probabilistic interpretation of attention as a weighted average over value vectors", "It normalizes the queries and keys to unit length before computing their dot product, ensuring the result is a cosine similarity bounded between -1 and 1", "It prevents the dot products from growing large in magnitude, which would push softmax into saturated regions where gradients vanish", "It compensates for the fact that values have a different dimensionality than queries and keys, rescaling the attention logits to match the value projection's output scale"],
      correct: 2,
      explanation: "When $d_k$ is large, dot products $q \\cdot k$ have variance proportional to $d_k$ (assuming unit-variance components). Without scaling, large dot products push softmax into regions where one logit dominates — the output approaches a one-hot vector, entropy collapses to near zero, and gradients through the softmax become vanishingly small. Dividing by $\\sqrt{d_k}$ keeps the variance of the logits at $O(1)$ regardless of dimension."
    },
    {
      type: "mc",
      question: "RoPE (Rotary Position Embeddings) encodes position by rotating query/key vectors in 2D subspaces of the embedding. Why does this enable better length generalization than learned absolute positional embeddings?",
      options: ["RoPE uses fewer parameters than learned embeddings, so it is less prone to overfitting on short sequences and can extrapolate to unseen lengths through its compact parameterization", "RoPE embeddings decay smoothly to zero at long distances, acting as an implicit windowed attention mechanism that gracefully handles positions beyond the training length", "RoPE projects positions into a Fourier basis with fixed frequencies, which is inherently periodic and can therefore represent any sequence length without encountering unseen position indices", "RoPE makes the dot product $q_m^\\top k_n$ depend only on the relative distance $(m - n)$ via a rotation in the complex plane, so the model never sees an absolute position index it hasn't trained on"],
      correct: 3,
      explanation: "RoPE applies a rotation $R_{\\theta, m}$ to position $m$, so $q_m^\\top k_n = (R_{\\theta,m} q)^\\top (R_{\\theta,n} k) = q^\\top R_{\\theta, n-m} k$. The attention logit depends only on the relative offset $(m - n)$, not the absolute positions. Learned absolute embeddings assign a fixed vector to each position index, so at inference time positions beyond the training length have never been seen. RoPE's relative formulation avoids this, though it still degrades at very long distances without further techniques (e.g., NTK-aware scaling, YaRN)."
    },
    {
      type: "mc",
      question: "Anthropic's \"circuits\" view describes the residual stream as a communication bus. Under this interpretation, what is the role of attention heads?",
      options: [
        "They compute nonlinear activation functions that gate information flow between layers, controlling which features pass through to subsequent computations",
        "They read from and write to the residual stream, moving information between token positions — each head implements a specific information-routing operation",
        "They normalize the residual stream to prevent feature magnitudes from diverging across layers, maintaining stable signal propagation through the network",
        "They act as memory banks that store factual knowledge retrieved by the FFN layers, enabling recall of learned associations during the forward pass"
      ],
      correct: 1,
      explanation: "In the residual stream view, the stream is a shared memory bus of dimension $d_{\\text{model}}$. Attention heads read from source positions (via $W_Q, W_K$ selecting what to attend to, and $W_V$ selecting what to read) and write to destination positions (via $W_O$). Each head moves specific types of information between positions — e.g., induction heads copy tokens that followed similar patterns. The FFN layers then read from the stream to process information locally at each position."
    },
    {
      type: "mc",
      question: "SwiGLU, used in LLaMA and PaLM, replaces the standard FFN with $\\text{SwiGLU}(x) = (\\text{Swish}(xW_1) \\odot xV) W_2$. What is the key structural difference from a vanilla two-layer FFN with ReLU?",
      options: [
        "SwiGLU uses three weight matrices instead of two, introducing a gating mechanism where one linear projection controls the information flow of another",
        "SwiGLU reduces the parameter count by factoring the two weight matrices into a single shared projection that serves both gating and value computation",
        "SwiGLU eliminates the nonlinearity entirely, relying purely on the element-wise product between two linear branches for all representational expressivity",
        "SwiGLU applies layer normalization between the two linear projections, conditioning the hidden activations before the second matrix multiplication"
      ],
      correct: 0,
      explanation: "A standard FFN uses two matrices: $\\text{ReLU}(xW_1)W_2$. SwiGLU introduces a third matrix $V$ and uses a gating mechanism: one branch $\\text{Swish}(xW_1)$ produces gating values, and the other branch $xV$ produces candidate values, combined via element-wise product. This gated linear unit structure (from Dauphin et al., 2017) gives the network finer control over information flow. To keep parameter count comparable, the hidden dimension is typically reduced from $4d$ to $\\frac{8}{3}d$. The \"key-value memory\" interpretation views each hidden unit as storing a (key, value) pair where the key determines when the unit activates."
    },
    {
      type: "mc",
      question: "Pre-LN (LayerNorm before attention/FFN) is now standard over Post-LN (LayerNorm after). What training stability problem does Pre-LN solve?",
      options: ["Pre-LN eliminates the need for residual connections entirely by normalizing each sub-layer's output to zero mean and unit variance, making the additive skip connection redundant for signal propagation", "Pre-LN is mathematically equivalent to Post-LN but uses less memory during backpropagation because the normalization statistics can be recomputed cheaply instead of stored as part of the activation checkpoint", "Pre-LN prevents activation magnitudes from growing unboundedly across layers, allowing stable training without careful learning rate warmup — Post-LN amplifies gradients in early layers, causing divergence", "Pre-LN ensures that the output distribution matches the input distribution at initialization for every sub-layer, fully solving the internal covariate shift problem that motivated batch normalization"],
      correct: 2,
      explanation: "In Post-LN, the residual path feeds raw (unnormalized) outputs back into the stream, and normalization happens after the addition. This causes gradient magnitudes to vary dramatically across layers — early layers receive amplified gradients, leading to instability. Pre-LN normalizes inputs before each sub-layer, ensuring the residual path carries well-behaved signals. Xiong et al. (2020) showed Pre-LN removes the need for careful warmup schedules. The tradeoff: some work suggests Post-LN produces slightly better final performance when it can be stabilized (e.g., with careful init or additional techniques)."
    },
    {
      type: "mc",
      question: "Grouped Query Attention (GQA) with $G$ groups uses $G$ key-value heads shared across $H$ query heads (where $G < H$). If a model has $H = 32$ query heads and uses GQA with $G = 8$, what is the KV cache memory reduction compared to standard MHA, and how does this compare to MQA?",
      options: ["KV cache is reduced by $32\\times$ vs MHA because each GQA group shares keys and values across all 32 query heads; MQA would only reduce it by $8\\times$", "KV cache is reduced by $8\\times$ vs MHA because the group size equals the reduction factor; MQA and GQA have identical memory footprints since both share KV heads", "KV cache is the same size as MHA because GQA still stores distinct key-value pairs per query head; the savings come only from reduced compute in the attention scores", "KV cache is reduced by $4\\times$ vs MHA ($8/32$); MQA ($G=1$) would reduce it by $32\\times$ but with greater quality degradation"],
      correct: 3,
      explanation: "In MHA, we store $H = 32$ distinct key-value pairs per layer per token. GQA with $G = 8$ stores only 8, so the KV cache shrinks by $32/8 = 4\\times$. MQA uses $G = 1$ (a single shared KV head), shrinking it by $32\\times$. GQA is a middle ground: it recovers most of MQA's memory savings while preserving quality closer to MHA. LLaMA 2 70B uses GQA with $G = 8$ for exactly this tradeoff. The quality gap between MQA and MHA is measurable on long-form generation tasks where KV diversity matters."
    },
    {
      type: "mc",
      question: "The concept of **superposition** in transformer residual streams refers to the hypothesis that:",
      options: [
        "Residual streams encode features in a strict one-to-one mapping, with each dimension representing exactly one interpretable feature and no sharing across directions",
        "The model represents more features than it has dimensions by encoding them as nearly orthogonal directions, tolerating small interference — analogous to compressed sensing",
        "Multiple attention heads attend to the same positions independently, creating redundant representations that waste capacity through duplicated information routing",
        "The residual stream superposes the outputs of all previous layers via simple addition, with no learned feature-level structure beyond the cumulative sum"
      ],
      correct: 1,
      explanation: "Superposition (Elhage et al., 2022) is the hypothesis that transformers represent $m \\gg d$ features using $d$-dimensional residual streams by assigning nearly orthogonal directions to different features. This works because most features are sparse (rarely active simultaneously), so the interference from non-orthogonality rarely causes problems. This is closely related to compressed sensing and the Johnson-Lindenstrauss lemma — in high dimensions, random vectors are nearly orthogonal. Superposition makes interpretability hard because features don't align with individual neurons."
    },
    {
      type: "mc",
      question: "Consider a transformer with $L = 32$ layers, $d_{\\text{model}} = 4096$, $H = 32$ heads, and SwiGLU FFN with hidden dim $\\frac{8}{3} \\times 4096 \\approx 10923$. Approximately how many parameters are in the model (ignoring embeddings and final layer norm)?",
      options: ["About 7B — each layer has roughly 220M parameters from attention ($4d^2$) plus FFN ($3 \\times d \\times d_{\\text{ff}}$)", "About 1.5B — each layer has roughly 50M parameters because GQA reduces the attention projection matrices by sharing key-value heads", "About 13B — the SwiGLU FFN dominates with three matrices totaling $8d^2$ per layer while attention contributes only a small fraction", "About 30B — attention and FFN contribute equally at approximately $8d^2$ each, giving $16d^2$ per layer times 32 layers"],
      correct: 0,
      explanation: "Per layer: Attention has $W_Q, W_K, W_V, W_O$ each of size $d^2$, totaling $4d^2 \\approx 4 \\times 16.8M \\approx 67M$. SwiGLU FFN has three matrices: $W_1, V$ of size $d \\times d_{\\text{ff}}$ and $W_2$ of size $d_{\\text{ff}} \\times d$, totaling $3 \\times d \\times d_{\\text{ff}} \\approx 3 \\times 4096 \\times 10923 \\approx 134M$. Plus two LayerNorms at $2 \\times 2d \\approx 16K$. Per layer $\\approx 201M$. Multiply by 32 layers $\\approx 6.4B$. Adding embeddings ($V \\times d$) brings it near 7B. This matches LLaMA 7B's architecture."
    },
    {
      type: "mc",
      question: "An \"induction head\" is a two-head circuit that implements a specific attention pattern. What computation does it perform, and why is it considered a key mechanism for in-context learning?",
      options: ["It attends to the most frequent token in the context and copies it to the output, implementing a unigram frequency estimator that tracks token counts across the sequence", "It computes the weighted average of all key vectors to produce a compressed summary representation of the full context, acting as a bottleneck that forces information compression", "It uses one head to identify previous occurrences of the current token and a second head to copy what followed those occurrences — implementing the rule 'if A B ... A, predict B'", "It detects syntactic patterns like subject-verb agreement by attending to structurally related positions, using learned positional features to identify grammatical dependencies"],
      correct: 2,
      explanation: "Induction heads (Olsson et al., 2022) are a two-step circuit: (1) a \"previous-token head\" shifts information backward so each token's residual stream contains info about the preceding token; (2) an \"induction head\" uses this to find previous instances of the current token and attend to what came after them. This implements \"A B ... A $\\to$ B\" — a bigram copy operation that generalizes to longer patterns. Olsson et al. found that induction heads form during a phase transition early in training that coincides with a sharp drop in loss, and they account for the majority of in-context learning ability."
    },
    {
      type: "mc",
      question: "Why is the attention mechanism $O(n^2 d)$ in sequence length $n$, and which component specifically creates the quadratic bottleneck?",
      options: ["The residual connections require adding $n^2$ vectors together because each token's residual accumulates contributions from all other tokens across all attention heads", "The value projection $W_V$ has $n^2$ parameters that must be learned, because the projection must account for all pairwise token interactions in the sequence", "The softmax normalization requires summing over $n^2$ elements in total across all rows, making the normalization step alone the computational bottleneck of the attention mechanism", "The $QK^\\top$ matrix multiplication produces an $n \\times n$ attention matrix, and both computing and storing this matrix scale quadratically in sequence length"],
      correct: 3,
      explanation: "The attention score matrix $A = \\text{softmax}(QK^\\top / \\sqrt{d_k})$ has shape $n \\times n$. Computing the $n \\times n$ dot products costs $O(n^2 d_k)$, storing the matrix costs $O(n^2)$ memory, and the matrix-vector product $AV$ costs $O(n^2 d_v)$. For long contexts (e.g., $n = 128K$), this $n^2$ term dominates everything. This is why methods like FlashAttention (which avoids materializing the full attention matrix via tiling), linear attention, and sparse attention exist — they aim to reduce either the compute or memory from $O(n^2)$."
    }
  ]
};
