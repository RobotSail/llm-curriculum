// Focused learning module: Self-Attention Mechanism
// Section 1.1: Transformer Architecture
// Covers: dot-product attention, Q/K/V projections, softmax weighting,
// the sqrt(d_k) scaling, and attention as soft dictionary lookup.
// Single-concept module building from first principles.

export const selfAttentionLearning = {
  id: "1.1-self-attention-learning-easy",
  sectionId: "1.1",
  title: "Self-Attention Mechanism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Motivation — why attention?
    {
      type: "info",
      title: "The Problem: Fixed-Width Bottlenecks",
      content: "Before transformers, sequence models like RNNs processed tokens sequentially, compressing all past context into a single fixed-width hidden state $h_t \\in \\mathbb{R}^d$. To predict token $t$, the model could only access information that survived compression through the $h_1 \\to h_2 \\to \\cdots \\to h_t$ chain.\n\nThis creates a **bottleneck**: information from early tokens must be packed into the same $d$-dimensional vector that also carries recent context. Long-range dependencies — like a pronoun referring to a noun 200 tokens back — are lost because the hidden state has finite capacity and no mechanism to selectively retrieve distant information.\n\n**Self-attention** solves this by giving each token **direct access** to every other token in the sequence. Instead of passing information through a chain, token $t$ can look at all positions $1, \\ldots, n$ simultaneously and decide which ones are relevant. No information needs to survive a sequential bottleneck — it is available on demand."
    },
    // Step 2: MC — understanding the bottleneck
    {
      type: "mc",
      question: "An RNN processes a 500-token document. At position 500, the model needs to recall a specific fact stated at position 12. Why is this fundamentally difficult for the RNN, regardless of its hidden state size $d$?",
      options: [
        "The hidden state at position 500 has been updated 488 times since position 12 — each update may overwrite information to store newer content, and there is no mechanism to protect or selectively retrieve old information",
        "RNNs can only attend to a fixed window of recent tokens, typically 64-128 positions, making position 12 completely inaccessible by the time the model reaches position 500",
        "The gradient signal from position 500 to position 12 passes through 488 matrix multiplications, so the model cannot learn to store that information during training even if the hidden state has capacity",
        "RNNs process tokens in parallel batches, so position 12 and position 500 are processed independently without any information flow between them"
      ],
      correct: 0,
      explanation: "The core issue is the fixed-width bottleneck: the hidden state must serve as both memory and communication channel. Each of the 488 intervening updates can overwrite previously stored information, and the model has no content-addressable retrieval mechanism — it cannot say 'give me back what was at position 12.' Option C describes a training problem (vanishing gradients), which is real but separate from the inference-time bottleneck. Even a perfectly trained RNN faces the storage limitation at inference."
    },
    // Step 3: Attention as soft lookup
    {
      type: "info",
      title: "Attention as Soft Dictionary Lookup",
      content: "The key insight behind attention: think of it as a **soft dictionary lookup**.\n\nIn a regular dictionary (hash map), you have a query, you find the matching key, and you retrieve its value. Self-attention does the same thing, but *softly* — instead of exact matching, the query is compared to all keys simultaneously, and the result is a **weighted combination** of all values.\n\nConcretely, each token $i$ in the sequence produces three vectors:\n- **Query** $q_i = x_i W_Q$: \"What am I looking for?\"\n- **Key** $k_i = x_i W_K$: \"What do I contain that others might want?\"\n- **Value** $v_i = x_i W_V$: \"What information do I provide if selected?\"\n\nwhere $x_i \\in \\mathbb{R}^{d_{\\text{model}}}$ is the input embedding and $W_Q, W_K \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}$, $W_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_v}$ are learned projection matrices.\n\nThe output for token $i$ is a weighted sum of all values, where the weight on each value $v_j$ is determined by how well query $q_i$ matches key $k_j$."
    },
    // Step 4: MC — Q/K/V roles
    {
      type: "mc",
      question: "In self-attention, why are there three separate projections (Q, K, V) rather than simply using the raw token embeddings $x_i$ for all three roles?",
      options: [
        "Three projections reduce the total parameter count compared to using full $d_{\\text{model}}$-dimensional embeddings, since $d_k$ and $d_v$ are typically smaller than $d_{\\text{model}}$",
        "Using the same vector for querying and being queried creates a symmetric attention pattern ($\\alpha_{ij} = \\alpha_{ji}$), which prevents the model from learning directional relationships like 'adjective attends to its noun but not vice versa'",
        "Separate projections allow the model to learn independent subspaces for matching (Q, K) and information transfer (V) — what makes two tokens relevant to each other is different from what information should flow between them",
        "The three projections implement a form of dropout regularization by projecting into lower-dimensional spaces before computing attention, preventing overfitting to individual dimensions"
      ],
      correct: 2,
      explanation: "The Q and K projections jointly determine *which* tokens attend to which (relevance), while the V projection determines *what information* flows from attended tokens. These are fundamentally different functions: a pronoun might attend to a noun because of syntactic structure (captured by Q/K), but the information it needs is the noun's semantic content (captured by V). Using raw embeddings for all three would force a single representation to serve all three roles, severely limiting the model's expressivity."
    },
    // Step 5: Scaled dot-product attention
    {
      type: "info",
      title: "Scaled Dot-Product Attention",
      content: "The attention mechanism computes:\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$$\n\nLet's unpack this step by step:\n\n**1. Compute attention scores**: $S = QK^\\top \\in \\mathbb{R}^{n \\times n}$. Entry $S_{ij} = q_i \\cdot k_j$ measures how much query $i$ matches key $j$. This is an $O(n^2 d_k)$ operation.\n\n**2. Scale**: Divide by $\\sqrt{d_k}$ to control the magnitude of the scores.\n\n**3. Softmax**: Apply softmax row-wise to get attention weights $\\alpha_{ij} = \\text{softmax}(S_i / \\sqrt{d_k})_j$. Each row sums to 1, so each token distributes a total weight of 1 across all positions.\n\n**4. Weighted sum**: Output $o_i = \\sum_j \\alpha_{ij} v_j$ — each token's output is a weighted average of all value vectors.\n\nThe attention weight $\\alpha_{ij}$ is the fraction of attention that token $i$ pays to token $j$. If $\\alpha_{ij} \\approx 1$ and all other weights are $\\approx 0$, token $i$ is essentially copying the value from position $j$."
    },
    // Step 6: MC — attention weight interpretation
    {
      type: "mc",
      question: "Token $i$ has attention weights $\\alpha_{i,1} = 0.02, \\alpha_{i,2} = 0.91, \\alpha_{i,3} = 0.03, \\alpha_{i,4} = 0.04$. What is the output $o_i$?",
      options: [
        "Exactly $v_2$, since token 2 has the highest attention weight and the softmax approximates argmax",
        "The average of all four value vectors, since softmax ensures equal contribution from all positions regardless of the attention scores",
        "A nonlinear transformation of $v_2$ determined by the attention weight magnitude 0.91",
        "Approximately $v_2$ with small contributions from $v_1, v_3, v_4$ — specifically $o_i = 0.02 v_1 + 0.91 v_2 + 0.03 v_3 + 0.04 v_4$"
      ],
      correct: 3,
      explanation: "The output is always a weighted sum: $o_i = \\sum_j \\alpha_{ij} v_j$. With $\\alpha_{i,2} = 0.91$, token 2 dominates but the other tokens still contribute. The operation is purely linear in the values — the attention weights determine the convex combination. If the weights were exactly $[0, 1, 0, 0]$ then $o_i = v_2$ exactly, but softmax never produces exact zeros (though they can be numerically negligible)."
    },
    // Step 7: The sqrt(d_k) scaling
    {
      type: "info",
      title: "Why Scale by $\\sqrt{d_k}$?",
      content: "The scaling factor $1/\\sqrt{d_k}$ is not cosmetic — it prevents a specific failure mode.\n\nAssume query and key components are independent with zero mean and unit variance. The dot product $q \\cdot k = \\sum_{j=1}^{d_k} q_j k_j$ is a sum of $d_k$ terms, each with zero mean and variance 1. By the CLT, $q \\cdot k$ has:\n- Mean: $0$\n- Variance: $d_k$\n- Standard deviation: $\\sqrt{d_k}$\n\nSo as $d_k$ grows, the raw dot products grow in magnitude proportionally to $\\sqrt{d_k}$. For $d_k = 64$ (common), typical dot products are on the order of $\\pm 8$.\n\nWhen softmax receives inputs of magnitude $\\pm 8$, the output becomes extremely peaked — nearly a one-hot vector. In this regime:\n- **Forward pass**: Attention collapses to near-hard attention, ignoring most positions\n- **Backward pass**: Softmax gradients $\\frac{\\partial \\alpha_i}{\\partial s_j} = \\alpha_i(\\delta_{ij} - \\alpha_j)$ vanish because the $\\alpha$ values are near 0 or 1\n\nDividing by $\\sqrt{d_k}$ normalizes the dot products back to $O(1)$ variance, keeping softmax in its informative (non-saturated) regime."
    },
    // Step 8: MC — scaling failure mode
    {
      type: "mc",
      question: "A researcher removes the $\\sqrt{d_k}$ scaling from attention in a model with $d_k = 128$. During training, they observe that attention weights quickly become very peaked (near one-hot). What is the most direct consequence for learning?",
      options: [
        "Gradients through the softmax vanish because $\\frac{\\partial \\alpha_i}{\\partial s_j} \\approx 0$ when attention weights are near 0 or 1, making the attention pattern unable to update",
        "Training becomes unstable because the loss function becomes non-differentiable when attention weights are exactly one-hot",
        "The model trains faster because hard attention creates cleaner gradient signals with less noise from irrelevant positions",
        "The model uses more memory because peaked attention weights cannot be compressed as efficiently during mixed-precision training"
      ],
      correct: 0,
      explanation: "The softmax Jacobian $\\frac{\\partial \\alpha_i}{\\partial s_j} = \\alpha_i(\\delta_{ij} - \\alpha_j)$ approaches zero when any $\\alpha_i$ is near 0 or 1. If $\\alpha_2 \\approx 1$ and all others $\\approx 0$, then $\\partial \\alpha_2 / \\partial s_j \\approx 1 \\cdot (0) = 0$ for $j=2$, and $\\approx 0$ for $j \\neq 2$. The attention pattern becomes frozen — gradients cannot flow through softmax to adjust which tokens attend where. With $d_k = 128$, unscaled dot products have std $\\approx 11.3$, pushing softmax deep into saturation."
    },
    // Step 9: Causal masking for autoregressive models
    {
      type: "info",
      title: "Causal Masking: Preventing Information Leakage",
      content: "In autoregressive language models (GPT-style), the model predicts each token $y_t$ given only the preceding tokens $y_1, \\ldots, y_{t-1}$. But self-attention naturally lets every position attend to every other position — including future tokens.\n\nTo enforce causality, we apply a **causal mask** before the softmax:\n\n$$S_{ij}^{\\text{masked}} = \\begin{cases} q_i \\cdot k_j / \\sqrt{d_k} & \\text{if } j \\leq i \\\\ -\\infty & \\text{if } j > i \\end{cases}$$\n\nSetting future positions to $-\\infty$ ensures $\\text{softmax}(-\\infty) = 0$ — the model cannot attend to tokens it hasn't yet generated.\n\nThe resulting attention matrix is **lower-triangular**: position 1 attends only to itself, position 2 to positions 1-2, position 3 to positions 1-3, and so on. This is equivalent to training $n$ models simultaneously — one for each prefix length — in a single forward pass.\n\nEncoder-only models (BERT) and the encoder in encoder-decoder models (T5's encoder) use **bidirectional** attention without causal masking, since they process complete input sequences where all positions are known."
    },
    // Step 10: MC — causal masking
    {
      type: "mc",
      question: "During training, a causal language model processes a sequence of $n$ tokens. The causal mask makes the attention matrix lower-triangular. How many next-token predictions does the model make in a single forward pass?",
      options: [
        "Just 1 — the model only predicts the token after the last position, since all other positions are used as context",
        "$n$ — every position predicts the next token, including position $n$ which predicts a special end-of-sequence token",
        "$n^2 / 2$ — each entry in the lower-triangular attention matrix corresponds to one prediction",
        "$n - 1$ — each position $t$ predicts $y_{t+1}$ using the causal context $y_1, \\ldots, y_t$, except the last position which has no target"
      ],
      correct: 3,
      explanation: "With teacher forcing, position $t$ receives the ground-truth prefix $y_1, \\ldots, y_t$ (enforced by the causal mask) and predicts $y_{t+1}$. This gives $n - 1$ predictions: position 1 predicts $y_2$, position 2 predicts $y_3$, ..., position $n-1$ predicts $y_n$. Position $n$ has no next token to predict. This is why causal LMs are efficient to train — a single forward pass produces $n-1$ training signals, unlike an RNN that would need $n-1$ sequential forward passes."
    },
    // Step 11: Attention complexity
    {
      type: "info",
      title: "Computational Cost of Self-Attention",
      content: "The dominant cost of self-attention is computing the $n \\times n$ attention matrix $QK^\\top$, which requires $O(n^2 d_k)$ multiply-adds.\n\n**Memory**: Storing the full attention matrix requires $O(n^2)$ memory per head. For a model with $h$ heads and $L$ layers, the total attention memory is $O(L \\cdot h \\cdot n^2)$. At sequence length $n = 8192$ with 32 heads and 32 layers, this is $32 \\times 32 \\times 8192^2 \\approx 69$ billion entries.\n\n**Compute**: The matrix multiplications $QK^\\top$ and $(\\text{softmax})V$ each cost $O(n^2 d_k)$ per head. For short sequences ($n \\ll d_{\\text{model}}$), the FFN layers dominate compute. But as $n$ grows, the quadratic attention cost overtakes everything else.\n\nThis quadratic scaling is why long-context models are expensive and why techniques like **FlashAttention** (which reduces memory by not materializing the full $n \\times n$ matrix) and **KV caching** (which avoids recomputing past keys/values during generation) are essential for practical deployment.\n\nAt inference time during autoregressive generation, each new token only needs to compute one row of the attention matrix (its attention over all previous tokens), making the per-token cost $O(n \\cdot d_k)$ — linear in the current sequence length."
    },
    // Step 12: MC — attention complexity
    {
      type: "mc",
      question: "A model processes sequences of length $n$ with attention dimension $d_k = 128$. Doubling the sequence length from 4096 to 8192 will increase the attention computation by approximately:",
      options: [
        "2x — attention computation scales linearly with sequence length since each token attends to a fixed number of neighbors",
        "4x — the attention matrix grows from $4096^2$ to $8192^2$ entries, and the matrix multiplications to fill it scale quadratically",
        "8x — attention scales as $O(n^3)$ because both the matrix dimensions and the inner dimension grow with $n$",
        "128x — the cost is dominated by $d_k$, so doubling $n$ creates $d_k = 128$ times more work per new attention entry"
      ],
      correct: 1,
      explanation: "The attention computation $QK^\\top$ has cost $O(n^2 d_k)$. With $d_k$ fixed, doubling $n$ increases the cost by $4\\times$. The matrix goes from $4096 \\times 4096$ to $8192 \\times 8192$ — four times as many entries, each requiring the same $d_k$-dimensional dot product. This quadratic scaling is why context length is one of the most important efficiency considerations for transformers."
    },
    // Step 13: Putting it together
    {
      type: "info",
      title: "Self-Attention in the Transformer Block",
      content: "In a full transformer block, self-attention is one of two sub-layers:\n\n$$\\text{Block}(x) = \\text{FFN}(\\text{LN}(x + \\text{Attn}(\\text{LN}(x))))$$\n\n(using Pre-LN convention). The attention sub-layer handles **inter-token communication** — moving information between positions. The FFN sub-layer handles **per-token computation** — processing information locally at each position.\n\nThe **residual connection** ($x + \\text{Attn}(\\cdot)$) is crucial: it means each layer only needs to compute a *correction* to the current representation, not a complete new representation. This creates the **residual stream** — a running sum that accumulates contributions from all layers.\n\nStacking $L$ such blocks gives each token $L$ opportunities to gather information from other positions (via attention) and process it (via FFN). Deep transformers can implement multi-step reasoning: layer 1 might resolve coreference (\"he\" → \"John\"), layer 2 might compose attributes (\"tall John\"), layer 3 might perform inference (\"tall → plays basketball\").\n\nThe entire self-attention mechanism — queries, keys, values, softmax, scaling — is differentiable end-to-end, so all projection matrices $W_Q, W_K, W_V, W_O$ are learned jointly through backpropagation."
    },
    // Step 14: MC — integration question
    {
      type: "mc",
      question: "In a 32-layer transformer processing the sentence \"The cat sat on the mat because it was tired\", which statement best describes how self-attention enables the model to resolve what \"it\" refers to?",
      options: [
        "A single attention head in one layer learns to match pronouns to their antecedents — it assigns high attention weight from \"it\" to \"cat\" based on syntactic features encoded in the Q/K projections",
        "The model does not use attention for coreference — instead, the FFN layers store a lookup table mapping pronouns to likely antecedents based on training data statistics",
        "Across multiple layers and heads, the model builds up the resolution: early layers may encode syntactic structure and semantic features, while later layers compose these to route information from \"cat\" to \"it\" via the residual stream",
        "The causal mask prevents \"it\" from attending to \"cat\" because \"cat\" appears later in the internal representation after the model reorders tokens by semantic importance"
      ],
      correct: 2,
      explanation: "Coreference resolution in transformers is typically a multi-layer, multi-head process. Early heads may identify syntactic roles (subject, object), encode semantic features (animate vs inanimate), or track positional relationships. Later heads compose these features to perform the actual resolution. The residual stream accumulates these incremental contributions. This distributed, multi-step computation is why transformers handle complex linguistic phenomena well — no single attention head needs to solve the full problem."
    }
  ]
};
