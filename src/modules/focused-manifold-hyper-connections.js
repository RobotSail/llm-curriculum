// Focused learning module: Manifold-Constrained Hyper-Connections (mHC)
// — stabilizing expanded residual streams using doubly stochastic mixing
// matrices constrained to the Birkhoff polytope.

export const manifoldHyperConnectionsLearning = {
  id: "B.2-mhc-learning-medium",
  sectionId: "B.2",
  title: "Manifold-Constrained Hyper-Connections: Stabilizing Wide Residual Streams",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — Standard residual connections' limitations
    {
      type: "info",
      title: "The Fixed-Weight Problem in Residual Connections",
      content: "Standard transformer residual connections follow a simple pattern:\n\n$$\\mathbf{h}_l = \\mathbf{h}_{l-1} + f_l(\\text{Norm}(\\mathbf{h}_{l-1}))$$\n\nUnrolling this recurrence across $L$ layers:\n\n$$\\mathbf{h}_L = \\mathbf{h}_0 + \\sum_{l=1}^{L} f_l(\\text{Norm}(\\mathbf{h}_{l-1}))$$\n\nThe final hidden state is the embedding $\\mathbf{h}_0$ plus a **uniformly-weighted sum** of all layer outputs. Every layer's contribution receives weight 1, regardless of its importance.\n\nThis creates two problems:\n\n1. **Hidden-state growth**: $\\|\\mathbf{h}_l\\|$ grows as $O(L)$ with depth. Each layer adds to the norm.\n\n2. **PreNorm dilution**: RMSNorm normalizes $\\mathbf{h}_l$ before each layer's computation, but since $\\|\\mathbf{h}_l\\|$ keeps growing, deeper layers must produce increasingly large outputs just to be \"heard\" above the accumulated sum of previous layers.\n\nThe research community has moved from fixed to learned, input-dependent weighting for **sequence mixing** (attention) and **expert routing** (MoE). But depth-wise aggregation of layer outputs still uses fixed unit weights — it is the last hardcoded aggregation in the transformer."
    },
    // Step 2: MC — Residual stream growth
    {
      type: "mc",
      question: "In a 60-layer transformer with standard residual connections, $\\mathbf{h}_L = \\mathbf{h}_0 + \\sum_{l=1}^{60} f_l(\\cdot)$. If each layer's output $f_l$ has roughly equal norm, how does $\\|\\mathbf{h}_L\\|$ compare to $\\|\\mathbf{h}_0\\|$?",
      options: [
        "$\\|\\mathbf{h}_L\\| \\approx \\|\\mathbf{h}_0\\|$ — the layer outputs cancel each other out due to random directions",
        "$\\|\\mathbf{h}_L\\|$ grows roughly as $O(\\sqrt{L})$ due to random walk behavior, reaching about $8\\times \\|\\mathbf{h}_0\\|$",
        "$\\|\\mathbf{h}_L\\|$ grows roughly as $O(L)$ because the layer outputs are not random — they are learned and tend to be positively correlated, reaching up to $60\\times \\|\\mathbf{h}_0\\|$",
        "$\\|\\mathbf{h}_L\\| = 60 \\|\\mathbf{h}_0\\|$ exactly, because each layer adds exactly one unit of norm"
      ],
      correct: 2,
      explanation: "Layer outputs are not independent random vectors — they are learned functions that tend to be correlated (they all operate on and modify the same residual stream). In practice, the norm grows roughly linearly with depth, not as $\\sqrt{L}$ (which would hold for truly random, independent additions). This $O(L)$ growth causes PreNorm dilution: later layers' contributions are diluted relative to the accumulated sum."
    },
    // Step 3: Info — Hyper-Connections: expanding the residual stream
    {
      type: "info",
      title: "Hyper-Connections: Multiple Residual Streams",
      content: "**Hyper-Connections (HC)**, introduced by DeepSeek, expand the residual stream from dimension $C$ to $n \\times C$ (with expansion rate $n$, typically $n = 4$). Instead of one residual path, the model maintains $n$ parallel streams.\n\nThe update rule becomes:\n\n$$\\mathbf{x}_{l+1} = \\mathbf{H}_l^{\\text{res}} \\mathbf{x}_l + (\\mathbf{H}_l^{\\text{post}})^T f_l(\\mathbf{H}_l^{\\text{pre}} \\mathbf{x}_l)$$\n\nwhere:\n- $\\mathbf{x}_l \\in \\mathbb{R}^{nC}$ is the expanded state (concatenation of $n$ streams)\n- $\\mathbf{H}_l^{\\text{pre}} \\in \\mathbb{R}^{C \\times nC}$: compresses $n$ streams → 1 stream as input to layer $f_l$\n- $\\mathbf{H}_l^{\\text{post}} \\in \\mathbb{R}^{C \\times nC}$: expands layer output back to $n$ streams\n- $\\mathbf{H}_l^{\\text{res}} \\in \\mathbb{R}^{n \\times n}$: mixes information between the $n$ streams directly (acts per-channel)\n\nHC gives significant quality improvements by increasing the **topological complexity** of the computation graph without adding FLOPs to individual layers (attention, FFN still operate on $C$-dimensional inputs).\n\n**The problem**: unconstrained learned mixing matrices $\\mathbf{H}^{\\text{res}}$ can amplify signals — their spectral norm can exceed 1. In experiments with a 27B model, signal gains exceeded **3000×**, causing training divergence."
    },
    // Step 4: MC — HC instability
    {
      type: "mc",
      question: "Hyper-Connections use a learned $n \\times n$ mixing matrix $\\mathbf{H}^{\\text{res}}$ to combine $n$ residual streams. With $L = 60$ layers, the effective residual mapping is $(\\mathbf{H}^{\\text{res}})^{60}$. What happens when $\\|\\mathbf{H}^{\\text{res}}\\|_2 = 1.1$ (spectral norm slightly above 1)?",
      options: [
        "The signal is attenuated by a factor of $0.9^{60} \\approx 0.002$, causing vanishing gradients",
        "The signal is amplified by $1.1^{60} \\approx 304$, potentially causing training instability",
        "The effect is negligible — a spectral norm of 1.1 is close enough to 1 to maintain stability",
        "The mixing matrix is applied independently per layer, so the spectral norms don't compound"
      ],
      correct: 1,
      explanation: "Matrix powers compound: $\\|(\\mathbf{H}^{\\text{res}})^L\\|_2 \\leq \\|\\mathbf{H}^{\\text{res}}\\|_2^L = 1.1^{60} \\approx 304$. Even a spectral norm slightly above 1 causes exponential amplification across layers. In practice, DeepSeek observed amplifications up to 3000× with unconstrained HC. This is the same instability mechanism as in RNNs (exploding gradients) — the mixing matrices play the role of the recurrent weight matrix."
    },
    // Step 5: Info — The Birkhoff polytope constraint
    {
      type: "info",
      title: "Constraining to the Birkhoff Polytope",
      content: "The key innovation of **mHC (manifold-Constrained Hyper-Connections)** is constraining $\\mathbf{H}^{\\text{res}}$ to lie within the **Birkhoff polytope** — the set of **doubly stochastic matrices**.\n\nA matrix $\\mathbf{M}$ is doubly stochastic if:\n- All entries are non-negative: $M_{ij} \\geq 0$\n- Every row sums to 1: $\\sum_j M_{ij} = 1$\n- Every column sums to 1: $\\sum_i M_{ij} = 1$\n\nThree properties make this constraint effective for stabilizing residual streams:\n\n1. **Norm-bounded**: $\\|\\mathbf{M}\\|_2 \\leq 1$ for any doubly stochastic matrix. The mapping is **non-expansive** — it cannot amplify signals. Each output is a convex combination of inputs.\n\n2. **Closed under multiplication**: The product of doubly stochastic matrices is doubly stochastic. So $(\\mathbf{H}^{\\text{res}})^L$ remains doubly stochastic regardless of $L$ — stability is guaranteed at **any depth**.\n\n3. **Geometric interpretation**: By Birkhoff's theorem, the Birkhoff polytope is the convex hull of all permutation matrices. So $\\mathbf{H}^{\\text{res}}$ acts as a learned convex combination of permutations — controlled stream mixing that preserves the information structure.\n\nResult: mHC reduces the maximum signal gain from ~3000× (unconstrained HC) to ~1.6× — three orders of magnitude improvement."
    },
    // Step 6: MC — Doubly stochastic properties
    {
      type: "mc",
      question: "A doubly stochastic matrix has all rows and columns summing to 1 with non-negative entries. Why does the closure-under-multiplication property matter for deep networks?",
      options: [
        "It ensures that the mixing matrix can be inverted at any layer, enabling exact gradient computation",
        "It guarantees that composing the mixing across any number of layers produces another doubly stochastic matrix, maintaining the $\\|\\mathbf{M}\\|_2 \\leq 1$ bound regardless of depth",
        "It means the mixing matrices commute, so the order of layer processing doesn't affect the output",
        "It allows the mixing matrices to be decomposed into a product of simpler operations for efficient computation"
      ],
      correct: 1,
      explanation: "If $\\mathbf{A}$ and $\\mathbf{B}$ are doubly stochastic, then $\\mathbf{AB}$ is also doubly stochastic. This means the composed residual mapping across $L$ layers is itself doubly stochastic, inheriting the $\\|\\cdot\\|_2 \\leq 1$ bound. Without closure, individual layer constraints wouldn't guarantee stability of the full network. Doubly stochastic matrices do not generally commute, and invertibility is not guaranteed (singular doubly stochastic matrices exist)."
    },
    // Step 7: Info — Sinkhorn-Knopp projection
    {
      type: "info",
      title: "Making It Differentiable: Sinkhorn-Knopp Iteration",
      content: "How do you make a neural network output a doubly stochastic matrix? mHC uses the **Sinkhorn-Knopp algorithm** (1967).\n\nStart with learnable logits $\\mathbf{L} \\in \\mathbb{R}^{n \\times n}$ (unconstrained). Convert to doubly stochastic via alternating normalization:\n\n1. Exponentiate: $\\mathbf{M}^{(0)} = \\exp(\\mathbf{L})$ (ensures non-negativity)\n2. Normalize rows: $\\mathbf{M}^{(k+1/2)} = \\text{diag}(\\mathbf{M}^{(k)} \\mathbf{1})^{-1} \\mathbf{M}^{(k)}$\n3. Normalize columns: $\\mathbf{M}^{(k+1)} = \\mathbf{M}^{(k+1/2)} \\text{diag}(\\mathbf{1}^T \\mathbf{M}^{(k+1/2)})^{-1}$\n4. Repeat steps 2–3 for $K$ iterations (typically $K = 20$ suffices)\n\nThe result converges to the unique doubly stochastic matrix closest to the original $\\exp(\\mathbf{L})$ in the KL-divergence sense.\n\nSince $n$ is small (typically 4), each Sinkhorn iteration is cheap: just normalizing a $4 \\times 4$ matrix. The entire projection adds negligible overhead.\n\nFor $\\mathbf{H}^{\\text{pre}}$ and $\\mathbf{H}^{\\text{post}}$, a simpler constraint is used: sigmoid activation ensures non-negativity ($H_{ij} \\geq 0$), preventing signal cancellation from mixing positive and negative coefficients. Full double stochasticity is only needed for $\\mathbf{H}^{\\text{res}}$ because it's the matrix that compounds across layers."
    },
    // Step 8: MC — Why Sinkhorn-Knopp
    {
      type: "mc",
      question: "mHC uses Sinkhorn-Knopp iteration (alternating row/column normalization) to project learned logits onto the Birkhoff polytope. Why is this preferred over directly parameterizing the matrix as, say, a softmax over rows?",
      options: [
        "Softmax over rows only produces row-stochastic matrices (rows sum to 1); it doesn't guarantee column sums equal 1, so the $\\|\\mathbf{M}\\|_2 \\leq 1$ bound would not hold",
        "Sinkhorn-Knopp is faster than softmax because it avoids computing exponentials",
        "Softmax cannot produce sparse matrices, while Sinkhorn-Knopp can produce exact permutation matrices",
        "Row-wise softmax would require $n^2$ parameters while Sinkhorn-Knopp needs only $n$ parameters"
      ],
      correct: 0,
      explanation: "Row-stochastic matrices (each row sums to 1) are not necessarily doubly stochastic — their spectral norm can exceed 1. For example, the row-stochastic matrix $[[0.9, 0.1], [0.9, 0.1]]$ has spectral norm 1, but $[[0.1, 0.9], [0.9, 0.1]]$ does too. However, without the column constraint, matrices like $[[1, 0], [1, 0]]$ are row-stochastic with spectral norm $\\sqrt{2} > 1$. Doubly stochastic guarantees both row and column sums equal 1, which ensures $\\|\\mathbf{M}\\|_2 \\leq 1$. Both methods use $n^2$ parameters."
    },
    // Step 9: Info — mHC architecture and training
    {
      type: "info",
      title: "The Full mHC Architecture",
      content: "Putting it together, each layer $l$ in mHC computes:\n\n$$\\mathbf{x}_{l+1} = \\mathbf{H}_l^{\\text{res}} \\mathbf{x}_l + (\\mathbf{H}_l^{\\text{post}})^T f_l(\\mathbf{H}_l^{\\text{pre}} \\mathbf{x}_l)$$\n\nwhere:\n- $\\mathbf{H}_l^{\\text{res}} \\in \\mathbb{R}^{n \\times n}$: doubly stochastic (via Sinkhorn-Knopp), mixes $n$ streams\n- $\\mathbf{H}_l^{\\text{pre}} \\in \\mathbb{R}^{1 \\times n}$: non-negative (via sigmoid), compresses streams for layer input\n- $\\mathbf{H}_l^{\\text{post}} \\in \\mathbb{R}^{1 \\times n}$: non-negative (via sigmoid), distributes output to streams\n- $f_l$: standard attention or FFN block operating on $C$-dimensional input\n\nThe learnable parameters added per layer: $n^2 + 2n$ scalars (for $n = 4$: 24 scalars). This is negligible compared to the millions of parameters in $f_l$.\n\n**Training overhead**: Only 6.7% additional wall-clock time despite 4× wider residual stream, achieved through kernel fusion (fusing the $n \\times n$ mixing into adjacent operations) and selective recomputation.\n\n**Results on MoE models (27B parameters)**:\n- BBH (reasoning): +7.2 points over baseline, +2.1 over unconstrained HC\n- GSM8K (math): +4.9 over baseline\n- MMLU (general knowledge): +2.6 over baseline\n- Training stable throughout — no divergence issues observed"
    },
    // Step 10: MC — mHC overhead
    {
      type: "mc",
      question: "mHC expands the residual stream by $n = 4\\times$ but adds only 6.7% training time overhead. How is this possible if the hidden state is 4× larger?",
      options: [
        "The wider residual stream uses FP8 precision, which is 4× cheaper than BF16",
        "The mixing operations are on $n \\times n = 4 \\times 4$ matrices, which are tiny; the expensive operations (attention, FFN) still operate on the original $C$-dimensional input after compression via $\\mathbf{H}^{\\text{pre}}$",
        "Only every 4th layer uses the expanded stream; the rest use standard residual connections",
        "The expansion is applied only to the KV cache, not to the full hidden state"
      ],
      correct: 1,
      explanation: "The key insight is that $\\mathbf{H}^{\\text{pre}}$ compresses the $n$ streams back to 1 before the expensive layer computation ($f_l$). The attention and FFN blocks — which dominate compute — still operate on $C$-dimensional inputs, unchanged. The $n \\times n$ mixing and the $1 \\times n$ compression/expansion are applied per-channel using small learned scalars, adding negligible compute. The 6.7% overhead comes mainly from the 4× larger residual stream occupying more memory bandwidth."
    },
    // Step 11: Info — Connection to depth-wise attention
    {
      type: "info",
      title: "Conceptual Connection: Residuals as Depth-wise Mixing",
      content: "mHC can be understood through a broader lens: **how should layer outputs be aggregated?**\n\n- **Standard residuals**: uniformly-weighted sum → depth-wise \"uniform attention\"\n- **Highway networks**: scalar gates per layer → depth-wise gated mixing\n- **Hyper-Connections**: learned $n \\times n$ matrices → depth-wise linear attention (unconstrained)\n- **mHC**: doubly stochastic matrices → depth-wise linear attention (norm-preserving)\n\nThe evolution parallels the sequence dimension: fixed weights → scalar gates → unconstrained linear → constrained/softmax. Just as softmax attention replaced fixed position weighting in the sequence dimension, learned depth-wise mixing replaces fixed residual connections in the depth dimension.\n\nThe doubly stochastic constraint in mHC plays a role analogous to softmax's normalization in sequence attention: it ensures the mixing weights are non-negative and sum to 1 (along both rows and columns), preventing any single stream from dominating or being silenced.\n\nThis perspective suggests that residual connections — while revolutionary when introduced — are a special case of a more general depth-wise mixing mechanism. As models scale to hundreds of layers, the fixed-weight assumption becomes increasingly restrictive."
    },
    // Step 12: MC — When mHC helps
    {
      type: "mc",
      question: "mHC shows the largest improvements on reasoning benchmarks (BBH +7.2) compared to general knowledge benchmarks (MMLU +2.6). What might explain this pattern?",
      options: [
        "Reasoning tasks require combining information across many layers, which benefits from learned depth-wise mixing that can emphasize the most relevant intermediate representations",
        "MMLU questions are easier and any architecture performs well on them, so there is less room for improvement",
        "The reasoning benchmarks were specifically included in mHC's training data, biasing the results",
        "mHC's expanded residual stream provides extra memory capacity that specifically benefits chain-of-thought reasoning by storing more intermediate states"
      ],
      correct: 0,
      explanation: "Reasoning tasks (multi-step inference, logical deduction) require the model to selectively combine outputs from specific layers — some layers may extract relevant facts while others perform logical operations. Standard residuals force all layers to contribute equally. mHC's learned mixing can upweight the layers whose outputs are most relevant for the current reasoning step. Knowledge benchmarks like MMLU rely more on retrieval from early layers, where the depth-wise mixing matters less."
    },
    // Step 13: MC — Practical considerations
    {
      type: "mc",
      question: "You want to add mHC to an existing 70B transformer model. Which implementation challenge is most significant?",
      options: [
        "The Sinkhorn-Knopp iteration requires 20 iterations per forward pass, adding substantial compute overhead",
        "The 4× wider residual stream requires redesigning every attention and FFN layer to accept $4C$-dimensional input",
        "The expanded residual stream increases activation memory by ~4× during training, which may require adjusting batch size, activation checkpointing, or parallelism strategy",
        "The doubly stochastic constraint prevents the model from learning identity residual connections, forcing every layer to mix streams"
      ],
      correct: 2,
      explanation: "The primary challenge is memory: a 4× wider residual stream means ~4× more activation memory between layers. For a 70B model already near GPU memory limits, this may require reducing batch size or adding activation checkpointing. The Sinkhorn iterations are on tiny $4 \\times 4$ matrices — negligible compute. Attention and FFN layers are unchanged — they receive $C$-dimensional input after $\\mathbf{H}^{\\text{pre}}$ compression. The identity connection IS a doubly stochastic matrix (a permutation), so it's within the constraint set."
    }
  ]
};
