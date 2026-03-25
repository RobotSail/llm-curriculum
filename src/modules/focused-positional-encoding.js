// Focused learning module: Positional Encoding Schemes
// Section 1.1: Transformer Architecture
// Covers: why position information is needed, sinusoidal encodings,
// learned absolute embeddings, relative position, and RoPE.
// Single-concept module building from first principles.

export const positionalEncodingLearning = {
  id: "1.1-positional-encoding-learning-easy",
  sectionId: "1.1",
  title: "Positional Encoding Schemes",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Why position matters
    {
      type: "info",
      title: "The Permutation Problem",
      content: "Self-attention computes $\\text{Attention}(Q, K, V) = \\text{softmax}(QK^\\top / \\sqrt{d_k})V$. Notice something: this operation treats the input as a **set**, not a sequence. If you permute the input tokens, the attention weights and outputs are permuted in the same way — attention is **permutation-equivariant**.\n\nConcretely, for any permutation $\\pi$:\n$$\\text{Attention}(Q_\\pi, K_\\pi, V_\\pi) = \\text{Attention}(Q, K, V)_\\pi$$\n\nThis means that without any additional signal, a transformer cannot distinguish \"The cat sat on the mat\" from \"mat the on sat cat The\". The order of tokens — which is essential for meaning — is invisible to the attention mechanism.\n\n**Positional encodings** solve this by injecting position information into the token representations before attention processes them. The question is: what is the best way to encode position?"
    },
    // Step 2: MC — understanding the problem
    {
      type: "mc",
      question: "A transformer without any positional encoding processes the inputs \"dog bites man\" and \"man bites dog\". What would happen?",
      options: [
        "The model would produce identical output representations at each position (after accounting for the permutation), because attention scores depend only on content, not position",
        "The model would produce different outputs because the embedding vectors for \"dog\", \"bites\", and \"man\" are different, giving the attention mechanism enough signal to distinguish the two",
        "The model would crash or produce NaN values because the attention mechanism requires position information to compute valid softmax probabilities",
        "The model would produce different outputs because the causal mask breaks the permutation symmetry, forcing a left-to-right ordering"
      ],
      correct: 0,
      explanation: "Without positional encoding, the attention operation is purely content-based. \"dog bites man\" and \"man bites dog\" are permutations of the same set of tokens, so the outputs at each position would be identical (after re-ordering to match the permutation). The model literally cannot distinguish subject from object. Option D is partially right — causal masking does break full permutation symmetry — but the model still cannot tell the difference between positions that see the same set of preceding tokens with different orderings."
    },
    // Step 3: Sinusoidal encodings
    {
      type: "info",
      title: "Sinusoidal Positional Encodings (Vaswani et al., 2017)",
      content: "The original Transformer paper proposed adding fixed sinusoidal functions to the token embeddings. For position $t$ and dimension $i$:\n\n$$PE_{(t, 2i)} = \\sin\\!\\left(\\frac{t}{10000^{2i/d}}\\right), \\quad PE_{(t, 2i+1)} = \\cos\\!\\left(\\frac{t}{10000^{2i/d}}\\right)$$\n\nEach pair of dimensions $(2i, 2i+1)$ traces out a circle at a different **frequency**. Low dimensions oscillate rapidly (high frequency), high dimensions oscillate slowly (low frequency). The geometric spacing of frequencies ($10000^{2i/d}$) creates wavelengths ranging from $2\\pi$ to $10000 \\cdot 2\\pi$.\n\nThe key property: for any fixed offset $k$, there exists a linear transformation $M_k$ such that $PE_{t+k} = M_k \\cdot PE_t$. This means **relative position shifts correspond to linear operations**, which the model can potentially learn to exploit via the $W_Q$ and $W_K$ projections.\n\nThe positional encoding is **added** to the token embedding: $x_t' = x_t + PE_t$. Both vectors live in $\\mathbb{R}^d$, so the model must learn to disentangle positional and content information from their sum."
    },
    // Step 4: MC — sinusoidal properties
    {
      type: "mc",
      question: "Sinusoidal encodings use frequencies that span several orders of magnitude (wavelengths from $2\\pi$ to $\\sim 10000 \\cdot 2\\pi$). Why is this range important?",
      options: [
        "High-frequency components encode the exact position of each token, while low-frequency components encode which document in the batch the token belongs to",
        "High frequencies help the model learn syntax (which operates locally) while low frequencies help with semantics (which operates over long distances), creating a natural separation of linguistic concerns",
        "All frequencies are needed to make each position's encoding unique via the uniqueness theorem for trigonometric polynomials, but the specific frequency values don't matter",
        "The multi-scale frequency spectrum allows the model to represent both fine-grained local position differences (nearby tokens) and coarse-grained global position (where in the sequence), analogous to a binary number system with different bit significances"
      ],
      correct: 3,
      explanation: "The multi-frequency design works like a binary position encoding with smooth transitions. High-frequency dimensions change rapidly between adjacent positions, letting the model distinguish tokens 1 apart. Low-frequency dimensions change slowly, letting the model determine roughly \"how far into the sequence\" a token is. This is analogous to the digits of a number: the ones digit (high frequency) changes every step, the tens digit (lower frequency) changes every 10 steps, etc. Without low frequencies, distant positions would appear identical; without high frequencies, nearby positions would be indistinguishable."
    },
    // Step 5: Learned absolute embeddings
    {
      type: "info",
      title: "Learned Absolute Positional Embeddings",
      content: "An alternative, used in GPT-2 and BERT, is to simply **learn** a separate embedding vector for each position:\n\n$$x_t' = x_t + E_t, \\quad E \\in \\mathbb{R}^{T_{\\max} \\times d}$$\n\nwhere $E_t$ is a learned vector for position $t$, and $T_{\\max}$ is the maximum sequence length supported (e.g., 1,024 for GPT-2, 512 for BERT).\n\n**Advantages**:\n- Simple to implement — just an extra embedding table\n- The model can learn arbitrary position-dependent patterns\n- No assumptions about what positional information is useful\n\n**Disadvantages**:\n- **Fixed maximum length**: The model cannot process sequences longer than $T_{\\max}$ because there are no learned embeddings for positions beyond it\n- **No length generalization**: Position 1,025 is completely out-of-distribution, even if the model handles position 1,024 perfectly\n- **Absolute, not relative**: The model must learn separately that \"token 5 attending to token 3\" and \"token 105 attending to token 103\" are both \"attending 2 positions back\" — this pattern isn't built into the representation"
    },
    // Step 6: MC — absolute vs relative
    {
      type: "mc",
      question: "A model trained with learned absolute positional embeddings ($T_{\\max} = 2048$) is given a 4096-token input at inference time using a simple extrapolation (repeating the embedding table). What is most likely to happen?",
      options: [
        "The model works perfectly because the learned embeddings capture general positional patterns that transfer to longer sequences",
        "The model produces garbage for positions 2049+ because it encounters embedding vectors associated with positions 0-2047 reused out of context, creating severe position confusion",
        "The model ignores the positional embeddings entirely and relies on causal masking alone, producing coherent but position-insensitive output",
        "The model works well for positions 2049-4096 but fails for positions 0-2048 because the repeated embeddings interfere with the original encodings"
      ],
      correct: 1,
      explanation: "Repeating the embedding table means position 2049 gets the same positional embedding as position 1. The model has learned that this embedding means \"first token in the sequence\" and will treat it accordingly — attention patterns, learned positional biases, and everything downstream will behave as if the sequence restarted. The model has no mechanism to understand that position 2049 is different from position 1. This is the fundamental limitation of absolute position encodings and the key motivation for relative position methods."
    },
    // Step 7: Relative position — the key insight
    {
      type: "info",
      title: "The Relative Position Insight",
      content: "In natural language, most syntactic and semantic relationships depend on **relative** position, not absolute position. The relationship between a verb and its subject doesn't change whether they appear at positions (3, 5) or (103, 105) — what matters is that the subject is 2 positions before the verb.\n\nRelative position methods modify the attention computation so that the attention score between positions $m$ and $n$ depends on the offset $(m - n)$ rather than on $m$ and $n$ individually:\n\n$$\\text{score}(m, n) = f(q_m, k_n, m - n)$$\n\nrather than:\n\n$$\\text{score}(m, n) = f(q_m + PE_m, k_n + PE_n)$$\n\nThis has a crucial benefit for **length generalization**: the model may encounter absolute position 10,000 for the first time, but it has seen the relative offset \"3 positions back\" countless times during training. Relative position representations are inherently reusable across sequence lengths.\n\nSeveral approaches implement this idea differently: T5's relative position bias adds learned scalar biases per offset, ALiBi adds a fixed linear penalty proportional to distance, and RoPE (Rotary Position Embeddings) encodes relative position through rotations in the complex plane."
    },
    // Step 8: MC — relative position reasoning
    {
      type: "mc",
      question: "ALiBi (Press et al., 2022) adds a fixed bias $-\\alpha \\cdot |m - n|$ to the attention logit between positions $m$ and $n$, where $\\alpha > 0$ is a per-head constant. What is the effect of this bias on the attention distribution?",
      options: [
        "It creates a soft distance penalty that exponentially decays attention to distant tokens after softmax, encouraging each position to attend more to nearby tokens with a head-specific decay rate",
        "It makes all attention weights exactly equal by canceling out content-based similarity scores, forcing uniform attention regardless of the query-key dot product",
        "It inverts the typical attention pattern so that distant tokens receive higher attention than nearby tokens, forcing the model to specialize in long-range dependencies",
        "It clips the attention to a fixed window of size $1/\\alpha$, making tokens beyond that window receive exactly zero attention weight"
      ],
      correct: 0,
      explanation: "The linear penalty $-\\alpha|m-n|$ in the logit becomes an exponential decay $e^{-\\alpha|m-n|}$ after softmax (since softmax exponentiates its inputs). Different heads use different $\\alpha$ values, so some heads have sharp local attention (large $\\alpha$, fast decay) and others have broad global attention (small $\\alpha$, slow decay). This is a simple inductive bias: nearby context is usually more relevant, but the model can override it through strong content-based scores. The fixed (non-learned) nature of ALiBi makes it naturally extrapolate to longer sequences."
    },
    // Step 9: RoPE — rotary embeddings
    {
      type: "info",
      title: "RoPE: Rotary Position Embeddings",
      content: "RoPE (Su et al., 2021) is now the dominant positional encoding for modern LLMs (LLaMA, Mistral, Qwen, etc.). Its core idea: encode position by **rotating** the query and key vectors, so that their dot product naturally depends on relative position.\n\nPair up dimensions and treat each pair $(x_{2i}, x_{2i+1})$ as a 2D vector (or equivalently, a complex number $x_{2i} + ix_{2i+1}$). For position $m$, multiply by a rotation:\n\n$$\\tilde{q}_m^{(i)} = q_m^{(i)} \\cdot e^{im\\theta_i}$$\n\nwhere $\\theta_i = 10000^{-2i/d}$ (same frequency spacing as sinusoidal encodings).\n\nThe key mathematical property: the dot product between the rotated query at position $m$ and rotated key at position $n$ depends only on the relative offset:\n\n$$\\langle \\tilde{q}_m^{(i)}, \\tilde{k}_n^{(i)} \\rangle = \\text{Re}\\left[q_m^{(i)} \\cdot \\overline{k_n^{(i)}} \\cdot e^{i(m-n)\\theta_i}\\right]$$\n\nThe factor $e^{i(m-n)\\theta_i}$ depends only on the offset $(m-n)$, not on $m$ or $n$ individually. The model gets relative position information for free through the geometry of rotations.\n\nUnlike additive encodings, RoPE modifies Q and K **multiplicatively** — it rotates the vectors rather than shifting them. This preserves the norm of the vectors while encoding position purely in their angle."
    },
    // Step 10: MC — RoPE mechanics
    {
      type: "mc",
      question: "RoPE encodes position by rotating query/key vectors in 2D subspaces using frequencies $\\theta_i = 10000^{-2i/d}$. What happens to the attention score between two tokens as their distance $|m - n|$ increases?",
      options: [
        "The attention score monotonically decreases because each rotation adds a fixed angular displacement, eventually rotating the vectors to be orthogonal at sufficiently large distances",
        "The attention score remains exactly the same regardless of distance because RoPE preserves the inner product magnitude — only the phase changes, not the amplitude",
        "The attention score oscillates as the rotation angle $(m-n)\\theta_i$ cycles through $0$ to $2\\pi$ for each frequency, with no guaranteed monotonic decay — the model must learn to handle these oscillations through the Q/K projections",
        "The attention score increases with distance because the rotation accumulates constructively across all frequency components, creating a long-range amplification effect"
      ],
      correct: 2,
      explanation: "Each 2D subspace contributes a cosine-modulated term: the dot product includes $\\cos((m-n)\\theta_i)$, which oscillates as distance grows. There is no built-in monotonic decay as in ALiBi. The different frequencies mean that short-distance oscillations cycle rapidly while long-distance oscillations are slow — the model learns (via $W_Q$ and $W_K$) to combine these oscillating signals to extract useful distance information. In practice, the superposition of many frequencies at different scales provides a rich distance signal, but RoPE can struggle with very long distances without modifications like NTK-aware scaling or YaRN."
    },
    // Step 11: Comparison and modern practice
    {
      type: "info",
      title: "Comparison: Which Encoding Wins?",
      content: "The landscape of positional encodings in modern LLMs has converged:\n\n| Method | Type | Length Generalization | Used In |\n|---|---|---|---|\n| Sinusoidal | Absolute, fixed | Limited | Original Transformer |\n| Learned absolute | Absolute, learned | None beyond $T_{\\max}$ | GPT-2, BERT |\n| T5 relative bias | Relative, learned | Moderate | T5, some early models |\n| ALiBi | Relative, fixed | Good | BLOOM, MPT |\n| **RoPE** | **Relative, fixed** | **Good (with scaling)** | **LLaMA, Mistral, GPT-NeoX, Qwen** |\n\n**RoPE has become the de facto standard** for autoregressive LLMs, for several reasons:\n\n1. **Efficient implementation**: The rotation is a cheap element-wise operation, adding negligible compute\n2. **Relative position**: Attention scores depend on offset, not absolute position\n3. **Extensible**: Context length can be extended beyond training length using techniques like NTK-aware scaling, YaRN, or simply fine-tuning with longer sequences and an adjusted $\\theta$ base\n4. **Compatible with KV caching**: Rotations are applied before caching, so cached keys already have position information baked in\n\nThe shift from absolute to relative position encodings reflects a broader lesson: **inductive biases that match the structure of the problem** (language depends on relative, not absolute position) enable better generalization."
    },
    // Step 12: MC — integration question
    {
      type: "mc",
      question: "A team is designing a new LLM that must handle 128K-token contexts, but training is only feasible on 8K-token sequences due to compute constraints. Which positional encoding strategy is most appropriate?",
      options: [
        "Learned absolute embeddings with $T_{\\max} = 128K$ — the extra embedding parameters are negligible relative to the model size, and positions will be well-trained from the 8K data",
        "Sinusoidal encodings — their mathematical structure guarantees perfect extrapolation to any length, as the sine and cosine functions are defined for all positive integers",
        "No positional encoding at all — the causal mask alone provides sufficient position information for autoregressive models, and omitting position encodings eliminates the length generalization problem entirely",
        "RoPE with a plan to extend context after pretraining via $\\theta$-base scaling (e.g., NTK-aware or YaRN), since RoPE's relative position representation provides a foundation for length extension while training on 8K sequences"
      ],
      correct: 3,
      explanation: "RoPE is the right choice because: (1) it encodes relative position, so the model learns patterns that transfer across lengths; (2) context extension techniques like NTK-aware scaling (adjusting the $\\theta$ base to spread the same rotation range over more positions) or YaRN allow post-training extension from 8K to 128K with minimal fine-tuning. Learned absolute embeddings (option A) would have positions 8K-128K completely untrained. Sinusoidal (option B) extrapolates in principle but underperforms in practice. No encoding (option C) severely limits the model since the causal mask only provides a partial ordering, not a metric of distance."
    }
  ]
};
