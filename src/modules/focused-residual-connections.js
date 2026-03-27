// Focused learning module: Residual Connections and the Residual Stream
// Section 1.1: Transformer Architecture
// Single concept: how residual (skip) connections enable gradient flow
// and create a shared communication bus in transformers.
// Grounded in He et al. (2016) and Anthropic's residual stream framework.

export const residualConnectionsLearning = {
  id: "1.1-residual-connections-learning-easy",
  sectionId: "1.1",
  title: "Residual Connections",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: The depth problem
    {
      type: "info",
      title: "The Depth Problem: Why Deep Networks Are Hard to Train",
      content: "Stacking more layers should make a network more expressive \u2014 deeper networks can represent more complex functions. But in practice, naively stacking layers causes training to fail.\n\nThe core issue is **gradient flow**. During backpropagation, gradients pass through every layer in reverse. In a network with $L$ layers, the gradient of the loss with respect to layer $l$'s parameters involves a product of $L - l$ Jacobian matrices:\n\n$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta_l} \\propto \\prod_{j=l+1}^{L} \\frac{\\partial h_j}{\\partial h_{j-1}}$$\n\nIf these Jacobians consistently have spectral norm $> 1$, gradients **explode** exponentially. If $< 1$, gradients **vanish** exponentially. Either way, early layers receive useless gradient signals.\n\nFor a 96-layer transformer (like GPT-3), gradients must flow through 96 such products. Without mitigation, training is essentially impossible \u2014 early layers would receive gradients that are either astronomically large or indistinguishably close to zero (Goodfellow et al., 2016, \u00a78.2.5)."
    },
    // Step 2: MC \u2014 gradient flow
    {
      type: "mc",
      question: "A 48-layer network without residual connections has layer Jacobians with average spectral norm 0.95. What is the approximate magnitude of the gradient signal reaching layer 1 relative to layer 48?",
      options: [
        "About $0.95 \\times 48 \\approx 46$ \u2014 linear decay with depth, so the gradient is roughly half its original magnitude",
        "About $0.95^{48} \\approx 0.085$ \u2014 the gradient is reduced to about 8.5% of its value at the last layer, making early-layer learning very slow",
        "About $0.95^{48} \\approx 0.085$, but adaptive optimizers like Adam fully compensate for this by normalizing per-parameter gradients, so it has no practical effect",
        "Exactly $0.95$ regardless of depth, because each layer independently scales the gradient and the total effect is determined by the worst single layer"
      ],
      correct: 1,
      explanation: "The gradient magnitude scales as $\\prod_{j} \\|J_j\\| \\approx 0.95^{47} \\approx 0.085$. This is the vanishing gradient problem: even with spectral norms close to 1, the exponential product over many layers causes severe attenuation. Adam helps by adapting per-parameter (dividing by $\\sqrt{v_t}$), but it cannot fully compensate \u2014 if the raw gradient is near zero, Adam's estimate of $v_t$ is also near zero, and the ratio can be noisy or unstable. Residual connections are the structural solution."
    },
    // Step 3: Residual connections
    {
      type: "info",
      title: "Residual Connections: The Identity Shortcut",
      content: "A **residual connection** (He et al., 2016) adds a skip path that bypasses each sub-layer:\n\n$$h_{l+1} = h_l + f_l(h_l)$$\n\ninstead of $h_{l+1} = f_l(h_l)$, where $f_l$ is the sub-layer's computation (attention or FFN).\n\nThis simple change has a profound effect on gradient flow. The Jacobian of the residual block is:\n\n$$\\frac{\\partial h_{l+1}}{\\partial h_l} = I + \\frac{\\partial f_l}{\\partial h_l}$$\n\nThe identity matrix $I$ guarantees that gradients can flow through the skip path unchanged, regardless of what $f_l$ does. Even if $\\frac{\\partial f_l}{\\partial h_l} \\approx 0$ (the sub-layer produces negligible gradients), the gradient still passes through via the identity path.\n\nOver $L$ layers, the gradient includes a **direct path** from the loss back to any layer \u2014 the product of identity matrices, which is just $I$. Each sub-layer can add or subtract from this gradient, but it cannot block it.\n\nIn transformers, there are **two residual connections per layer**: one around the attention sub-layer and one around the FFN sub-layer. A 32-layer transformer thus has 64 residual additions."
    },
    // Step 4: MC \u2014 residual mechanics
    {
      type: "mc",
      question: "In the residual connection $h_{l+1} = h_l + f_l(h_l)$, the sub-layer $f_l$ learns to compute a **correction** to the input $h_l$. At initialization, $f_l$ outputs near-zero values (due to small random weights). What does this mean for the network's behavior at the start of training?",
      options: [
        "The network approximately implements the identity function \u2014 each layer passes its input through nearly unchanged, and the model output is close to a simple projection of the input embeddings",
        "The network is effectively a random function because the small random outputs from each $f_l$ accumulate across layers, creating a random mapping from input to output",
        "The network cannot learn because the near-zero sub-layer outputs produce near-zero gradients, creating a chicken-and-egg problem where the network is stuck at initialization",
        "The network behaves like a single-layer model because only the last sub-layer has non-negligible output, and all earlier layers are effectively bypassed"
      ],
      correct: 0,
      explanation: "With $f_l \\approx 0$ at initialization, $h_{l+1} \\approx h_l + 0 = h_l$. The input representations flow through all layers nearly unchanged \u2014 the deep network starts as an approximate identity function. This is beneficial: the model begins from a well-behaved starting point (identity) and gradually learns layer-by-layer corrections. Without residual connections, the initialized network would apply a product of random transformations, producing chaotic outputs. The residual structure means the network can never be *worse* than identity at initialization."
    },
    // Step 5: Initialization scaling
    {
      type: "info",
      title: "Initialization Scaling: Controlling Residual Growth",
      content: "With residual connections, the output of a deep network is a sum of contributions from all sub-layers:\n\n$$x_{\\text{final}} = x_0 + \\sum_{l=1}^{L} f_l(\\cdot)$$\n\nIf each sub-layer output has variance $\\sigma^2$ at initialization, the total variance of the residual stream grows as $O(L \\sigma^2)$. For a 96-layer transformer with 192 sub-layers, this means activations could be $\\sim\\!14\\times$ their initial magnitude.\n\nThe standard fix (GPT-2, Radford et al., 2019) is to **scale the output projection** of each sub-layer by $1/\\sqrt{2L}$:\n\n$$f_l(x) = \\frac{1}{\\sqrt{2L}} W_l^{\\text{out}} \\cdot \\text{sublayer}_l(x)$$\n\nwhere the factor $2L$ accounts for two sub-layers (attention + FFN) per transformer block across $L$ blocks. This ensures the total variance of the residual stream remains $O(1)$ at initialization, regardless of depth.\n\nSome architectures (e.g., PaLM) use different scaling strategies, but the principle is the same: prevent the sum of residual contributions from blowing up at initialization."
    },
    // Step 6: MC \u2014 initialization scaling
    {
      type: "mc",
      question: "A 64-layer transformer uses the GPT-2 initialization scaling of $1/\\sqrt{2L}$ on output projections. If this scaling were accidentally omitted, what would likely happen during the first few training steps?",
      options: [
        "Training would proceed normally because the optimizer's learning rate already controls the magnitude of each layer's contribution, making the initialization scaling redundant",
        "The model would converge to a worse final loss but training would be stable, because residual connections still guarantee gradient flow regardless of activation magnitude",
        "Only the first layer would be affected since subsequent layers normalize their inputs, preventing any accumulation of magnitude across the network",
        "Activations in later layers would be $\\sim\\!11\\times$ larger than expected, likely causing overflow in float16, NaN losses, and immediate training divergence"
      ],
      correct: 3,
      explanation: "Without the $1/\\sqrt{2L} = 1/\\sqrt{128} \\approx 0.088$ scaling, each sub-layer contributes full-variance outputs. With 128 sub-layers, the residual stream variance grows to $\\sim\\!128\\times$ the per-layer variance, meaning activations are $\\sqrt{128} \\approx 11\\times$ larger than intended. In float16 (max representable value $\\approx 65504$), this can cause overflow, especially after the softmax in attention or in the output logits. Even in float32, the large activations can make the initial loss enormous, producing gradient explosions in the first backward pass."
    },
    // Step 7: The residual stream
    {
      type: "info",
      title: "The Residual Stream: A Shared Communication Bus",
      content: "Anthropic's \"circuits\" framework provides an elegant reinterpretation of residual connections. Instead of thinking of layers as sequential processors, view the $d_{\\text{model}}$-dimensional vector as a **residual stream** \u2014 a shared bus that flows through the entire network.\n\nEach sub-layer **reads** from and **writes** to this stream:\n\n$$x_0 \\xrightarrow{+\\text{attn}_1} x_1 \\xrightarrow{+\\text{ffn}_1} x_2 \\xrightarrow{+\\text{attn}_2} x_3 \\xrightarrow{+\\text{ffn}_2} x_4 \\cdots$$\n\nwhere $x_{2l+1} = x_{2l} + \\text{attn}_l(x_{2l})$ and $x_{2l+2} = x_{2l+1} + \\text{ffn}_l(x_{2l+1})$.\n\nThe output at the final layer is the sum of all contributions:\n\n$$x_{\\text{final}} = x_0 + \\sum_{l=1}^{L} \\text{attn}_l(\\cdot) + \\sum_{l=1}^{L} \\text{ffn}_l(\\cdot)$$\n\nThis means **every sub-layer can directly influence the output** \u2014 not through a chain of intermediaries, but by writing to a stream that is read by the final projection. Sub-layers that are later in the network don't have privileged access; they simply write last.\n\nThis also means sub-layers can communicate: attention in layer 5 can read what the FFN in layer 3 wrote to the stream, enabling multi-step computations composed from independent modules."
    },
    // Step 8: MC \u2014 residual stream reasoning
    {
      type: "mc",
      question: "In the residual stream view, the final output is $x_{\\text{final}} = x_0 + \\sum_l \\text{attn}_l(\\cdot) + \\sum_l \\text{ffn}_l(\\cdot)$. If you could ablate (set to zero) the contribution of a single attention head in layer 10, what would happen?",
      options: [
        "All subsequent layers would receive corrupted input, causing a cascade of failures that completely destroys the model's output quality",
        "Only the direct effect of that head is removed from the final output, but indirect effects (where later layers used that head's output) would also be affected \u2014 both direct and indirect contributions are lost",
        "Nothing would change because individual attention heads have negligible impact \u2014 the model has hundreds of heads and is robust to single-head removal",
        "Only the output of layer 10 changes \u2014 layers 11+ are computed from scratch using the new residual stream and are unaffected by the original head's contribution"
      ],
      correct: 1,
      explanation: "Ablating a head removes two types of contribution: (1) its **direct effect** \u2014 what it wrote to the stream that the final projection reads, and (2) its **indirect effects** \u2014 what later sub-layers read from the stream that included that head's contribution. A head that computes useful features may be relied upon by FFN layers in later blocks. In practice, importance varies enormously: some heads have large direct and indirect effects (e.g., induction heads), while others can be removed with minimal impact. This is the basis for attention head pruning."
    },
    // Step 9: Residual connections enable compositionality
    {
      type: "info",
      title: "Compositionality: How Residual Streams Enable Multi-Step Reasoning",
      content: "The residual stream view reveals why transformers can perform multi-step computations despite each sub-layer being a relatively simple function.\n\nConsider a task like resolving a pronoun: \"The trophy didn't fit in the suitcase because **it** was too big.\" The model must:\n1. Identify that \"it\" is a pronoun needing resolution\n2. Find candidate antecedents (\"trophy\", \"suitcase\")\n3. Use world knowledge to determine which one can be \"too big\"\n\nIn the residual stream framework, different sub-layers handle different steps:\n- An **attention head in an early layer** might mark \"it\" as a pronoun and write a query for its antecedent to the stream\n- An **attention head in a middle layer** reads this query and writes the token identity of the most likely antecedent\n- A **later FFN layer** reads the resolved reference and adjusts the token prediction accordingly\n\nEach sub-layer does a small, composable operation. The residual stream acts as **shared memory** that accumulates the results. Without the additive structure, each layer would need to preserve all previous computations in its output \u2014 a much harder task that compresses the information bottleneck.\n\nThis is why simply making networks deeper (without skip connections) doesn't scale: depth without residual connections forces each layer to be a complete representation, not an incremental contribution."
    },
    // Step 10: MC \u2014 compositionality
    {
      type: "mc",
      question: "Researchers find that ablating attention head 7.3 (layer 7, head 3) in a language model causes a large drop in performance on tasks requiring copying tokens from earlier in the context, but minimal effect on other tasks. Under the residual stream framework, what does this imply?",
      options: [
        "Head 7.3 is an \"induction head\" that writes token-copying information to the stream \u2014 later layers that relied on this information for copy-dependent predictions also lose their ability to contribute, amplifying the effect",
        "Layer 7 is the only layer responsible for copying, and all other layers handle different linguistic functions \u2014 the transformer has a strict division of labor across layers",
        "Head 7.3 must be the largest head in the model by parameter count, since only heads with the most parameters can have task-specific effects when ablated",
        "The model was undertrained \u2014 a fully converged model would have redundant circuits for token copying spread across many heads, making any single-head ablation negligible"
      ],
      correct: 0,
      explanation: "This describes the behavior of an \"induction head\" (Olsson et al., 2022). In the residual stream view, head 7.3 writes a specific type of information (copy signal) to the stream. Later FFN and attention layers read this signal to inform their predictions. The ablation disrupts both the direct contribution and all downstream computations that depended on it. Importantly, this doesn't mean no other heads do copying \u2014 but this particular head's contribution is non-redundant for copy-heavy tasks. Head importance is about function, not size."
    },
    // Step 11: Practical implications
    {
      type: "info",
      title: "Practical Implications for LLM Architecture",
      content: "The residual connection design has several practical consequences for modern LLMs:\n\n**Parallel sub-layers (PaLM, GPT-J)**: Since attention and FFN both read from and write to the same stream, some architectures compute them **in parallel** rather than sequentially:\n\n$$x_{l+1} = x_l + \\text{attn}_l(\\text{LN}(x_l)) + \\text{ffn}_l(\\text{LN}(x_l))$$\n\nThis saves one sequential operation per layer (the FFN doesn't wait for attention to finish), improving training throughput by $\\sim\\!15\\%$ with minimal quality loss. It works because both sub-layers read from the *same* residual stream state \u2014 the FFN doesn't need attention's output.\n\n**Pruning and distillation**: The additive structure means individual heads or even entire layers can be removed if their contributions to the stream are small. Layer pruning (removing entire transformer blocks) exploits the fact that the skip connection preserves the stream even when the sub-layer is deleted.\n\n**Feature visualization**: Researchers can decompose the final output into per-head and per-FFN contributions, attributing predictions to specific sub-layers. This is the basis of logit attribution in mechanistic interpretability."
    },
    // Step 12: MC \u2014 practical implications
    {
      type: "mc",
      question: "PaLM computes attention and FFN in parallel: $x_{l+1} = x_l + \\text{attn}_l(\\text{LN}(x_l)) + \\text{ffn}_l(\\text{LN}(x_l))$. Compared to the standard sequential approach where FFN processes the post-attention residual, what is sacrificed?",
      options: [
        "Gradient flow is severely degraded because the two sub-layers now share the same Jacobian path, halving the effective number of gradient highways through the network",
        "The parallel approach uses twice as much memory because both sub-layers must store separate copies of the input activation for their backward passes",
        "The FFN cannot condition on what attention just wrote \u2014 within the same layer, the FFN and attention operate on identical inputs, losing one step of sequential composition",
        "Nothing is sacrificed \u2014 the parallel formulation is mathematically equivalent to the sequential one because the residual additions are commutative"
      ],
      correct: 2,
      explanation: "In sequential mode, the FFN sees $x_l + \\text{attn}_l(\\text{LN}(x_l))$ \u2014 the residual stream *after* attention has written to it. In parallel mode, the FFN sees only $x_l$ (via LayerNorm) \u2014 the stream *before* attention. This means within-layer attention-to-FFN communication is lost. In practice, this is a minor cost: inter-layer communication (attention in layer $l+1$ reading FFN output from layer $l$) is preserved, and the $\\sim\\!15\\%$ training speedup outweighs the small quality regression observed empirically."
    }
  ]
};
