// Section B.2: Architecture Innovations Assessment

export const architectureAssessment = {
  id: "B.2-assess",
  sectionId: "B.2",
  title: "Assessment: Architecture Innovations",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In a Mixture-of-Experts (MoE) Transformer, the gating network selects a subset of expert FFN layers for each token. The primary computational advantage is:",
      options: ["Total parameters increase but FLOPs per token stay constant — only $k$ of $E$ experts activate per token, decoupling parameter count from compute cost", "The attention layers become cheaper because experts handle most of the computational work that attention would otherwise perform", "Experts share weights with each other, reducing total memory requirements compared to having independent fully-connected layers", "The vocabulary size can be reduced because experts specialize in different token types, enabling each expert to use a smaller subvocabulary"],
      correct: 0,
      explanation: "MoE decouples model capacity (total parameters) from per-token compute. A model with $E$ experts but top-$k$ routing uses $k/E$ of the FFN FLOPs per token. For example, Mixtral 8x7B has ~47B total parameters but uses only ~13B per token (top-2 of 8 experts). The attention layers are unchanged."
    },
    {
      type: "mc",
      question: "Routing collapse in MoE models is a failure mode where:",
      options: [
        "All experts converge to identical weights through gradient averaging, effectively reducing the model to a single expert and wasting the additional parameter capacity",
        "The router assigns nearly all tokens to a small subset of experts, leaving most experts undertrained while overloaded experts become bottlenecks",
        "The gating network outputs uniform probabilities across all experts, ignoring input features and distributing tokens randomly rather than by content",
        "Experts become too narrowly specialized on specific token types and cannot generalize to new domains, causing sharp accuracy drops on distribution shifts"
      ],
      correct: 1,
      explanation: "Routing collapse is a rich-get-richer dynamic: experts that receive more tokens learn faster, causing the router to send even more tokens their way. This leaves most experts undertrained. Load-balancing losses (auxiliary losses that penalize uneven expert utilization) are the standard mitigation, but they introduce a tension between routing quality and load balance."
    },
    {
      type: "mc",
      question: "Standard self-attention computes $\\text{Softmax}(QK^\\top / \\sqrt{d})V$, which is $O(n^2 d)$ in sequence length $n$. Linear attention replaces this with $\\phi(Q)(\\phi(K)^\\top V)$, achieving $O(nd^2)$. Why does linear attention underperform standard attention on retrieval-intensive tasks?",
      options: ["Linear attention cannot represent positional information because the kernel map destroys the ordering of tokens in the sequence", "Linear attention cannot be parallelized during training, making it slower than standard attention despite the lower asymptotic complexity", "Linear attention uses fewer parameters than standard attention, reducing model capacity below what is needed for retrieval tasks", "The kernel feature map $\\phi$ compresses the key-query interaction into a fixed-size matrix $\\phi(K)^\\top V \\in \\mathbb{R}^{d \\times d}$, which cannot store and retrieve arbitrary token-level associations from a long context"],
      correct: 3,
      explanation: "By associating right-to-left, linear attention maintains a $d \\times d$ state matrix that accumulates key-value associations. This is a fixed-size bottleneck regardless of sequence length — it cannot perfectly store $n$ distinct key-value pairs when $n > d$. Standard attention computes each query against all keys explicitly, enabling precise retrieval. This is why hybrid architectures pair linear attention (for efficiency) with some standard attention layers (for retrieval)."
    },
    {
      type: "mc",
      question: "Mamba and S4 are examples of state-space models (SSMs) for sequence modeling. Their key structural property, compared to Transformers, is:",
      options: ["They use convolutions instead of any recurrence, making them purely feedforward with fixed-length receptive fields determined by kernel size", "They replace attention with graph neural networks over token dependency trees, leveraging syntactic structure for efficient context modeling", "They model sequences through a latent continuous-time dynamical system $\\dot{h}(t) = Ah(t) + Bx(t)$, discretized for efficiency, enabling linear-time sequence processing with a fixed-size hidden state", "They use external memory modules to store long-range dependencies, reading and writing to a differentiable memory bank at each processing step"],
      correct: 2,
      explanation: "SSMs are grounded in continuous-time state-space equations: $\\dot{h} = Ah + Bx$, $y = Ch + Dx$. After discretization, these become linear recurrences that can be computed as convolutions during training (parallelizable) or step-by-step during inference (efficient autoregressive generation). Mamba adds input-dependent (selective) gating to the $A$, $B$, $C$ matrices, which is crucial for content-based reasoning."
    },
    {
      type: "mc",
      question: "What distinguishes Mamba's \"selective\" state-space mechanism from the original S4 model?",
      options: ["Mamba makes the SSM parameters ($B$, $C$, and $\\Delta$) input-dependent, allowing the model to selectively filter or retain information based on content rather than using fixed dynamics", "Mamba uses a larger state dimension to store more information per step, compensating for the lack of explicit attention over previous tokens", "Mamba replaces the HiPPO initialization with random initialization, relying on gradient descent to learn the optimal state transition dynamics", "Mamba adds self-attention layers between SSM layers to combine the linear-time efficiency of SSMs with the retrieval capabilities of attention"],
      correct: 0,
      explanation: "S4's $A$, $B$, $C$ matrices are fixed (input-independent) — the same dynamics apply to every input. Mamba makes $B$, $C$, and the discretization step $\\Delta$ functions of the input, enabling content-aware filtering. This selectivity is analogous to gating in LSTMs and is essential for tasks that require ignoring irrelevant tokens. The cost is losing the convolution-mode parallelism of S4, which Mamba compensates with a hardware-aware scan algorithm."
    },
    {
      type: "mc",
      question: "Mixture-of-Depths (MoD) applies a routing mechanism across layers rather than across experts. The core idea is:",
      options: [
        "Different layers use different hidden dimensions to save compute, with narrower layers for simpler transformations and wider layers for complex reasoning",
        "A learned router decides, per token, whether to apply the full Transformer block or skip it (via a residual bypass), allowing easy tokens to exit early and hard tokens to receive more computation",
        "The model has a variable number of layers that changes dynamically during training based on the current loss landscape and gradient statistics",
        "Attention heads at deeper layers are pruned to reduce compute, since empirically the deeper layers contribute less to the final token predictions"
      ],
      correct: 1,
      explanation: "MoD (Raposo et al., 2024) routes each token at each layer: the token either passes through the full block (attention + FFN) or takes a residual skip. This means \"easy\" tokens (e.g., deterministic function words) consume fewer FLOPs than \"hard\" tokens (e.g., content words requiring reasoning). Unlike early exit (which stops processing entirely), MoD allows tokens to re-enter processing at later layers."
    },
    {
      type: "mc",
      question: "The load-balancing auxiliary loss used in MoE training typically takes the form $\\mathcal{L}_{\\text{aux}} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot p_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $p_i$ is the average router probability for expert $i$. Why is this formulation used instead of directly penalizing the variance of $f_i$?",
      options: ["Variance penalization would be too expensive to compute at scale since it requires tracking running statistics across all experts per batch", "The $f_i \\cdot p_i$ formulation also regularizes the expert weights directly, preventing any single expert from developing disproportionately large parameters", "Variance penalization would force all experts to learn identical representations, eliminating the specialization that makes MoE models effective", "The $f_i \\cdot p_i$ product is differentiable with respect to the router's logits (through $p_i$), whereas $f_i$ alone involves a non-differentiable argmax/top-k selection"],
      correct: 3,
      explanation: "The token-to-expert assignment $f_i$ involves discrete top-$k$ selection, which is not differentiable. But the router probability $p_i$ (the softmax output before the discrete decision) is differentiable. By multiplying $f_i \\cdot p_i$, the gradient flows through $p_i$ to update the router, encouraging it to spread probability mass more evenly. Minimizing $\\sum f_i p_i$ is minimized when both fractions and probabilities are uniform ($1/E$ each)."
    },
    {
      type: "mc",
      question: "Hybrid architectures that combine attention with linear recurrences (e.g., Jamba, Griffin) typically interleave the two layer types. The design rationale is:",
      options: ["Attention layers are used only in the first few layers to establish position encoding, while recurrent layers handle the remaining computation", "Recurrent layers are only needed during inference for efficient KV-cache management, while training uses pure attention throughout the network", "Attention provides precise in-context retrieval for tasks that need it, while recurrent layers provide efficient long-range context compression — the hybrid gets both capabilities at lower total cost than pure attention", "The combination enables the model to process images and text simultaneously, with attention handling visual tokens and recurrence handling language"],
      correct: 2,
      explanation: "Pure recurrent models (Mamba, RWKV) struggle with tasks requiring precise recall from long contexts (e.g., \"what was the 3rd item in the list?\") because their fixed-size state compresses information. Pure attention is $O(n^2)$. Hybrids use a few attention layers (often every 4th or 8th layer) to handle retrieval while the recurrent layers efficiently process the majority of the context at $O(n)$ cost."
    },
    {
      type: "mc",
      question: "Early exit strategies in Transformer inference allow the model to produce an output token from an intermediate layer rather than processing through all $L$ layers. A key practical challenge is:",
      options: ["The hidden representations at early layers live in a different space than what the final output head expects, requiring either separate output heads per layer or representation alignment techniques that add training complexity", "Early layers do not have enough parameters to make predictions because the parameter count scales linearly with layer index in standard Transformer designs", "Early exit requires a different tokenizer for each exit point because intermediate representations encode tokens at different granularity levels", "Gradient computation is impossible with early exit during training because the computational graph is truncated at the exit point, blocking backpropagation"],
      correct: 0,
      explanation: "The output projection (unembedding) is trained against the final layer's representations. Intermediate representations may not yet encode the information needed for prediction, or may encode it in a different subspace. Solutions include: training separate lightweight classifiers at each potential exit layer, using shared output heads with representation alignment losses, or \"overthinking\" classifiers that decide when additional layers would not change the prediction."
    },
    {
      type: "mc",
      question: "In an MoE model using top-2 routing with 8 experts, what happens during training if one expert's router logits are consistently 10x larger than the others?",
      options: [
        "The model automatically normalizes the router logits via softmax temperature scaling, preventing any single expert from dominating the selection process",
        "That expert and one other will be selected for nearly all tokens, the remaining 6 experts will receive almost no gradient signal, and model capacity will be severely underutilized",
        "Training will diverge immediately due to gradient explosion in the dominant expert, as the concentrated token flow creates unbounded activation magnitudes",
        "The other experts will quickly catch up because they receive cleaner, less noisy gradients from the tokens that the dominant expert does not process"
      ],
      correct: 1,
      explanation: "With top-2 routing, the dominant expert gets selected for almost every token. The second slot gets competed for among the remaining experts, but with much smaller logits, one or two runners-up will also dominate. The result: 5-6 experts are effectively dead. This is routing collapse. The load-balancing loss combats this by penalizing uneven utilization, but if the coefficient $\\alpha$ is too small, collapse still occurs. If $\\alpha$ is too large, routing quality degrades because load balance overrides content-based routing."
    }
  ]
};
