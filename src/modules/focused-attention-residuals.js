// Focused learning module: Attention Residuals (AttnRes) — replacing fixed
// unit-weight residual connections with learned depth-wise softmax attention
// over preceding layer outputs.

export const attentionResidualsLearning = {
  id: "B.2-attnres-learning-medium",
  sectionId: "B.2",
  title: "Attention Residuals: Learned Depth-wise Aggregation of Layer Outputs",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — PreNorm dilution problem
    {
      type: "info",
      title: "The PreNorm Dilution Problem",
      content: "Modern transformers use **PreNorm** residual connections:\n\n$$\\mathbf{h}_l = \\mathbf{h}_{l-1} + f_l(\\text{RMSNorm}(\\mathbf{h}_{l-1}))$$\n\nUnrolling across $L$ layers:\n\n$$\\mathbf{h}_L = \\mathbf{h}_0 + \\sum_{l=1}^{L} f_l(\\text{RMSNorm}(\\mathbf{h}_{l-1}))$$\n\nTwo issues emerge as $L$ grows:\n\n1. **Norm growth**: $\\|\\mathbf{h}_l\\|$ grows roughly as $O(L)$. Each layer adds to the accumulator.\n\n2. **Dilution**: RMSNorm normalizes $\\mathbf{h}_l$ to unit scale before each layer sees it. But $\\mathbf{h}_l$'s actual norm grows with $l$, so the normalized input is an increasingly small fraction of the accumulated state. For layer $l$ to have a noticeable effect on $\\mathbf{h}_L$, its output $f_l$ must compete against the accumulated sum of all previous outputs — which has norm proportional to $l$.\n\nThe result: **later layers' contributions are diluted** relative to earlier ones. In the final representation $\\mathbf{h}_L$, the embedding $\\mathbf{h}_0$ and early layers dominate simply because they were added first to a small accumulator, not because they are more important.\n\nThis is a fundamental limitation of the fixed unit-weight residual: all layers contribute equally by construction, regardless of whether their output is relevant for the current input."
    },
    // Step 2: MC — PreNorm dilution
    {
      type: "mc",
      question: "In a 100-layer PreNorm transformer, layer 90's output $f_{90}$ is added to $\\mathbf{h}_{89}$, which has accumulated contributions from all 89 previous layers. How does this affect layer 90's relative influence on the final output?",
      options: [
        "Layer 90 has equal influence to layer 1 because RMSNorm normalizes the input to each layer identically",
        "Layer 90 has greater influence because it can access a richer representation that incorporates all previous layers' processing",
        "Layer 90's influence depends only on the magnitude of its output, not on the accumulated norm",
        "Layer 90's output contributes $\\|f_{90}\\| / \\|\\mathbf{h}_{89}\\|$ of the total signal — a decreasing fraction as the accumulated norm grows, regardless of the layer's importance"
      ],
      correct: 3,
      explanation: "Layer 90 adds $f_{90}$ to $\\mathbf{h}_{89}$. Its fractional contribution to $\\|\\mathbf{h}_{90}\\|$ is roughly $\\|f_{90}\\| / \\|\\mathbf{h}_{89}\\|$. Since $\\|\\mathbf{h}_{89}\\|$ grows with the number of accumulated layers, each successive layer's relative contribution shrinks — even if $\\|f_l\\|$ is constant. RMSNorm normalizes the input to each layer but doesn't control the output's relative impact on the growing accumulator."
    },
    // Step 3: Info — Attention Residuals core idea
    {
      type: "info",
      title: "Attention Residuals: Softmax Over Layer Outputs",
      content: "**Attention Residuals (AttnRes)** replaces the fixed unit-weight sum with **learned softmax attention** over all preceding layer outputs.\n\nDefine $\\mathbf{v}_i = f_i(\\cdot)$ as layer $i$'s output (with $\\mathbf{v}_0 = \\mathbf{h}_0$ as the embedding). Instead of:\n\n$$\\mathbf{h}_l = \\sum_{i=0}^{l-1} \\mathbf{v}_i \\quad \\text{(standard residuals)}$$\n\nAttnRes computes:\n\n$$\\mathbf{h}_l = \\sum_{i=0}^{l-1} \\alpha_{i \\to l} \\cdot \\mathbf{v}_i$$\n\nwhere the weights $\\alpha_{i \\to l}$ are computed via attention:\n\n$$\\alpha_{i \\to l} = \\text{softmax}_i\\left(\\mathbf{w}_l^T \\text{RMSNorm}(\\mathbf{v}_i)\\right)$$\n\nHere $\\mathbf{w}_l \\in \\mathbb{R}^d$ is a **learned pseudo-query** for layer $l$, and $\\text{RMSNorm}(\\mathbf{v}_i)$ serves as the key. The softmax is over the depth dimension (across all preceding layers).\n\n**Key design choices**:\n- **Single-head**: Multi-head depth attention was tried but hurts performance. The optimal depth-wise mixture is largely uniform across channels.\n- **Zero initialization**: $\\mathbf{w}_l$ starts at zero, so initial $\\alpha_{i \\to l} = 1/l$ (uniform). The model starts as a standard equal-weight average and gradually learns to specialize.\n- **RMSNorm on keys**: Prevents layers with larger output norms from dominating the softmax purely due to scale."
    },
    // Step 4: MC — AttnRes design
    {
      type: "mc",
      question: "AttnRes initializes the pseudo-queries $\\mathbf{w}_l$ to zero, so initial attention weights are $\\alpha_{i \\to l} = 1/l$ (uniform). Why is this initialization important?",
      options: [
        "It ensures the model starts at a known-good configuration (equal-weight averaging, similar to standard residuals) and gradually deviates only where learned depth-weighting improves the loss",
        "Zero initialization is standard practice for all transformer parameters and has no special significance here",
        "It forces the model to learn depth-wise attention from scratch, preventing interference from pretrained residual connection patterns",
        "Uniform initial weights guarantee that gradients flow equally to all layers during the first training steps, preventing early-layer bias"
      ],
      correct: 0,
      explanation: "With uniform weights, the initial behavior is a running average of all layer outputs — close to (but not identical to) standard residual connections. This makes AttnRes a smooth generalization: it starts at a reasonable baseline and the model only develops non-uniform depth-weighting where it empirically helps. Starting with random or non-uniform weights would destabilize early training. This is analogous to how LoRA initializes one projection to zero to start at the pretrained model."
    },
    // Step 5: Info — Structured matrix perspective
    {
      type: "info",
      title: "A Unified View: Residuals as Depth-wise Attention",
      content: "AttnRes reveals a conceptual unification. The depth-wise aggregation can be written as a matrix $\\mathbf{M} \\in \\mathbb{R}^{L \\times L}$ that maps stacked layer outputs to hidden states:\n\n$$\\begin{bmatrix} \\mathbf{h}_1 \\\\ \\mathbf{h}_2 \\\\ \\vdots \\\\ \\mathbf{h}_L \\end{bmatrix} = \\mathbf{M} \\begin{bmatrix} \\mathbf{v}_0 \\\\ \\mathbf{v}_1 \\\\ \\vdots \\\\ \\mathbf{v}_{L-1} \\end{bmatrix}$$\n\n**Standard residuals**: $\\mathbf{M}$ is a lower-triangular all-ones matrix — every entry below the diagonal is 1. This is equivalent to **depth-wise linear attention with uniform weights**.\n\n**Highway networks**: $\\mathbf{M}$ uses learned scalar gates — depth-wise linear attention with learned but input-independent weights.\n\n**AttnRes**: $\\mathbf{M}$ uses **softmax attention weights** that are input-dependent — each position in the depth dimension attends to all previous positions.\n\nThis mirrors the historical evolution in the sequence dimension:\n- Fixed positional weighting → Learned positional weighting → Softmax attention\n\nAttnRes completes this transition for the depth dimension. The paper argues that fixed residual connections are to depth-wise mixing what bag-of-words is to sequence modeling: a reasonable first approximation that leaves significant performance on the table."
    },
    // Step 6: MC — Structured matrix view
    {
      type: "mc",
      question: "Under the structured matrix view, standard residual connections correspond to a lower-triangular all-ones matrix $\\mathbf{M}$. AttnRes replaces this with an input-dependent softmax attention matrix. What structural constraint does $\\mathbf{M}$ still satisfy in AttnRes?",
      options: [
        "$\\mathbf{M}$ is symmetric, because depth-wise attention is bidirectional",
        "$\\mathbf{M}$ is lower-triangular (causal): layer $l$'s hidden state only depends on outputs from layers $0$ through $l-1$, not future layers",
        "$\\mathbf{M}$ is orthogonal, preserving the norm of the stacked output vector",
        "$\\mathbf{M}$ is doubly stochastic, with both rows and columns summing to 1"
      ],
      correct: 1,
      explanation: "AttnRes is causal in depth: $\\mathbf{h}_l = \\sum_{i=0}^{l-1} \\alpha_{i \\to l} \\mathbf{v}_i$ — layer $l$ only aggregates from layers before it (it can't attend to future layer outputs that haven't been computed yet). The matrix $\\mathbf{M}$ is lower-triangular. Each row sums to 1 (softmax normalization), but columns don't — it's row-stochastic, not doubly stochastic. The weights are input-dependent, unlike the fixed all-ones pattern."
    },
    // Step 7: Info — Depth-wise attention sinks
    {
      type: "info",
      title: "Depth-wise Attention Sinks",
      content: "When analyzing trained AttnRes models, a striking pattern emerges: **certain layers consistently attract high attention weight across all inputs**, regardless of the content being processed.\n\nThis is the depth-wise analogue of **attention sinks** in sequence-wise attention, where early tokens (especially the first token) receive disproportionate attention weight even when semantically irrelevant.\n\nIn depth-wise attention:\n- The **embedding layer** ($\\mathbf{v}_0$) and certain early layers often become sinks\n- Later layers tend to attend heavily to a few specific predecessors\n- The pattern is consistent across different inputs but varies across attention vs. FFN sublayers\n\nThis suggests that the transformer's residual stream has an inherent structure: some layers produce generally useful representations (\"infrastructure layers\" that establish features used by many subsequent layers), while others are more specialized. Standard residual connections can't distinguish these — they weight all layers equally. AttnRes can allocate more weight to the infrastructure layers.\n\nThe attention sink phenomenon also explains why **zero-initialization works**: starting uniform lets the model discover which layers are sinks through gradient descent, rather than baking in potentially wrong assumptions."
    },
    // Step 8: MC — Attention sinks
    {
      type: "mc",
      question: "Trained AttnRes models show \"depth-wise attention sinks\" — certain layers consistently receive high weight. This is analogous to sequence-wise attention sinks on early tokens. What does this imply about the structure of residual streams?",
      options: [
        "Most layers are redundant and could be pruned without quality loss",
        "The sinks are an artifact of the zero initialization and would disappear with random initialization",
        "All layers produce equally important features, but the attention mechanism is biased toward early layers due to softmax saturation",
        "The transformer develops a hierarchical structure where certain layers produce broadly useful features that many later layers depend on, while others contribute more locally"
      ],
      correct: 3,
      explanation: "Depth-wise sinks indicate that some layers produce representations that are broadly useful across many contexts (\"infrastructure\" features), while others contribute more specialized processing that's relevant only to certain layers. This matches interpretability findings that early layers extract basic features (syntax, entity types) used throughout the network, while later layers perform more task-specific reasoning. The sinks persist across different initialization strategies, indicating they reflect genuine network structure."
    },
    // Step 9: Info — Block AttnRes: the practical variant
    {
      type: "info",
      title: "Block AttnRes: Practical Scaling",
      content: "Full AttnRes requires storing all $L$ layer outputs simultaneously ($\\mathbf{v}_0, \\ldots, \\mathbf{v}_{L-1}$), adding $O(L \\cdot d)$ memory per token. For large models with hundreds of layers, this is prohibitive.\n\n**Block AttnRes** partitions the $L$ layers into ~$N$ blocks (typically $N = 8$):\n\n- **Within each block**: Standard residual connections (fixed weight-1 addition)\n- **At block boundaries**: Softmax attention over the $N$ block-level representations\n\nThe algorithm at each block boundary:\n1. Stack the $N$ completed block outputs plus the current partial accumulation\n2. Apply RMSNorm to get keys\n3. Compute attention logits via $\\mathbf{w}_l^T \\text{RMSNorm}(\\mathbf{v}_i)$\n4. Softmax over the depth (block) dimension\n5. Weighted sum produces the new hidden state for the next block\n\n**Why this works**: Block AttnRes confines the uncontrolled norm growth **within each block** (e.g., 8 layers). At block boundaries, the selective aggregation effectively resets the accumulator by choosing a weighted combination of block outputs. The norm can grow within a block but is reset periodically.\n\nMemory overhead: only $N$ block representations stored instead of $L$ layer outputs — typically $8 \\times d$ instead of $60 \\times d$ per token."
    },
    // Step 10: MC — Block AttnRes
    {
      type: "mc",
      question: "Block AttnRes with $N = 8$ blocks in a 64-layer model applies depth-wise attention every 8 layers. Standard residuals grow the hidden state norm as $O(L)$. How does Block AttnRes change this growth pattern?",
      options: [
        "The norm still grows as $O(L)$ but with a smaller constant because the attention weights are less than 1",
        "The norm stays constant because softmax attention preserves the input norm exactly",
        "The norm grows as $O(B)$ within each block of $B = 8$ layers, then is reset at block boundaries by the selective aggregation — limiting growth to $O(B)$ instead of $O(L)$",
        "The norm grows as $O(\\sqrt{L})$ because block boundaries create independent random contributions"
      ],
      correct: 2,
      explanation: "Within each block, standard residuals cause $O(B)$ growth ($B = 8$ layers). At block boundaries, softmax attention produces a convex combination of the block representations (weights sum to 1), which constrains the output norm to the weighted average of inputs — not a sum. This resets the accumulation. The overall growth is bounded by the within-block growth $O(B)$, not the total network depth $O(L = 64)$."
    },
    // Step 11: Info — Performance results
    {
      type: "info",
      title: "Performance: Quality Gains from Depth-wise Attention",
      content: "AttnRes was evaluated on large-scale language models trained by the Kimi/Moonshot team:\n\n**Pretraining results** (measured in loss and downstream benchmarks):\n- AttnRes consistently achieves lower training loss than standard residuals\n- The improvement scales with model depth: deeper models benefit more, consistent with the PreNorm dilution analysis\n- Block AttnRes with $N = 8$ blocks captures most of the full AttnRes benefit\n\n**Key qualitative findings**:\n- AttnRes helps most on tasks requiring **multi-step reasoning** (where information must be combined across many layers)\n- Improvements are consistent across model scales (not just a small-model phenomenon)\n- The learned attention patterns are **interpretable** — you can visualize which layers attend to which predecessors and understand the model's depth-wise information flow\n- Training overhead is minimal: the pseudo-queries add $L \\times d$ parameters (negligible) and the softmax attention over $N = 8$ blocks is cheap\n\nAttnRes demonstrates that even fundamental architectural assumptions (like fixed residual weights) can be questioned and improved. The residual connection was a crucial innovation for training deep networks, but its fixed-weight form is not the final word — learned depth-wise aggregation is a strict generalization."
    },
    // Step 12: MC — AttnRes vs alternatives
    {
      type: "mc",
      question: "AttnRes uses softmax attention over layer outputs, while mHC uses doubly stochastic matrices for stream mixing. How do their approaches to the residual connection problem differ?",
      options: [
        "They solve different problems: AttnRes addresses depth-wise aggregation weighting, while mHC addresses signal stability when using multiple parallel residual streams; both generalize the standard residual but in different dimensions",
        "They are mathematically equivalent — doubly stochastic mixing is a special case of softmax attention",
        "AttnRes is strictly better because softmax attention is more expressive than doubly stochastic constraints",
        "mHC is strictly better because it guarantees norm preservation, while AttnRes can cause gradient explosion"
      ],
      correct: 0,
      explanation: "AttnRes replaces the fixed weight-1 summation with input-dependent softmax weights — the model learns which layers to emphasize for each input. mHC expands the residual stream to multiple parallel channels and constrains their mixing to prevent instability. They address different aspects: AttnRes makes depth-wise mixing input-dependent and selective; mHC makes depth-wise mixing norm-preserving across multiple streams. Both generalize standard residuals, and they could potentially be combined."
    },
    // Step 13: MC — When AttnRes matters
    {
      type: "mc",
      question: "You're deciding whether to add Block AttnRes to your 128-layer LLM. In which scenario would you expect the largest improvement?",
      options: [
        "A 12-layer model where depth is limited and PreNorm dilution is minimal",
        "A 128-layer model evaluated primarily on short-context factual recall tasks where early layers dominate",
        "A 128-layer model evaluated on complex reasoning tasks that require integrating information across many layers, where the ability to selectively weight layer contributions matters most",
        "A 128-layer model with MoE layers, where expert routing already provides input-dependent computation"
      ],
      correct: 2,
      explanation: "AttnRes addresses PreNorm dilution, which worsens with depth. A 128-layer model suffers more than a 12-layer model. Reasoning tasks require combining information across many layers — exactly where learned depth-wise weighting helps. Factual recall relies heavily on early layers and would benefit less from reweighting. While MoE provides input-dependent routing for width, it doesn't address depth-wise aggregation — AttnRes and MoE are complementary."
    }
  ]
};
