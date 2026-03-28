// Focused learning module: Multi-Latent Attention (MLA) — how DeepSeek-V2/V3
// compresses the KV cache by jointly projecting keys and values into a low-rank
// latent space, achieving MQA-level cache size with MHA-level expressiveness.

export const multiLatentAttentionLearning = {
  id: "B.2-mla-learning-medium",
  sectionId: "B.2",
  title: "Multi-Latent Attention: Low-Rank KV Cache Compression",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Info — The KV cache bottleneck
    {
      type: "info",
      title: "The KV Cache Memory Bottleneck",
      content: "During autoregressive inference, a transformer caches the key and value tensors from all previous tokens to avoid recomputation. This **KV cache** grows linearly with sequence length, batch size, number of layers, and number of heads.\n\nFor a model with $n_h$ heads, head dimension $d_h$, and $L$ layers, the KV cache per token is:\n\n$$\\text{KV cache per token} = 2 \\times n_h \\times d_h \\times L \\times \\text{bytes}$$\n\nFor DeepSeek-V2's scale ($n_h = 128$, $d_h = 128$, $L = 60$, BF16):\n\n$$2 \\times 128 \\times 128 \\times 60 \\times 2 = 3{,}932{,}160 \\text{ bytes} \\approx 3.75 \\text{ MB per token}$$\n\nAt 32K context length: $3.75 \\times 32{,}768 \\approx 120$ GB of KV cache alone — more than one H100's memory.\n\nThe KV cache creates two problems:\n1. **Memory**: Large caches limit batch size and context length\n2. **Bandwidth**: During inference, the KV cache must be read from HBM for every generated token. Reading 120 GB at 3 TB/s takes 40ms — longer than the computation itself. Inference becomes **memory-bandwidth bound**."
    },
    // Step 2: MC — KV cache scaling
    {
      type: "mc",
      question: "A 70B model with 64 heads, $d_h = 128$, and 80 layers uses BF16. What is the KV cache size for a single sequence of 8K tokens?",
      options: [
        "~640 MB — the per-token per-layer cache is $2 \\times d_h \\times 2 = 512$ bytes, summed across 8K tokens and 80 layers",
        "~160 MB — modern inference engines quantize cached keys and values to INT4, compressing the 64-head cache by 4×",
        "~2.5 GB — sliding window attention retains only the nearest 1K tokens per layer, drastically capping total cache size",
        "~20 GB — per token per layer the cache stores $2 \\times 64 \\times 128 \\times 2$ bytes, summed across 8K tokens and 80 layers"
      ],
      correct: 3,
      explanation: "Per token per layer: $2 \\times 64 \\times 128 \\times 2 = 32{,}768$ bytes. Per token all layers: $32{,}768 \\times 80 = 2{,}621{,}440$ bytes $\\approx 2.5$ MB. For 8K tokens: $2.5 \\times 8192 \\approx 20$ GB. This is for a single sequence — batch size 8 would need 160 GB. The KV cache often exceeds the model weights themselves for long contexts."
    },
    // Step 3: Info — Prior solutions: MQA and GQA
    {
      type: "info",
      title: "Prior Solutions: MQA and GQA",
      content: "Two established techniques reduce KV cache by sharing key/value heads:\n\n**Multi-Query Attention (MQA)**: All query heads share a **single** key head and a single value head. Cache reduction: $n_h \\times$ smaller.\n\n$$\\text{MQA cache per token} = 2 \\times 1 \\times d_h \\times L$$\n\n**Grouped-Query Attention (GQA)**: Query heads are divided into $g$ groups, each sharing one key and one value head. Cache reduction: $n_h / g$ times smaller.\n\n$$\\text{GQA cache per token} = 2 \\times g \\times d_h \\times L$$\n\nFor Llama 3.1 70B ($g = 8$, $n_h = 64$): GQA reduces cache by $64/8 = 8\\times$.\n\n**The tradeoff**: MQA and GQA reduce expressiveness. With MQA, every head computes attention using the same keys and values — only the queries differ. This constrains the model's ability to attend to different aspects of the context from different heads. Empirically, MQA degrades quality noticeably; GQA with $g \\geq 4$ is a reasonable compromise but still sacrifices capacity.\n\nThe question is: can we get MQA-level cache size **without** losing MHA-level expressiveness?"
    },
    // Step 4: MC — MQA/GQA tradeoff
    {
      type: "mc",
      question: "GQA with $g = 8$ groups and $n_h = 64$ query heads reduces KV cache by 8×. Each group of 8 query heads shares one K and one V head. What expressiveness is lost compared to full MHA?",
      options: [
        "The 8 query heads in each group are forced to compute identical attention scores over the context, since they share the same keys; only the value mixing differs",
        "No expressiveness is lost — the query heads within a group can still compute distinct attention patterns through their different query projections",
        "Each group can attend to different positions, but all 8 groups must agree on a single attention distribution",
        "GQA limits the model to attending to at most $g = 8$ distinct positions per layer"
      ],
      correct: 0,
      explanation: "Attention scores are $\\mathbf{q}^T \\mathbf{k} / \\sqrt{d}$. With shared keys, the 8 query heads in a group attend to the same key vectors. Despite having different $\\mathbf{q}$ projections, they compute $\\mathbf{q}_i^T \\mathbf{k}$ with the same $\\mathbf{k}$ for all $i$ in the group. The attention distributions are identical within a group because the softmax input is the same. Different groups (with different shared keys) can attend differently, but within each group, the 8 heads are constrained to the same attention pattern. In MHA, all 64 heads would be independent."
    },
    // Step 5: Info — MLA core idea: joint low-rank compression
    {
      type: "info",
      title: "MLA: Joint Low-Rank Compression of K and V",
      content: "Multi-Latent Attention (MLA), introduced in DeepSeek-V2, takes a fundamentally different approach. Instead of sharing heads, it **compresses** K and V jointly into a low-rank latent vector.\n\nFor each token $t$ with hidden state $\\mathbf{h}_t \\in \\mathbb{R}^d$:\n\n$$\\mathbf{c}_t^{KV} = \\mathbf{W}^{DKV} \\mathbf{h}_t \\quad \\in \\mathbb{R}^{d_c}$$\n\nwhere $\\mathbf{W}^{DKV} \\in \\mathbb{R}^{d_c \\times d}$ is the **down-projection** and $d_c \\ll n_h \\cdot d_h$ is the compression dimension.\n\n**Only $\\mathbf{c}_t^{KV}$ is cached** — a single vector of dimension $d_c$ per token per layer.\n\nDuring attention, K and V are **reconstructed** from the latent:\n\n$$\\mathbf{k}_t = \\mathbf{W}^{UK} \\mathbf{c}_t^{KV}, \\quad \\mathbf{v}_t = \\mathbf{W}^{UV} \\mathbf{c}_t^{KV}$$\n\nwhere $\\mathbf{W}^{UK}, \\mathbf{W}^{UV} \\in \\mathbb{R}^{(n_h \\cdot d_h) \\times d_c}$ are **per-head up-projections**.\n\nThe key insight: each head has its own up-projection matrices $\\mathbf{W}^{UK}_i$ and $\\mathbf{W}^{UV}_i$, so **each head reconstructs distinct keys and values** from the shared latent. This preserves MHA-level expressiveness while caching only the compressed latent.\n\nFor DeepSeek-V2: $d_c = 512$ vs. $n_h \\cdot d_h = 128 \\times 128 = 16{,}384$ for standard MHA keys — a **32× compression** for keys alone (and the same latent serves both keys and values)."
    },
    // Step 6: MC — MLA compression
    {
      type: "mc",
      question: "MLA caches a latent vector $\\mathbf{c}_t^{KV} \\in \\mathbb{R}^{d_c}$ per token instead of separate K and V tensors. With $d_c = 512$, $n_h = 128$, $d_h = 128$ in BF16, what is the cache compression ratio compared to standard MHA?",
      options: [
        "4× — because $d_c = 4 d_h$ and MLA still stores both keys and values",
        "32× — because $d_c = 512$ replaces $n_h d_h = 16{,}384$ for keys, but V is stored separately",
        "~57× — because $d_c = 512$ (plus a small RoPE component) replaces $2 n_h d_h = 32{,}768$ for both K and V combined",
        "128× — because each of the 128 heads' keys is replaced by a single shared scalar"
      ],
      correct: 2,
      explanation: "Standard MHA caches $\\mathbf{K} \\in \\mathbb{R}^{n_h \\times d_h}$ and $\\mathbf{V} \\in \\mathbb{R}^{n_h \\times d_h}$ per token: $2 \\times 128 \\times 128 = 32{,}768$ values. MLA caches $\\mathbf{c}_t^{KV} \\in \\mathbb{R}^{512}$ plus a small RoPE key $\\mathbf{k}_t^R \\in \\mathbb{R}^{64}$: total 576 values. Compression ratio: $32{,}768 / 576 \\approx 57\\times$. This is larger than MQA's ratio because MLA compresses both K and V into a single latent."
    },
    // Step 7: Info — Decoupled RoPE
    {
      type: "info",
      title: "Decoupled RoPE: Why Position Needs Special Handling",
      content: "Most modern LLMs use **RoPE (Rotary Positional Embeddings)** to encode position. RoPE applies a position-dependent rotation to queries and keys:\n\n$$\\mathbf{q}_t^{\\text{rope}} = \\mathbf{R}_t \\mathbf{q}_t, \\quad \\mathbf{k}_t^{\\text{rope}} = \\mathbf{R}_t \\mathbf{k}_t$$\n\nwhere $\\mathbf{R}_t$ is a rotation matrix that depends on position $t$.\n\nMLA has a problem with RoPE. Consider the attention score:\n\n$$\\text{score} = (\\mathbf{R}_t \\mathbf{W}^{UQ} \\mathbf{c}_t^Q)^T (\\mathbf{R}_s \\mathbf{W}^{UK} \\mathbf{c}_s^{KV})$$\n\nThe position-dependent rotations $\\mathbf{R}_t$ and $\\mathbf{R}_s$ are sandwiched between the up-projections and the latent vectors, preventing a key optimization called **weight absorption** (explained next).\n\nMLA's solution is **decoupled RoPE**: split each key and query into two parts:\n\n1. **Content component** (no positional encoding): $\\mathbf{q}^C = \\mathbf{W}^{UQ} \\mathbf{c}^Q$, $\\mathbf{k}^C = \\mathbf{W}^{UK} \\mathbf{c}^{KV}$\n2. **Positional component** (with RoPE): $\\mathbf{q}^R = \\text{RoPE}(\\mathbf{W}^{QR} \\mathbf{c}^Q)$, $\\mathbf{k}^R = \\text{RoPE}(\\mathbf{W}^{KR} \\mathbf{h}_t)$\n\nThe final attention score is:\n$$\\text{score} = (\\mathbf{q}^C)^T \\mathbf{k}^C + (\\mathbf{q}^R)^T \\mathbf{k}^R$$\n\nThe RoPE key $\\mathbf{k}^R \\in \\mathbb{R}^{d_h^R}$ (dimension 64) is **shared across all heads** (MQA-style) and cached alongside $\\mathbf{c}^{KV}$. Total cache: $d_c + d_h^R = 512 + 64 = 576$ values per token."
    },
    // Step 8: MC — Decoupled RoPE
    {
      type: "mc",
      question: "MLA decouples position encoding into a separate RoPE component with its own small key vector $\\mathbf{k}^R \\in \\mathbb{R}^{64}$, shared across all heads. Why can't RoPE be applied directly to the reconstructed keys $\\mathbf{k} = \\mathbf{W}^{UK} \\mathbf{c}^{KV}$?",
      options: [
        "RoPE rotations are only defined for specific dimensions (64 or 128), and the reconstructed keys may not match these dimensions",
        "Applying RoPE after up-projection prevents the weight absorption optimization: the position-dependent rotation would prevent folding $\\mathbf{W}^{UK}$ into the query projection",
        "The up-projection $\\mathbf{W}^{UK}$ would amplify the RoPE rotation angles, causing positional encodings to overflow",
        "RoPE requires the key to have the same dimension as the query, which isn't guaranteed after up-projection from the latent space"
      ],
      correct: 1,
      explanation: "Weight absorption (Section 9) folds $\\mathbf{W}^{UK}$ into $\\mathbf{W}^{UQ}$ so that attention operates directly on the latent $\\mathbf{c}^{KV}$ without explicit decompression. If RoPE is applied after up-projection, the product $(\\mathbf{R}_t \\mathbf{W}^{UQ})^T (\\mathbf{R}_s \\mathbf{W}^{UK})$ can't be collapsed into a single matrix because $\\mathbf{R}_t$ and $\\mathbf{R}_s$ depend on different positions. Decoupling RoPE into a separate additive component keeps the content path absorption-friendly."
    },
    // Step 9: Info — Weight absorption
    {
      type: "info",
      title: "Weight Absorption: Operating Directly on Latents",
      content: "The **weight absorption** trick is what makes MLA efficient at inference, not just memory-efficient.\n\nThe content attention score for head $i$ is:\n\n$$(\\mathbf{q}_t^C)_i^T (\\mathbf{k}_s^C)_i = (\\mathbf{W}_i^{UQ} \\mathbf{c}_t^Q)^T (\\mathbf{W}_i^{UK} \\mathbf{c}_s^{KV})$$\n$$= (\\mathbf{c}_t^Q)^T \\underbrace{(\\mathbf{W}_i^{UQ})^T \\mathbf{W}_i^{UK}}_{\\mathbf{W}_i^{QK}} \\mathbf{c}_s^{KV}$$\n\nThe product $\\mathbf{W}_i^{QK} = (\\mathbf{W}_i^{UQ})^T \\mathbf{W}_i^{UK}$ is **independent of the input** — it can be precomputed once.\n\nSimilarly, for the value path, $\\mathbf{W}^{UV}$ is absorbed into the output projection:\n$$\\mathbf{W}_i^{VO} = \\mathbf{W}_i^O \\mathbf{W}_i^{UV}$$\n\nWith weight absorption, the inference computation becomes:\n1. Compute $\\text{score}_i = (\\mathbf{c}_t^Q)^T \\mathbf{W}_i^{QK} \\mathbf{c}_s^{KV} + (\\mathbf{q}_t^R)_i^T \\mathbf{k}_s^R$\n2. Compute output: $\\mathbf{o}_i = \\sum_s \\alpha_{ts} \\mathbf{W}_i^{VO} \\mathbf{c}_s^{KV}$\n\n**Keys and values are never explicitly reconstructed.** The computation operates directly on the cached latent $\\mathbf{c}_s^{KV}$. The up-projection cost is folded into the query/output projections, which operate on the current token only (not on the entire cached context).\n\nThis means MLA's inference is not just memory-efficient — it's also compute-efficient: the per-token matmuls during generation operate on $d_c$-dimensional latents rather than $n_h \\cdot d_h$-dimensional full representations."
    },
    // Step 10: MC — Weight absorption
    {
      type: "mc",
      question: "Weight absorption folds $\\mathbf{W}^{UK}$ into the query projection as $\\mathbf{W}^{QK} = (\\mathbf{W}^{UQ})^T \\mathbf{W}^{UK}$. What is the shape of $\\mathbf{W}^{QK}$ for a single head, and what does it replace?",
      options: [
        "$d_c^Q \\times d_c$ — it maps from query latent to KV latent space, replacing both the query up-projection and the key up-projection",
        "$d_h \\times d_h$ — it maps between per-head query and key spaces, same as standard attention's $\\mathbf{W}^Q$ and $\\mathbf{W}^K$",
        "$d_c \\times d_c$ — it's a square matrix in the KV latent space that replaces explicit key reconstruction",
        "$(n_h \\cdot d_h) \\times d_c$ — it maps from the KV latent to all heads' keys simultaneously"
      ],
      correct: 0,
      explanation: "$\\mathbf{W}_i^{UQ} \\in \\mathbb{R}^{d_h \\times d_c^Q}$ and $\\mathbf{W}_i^{UK} \\in \\mathbb{R}^{d_h \\times d_c}$, so $\\mathbf{W}_i^{QK} = (\\mathbf{W}_i^{UQ})^T \\mathbf{W}_i^{UK} \\in \\mathbb{R}^{d_c^Q \\times d_c}$. It maps from the query latent space to the KV latent space directly, replacing the need to up-project both Q and K to the $d_h$-dimensional head space before computing their dot product. The attention score becomes $\\mathbf{c}^{Q^T} \\mathbf{W}^{QK} \\mathbf{c}^{KV}$ — a bilinear form on the latent vectors."
    },
    // Step 11: Info — MLA vs MHA/MQA/GQA comparison
    {
      type: "info",
      title: "Comparing MLA with MHA, MQA, and GQA",
      content: "Here's how MLA stacks up against alternative attention mechanisms:\n\n| Property | MHA | MQA | GQA ($g$ groups) | MLA ($d_c$) |\n|----------|-----|-----|-----------|-----|\n| KV cache / token / layer | $2 n_h d_h$ | $2 d_h$ | $2 g d_h$ | $d_c + d_h^R$ |\n| Per-head expressiveness | Full | Shared K,V | Shared within group | Full (per-head $\\mathbf{W}^U$) |\n| Inference compute | Standard | Reduced | Reduced | Reduced (latent-space ops) |\n\nWith $n_h = 128$, $d_h = 128$, $d_c = 512$, $d_h^R = 64$:\n- MHA: 32,768 values cached per token per layer\n- MQA: 256 values (128× compression)\n- GQA ($g=8$): 2,048 values (16× compression)\n- **MLA: 576 values (~57× compression)**\n\nMLA achieves compression between MQA and GQA in absolute cache size, but unlike MQA, it preserves **full per-head expressiveness**. Each head reconstructs distinct K and V via its own up-projection — the heads are not constrained to use the same attention pattern.\n\nDeepSeek-V2's experiments confirm that MLA matches or exceeds MHA quality while reducing KV cache by 93.3%, improving generation throughput by 5.76× over their baseline."
    },
    // Step 12: MC — MLA advantage
    {
      type: "mc",
      question: "MLA achieves ~57× KV cache compression while maintaining per-head expressiveness. GQA with $g = 2$ groups achieves 32× compression. Why might MLA be preferred despite GQA's simpler implementation?",
      options: [
        "MLA is always faster because it eliminates the KV cache entirely during inference",
        "GQA cannot be combined with RoPE, while MLA can via decoupled positional encoding",
        "MLA uses less compute during training because the latent projection is cheaper than GQA's grouped projections",
        "GQA with $g = 2$ forces 64 query heads to share each KV head, severely limiting the model's ability to attend to different context features across heads; MLA gives every head independent key/value reconstructions from the shared latent"
      ],
      correct: 3,
      explanation: "With GQA $g = 2$, 64 query heads share each KV head — within each group, all heads compute identical attention patterns (same keys → same softmax). MLA's per-head up-projections ($\\mathbf{W}_i^{UK}$, $\\mathbf{W}_i^{UV}$) reconstruct distinct keys and values for each head from the shared latent, so all 128 heads can attend independently. GQA does work with RoPE, and MLA doesn't eliminate the cache — it compresses it. The key advantage is expressiveness at comparable compression."
    },
    // Step 13: MC — MLA latent dimension
    {
      type: "mc",
      question: "In DeepSeek-V2, the KV latent dimension is $d_c = 512$ while the total K dimension across all heads is $n_h d_h = 16{,}384$. What determines the right choice of $d_c$?",
      options: [
        "It must equal the model's hidden dimension $d$ to preserve the residual stream's information capacity",
        "It must be at least $n_h$ so that each head can reconstruct an independent key from the latent",
        "It's a hyperparameter trading compression against reconstruction quality — too small loses information, too large provides less cache reduction",
        "It must be a power of 2 for efficient GPU memory alignment, and 512 is the largest power of 2 smaller than $d_h = 128$"
      ],
      correct: 2,
      explanation: "$d_c$ controls the information bottleneck. A larger $d_c$ preserves more key/value information but reduces cache savings. A smaller $d_c$ compresses more aggressively but may lose subtle attention patterns. DeepSeek-V2 chose $d_c = 512 = 4 d_h$, found empirically to maintain quality while achieving >50× compression. This is analogous to choosing the rank in LoRA — it's a capacity-efficiency tradeoff determined by ablation."
    }
  ]
};
