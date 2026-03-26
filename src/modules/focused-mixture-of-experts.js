// Focused learning module: Mixture of Experts (MoE)
// Section B.2: Architecture Innovations
// Covers: conditional computation motivation, sparse gating, top-k routing,
// expert specialization, load balancing losses, and routing collapse.
// Grounded in Shazeer et al. (2017), Switch Transformer (Fedus et al., 2022),
// and Mixtral (Jiang et al., 2024).

export const mixtureOfExpertsLearning = {
  id: "B.2-moe-learning-easy",
  sectionId: "B.2",
  title: "Mixture of Experts: Conditional Computation",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Info — Why conditional computation
    {
      type: "info",
      title: "The Scaling Dilemma: Parameters vs. Compute",
      content: "Scaling laws tell us that larger models perform better — but larger models also cost more to train and serve. A 70B dense model requires every parameter to participate in every forward pass, for every token. The compute cost scales linearly with parameter count.\n\n**Conditional computation** breaks this coupling: what if only a fraction of the model's parameters activated for each input? A model with 400B total parameters but only 70B active per token would have the quality of a 400B model with the compute cost of a 70B model.\n\nThis is the core idea behind **Mixture of Experts (MoE)**. Instead of one large FFN in each transformer layer, MoE uses $N$ smaller **expert** FFNs and a **router** that selects which experts process each token. Most experts are skipped for any given token — they exist in memory but don't consume compute.\n\nThe practical impact is dramatic: Mixtral 8x7B has 46.7B total parameters but only ~12.9B active per token (2 of 8 experts active). It matches or exceeds the quality of LLaMA-2 70B at roughly 1/5 the inference compute. The tradeoff: you still need enough memory to store all 46.7B parameters."
    },
    // Step 2: MC
    {
      type: "mc",
      question: "A dense 70B model and an MoE model with 400B total parameters but 70B active per token are both served in float16. How do their memory and compute requirements compare?",
      options: [
        "Both use 140 GB memory and the same FLOPs per token — MoE only helps during training, not inference",
        "The MoE model uses 800 GB memory (all parameters stored) but the same FLOPs per token as the dense model — MoE trades memory for compute efficiency",
        "The MoE model uses ~800 GB memory but ~70B parameters worth of FLOPs per token — it needs more memory to store all experts but uses similar compute since only a subset activates",
        "The MoE model uses 140 GB memory (only active parameters are stored) and 1/5 the FLOPs — inactive experts are paged out to CPU"
      ],
      correct: 2,
      explanation: "All 400B parameters must reside in GPU memory (800 GB in float16) even though only ~70B activate per token. The per-token compute is proportional to the active parameters (~70B), not the total. This is the fundamental MoE tradeoff: you get the quality of a very large model at the compute cost of a smaller one, but the memory requirement reflects the total size. This is why MoE models benefit greatly from quantization and efficient serving infrastructure."
    },
    // Step 3: Info — MoE layer architecture
    {
      type: "info",
      title: "MoE Layer Architecture",
      content: "In a standard transformer, each layer has two sub-layers: multi-head attention and a feed-forward network (FFN). In an MoE transformer, the **FFN is replaced** by a mixture of experts:\n\n$$\\text{MoE}(x) = \\sum_{i=1}^{N} g_i(x) \\cdot E_i(x)$$\n\nwhere:\n- $E_1, \\ldots, E_N$ are $N$ **expert networks** (each is a standard FFN: two linear layers with an activation function)\n- $g(x) = \\text{Router}(x)$ produces **gating weights** $g_i(x) \\geq 0$ with $\\sum_i g_i = 1$ for sparsity\n- In practice, most $g_i(x) = 0$ — the router selects only the **top-$k$** experts\n\nThe attention layers are shared (not duplicated) — they remain dense. This is because attention's role is to mix information across token positions, which benefits from a single shared mechanism. The FFN layers, which transform each token's representation independently, are where capacity can be added via experts.\n\nTypical configurations:\n- **Mixtral 8x7B**: 8 experts, top-2 routing, every other layer is MoE\n- **Switch Transformer**: 128 experts, top-1 routing, every FFN layer\n- **GShard**: 2048 experts distributed across a cluster, top-2 routing"
    },
    // Step 4: MC
    {
      type: "mc",
      question: "In an MoE transformer, why are the attention layers kept dense (shared across all tokens) while only the FFN layers use mixture of experts?",
      options: [
        "Attention layers have too few parameters to benefit from expert routing — the Q, K, V projections are small relative to the FFN",
        "The attention mechanism already performs a form of conditional computation via the attention weights — each token selectively attends to different positions, so adding expert routing would be redundant",
        "Attention layers require cross-token interaction that cannot be split across independent experts, while FFN layers process each token independently and can be naturally parallelized into expert copies",
        "Hardware limitations prevent expert routing in attention — the softmax operation is incompatible with sparse gating"
      ],
      correct: 2,
      explanation: "FFN layers apply the same transformation to each token independently ($\\text{FFN}(x_t)$ for each position $t$). This makes them natural candidates for conditional computation: different tokens can be routed to different expert FFNs without interaction effects. Attention, by contrast, computes interactions between all token positions — splitting it into independent experts would break the cross-token information flow that is attention's purpose. Some recent work explores sparse attention variants, but standard MoE keeps attention dense."
    },
    // Step 5: Info — The router (gating network)
    {
      type: "info",
      title: "The Router: Top-k Sparse Gating",
      content: "The router is a simple linear layer that maps each token's hidden state to expert scores:\n\n$$h(x) = W_r \\cdot x, \\quad W_r \\in \\mathbb{R}^{N \\times d_{\\text{model}}}$$\n\nFor **top-$k$** routing, the $k$ highest-scoring experts are selected and their scores are renormalized:\n\n$$g_i(x) = \\begin{cases} \\frac{e^{h_i(x)}}{\\sum_{j \\in \\text{TopK}} e^{h_j(x)}} & \\text{if } i \\in \\text{TopK}(h(x), k) \\\\ 0 & \\text{otherwise} \\end{cases}$$\n\nThe output is the weighted sum of only the selected experts:\n\n$$\\text{MoE}(x) = \\sum_{i \\in \\text{TopK}} g_i(x) \\cdot E_i(x)$$\n\n**Top-1 routing** (Switch Transformer) is the most compute-efficient — each token goes to exactly one expert. But it puts all eggs in one basket: if the router makes a bad choice, the token gets a poor representation.\n\n**Top-2 routing** (Mixtral, GShard) provides a safety net: even if one expert is suboptimal, the weighted combination with a second expert can compensate. The compute cost is 2x that of a single expert, but the quality improvement is substantial.\n\nThe router's parameter count is tiny: $N \\times d_{\\text{model}}$ (e.g., $8 \\times 4096 = 32{,}768$ parameters). The routing decision is the cheapest part of the MoE computation."
    },
    // Step 6: MC
    {
      type: "mc",
      question: "With top-2 routing across 8 experts, what fraction of the total expert parameters is active for each token?",
      options: [
        "2/8 = 25% — two out of eight experts activate, so a quarter of the expert parameters participate in each token's computation",
        "1/8 = 12.5% — top-2 routing selects 2 experts, but only the highest-scoring one is actually used for computation; the second is used only for the gradient signal",
        "2/8 = 25% of expert parameters, but ~50% of total layer parameters because the shared attention layer's parameters are always fully active",
        "All 8 experts activate at reduced precision — top-2 routing means the top 2 run in full precision while the other 6 run in lower precision to save compute"
      ],
      correct: 0,
      explanation: "With top-2 routing and 8 experts, exactly 2 experts compute their outputs for each token. Since all experts have the same architecture, 2/8 = 25% of the expert parameters are active. Both selected experts run at full precision and their outputs are combined with the router's gating weights. The remaining 6 experts are completely skipped — no computation, no memory bandwidth consumed for their weights (beyond what's needed to store them). This is why MoE is compute-efficient: 75% of the expert parameters don't participate."
    },
    // Step 7: Info — Load balancing
    {
      type: "info",
      title: "The Load Balancing Problem",
      content: "Without intervention, MoE training is unstable because of **routing collapse**: a few experts get most of the tokens while others are starved. This happens through a positive feedback loop:\n\n1. Early in training, random initialization causes some experts to perform slightly better\n2. The router learns to send more tokens to these better experts\n3. These experts get more gradient updates and improve faster\n4. The router sends even more tokens to them\n5. Eventually, 1-2 experts handle all tokens and the rest are never used\n\nThis defeats the purpose of MoE — you pay the memory cost of $N$ experts but get the compute benefit of only 1-2.\n\nThe standard solution is an **auxiliary load balancing loss** added to the training objective:\n\n$$\\mathcal{L}_{\\text{balance}} = \\alpha \\cdot N \\cdot \\sum_{i=1}^{N} f_i \\cdot p_i$$\n\nwhere $f_i$ is the fraction of tokens routed to expert $i$ and $p_i$ is the average router probability for expert $i$. This loss is minimized when tokens are uniformly distributed ($f_i = 1/N$ for all $i$). The coefficient $\\alpha$ (typically 0.01) balances load balancing against the primary language modeling loss.\n\nThe loss encourages uniform token allocation without forcing it — experts can still specialize, but no expert should be completely starved or completely dominant."
    },
    // Step 8: MC
    {
      type: "mc",
      question: "The load balancing loss $\\mathcal{L}_{\\text{balance}} = \\alpha \\cdot N \\cdot \\sum_i f_i \\cdot p_i$ uses the product of token fraction $f_i$ and average probability $p_i$. Why is this product formulation used instead of simply penalizing the variance of $f_i$?",
      options: [
        "Computing the variance of $f_i$ requires tracking per-expert statistics across the entire training corpus, which is infeasible for distributed training",
        "The variance of $f_i$ is not differentiable because $f_i$ depends on discrete top-$k$ selection — the $f_i \\cdot p_i$ product is differentiable through $p_i$ (the soft router probabilities), allowing gradient flow to the router",
        "The product formulation is numerically more stable than variance when the number of experts exceeds 64, preventing underflow in the loss computation",
        "Penalizing variance would force all experts to process exactly the same number of tokens, preventing any specialization — the product formulation allows controlled imbalance"
      ],
      correct: 1,
      explanation: "The core issue is differentiability. The token fraction $f_i$ depends on the discrete top-$k$ selection and is not differentiable with respect to the router parameters. However, $p_i$ (the average soft probability before top-$k$ selection) is differentiable. By multiplying $f_i \\cdot p_i$, the gradient flows through $p_i$: if expert $i$ is receiving too many tokens (high $f_i$), the loss pushes $p_i$ down, which discourages the router from selecting expert $i$. This avoids the need to differentiate through the discrete routing decision."
    },
    // Step 9: Info — Expert specialization
    {
      type: "info",
      title: "Expert Specialization",
      content: "A natural question: do experts learn to specialize in different types of input? The answer is nuanced.\n\nIn **Switch Transformer** (Fedus et al., 2022), analysis showed that experts develop weak but consistent specializations:\n- Some experts preferentially handle tokens from specific domains (code, scientific text)\n- Some experts activate more for particular syntactic roles (punctuation, named entities)\n- But the specialization is **soft** — no expert handles only one type of token\n\nIn **Mixtral** (Jiang et al., 2024), the pattern is similar: experts show domain preferences but not hard boundaries. Importantly, the same token might be routed to different experts in different layers — there's no global \"code expert\" or \"math expert\" across the model.\n\nThe specialization is **emergent**, not designed. The router learns routing patterns that reduce the training loss, and these patterns happen to correlate with semantic categories. But they don't cleanly partition the input space.\n\nOne practical consequence: you can't easily prune specific experts to create a domain-specific model. Removing any expert degrades performance across all domains because every expert participates in some fraction of every domain's tokens."
    },
    // Step 10: MC
    {
      type: "mc",
      question: "A researcher wants to create a code-specific model by identifying the \"code expert\" in an 8-expert MoE model and pruning the other 7 experts. Why is this unlikely to work?",
      options: [
        "Expert specialization is soft and distributed — no single expert handles all code tokens, and each expert processes tokens from multiple domains, so removing 7 experts destroys representations needed for code",
        "MoE models don't develop any specialization — tokens are routed uniformly at random, so all experts are interchangeable and removing any 7 would halve the model's capacity",
        "The routing decisions are made by the attention layers, not the experts themselves, so pruning experts wouldn't affect which tokens go where",
        "Code tokens always use top-1 routing (only one expert), while natural language uses top-2, so the experts are shared across both modalities equally"
      ],
      correct: 0,
      explanation: "Empirical analyses of Mixtral and Switch Transformer consistently show that expert specialization is soft: a code-preferring expert might handle 30% of code tokens but also 15% of general text tokens. Code tokens are distributed across multiple experts, not concentrated in one. Pruning 7 of 8 experts would remove most of the routing destinations for code tokens (and all other tokens), causing catastrophic quality loss across all domains. Domain-specific model extraction from MoE requires more sophisticated techniques like distillation."
    },
    // Step 11: Info — Practical considerations and expert capacity
    {
      type: "info",
      title: "Expert Capacity and Token Dropping",
      content: "In distributed MoE training, each expert lives on a specific GPU. The router sends tokens to experts, but if many tokens route to the same expert, that GPU becomes a bottleneck while others sit idle.\n\nTo prevent this, MoE systems use an **expert capacity factor** $C$: each expert can process at most $C \\cdot (T / N)$ tokens per batch, where $T$ is the total token count and $N$ is the number of experts. Tokens that exceed an expert's capacity are **dropped** — they skip the MoE layer entirely (their representation passes through unchanged, or a residual connection).\n\nTypical capacity factors:\n- $C = 1.0$: no headroom — any imbalance causes drops\n- $C = 1.25$: 25% headroom — tolerates mild imbalance\n- $C = 2.0$: generous headroom — fewer drops but wastes memory\n\nToken dropping creates a tension: the load balancing loss prevents extreme imbalance, but some imbalance is natural and useful (harder tokens may genuinely need specific experts). Setting $C$ too low drops informative routing decisions; setting it too high wastes compute on empty expert capacity.\n\n**Megablocks** (Gale et al., 2023) sidesteps this with block-sparse matrix operations that handle variable-length expert assignments without padding or dropping, improving both efficiency and quality."
    },
    // Step 12: MC
    {
      type: "mc",
      question: "With 8 experts, 1024 tokens per batch, and capacity factor $C = 1.25$, the maximum tokens per expert is $1.25 \\times 1024/8 = 160$. If the router assigns 200 tokens to expert 3, what happens to the excess 40 tokens?",
      options: [
        "They are redistributed to the least-loaded expert, ensuring no compute is wasted",
        "They queue and are processed in a second pass after all other experts finish, adding latency but preserving the routing decision",
        "They are dropped — their representations skip the MoE layer entirely via the residual connection, receiving no expert processing for this layer",
        "The router is re-run with expert 3 masked out, forcing the 40 tokens to choose their next-best expert"
      ],
      correct: 2,
      explanation: "Standard MoE implementations (Switch Transformer, GShard) simply drop overflow tokens. The dropped tokens' representations pass through unchanged via the residual stream — they get the attention sub-layer's output but no FFN transformation for this layer. This is a quality cost: dropped tokens miss one layer of expert processing. The load balancing loss aims to minimize drops, and the capacity factor provides headroom, but some dropping is inevitable during training. This is one motivation for approaches like Megablocks that avoid dropping entirely."
    },
    // Step 13: Info — MoE vs Dense scaling
    {
      type: "info",
      title: "MoE vs. Dense: When and Why",
      content: "MoE models don't strictly dominate dense models — each has advantages:\n\n**MoE advantages**:\n- Better quality per FLOP: Mixtral 8x7B matches LLaMA-2 70B quality at ~5x less compute per token\n- Faster inference (per token): fewer active parameters means faster generation\n- Scalability: can grow parameter count without proportional compute increase\n\n**MoE disadvantages**:\n- **Memory**: all expert parameters must be in memory, even though most are idle per token. The 46.7B total parameters of Mixtral need full storage.\n- **Batch efficiency**: with small batches, different tokens may route to different experts, creating load imbalance and wasted GPU cycles\n- **Communication**: in distributed settings, tokens must be sent to the GPU hosting their assigned expert (all-to-all communication)\n- **Training instability**: routing collapse, token dropping, and sensitivity to the balance loss coefficient $\\alpha$ make training harder to tune\n- **Fine-tuning**: standard fine-tuning can destabilize learned routing patterns; specialized techniques (e.g., expert-level LoRA) are needed\n\nThe trend in frontier models is toward MoE: GPT-4 is widely reported to use MoE, and most leading labs are investing heavily in sparse architectures. The compute savings at scale outweigh the engineering complexity."
    },
    // Step 14: MC
    {
      type: "mc",
      question: "A team is choosing between training a 70B dense model and a Mixtral-style 8x7B MoE model (46.7B total, ~12.9B active). Both will be served on a single 80 GB GPU in float16. Which statement is most accurate?",
      options: [
        "The dense model won't fit (140 GB) but the MoE model will (25.8 GB for active parameters), so MoE is the only viable option for single-GPU serving",
        "Neither model fits — the dense model needs 140 GB and the MoE model needs ~93 GB for all parameters — but the MoE model is closer to fitting and benefits more from 4-bit quantization",
        "Both models fit — the dense model uses 140 GB of virtual memory with GPU offloading, and the MoE model stores inactive experts on CPU",
        "The MoE model requires 8x the memory of the dense model because each expert is a full 7B model, totaling 560 GB"
      ],
      correct: 1,
      explanation: "The dense 70B model needs $70 \\times 10^9 \\times 2 = 140$ GB — won't fit on 80 GB. The MoE 46.7B total needs $46.7 \\times 10^9 \\times 2 \\approx 93$ GB — also doesn't fit, but it's closer. With 4-bit quantization: the dense model drops to ~35 GB (fits), and the MoE model drops to ~23 GB (fits easily with room for KV cache). The key insight: MoE's memory cost is determined by **total** parameters, not active parameters — all experts must be in memory even though most are idle. But the compute per token is proportional to active parameters, so inference is fast once the model fits."
    }
  ]
};
