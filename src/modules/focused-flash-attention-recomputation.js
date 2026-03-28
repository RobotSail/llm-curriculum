// Focused learning module: Recomputation in FlashAttention — trading
// compute for memory by recomputing the attention matrix in the backward
// pass instead of storing it from the forward pass.

export const flashAttentionRecomputationLearning = {
  id: "G.3-fa-recomputation-learning-medium",
  sectionId: "G.3",
  title: "FlashAttention Recomputation: Trading FLOPs for Memory in the Backward Pass",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — The backward pass problem
    {
      type: "info",
      title: "The Backward Pass Memory Problem",
      content: "In the forward pass, FlashAttention avoids materializing the $N \\times N$ attention matrix $\\mathbf{P}$ by computing attention in tiles. But standard backpropagation through attention **needs** $\\mathbf{P}$ to compute gradients.\n\nRecall the attention computation:\n$$\\mathbf{S} = \\mathbf{Q}\\mathbf{K}^T / \\sqrt{d}, \\quad \\mathbf{P} = \\text{softmax}(\\mathbf{S}), \\quad \\mathbf{O} = \\mathbf{P}\\mathbf{V}$$\n\nThe backward pass computes gradients $d\\mathbf{Q}, d\\mathbf{K}, d\\mathbf{V}$ given the output gradient $d\\mathbf{O}$:\n$$d\\mathbf{V} = \\mathbf{P}^T d\\mathbf{O}$$\n$$d\\mathbf{P} = d\\mathbf{O} \\mathbf{V}^T$$\n$$d\\mathbf{S} = \\text{dsoftmax}(d\\mathbf{P}, \\mathbf{P})$$\n$$d\\mathbf{Q} = d\\mathbf{S} \\cdot \\mathbf{K} / \\sqrt{d}, \\quad d\\mathbf{K} = d\\mathbf{S}^T \\cdot \\mathbf{Q} / \\sqrt{d}$$\n\nEvery term above involves $\\mathbf{P}$ or $\\mathbf{S}$, which are $N \\times N$. Standard implementations save $\\mathbf{P}$ from the forward pass (in HBM) for use in the backward pass. For sequence length 8K with 32 heads, that's $32 \\times 8192^2 \\times 2$ bytes $\\approx 4$ GB of saved activations."
    },
    // Step 2: MC — Why save P
    {
      type: "mc",
      question: "In standard attention training, the forward pass saves $\\mathbf{P} = \\text{softmax}(\\mathbf{QK}^T / \\sqrt{d})$ for use in the backward pass. What is the memory cost of storing $\\mathbf{P}$ for a model with 32 attention heads, sequence length $N = 4096$, in FP16?",
      options: [
        "32 MB — each head stores an $N \\times d$ matrix at $4096 \\times 128 \\times 2$ bytes",
        "1 GB — each head stores an $N \\times N$ matrix at $4096^2 \\times 2$ bytes, times 32 heads",
        "128 MB — only one head's attention matrix is stored at a time and reused",
        "8 GB — the matrix is stored in FP32 for gradient precision, doubling the cost"
      ],
      correct: 1,
      explanation: "$\\mathbf{P}$ has shape $N \\times N$ per head. Total storage: $32 \\times 4096^2 \\times 2$ bytes $= 32 \\times 16{,}777{,}216 \\times 2 \\approx 1.07$ GB. This is per-layer — a 32-layer transformer stores ~34 GB of attention matrices. This $O(N^2)$ memory cost is what limits sequence length during training, and it's what FlashAttention's recomputation strategy eliminates."
    },
    // Step 3: Info — Activation checkpointing background
    {
      type: "info",
      title: "Activation Checkpointing: The Precedent",
      content: "FlashAttention's recomputation strategy is related to **activation checkpointing** (also called gradient checkpointing), a well-established technique in deep learning.\n\nThe idea: instead of saving all intermediate activations during the forward pass, save only a subset of **checkpoints**. During the backward pass, recompute the missing activations from the nearest checkpoint.\n\n**Standard training**: save all activations → $O(L)$ memory for $L$ layers, one backward pass.\n\n**Checkpointed training**: save activations every $k$ layers → $O(L/k)$ memory, but recompute $k-1$ layers' activations during backward → extra compute.\n\nThe tradeoff: **more FLOPs** (roughly 33% overhead for full checkpointing) in exchange for **less memory**. This is worthwhile because:\n1. Memory is often the binding constraint for batch size and sequence length\n2. The recomputation overlaps with other operations, partially hiding its cost\n3. Training larger models or using longer sequences can improve quality enough to justify the overhead\n\nFlashAttention applies this principle specifically to the attention matrix — but with an important twist."
    },
    // Step 4: MC — Checkpointing tradeoff
    {
      type: "mc",
      question: "Activation checkpointing saves memory by recomputing activations during the backward pass. If full checkpointing is applied to every layer, roughly how much extra forward-pass computation does this add?",
      options: [
        "33% — one additional forward pass is needed, but only segments between checkpoints are recomputed rather than the full network",
        "100% — the entire forward pass is repeated, doubling total compute",
        "0% — recomputation happens on idle GPU cores that would otherwise be unused during backward",
        "50% — exactly half the forward pass must be recomputed on average"
      ],
      correct: 0,
      explanation: "With standard checkpointing, you do one complete forward pass, then during backward you recompute each segment's forward activations. This adds roughly one extra forward pass worth of compute. Since a standard training step is ~1 forward + ~2 backward (backward ≈ 2× forward FLOPs), the overhead is roughly 1/(1+2) ≈ 33% of total training FLOPs. In practice, the overhead can be hidden by pipelining."
    },
    // Step 5: Info — FlashAttention's recomputation approach
    {
      type: "info",
      title: "FlashAttention's Recomputation: Cheaper Than You'd Think",
      content: "FlashAttention saves only $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$ (each $N \\times d$) and the softmax normalization statistics $m, \\ell$ (each $N \\times 1$) from the forward pass. It does **not** save the $N \\times N$ attention matrix $\\mathbf{P}$.\n\nDuring the backward pass, it **recomputes** $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T$ and $\\mathbf{P}_{ij} = \\text{softmax}(\\mathbf{S}_{ij})$ block by block, using the saved $m$ and $\\ell$ for correct normalization. The recomputed tiles never leave SRAM.\n\nMemory savings:\n- Standard attention stores $\\mathbf{P}$: $O(N^2)$ per head per layer\n- FlashAttention stores $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}, m, \\ell$: $O(Nd)$ per head per layer\n\nFor $N = 8192$ and $d = 128$: standard stores $8192^2 \\approx 67M$ values; FlashAttention stores $3 \\times 8192 \\times 128 + 2 \\times 8192 \\approx 3.2M$ values — a **20× reduction**.\n\nThe extra FLOPs from recomputation are small relative to the total training FLOPs, and because the recomputation happens in SRAM (fast), the wall-clock overhead is often negligible. You trade $O(N^2)$ slow HBM memory for a modest amount of fast SRAM compute."
    },
    // Step 6: MC — What FlashAttention saves
    {
      type: "mc",
      question: "FlashAttention's forward pass saves $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$ and the softmax statistics $m$ (row maxima) and $\\ell$ (row sums). Why are $m$ and $\\ell$ sufficient to correctly recompute the softmax during the backward pass?",
      options: [
        "They allow reconstructing the full attention matrix $\\mathbf{P}$ by applying $P_{ij} = e^{S_{ij} - m_i} / \\ell_i$ to any recomputed score $S_{ij}$",
        "They encode a compressed approximation of $\\mathbf{P}$ that is accurate enough for gradient computation",
        "They store the top-$k$ attention weights per row, which account for most of the gradient signal",
        "They record which blocks were processed in which order, allowing the tiling pattern to be replayed"
      ],
      correct: 0,
      explanation: "Given $\\mathbf{Q}$ and $\\mathbf{K}$, any block $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T$ can be recomputed. The saved statistics $m_i$ (global row max) and $\\ell_i$ (global softmax denominator) convert any score $S_{ij}$ to its exact softmax probability: $P_{ij} = e^{S_{ij} - m_i} / \\ell_i$. This is exact, not approximate — the same values as if the full attention matrix had been computed and softmaxed."
    },
    // Step 7: Info — The backward pass algorithm
    {
      type: "info",
      title: "FlashAttention Backward Pass Algorithm",
      content: "The backward pass mirrors the forward pass's tiling structure. Given output gradient $d\\mathbf{O}$ and saved values $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}, \\mathbf{O}, m, \\ell$:\n\nFirst, compute a useful intermediate per query position:\n$$D_i = \\sum_j O_{ij} \\cdot dO_{ij} \\quad \\text{(row-wise dot product of } \\mathbf{O} \\text{ and } d\\mathbf{O}\\text{)}$$\n\nThen, in tiled loops over blocks of Q (indexed by $i$) and K/V (indexed by $j$):\n\n1. **Recompute** $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T / \\sqrt{d}$ in SRAM\n2. **Recompute** $\\mathbf{P}_{ij} = \\text{diag}(\\ell_i)^{-1} \\cdot e^{\\mathbf{S}_{ij} - m_i}$ in SRAM\n3. **Compute** local gradients:\n   - $d\\mathbf{V}_j \\mathrel{+}= \\mathbf{P}_{ij}^T \\cdot d\\mathbf{O}_i$\n   - $d\\mathbf{P}_{ij} = d\\mathbf{O}_i \\cdot \\mathbf{V}_j^T$\n   - $d\\mathbf{S}_{ij} = \\mathbf{P}_{ij} \\odot (d\\mathbf{P}_{ij} - D_i)$ (softmax Jacobian)\n   - $d\\mathbf{Q}_i \\mathrel{+}= d\\mathbf{S}_{ij} \\cdot \\mathbf{K}_j / \\sqrt{d}$\n   - $d\\mathbf{K}_j \\mathrel{+}= d\\mathbf{S}_{ij}^T \\cdot \\mathbf{Q}_i / \\sqrt{d}$\n\nAll intermediate matrices ($\\mathbf{S}_{ij}$, $\\mathbf{P}_{ij}$, $d\\mathbf{P}_{ij}$, $d\\mathbf{S}_{ij}$) live in SRAM. Only the gradients $d\\mathbf{Q}, d\\mathbf{K}, d\\mathbf{V}$ accumulate in HBM."
    },
    // Step 8: MC — Softmax backward
    {
      type: "mc",
      question: "The softmax backward formula uses $d\\mathbf{S}_{ij} = \\mathbf{P}_{ij} \\odot (d\\mathbf{P}_{ij} - D_i)$ where $D_i = \\text{rowsum}(\\mathbf{O}_i \\odot d\\mathbf{O}_i)$. Why is the $D_i$ term needed?",
      options: [
        "It provides numerical stability by centering the gradients, similar to how subtracting the max stabilizes the forward softmax",
        "It compensates for dropout applied to the attention weights during training",
        "It corrects for the online softmax approximation error accumulated during the forward pass",
        "It accounts for the fact that softmax outputs sum to 1 — changing one logit affects all probabilities in the row, and $D_i$ captures this coupling"
      ],
      correct: 3,
      explanation: "The softmax Jacobian is $\\frac{\\partial P_j}{\\partial S_k} = P_j(\\delta_{jk} - P_k)$. Because all probabilities sum to 1, increasing logit $k$ increases $P_k$ but decreases all other $P_j$ — the outputs are coupled. The $D_i = \\sum_j P_{ij} \\cdot dL/dP_{ij} = \\sum_j O_{ij} \\cdot dO_{ij}$ term captures this row-wise coupling. Without it, the gradient would ignore the constraint that attention weights sum to 1."
    },
    // Step 9: Info — Memory vs compute tradeoff analysis
    {
      type: "info",
      title: "Quantifying the Tradeoff",
      content: "Let's quantify what FlashAttention's recomputation costs and saves.\n\n**Memory saved per layer** (for one attention head, sequence length $N$, head dim $d$):\n- Eliminated: $\\mathbf{P} \\in \\mathbb{R}^{N \\times N}$ → $N^2$ elements\n- Added: $m, \\ell \\in \\mathbb{R}^N$ → $2N$ elements\n- Net savings: $N^2 - 2N \\approx N^2$ elements\n\nFor $N = 8192$: saving ~134 MB per head per layer (FP16). With 32 heads and 32 layers: **~137 GB saved**. This is the difference between fitting training in memory or not.\n\n**Extra FLOPs from recomputation**:\n- Recomputing $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T$: $O(N^2 d)$ — same as forward\n- Recomputing $\\mathbf{P}_{ij}$ from $\\mathbf{S}_{ij}$: $O(N^2)$ elementwise ops\n\nThis roughly doubles the attention FLOPs. But attention is typically 10-30% of total transformer FLOPs (the rest is FFN layers, embeddings, etc.), so the overall training overhead is modest: roughly **5-15% more total FLOPs**.\n\n**Net effect**: The memory savings allow either longer sequences or larger batch sizes, which typically more than compensate for the extra FLOPs in overall training efficiency."
    },
    // Step 10: MC — Memory savings
    {
      type: "mc",
      question: "A 30-layer transformer has 32 attention heads per layer with sequence length $N = 4096$ and head dimension $d = 128$ (FP16 training). Roughly how much memory does FlashAttention's recomputation strategy save compared to storing all attention matrices?",
      options: [
        "~480 MB — the attention matrices are small relative to the model weights",
        "~120 GB — the attention matrices dominate memory at this sequence length",
        "~30 GB — each of the 960 attention heads saves a 32 MB attention matrix",
        "~1 GB — recomputation only saves the dropout mask, which is small"
      ],
      correct: 2,
      explanation: "Per head: $N^2 \\times 2$ bytes $= 4096^2 \\times 2 \\approx 33.6$ MB. Total heads: $30 \\times 32 = 960$. Total memory: $960 \\times 33.6$ MB $\\approx 32.3$ GB. FlashAttention replaces this with $m, \\ell$ vectors: $960 \\times 4096 \\times 2 \\times 2 \\approx 15$ MB — negligible. The ~30 GB saving is critical: it's often the difference between fitting a training batch in GPU memory or running out."
    },
    // Step 11: Info — Why recomputation is fast in practice
    {
      type: "info",
      title: "Why Recomputation Is Fast Despite Extra FLOPs",
      content: "You might expect doubling the attention FLOPs to add 100% overhead to the attention time. In practice, FlashAttention's backward pass (with recomputation) is often **faster** than the standard backward pass. How?\n\nThe key insight: **the standard backward pass is memory-bound, not compute-bound**.\n\nStandard backward:\n- Reads $\\mathbf{P}$ ($N^2$ elements) from HBM — slow\n- Reads $d\\mathbf{O}$, $\\mathbf{V}$ from HBM\n- Computes $d\\mathbf{V} = \\mathbf{P}^T d\\mathbf{O}$ — fast (compute)\n- Writes $d\\mathbf{V}$ to HBM — slow\n- Reads $\\mathbf{P}$ again for $d\\mathbf{P}$ computation — slow\n- ... more HBM round-trips for $d\\mathbf{S}$, $d\\mathbf{Q}$, $d\\mathbf{K}$\n\nFlashAttention backward:\n- Loads small blocks of $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$ into SRAM — moderate HBM cost\n- Recomputes $\\mathbf{S}_{ij}$ and $\\mathbf{P}_{ij}$ in SRAM — fast (compute, not memory)\n- All gradient computations happen in SRAM — fast\n- Only writes final $d\\mathbf{Q}, d\\mathbf{K}, d\\mathbf{V}$ to HBM\n\nThe recomputation happens at SRAM bandwidth (~19 TB/s), while the saved HBM reads would have occurred at ~2 TB/s. **Recomputing in fast memory beats reading from slow memory.** This counterintuitive result is the essence of IO-aware algorithm design."
    },
    // Step 12: MC — Why recomputation can be faster
    {
      type: "mc",
      question: "FlashAttention recomputes $\\mathbf{S}$ and $\\mathbf{P}$ during the backward pass instead of reading the saved $\\mathbf{P}$ from HBM. Despite the extra FLOPs, this can be faster. Which explanation is most accurate?",
      options: [
        "GPU cores are underutilized during backward, so the recomputation fills otherwise-idle cycles without adding wall-clock time",
        "The recomputed values are lower precision, so they require fewer FLOPs than the original computation",
        "Modern GPUs have dedicated recomputation hardware that can replay forward computations at no cost",
        "The attention backward pass is memory-bound — eliminating the $O(N^2)$ HBM reads of $\\mathbf{P}$ saves more time than the recomputation costs, because recomputation happens in fast SRAM"
      ],
      correct: 3,
      explanation: "The standard backward pass reads $\\mathbf{P}$ (an $N \\times N$ matrix) from HBM, which is the bottleneck for a memory-bound operation. FlashAttention replaces this with recomputation in SRAM, which is ~10× faster bandwidth. The extra FLOPs are absorbed by the GPU's underutilized compute units during what was a memory-bound operation. The values are the same precision — this is about memory hierarchy, not precision."
    },
    // Step 13: MC — Practical implications
    {
      type: "mc",
      question: "A researcher is training a 7B parameter model and wants to increase context length from 4K to 16K tokens. Without FlashAttention, the $O(N^2)$ attention matrix storage fills GPU memory. With FlashAttention's recomputation, what is the primary benefit?",
      options: [
        "Training becomes 4× faster because FlashAttention eliminates all quadratic operations",
        "The 16K context requires gradient accumulation across 4 micro-batches, which FlashAttention enables by halving the per-step memory",
        "The model can now handle the 16K context because attention memory scales as $O(N)$ instead of $O(N^2)$, at the cost of modest extra compute",
        "FlashAttention enables 16K context by using sparse attention that skips 75% of token pairs"
      ],
      correct: 2,
      explanation: "Going from 4K to 16K increases the attention matrix from $O(4096^2)$ to $O(16384^2)$ — a 16× increase in memory. FlashAttention's recomputation reduces this to $O(N \\cdot d)$, growing only linearly with $N$. The saved memory directly enables the longer context. The compute does grow quadratically (attention is still $O(N^2 d)$ FLOPs), but the memory bottleneck is what was preventing the longer context. FlashAttention computes exact, dense attention — no sparsity."
    }
  ]
};
