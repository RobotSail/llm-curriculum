// Focused learning module: FlashAttention Tiling — how to compute exact
// attention without materializing the N×N matrix by processing blocks
// and using the online softmax algorithm.

export const flashAttentionTilingLearning = {
  id: "G.3-fa-tiling-learning-medium",
  sectionId: "G.3",
  title: "FlashAttention Tiling: Block-wise Exact Attention via Online Softmax",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Info — The softmax barrier
    {
      type: "info",
      title: "The Softmax Barrier to Tiling",
      content: "We want to fuse the entire attention computation ($\\mathbf{S} = \\mathbf{QK}^T$, $\\mathbf{P} = \\text{softmax}(\\mathbf{S})$, $\\mathbf{O} = \\mathbf{PV}$) into a single kernel that never writes the $N \\times N$ attention matrix to HBM. The obstacle is **softmax**.\n\nSoftmax over a row $\\mathbf{s} \\in \\mathbb{R}^N$ requires:\n\n$$\\text{softmax}(s_j) = \\frac{e^{s_j}}{\\sum_{k=1}^{N} e^{s_k}}$$\n\nThe denominator sums over **all** $N$ entries. If we process the $K$ columns in blocks (tiles), we don't have the full row when processing the first tile. We only see partial sums.\n\nNaively, this means we must compute all $N$ dot products in $\\mathbf{QK}^T$ for a given query row before applying softmax — which forces us to either (a) store the full row in HBM, or (b) fit the full row in SRAM (impossible for large $N$).\n\nThe **online softmax** algorithm breaks this barrier by computing softmax incrementally, updating a running maximum and sum as new blocks arrive."
    },
    // Step 2: MC — Why softmax blocks fusion
    {
      type: "mc",
      question: "Why can't you naively fuse the attention computation into tiles that each process a block of $K$ columns independently?",
      options: [
        "Matrix multiplication requires the full matrices to be present in memory simultaneously",
        "Each tile produces partial attention scores, but softmax normalization requires the sum over ALL columns in the row, which isn't available until every tile has been processed",
        "GPU hardware restricts shared memory access to one tile at a time, preventing overlapping computation",
        "The $\\mathbf{QK}^T$ products within each tile are not independent due to key-query cross-correlations"
      ],
      correct: 1,
      explanation: "The matmul $\\mathbf{QK}^T$ is trivially blockable — each block of columns can be computed independently. The bottleneck is softmax: to normalize attention score $s_j$, you need $\\sum_k e^{s_k}$ across ALL $N$ columns. After processing one tile, you only have a partial sum. The online softmax algorithm solves this by maintaining running statistics that allow corrections when new tiles arrive."
    },
    // Step 3: Info — Safe softmax (numerical stability)
    {
      type: "info",
      title: "Numerically Stable Softmax",
      content: "Before understanding online softmax, recall the standard **safe softmax** trick for numerical stability. Computing $e^{s_j}$ directly overflows when $s_j$ is large (e.g., $e^{100} \\approx 10^{43}$).\n\nThe fix: subtract the row maximum $m = \\max_j s_j$:\n\n$$\\text{softmax}(s_j) = \\frac{e^{s_j - m}}{\\sum_{k=1}^{N} e^{s_k - m}}$$\n\nThis is mathematically identical (the $e^{-m}$ factors cancel) but now all exponents are $\\leq 0$, preventing overflow.\n\nThe standard safe softmax requires **three passes** over the data:\n1. **Pass 1**: Find $m = \\max_j s_j$\n2. **Pass 2**: Compute $\\ell = \\sum_j e^{s_j - m}$\n3. **Pass 3**: Output $e^{s_j - m} / \\ell$ for each $j$\n\nEach pass reads the full row from memory. Three passes over an $N \\times N$ matrix means $3 \\times O(N^2)$ HBM accesses — exactly what we want to avoid."
    },
    // Step 4: MC — Safe softmax
    {
      type: "mc",
      question: "The safe softmax subtracts $m = \\max_j s_j$ from all scores before exponentiation. Why does this not change the output values?",
      options: [
        "It does change the values slightly, but the error is negligible for practical purposes",
        "Subtracting a constant shifts the distribution's mean without affecting relative probabilities",
        "The $e^{-m}$ factor appears in both numerator and denominator, canceling exactly: $\\frac{e^{s_j - m}}{\\sum_k e^{s_k - m}} = \\frac{e^{s_j} \\cdot e^{-m}}{e^{-m} \\cdot \\sum_k e^{s_k}}$",
        "Softmax is invariant to any monotonic transformation of its inputs, not just constant subtraction"
      ],
      correct: 2,
      explanation: "The $e^{-m}$ factor is a common multiplier in both the numerator and the sum in the denominator: $\\frac{e^{s_j - m}}{\\sum_k e^{s_k - m}} = \\frac{e^{s_j} e^{-m}}{\\sum_k e^{s_k} e^{-m}} = \\frac{e^{s_j}}{\\sum_k e^{s_k}}$. The cancellation is exact, not approximate. Softmax is invariant to adding any constant to all inputs (translation invariance), but not to arbitrary monotonic transformations."
    },
    // Step 5: Info — Online softmax algorithm
    {
      type: "info",
      title: "Online Softmax: One Pass to Rule Them All",
      content: "The **online softmax** algorithm (Milakov & Gimelshein, 2018) computes safe softmax in a **single pass** by maintaining a running maximum and running sum.\n\nProcess elements one at a time (or one block at a time). After seeing elements $s_1, \\ldots, s_j$:\n- $m_j = \\max(m_{j-1}, s_j)$ — update the running maximum\n- $\\ell_j = \\ell_{j-1} \\cdot e^{m_{j-1} - m_j} + e^{s_j - m_j}$ — update the running sum, correcting previous terms\n\nThe correction factor $e^{m_{j-1} - m_j}$ is the key: when a new maximum is encountered, all previously accumulated terms in $\\ell$ are rescaled to use the new maximum. If the max doesn't change ($m_j = m_{j-1}$), the correction is $e^0 = 1$ — no rescaling needed.\n\nAfter processing all $N$ elements, $m_N$ is the true maximum and $\\ell_N = \\sum_{k=1}^N e^{s_k - m_N}$ is the correct denominator. The softmax output is $e^{s_j - m_N} / \\ell_N$.\n\nFlashAttention extends this from element-wise to **block-wise**: instead of processing one $s_j$ at a time, it processes a block of $B_c$ columns at once, updating $m$ and $\\ell$ per block."
    },
    // Step 6: MC — Online softmax update
    {
      type: "mc",
      question: "In the online softmax algorithm, after processing block 1 you have running max $m_1 = 5.0$ and running sum $\\ell_1 = 100.0$. Block 2 has a new maximum element of $8.0$. How must $\\ell_1$ be corrected before incorporating block 2's contribution?",
      options: [
        "No correction needed — just add block 2's partial sum directly to $\\ell_1$",
        "Multiply $\\ell_1$ by $e^{5.0 - 8.0} = e^{-3}$ to rescale all previous terms to the new maximum",
        "Divide $\\ell_1$ by $e^{8.0}$ to normalize it relative to the new maximum",
        "Subtract $e^{5.0}$ and add $e^{8.0}$ to replace the old maximum's contribution with the new one"
      ],
      correct: 1,
      explanation: "The running sum $\\ell_1 = \\sum_{k \\in \\text{block 1}} e^{s_k - m_1} = \\sum_k e^{s_k - 5}$. Under the new maximum $m_2 = 8$, each term should be $e^{s_k - 8} = e^{s_k - 5} \\cdot e^{5 - 8} = e^{s_k - 5} \\cdot e^{-3}$. So the entire sum is rescaled by $e^{m_1 - m_2} = e^{-3} \\approx 0.05$. This multiplicative correction is efficient — a single scalar multiply on the accumulated sum."
    },
    // Step 7: Info — The FlashAttention tiling scheme
    {
      type: "info",
      title: "FlashAttention's Tiling Scheme",
      content: "FlashAttention divides $\\mathbf{Q}$, $\\mathbf{K}$, $\\mathbf{V}$ into blocks that fit in SRAM:\n- $\\mathbf{Q}$ is divided into blocks of $B_r$ rows (query block size)\n- $\\mathbf{K}$ and $\\mathbf{V}$ are divided into blocks of $B_c$ rows (key/value block size)\n\nThe outer loop iterates over $\\mathbf{K}, \\mathbf{V}$ blocks. The inner loop iterates over $\\mathbf{Q}$ blocks. For each pair of blocks:\n\n1. **Load** blocks $\\mathbf{Q}_i \\in \\mathbb{R}^{B_r \\times d}$, $\\mathbf{K}_j \\in \\mathbb{R}^{B_c \\times d}$, $\\mathbf{V}_j \\in \\mathbb{R}^{B_c \\times d}$ from HBM to SRAM\n2. **Compute** the block attention scores: $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T \\in \\mathbb{R}^{B_r \\times B_c}$ in SRAM\n3. **Update** the online softmax statistics $(m, \\ell)$ using $\\mathbf{S}_{ij}$\n4. **Update** the output accumulator: $\\mathbf{O}_i \\leftarrow \\text{diag}(\\text{correction}) \\cdot \\mathbf{O}_i + \\widetilde{\\mathbf{P}}_{ij} \\mathbf{V}_j$\n5. **Write** updated $\\mathbf{O}_i, m_i, \\ell_i$ back to HBM\n\nThe block size $B_r \\times B_c$ is chosen so that $\\mathbf{S}_{ij}$ fits in SRAM. Crucially, the full $N \\times N$ attention matrix is **never** materialized — only one $B_r \\times B_c$ tile exists at any time.\n\nThe output $\\mathbf{O}$ is **exact** — identical to standard attention, bit-for-bit (up to floating-point non-associativity). No approximation is involved."
    },
    // Step 8: MC — Tiling and SRAM
    {
      type: "mc",
      question: "FlashAttention chooses block sizes $B_r$ and $B_c$ such that certain matrices fit in SRAM. Which matrices must simultaneously reside in SRAM during a single tile's computation?",
      options: [
        "Only the attention tile $\\mathbf{S}_{ij} \\in \\mathbb{R}^{B_r \\times B_c}$, since all other data is streamed from HBM",
        "The full $\\mathbf{Q}$ and $\\mathbf{K}$ matrices, since all query-key dot products must be computed before softmax",
        "Blocks $\\mathbf{Q}_i$, $\\mathbf{K}_j$, $\\mathbf{V}_j$, the tile $\\mathbf{S}_{ij}$, and the output accumulator $\\mathbf{O}_i$ — all must coexist in SRAM",
        "Only $\\mathbf{K}_j$ and $\\mathbf{V}_j$ — the query block is kept in registers, not shared memory"
      ],
      correct: 2,
      explanation: "To compute a tile, SRAM must hold: $\\mathbf{Q}_i$ ($B_r \\times d$), $\\mathbf{K}_j$ ($B_c \\times d$), $\\mathbf{V}_j$ ($B_c \\times d$), the attention tile $\\mathbf{S}_{ij}$ ($B_r \\times B_c$), and the running output $\\mathbf{O}_i$ ($B_r \\times d$). The total SRAM requirement is roughly $(B_r + 2B_c) \\times d + B_r \\times B_c + B_r \\times d$ elements. Block sizes are chosen to keep this within the available SRAM budget."
    },
    // Step 9: Info — The output correction step
    {
      type: "info",
      title: "Correcting the Output Accumulator",
      content: "The trickiest part of FlashAttention is the output update. After each new $\\mathbf{K}_j, \\mathbf{V}_j$ block is processed, the running output must be corrected because the softmax denominator has changed.\n\nLet's trace through the update for query block $i$ as we process key/value blocks $j = 1, 2, \\ldots$\n\nAfter block $j$, we have:\n- $m_i^{(j)}$ — running row-wise max of attention scores seen so far\n- $\\ell_i^{(j)}$ — running row-wise sum $\\sum_{k \\leq j} e^{s_{ik} - m_i^{(j)}}$\n- $\\mathbf{O}_i^{(j)}$ — the **unnormalized** output accumulator\n\nWhen processing block $j+1$ with new scores $\\mathbf{S}_{i,j+1}$:\n\n$$m_i^{(j+1)} = \\max(m_i^{(j)},\\; \\text{rowmax}(\\mathbf{S}_{i,j+1}))$$\n$$\\ell_i^{(j+1)} = e^{m_i^{(j)} - m_i^{(j+1)}} \\cdot \\ell_i^{(j)} + \\text{rowsum}(e^{\\mathbf{S}_{i,j+1} - m_i^{(j+1)}})$$\n$$\\mathbf{O}_i^{(j+1)} = e^{m_i^{(j)} - m_i^{(j+1)}} \\cdot \\mathbf{O}_i^{(j)} + e^{\\mathbf{S}_{i,j+1} - m_i^{(j+1)}} \\cdot \\mathbf{V}_{j+1}$$\n\nThe factor $e^{m_i^{(j)} - m_i^{(j+1)}}$ rescales all previously accumulated terms to the new maximum. After all blocks are processed, the final output is $\\mathbf{O}_i / \\ell_i$ — dividing by the total softmax denominator.\n\nThis is the full online softmax-weighted sum, computed without ever storing the full $N \\times N$ matrix."
    },
    // Step 10: MC — Output correction
    {
      type: "mc",
      question: "After processing KV blocks 1 and 2, you have $m^{(2)} = 10$, $\\ell^{(2)} = 500$, and unnormalized output $\\mathbf{O}^{(2)}$. Block 3 has row-max 7. What is the correction factor applied to $\\mathbf{O}^{(2)}$?",
      options: [
        "$e^{10 - 7} = e^3 \\approx 20.1$ — the old accumulator is scaled up because the new block's max is smaller",
        "$e^{7 - 10} = e^{-3} \\approx 0.05$ — the accumulator is scaled down to match the new, lower maximum",
        "$e^{0} = 1$ — the correction is always 1 because the running max $m^{(2)} = 10$ already exceeds block 3's max of 7",
        "$e^{7} / e^{10}$ applied only to block 3's terms, while $\\mathbf{O}^{(2)}$ remains unchanged"
      ],
      correct: 2,
      explanation: "The new running max is $m^{(3)} = \\max(10, 7) = 10$ — unchanged. The correction factor is $e^{m^{(2)} - m^{(3)}} = e^{10 - 10} = e^0 = 1$. No rescaling of the accumulator is needed because the new block didn't introduce a larger maximum. Block 3's own terms are computed as $e^{s_k - 10}$, which are all small since $s_k \\leq 7$. Corrections only occur when a new block has a larger maximum than all previous blocks."
    },
    // Step 11: Info — HBM access complexity
    {
      type: "info",
      title: "Memory Access Analysis",
      content: "Let's analyze FlashAttention's HBM access precisely.\n\n**Standard attention** reads/writes the $N \\times N$ matrices $\\mathbf{S}$ and $\\mathbf{P}$:\n$$\\text{HBM access} = O(N^2 + Nd) \\approx O(N^2) \\quad \\text{(since } N \\gg d \\text{)}$$\n\n**FlashAttention** has two nested loops:\n- Outer loop: $T_c = \\lceil N/B_c \\rceil$ iterations over KV blocks\n- Inner loop: $T_r = \\lceil N/B_r \\rceil$ iterations over Q blocks\n\nEach iteration loads blocks of Q, K, V from HBM. The total reads:\n- $\\mathbf{K}_j, \\mathbf{V}_j$: each loaded once per outer iteration → $O(Nd)$ total\n- $\\mathbf{Q}_i, \\mathbf{O}_i$: loaded $T_c$ times each → $O(T_c \\cdot Nd) = O(N^2 d / B_c)$\n\nWith block sizes optimized for SRAM size $M$: $B_c = \\Theta(M/d)$, giving:\n\n$$\\text{HBM access} = O\\left(\\frac{N^2 d^2}{M}\\right)$$\n\nSince $d$ (head dimension, typically 64–128) is much smaller than $\\sqrt{M}$ (for typical SRAM sizes), this is a significant reduction over $O(N^2)$. For example, with $d = 128$ and $M = 100\\text{KB}$: the ratio is $d^2/M \\approx 128^2 / 100{,}000 \\approx 0.16$, roughly a **6× reduction** in HBM traffic."
    },
    // Step 12: MC — Memory complexity
    {
      type: "mc",
      question: "FlashAttention's HBM access is $O(N^2 d^2 / M)$ where $M$ is SRAM size. Standard attention's HBM access is $O(N^2)$. Under what condition does FlashAttention provide the largest relative improvement?",
      options: [
        "When $d$ is large relative to $\\sqrt{M}$, maximizing the ratio $d^2/M$",
        "When $N$ is very small, since the quadratic term is negligible",
        "When $M$ is large relative to $d^2$, making the factor $d^2/M$ much less than 1",
        "When $N \\approx d$, so that the attention matrix is nearly square and tiling is most efficient"
      ],
      correct: 2,
      explanation: "The relative improvement is $O(N^2) / O(N^2 d^2/M) = O(M/d^2)$. This ratio grows as SRAM size $M$ increases or head dimension $d$ decreases. Larger SRAM means bigger tiles, which means fewer passes over the data. Typical values give $M/d^2 \\approx 100{,}000 / 128^2 \\approx 6$, so FlashAttention uses roughly 6× fewer HBM accesses. On newer GPUs with more SRAM, the advantage grows."
    },
    // Step 13: MC — FlashAttention properties
    {
      type: "mc",
      question: "Which statement about FlashAttention's tiling algorithm is correct?",
      options: [
        "It produces approximate attention — the online softmax introduces a controllable error proportional to the number of tiles",
        "It requires storing the $N \\times N$ attention matrix in HBM for the backward pass, even though the forward pass avoids it",
        "It computes mathematically exact attention while using $O(N)$ additional memory instead of $O(N^2)$ for the attention matrix",
        "It only works when sequence length $N$ is a multiple of the block size $B_c$, requiring padding that wastes computation"
      ],
      correct: 2,
      explanation: "FlashAttention computes exact attention — the online softmax produces identical results to standard softmax (up to floating-point non-associativity). It uses $O(N)$ extra memory: just the running statistics $m$ and $\\ell$ per query position, plus the output accumulator. No $N \\times N$ matrix is ever materialized. For the backward pass, it uses recomputation (a separate concept) rather than storing the attention matrix. Sequences not divisible by block size are handled with masking, not wasteful padding."
    }
  ]
};
