// Focused learning module: FlashAttention-2 Work Partitioning — how FA2
// improves on FA1 by swapping loop order, reducing non-matmul FLOPs,
// and improving GPU parallelism to close the gap to theoretical peak.

export const flashAttention2Partitioning = {
  id: "G.3-fa2-partitioning-learning-medium",
  sectionId: "G.3",
  title: "FlashAttention-2: Better Parallelism and Work Partitioning",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — FA1's utilization gap
    {
      type: "info",
      title: "FlashAttention-1's Utilization Gap",
      content: "FlashAttention-1 achieves 2–4× speedup over standard attention by reducing HBM accesses. But it still reaches only **25–40% of the GPU's theoretical peak FLOPS** on an A100.\n\nWhere do the remaining cycles go? Three sources of inefficiency:\n\n1. **Non-matmul FLOPs**: FA1 performs many non-matrix-multiply operations (softmax rescaling, exponentials, row max/sum updates). On modern GPUs, matrix multiply units (tensor cores) have **16× higher throughput** than general-purpose FP16 units. Every FLOP spent outside of matmuls is 16× more expensive in terms of hardware utilization.\n\n2. **Suboptimal loop ordering**: FA1's outer loop iterates over K/V blocks and the inner loop over Q blocks. This means the output $\\mathbf{O}$ must be repeatedly read from and written to HBM, and the softmax rescaling (the $e^{m_{\\text{old}} - m_{\\text{new}}}$ correction) must be applied at every inner iteration.\n\n3. **Parallelism limitations**: FA1 parallelizes over batch size and number of heads but not over sequence length. For long sequences with small batch sizes (common during inference), the GPU is underutilized.\n\nFlashAttention-2 addresses all three issues."
    },
    // Step 2: MC — FA1 bottleneck
    {
      type: "mc",
      question: "FlashAttention-1 reaches only 25–40% of peak GPU FLOPS despite being faster than standard attention. On an A100, tensor core matmul throughput is 312 TFLOPS while non-matmul FP16 throughput is ~19.5 TFLOPS. What does this 16× gap imply?",
      options: [
        "Even a small fraction of non-matmul operations (rescaling, softmax, reductions) can disproportionately reduce hardware utilization because they run at 1/16th the throughput",
        "Attention should be computed entirely using integer arithmetic to avoid the FP16 bottleneck",
        "The 312 TFLOPS peak is only achievable for batch sizes above 256, making it irrelevant for typical training",
        "FlashAttention should approximate softmax with a matmul-friendly function to eliminate non-matmul operations entirely"
      ],
      correct: 0,
      explanation: "If 5% of FLOPs are non-matmul, those 5% take as long as 5% × 16 = 80% of matmul FLOPs would. The non-matmul operations become a significant bottleneck even though they're a small fraction of total FLOPs. FA2's strategy is to reduce the number of non-matmul FLOPs rather than replace softmax — the exact computation is preserved."
    },
    // Step 3: Info — Swapping the loop order
    {
      type: "info",
      title: "Swapping the Loop Order: Q in the Outer Loop",
      content: "FlashAttention-1's loop structure:\n- **Outer loop**: iterate over K/V blocks (index $j$)\n- **Inner loop**: iterate over Q blocks (index $i$)\n\nThis means for each Q block, the output $\\mathbf{O}_i$ is updated $T_c$ times (once per K/V block), requiring the rescaling correction $e^{m_i^{\\text{old}} - m_i^{\\text{new}}}$ at each step. Each update reads and writes $\\mathbf{O}_i$ to HBM.\n\nFlashAttention-2 **reverses the loops**:\n- **Outer loop**: iterate over Q blocks (index $i$) — each Q block assigned to one thread block\n- **Inner loop**: iterate over K/V blocks (index $j$)\n\nNow each thread block owns a single Q block and accumulates its output $\\mathbf{O}_i$ **in SRAM** across all K/V blocks, only writing to HBM once at the end. Benefits:\n\n1. **No repeated HBM writes**: $\\mathbf{O}_i$ stays in shared memory/registers for the entire inner loop\n2. **Fewer rescaling operations**: the running max $m_i$ and sum $\\ell_i$ are updated in registers, not in HBM\n3. **Better data reuse**: each thread block processes one Q block completely before moving on\n\nThe final rescaling (dividing by $\\ell_i$) happens once, in SRAM, before writing $\\mathbf{O}_i$ to HBM."
    },
    // Step 4: MC — Loop order benefit
    {
      type: "mc",
      question: "FA1 loops over K/V blocks in the outer loop and Q blocks in the inner loop. FA2 reverses this. What is the primary advantage of putting Q blocks in the outer loop?",
      options: [
        "K/V blocks are larger than Q blocks, so iterating over them in the inner loop reduces total iterations",
        "The output $\\mathbf{O}_i$ for each Q block can be accumulated entirely in SRAM, eliminating repeated HBM reads/writes of the output",
        "Q blocks can be processed independently on different GPUs, enabling tensor parallelism over the sequence dimension",
        "The reversed order allows skipping blocks where the query-key dot products are known to be small"
      ],
      correct: 1,
      explanation: "With Q in the outer loop, each thread block keeps one Q block's output $\\mathbf{O}_i$ in SRAM while iterating through all K/V blocks. The output is written to HBM only once at the end. In FA1's ordering, $\\mathbf{O}_i$ was updated incrementally with HBM round-trips at each K/V iteration. The blocks aren't different sizes — both are tuned to SRAM. The benefit is purely about where the output accumulator lives."
    },
    // Step 5: Info — Reducing non-matmul FLOPs
    {
      type: "info",
      title: "Reducing Non-matmul FLOPs",
      content: "The second optimization in FA2 targets the softmax rescaling operations. In FA1, every time the inner loop processes a new block, the output accumulator is rescaled:\n\n$$\\mathbf{O}_i \\leftarrow \\text{diag}(e^{m_i^{\\text{old}} - m_i^{\\text{new}}}) \\cdot \\mathbf{O}_i + \\widetilde{\\mathbf{P}}_{ij} \\mathbf{V}_j$$\n\nThis rescaling requires a diagonal matrix multiply (elementwise scaling of each row) — a non-matmul operation.\n\nFA2 delays this rescaling. Instead of rescaling after each block, it keeps track of the cumulative correction factor and applies it **once** at the end:\n\n$$\\mathbf{O}_i \\leftarrow \\text{diag}(\\ell_i)^{-1} \\cdot \\mathbf{O}_i$$\n\nDuring the inner loop, FA2 accumulates:\n$$\\mathbf{O}_i^{\\text{unnorm}} \\leftarrow e^{m_i^{\\text{prev}} - m_i^{\\text{new}}} \\cdot \\mathbf{O}_i^{\\text{unnorm}} + e^{\\mathbf{S}_{ij} - m_i^{\\text{new}}} \\cdot \\mathbf{V}_j$$\n\nThe rescaling when the max changes still happens, but the final division by $\\ell_i$ is deferred. This eliminates one non-matmul pass over $\\mathbf{O}_i$ per block iteration.\n\nWhile this seems minor, remember: non-matmul FLOPs cost 16× more than matmul FLOPs on tensor cores. Every eliminated rescaling directly improves utilization."
    },
    // Step 6: MC — Non-matmul cost
    {
      type: "mc",
      question: "FA2 reduces non-matmul FLOPs by deferring the final $\\ell_i$ division to the end of the inner loop. If there are $T_c = 32$ K/V blocks in the inner loop, how many per-block elementwise rescaling passes of $\\mathbf{O}_i$ does this save?",
      options: [
        "32 — one per inner loop iteration, since the division was previously applied at each step",
        "31 — the rescaling is still needed at the first iteration, saving passes for the remaining 31",
        "16 — only half the rescaling operations can be deferred due to numerical stability requirements",
        "1 — only the very last rescaling is moved, from inside the loop to after it"
      ],
      correct: 0,
      explanation: "In FA1, the output is rescaled by $\\text{diag}(\\ell_i)^{-1}$ at every inner iteration — that's a full elementwise pass over $\\mathbf{O}_i$ each time. By deferring to the end, all 32 per-block divisions are replaced by a single division after the loop. Note: the max-change rescaling ($e^{m_{\\text{old}} - m_{\\text{new}}}$) still happens within the loop when the max changes, but the $\\ell_i$ normalization is deferred."
    },
    // Step 7: Info — Parallelism over sequence length
    {
      type: "info",
      title: "Parallelism Over Sequence Length",
      content: "FA1 parallelizes across the **batch** and **head** dimensions only. Each attention head runs on a single thread block (or set of thread blocks for large sequences). The number of thread blocks is:\n$$\\text{FA1 blocks} = B \\times H$$\n\nwhere $B$ is batch size and $H$ is number of heads. An A100 has 108 SMs. If $B \\times H < 108$ (e.g., batch 1 with 32 heads), many SMs are idle.\n\nFA2 adds parallelism over the **sequence dimension**. With Q blocks in the outer loop, each Q block is assigned to its own thread block:\n$$\\text{FA2 blocks} = B \\times H \\times \\lceil N / B_r \\rceil$$\n\nFor $B = 1$, $H = 32$, $N = 8192$, $B_r = 128$: FA1 uses 32 blocks, FA2 uses $32 \\times 64 = 2048$ blocks — far exceeding the 108 SMs and enabling full GPU saturation.\n\nThis is especially critical for **long-sequence inference** and **small-batch training** where the batch × heads product alone doesn't provide enough parallelism.\n\nThe forward pass parallelizes over Q blocks trivially (each is independent). The backward pass is trickier — multiple Q blocks contribute to the same $d\\mathbf{K}_j$ and $d\\mathbf{V}_j$, requiring atomic additions or a split-K approach."
    },
    // Step 8: MC — Parallelism improvement
    {
      type: "mc",
      question: "During inference with batch size 1 and 32 attention heads on an A100 (108 SMs), FA1 launches $1 \\times 32 = 32$ thread blocks. FA2 with sequence length 8192 and block size 128 launches $32 \\times 64 = 2048$ blocks. What is the practical consequence?",
      options: [
        "FA2 is slower because launching 2048 thread blocks has higher kernel launch overhead than 32 blocks",
        "The improvement only matters for training; inference doesn't benefit from additional parallelism",
        "FA2 requires 64× more SRAM because each thread block needs its own Q block in shared memory",
        "FA2 achieves higher SM occupancy — all 108 SMs stay busy because there are many more blocks than SMs to schedule, hiding latency"
      ],
      correct: 3,
      explanation: "With only 32 blocks, 76 of 108 SMs sit idle — the GPU is ~30% utilized. With 2048 blocks, the GPU scheduler can keep all 108 SMs busy by assigning ~19 blocks per SM, hiding memory latency through warp switching. SRAM is per-SM (not per-block in total), so each SM's shared memory is time-shared across its blocks. Inference with small batches benefits enormously from this parallelism gain."
    },
    // Step 9: Info — Causal masking optimization
    {
      type: "info",
      title: "Causal Masking Without Wasted Compute",
      content: "Autoregressive language models use **causal masking**: token $i$ can only attend to tokens $j \\leq i$. The attention matrix is lower-triangular.\n\nIn standard attention, causal masking is applied after computing the full $\\mathbf{QK}^T$ matrix — half the computed values (the upper triangle) are masked to $-\\infty$ before softmax. This wastes ~50% of the FLOPs.\n\nFA1 applies the mask within each tile but still computes all tiles, including those that are **entirely** in the upper triangle (all masked out).\n\nFA2 optimizes this: for each Q block $i$, it only iterates over K/V blocks $j$ where $j \\leq i$ (plus the partially masked diagonal block). Fully masked blocks are **skipped entirely**.\n\nFor a sequence of length $N$ divided into $T$ blocks:\n- Without optimization: $T^2$ tiles computed\n- With causal skipping: $\\sim T^2 / 2$ tiles computed\n\nThis gives roughly a **2× speedup for causal attention** compared to bidirectional attention — or equivalently, causal attention runs at the same speed as bidirectional attention on a sequence of half the length.\n\nThe partial diagonal blocks (where some positions are masked and others aren't) still require masking within the tile, but these are only $T$ out of $T^2/2$ total tiles — a negligible fraction."
    },
    // Step 10: MC — Causal masking
    {
      type: "mc",
      question: "FA2 skips tile computations that fall entirely within the masked (upper-triangular) region of causal attention. For a sequence of length 8192 with block size 128 ($T = 64$ blocks), approximately how many tile computations does this save compared to full bidirectional attention?",
      options: [
        "~64 tiles — only the diagonal tiles contain any masked elements",
        "~4096 tiles — three-quarters of tiles are masked in causal attention",
        "~2048 tiles — roughly half of $T^2 = 4096$ tiles are fully masked and skipped",
        "~32 tiles — only the first and last block rows interact differently under causal masking"
      ],
      correct: 2,
      explanation: "With $T = 64$ blocks, bidirectional attention computes $T^2 = 4096$ tiles. For causal masking, the lower triangle (including diagonal) has $T(T+1)/2 = 2080$ tiles. The upper triangle has $T(T-1)/2 = 2016$ tiles that are fully masked and can be skipped. This saves approximately half the computation. In practice, the diagonal blocks are partially masked, but that's only 64 tiles — the savings come from skipping the ~2000 fully-masked tiles."
    },
    // Step 11: Info — Warp-level partitioning (within a thread block)
    {
      type: "info",
      title: "Within-Block Work Partitioning: Warps",
      content: "A GPU thread block consists of multiple **warps** (groups of 32 threads). An A100 thread block might use 4 or 8 warps. How work is divided among warps within a single tile computation matters for performance.\n\nFA1 splits work among warps by partitioning the Q block — different warps compute different rows of $\\mathbf{S}_{ij} = \\mathbf{Q}_i \\mathbf{K}_j^T$. But each warp needs access to the **same** $\\mathbf{K}_j$ and $\\mathbf{V}_j$ blocks. With multiple warps accessing the same shared memory, this causes bank conflicts and synchronization overhead.\n\nFA2 changes the partitioning: warps split the **K/V** dimension instead. Each warp computes a partial result for $\\mathbf{O}_i$ using a different subset of the key/value positions. Benefits:\n\n1. **Reduced shared memory reads**: $\\mathbf{Q}_i$ is read once by all warps (broadcast-friendly), while different warps access different $\\mathbf{K}_j, \\mathbf{V}_j$ slices without conflict\n2. **No warp synchronization needed for intermediate results**: each warp independently computes its partial attention output\n3. **The final reduction** (summing partial outputs across warps) is a single, fast operation\n\nThis warp-level optimization contributes to FA2 reaching **50–73% of peak FLOPS**, up from FA1's 25–40%."
    },
    // Step 12: MC — Warp partitioning
    {
      type: "mc",
      question: "FA1 splits Q rows across warps; FA2 splits K/V columns across warps. Why does splitting K/V lead to fewer shared memory conflicts?",
      options: [
        "K/V blocks are smaller than Q blocks, so each warp accesses less shared memory overall",
        "Modern GPUs have dedicated K/V memory banks that don't conflict, unlike the general-purpose memory used for Q",
        "K/V values are read-only during the forward pass, while Q values require read-write access that causes conflicts",
        "Splitting K/V means all warps read the same Q block (which can be broadcast efficiently) while accessing disjoint K/V regions, avoiding concurrent reads to the same addresses"
      ],
      correct: 3,
      explanation: "When warps split Q rows, each warp needs the full K block — multiple warps reading the same K data simultaneously causes shared memory bank conflicts. When warps split K/V instead, all warps read the same Q data (a broadcast pattern that GPUs handle efficiently) and access disjoint K/V slices (no conflicts). The warps compute independent partial attention outputs that are combined at the end."
    },
    // Step 13: MC — Overall FA2 improvement
    {
      type: "mc",
      question: "FlashAttention-2 achieves 50–73% of A100 peak FLOPS, up from FA1's 25–40%. Which combination of improvements is responsible?",
      options: [
        "Approximate softmax that eliminates all non-matmul operations, plus sparse attention that skips low-weight connections",
        "Moving from CUDA to Triton, which provides automatic kernel fusion and memory management superior to hand-written CUDA",
        "Swapping the loop order to keep $\\mathbf{O}$ in SRAM, reducing non-matmul rescaling operations, better warp-level K/V partitioning, and causal masking block skipping",
        "Using FP8 tensor cores instead of FP16, doubling the effective FLOPS while the algorithm remains unchanged"
      ],
      correct: 2,
      explanation: "FA2's improvements are all about better utilization of the existing hardware: (1) reversed loop order keeps $\\mathbf{O}$ in SRAM, (2) deferred rescaling reduces non-matmul FLOPs that cost 16× more on tensor cores, (3) K/V warp partitioning reduces shared memory conflicts, and (4) causal block skipping avoids ~50% of wasted tiles. The computation is still exact FP16/BF16 attention — no approximation, no precision change. FA2 is written in CUDA, not Triton."
    }
  ]
};
