// Focused learning module: IO-Awareness — why memory access patterns
// matter more than FLOP count for GPU attention performance.
// Concept: The GPU memory hierarchy (SRAM vs HBM) creates a bottleneck
// where attention is memory-bound, not compute-bound.

export const ioAwarenessLearning = {
  id: "G.3-io-awareness-learning-easy",
  sectionId: "G.3",
  title: "IO-Awareness: Memory Hierarchy as the True Bottleneck",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — The gap between compute and memory
    {
      type: "info",
      title: "The Compute-Memory Gap",
      content: "Modern GPUs are extraordinarily fast at arithmetic. An A100 GPU performs **312 TFLOPS** of FP16 computation — that's $3.12 \\times 10^{14}$ floating-point operations per second. But its high-bandwidth memory (HBM) delivers only **2 TB/s** of bandwidth.\n\nLet's do the math. The ratio of compute to memory bandwidth is:\n\n$$\\text{Arithmetic intensity threshold} = \\frac{312 \\text{ TFLOPS}}{2 \\text{ TB/s}} = 156 \\text{ FLOPs/byte}$$\n\nThis means: for every byte loaded from HBM, the GPU can perform 156 floating-point operations in the same time. If an operation does fewer than 156 FLOPs per byte of data it reads, the GPU sits idle waiting for memory — it is **memory-bound** (also called bandwidth-bound). Only operations that do more than 156 FLOPs per byte actually keep the compute units busy — those are **compute-bound**.\n\nThis gap has widened over GPU generations. Compute has grown faster than memory bandwidth, making the memory bottleneck increasingly severe."
    },
    // Step 2: MC — Compute vs memory bound
    {
      type: "mc",
      question: "An operation loads 4 MB of data from HBM and performs $2 \\times 10^8$ FLOPs. On an A100 (312 TFLOPS compute, 2 TB/s bandwidth), is this operation compute-bound or memory-bound?",
      options: [
        "Memory-bound — the arithmetic intensity is 50 FLOPs/byte, well below the 156 FLOPs/byte threshold",
        "Compute-bound — $2 \\times 10^8$ FLOPs is a large amount of computation that will saturate the GPU cores",
        "Neither — the operation is perfectly balanced between compute and memory",
        "Cannot determine without knowing the kernel's occupancy and thread-block configuration"
      ],
      correct: 0,
      explanation: "Arithmetic intensity = $2 \\times 10^8$ FLOPs / $4 \\times 10^6$ bytes = 50 FLOPs/byte. Since 50 < 156 (the A100's compute-to-bandwidth ratio), the GPU finishes computation before the next batch of data arrives from HBM. The operation is memory-bound: making it faster requires reducing memory traffic, not reducing FLOPs."
    },
    // Step 3: Info — GPU memory hierarchy
    {
      type: "info",
      title: "The GPU Memory Hierarchy",
      content: "A GPU has (at least) two levels of memory that matter for understanding attention performance:\n\n**HBM (High-Bandwidth Memory)** — the GPU's main memory.\n- Capacity: 40–80 GB (A100), up to 188 GB (B200)\n- Bandwidth: 2–3.35 TB/s\n- This is where tensors (activations, weights, KV cache) live\n\n**SRAM (on-chip shared memory)** — small, fast scratchpad memory on each streaming multiprocessor (SM).\n- Capacity: ~192 KB per SM, roughly **20 MB total** across all SMs on an A100\n- Bandwidth: approximately **19 TB/s** — nearly 10× faster than HBM\n- Each thread block can use its SM's shared memory as a programmer-controlled cache\n\nThe key insight: **SRAM is ~10× faster than HBM, but ~1000× smaller.** Standard PyTorch operations treat HBM as the only memory — they write intermediate results to HBM and read them back. But if you can tile your computation to keep intermediates in SRAM, you avoid the expensive HBM round-trips entirely.\n\nThis is the core principle of **IO-awareness**: designing algorithms around memory access patterns, not just FLOP counts."
    },
    // Step 4: MC — Memory hierarchy
    {
      type: "mc",
      question: "An A100 GPU has ~20 MB of total SRAM across all SMs and 80 GB of HBM. SRAM bandwidth is ~19 TB/s while HBM bandwidth is ~2 TB/s. A kernel needs to store a 50 MB intermediate tensor. What is the performance consequence?",
      options: [
        "The kernel can split the 50 MB across SMs, fitting entirely in SRAM with no HBM access needed",
        "SRAM and HBM bandwidth are combined additively, so the effective bandwidth is 21 TB/s regardless of where data resides",
        "The GPU automatically compresses the 50 MB tensor to fit in SRAM using hardware-level quantization",
        "The intermediate must spill to HBM since 50 MB exceeds SRAM capacity, forcing reads/writes at the slower 2 TB/s rate"
      ],
      correct: 3,
      explanation: "The 50 MB intermediate exceeds the ~20 MB total SRAM capacity, so it must be written to HBM and read back later. This means those memory accesses happen at ~2 TB/s instead of ~19 TB/s — roughly a 10× slowdown for the memory-bound portions. There is no automatic compression; bandwidth doesn't combine across memory levels. This is exactly the problem FlashAttention solves for the attention computation."
    },
    // Step 5: Info — Standard attention's memory problem
    {
      type: "info",
      title: "Standard Attention: A Memory Disaster",
      content: "The standard attention computation is:\n\n$$\\mathbf{O} = \\text{softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d}}\\right) \\mathbf{V}$$\n\nwhere $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V} \\in \\mathbb{R}^{N \\times d}$ with sequence length $N$ and head dimension $d$.\n\nIn a standard implementation (e.g., PyTorch), this executes as separate kernel calls:\n1. Compute $\\mathbf{S} = \\mathbf{Q}\\mathbf{K}^T / \\sqrt{d}$ → write $\\mathbf{S} \\in \\mathbb{R}^{N \\times N}$ to HBM\n2. Compute $\\mathbf{P} = \\text{softmax}(\\mathbf{S})$ → read $\\mathbf{S}$ from HBM, write $\\mathbf{P} \\in \\mathbb{R}^{N \\times N}$ to HBM\n3. Compute $\\mathbf{O} = \\mathbf{P}\\mathbf{V}$ → read $\\mathbf{P}$ from HBM\n\nThe $N \\times N$ matrices $\\mathbf{S}$ and $\\mathbf{P}$ are the problem. For $N = 8192$ (a typical context length) with FP16, each is:\n\n$$8192^2 \\times 2 \\text{ bytes} = 128 \\text{ MB}$$\n\nThese intermediate matrices are **materialized** in HBM — written out by one kernel, then read back by the next. This creates $O(N^2)$ HBM reads and writes, even though the final output $\\mathbf{O}$ is only $O(Nd)$."
    },
    // Step 6: MC — Attention memory scaling
    {
      type: "mc",
      question: "For standard attention with sequence length $N = 4096$ and head dimension $d = 128$ in FP16, what is the approximate size of the intermediate attention matrix $\\mathbf{S} = \\mathbf{Q}\\mathbf{K}^T$?",
      options: [
        "1 MB — it's $N \\times d = 4096 \\times 128$ elements at 2 bytes each",
        "32 MB — it's $N \\times N = 4096^2$ elements at 2 bytes each",
        "128 MB — it's $N^2 \\times d$ elements since each attention score involves a $d$-dimensional dot product",
        "512 KB — attention scores are scalar so only $N$ values are stored"
      ],
      correct: 1,
      explanation: "$\\mathbf{S}$ has shape $N \\times N = 4096 \\times 4096 = 16{,}777{,}216$ elements. At 2 bytes per FP16 element, that's $\\approx 32$ MB. This is just for one attention head — with 32 heads, you'd need ~1 GB for the intermediate matrices alone. Compare this to the input/output matrices $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}, \\mathbf{O}$ which are each only $4096 \\times 128 \\times 2 = 1$ MB."
    },
    // Step 7: Info — The roofline model
    {
      type: "info",
      title: "The Roofline Model: Diagnosing the Bottleneck",
      content: "The **roofline model** provides a simple framework for identifying whether a kernel is compute-bound or memory-bound.\n\nDefine the **arithmetic intensity** of an operation:\n\n$$\\text{AI} = \\frac{\\text{FLOPs}}{\\text{bytes accessed from HBM}}$$\n\nThe GPU has a characteristic **ridge point**:\n\n$$\\text{AI}_{\\text{ridge}} = \\frac{\\text{peak FLOPS}}{\\text{peak bandwidth}}$$\n\nIf $\\text{AI} < \\text{AI}_{\\text{ridge}}$, the operation is **memory-bound** — time is dominated by memory access, and reducing FLOPs won't help.\n\nIf $\\text{AI} > \\text{AI}_{\\text{ridge}}$, the operation is **compute-bound** — time is dominated by arithmetic, and reducing memory access won't help.\n\nFor standard attention, the elementwise softmax has arithmetic intensity ~4-8 FLOPs/byte (one exp, one divide per element loaded). The matrix multiplications have higher arithmetic intensity. But because the intermediate $N \\times N$ matrices force extra HBM round-trips, the **overall** attention operation is memory-bound for typical sequence lengths.\n\nThis is the key diagnostic: standard attention is slow not because it does too many FLOPs, but because it moves too much data to and from HBM."
    },
    // Step 8: MC — Roofline analysis
    {
      type: "mc",
      question: "A kernel performs $10^{10}$ FLOPs and accesses $10^8$ bytes from HBM. The GPU's ridge point is at 156 FLOPs/byte. You want to make this kernel 2× faster. Which optimization strategy is most effective?",
      options: [
        "Reduce HBM accesses by 50% through tiling and data reuse in SRAM, since the kernel is memory-bound",
        "Use a lower precision datatype (FP16 → INT8) to double the peak FLOPS throughput",
        "Reduce the FLOP count by 50% through algorithmic improvements, cutting compute time in half",
        "Increase GPU clock speed to boost both compute and memory bandwidth proportionally"
      ],
      correct: 0,
      explanation: "The kernel's arithmetic intensity is $10^{10} / 10^8 = 100$ FLOPs/byte, below the 156 FLOPs/byte ridge point. This means it's memory-bound — execution time is determined by memory access, not computation. Reducing FLOPs (option A) or increasing compute throughput (option B) won't help because the GPU is already waiting on memory. Reducing HBM traffic by tiling into SRAM directly reduces the bottleneck. This is exactly the IO-awareness principle."
    },
    // Step 9: Info — Kernel fusion and IO-awareness
    {
      type: "info",
      title: "Kernel Fusion: The IO-Aware Solution",
      content: "The fix for memory-bound operations is **kernel fusion** — combining multiple operations into a single GPU kernel that keeps intermediate results in SRAM rather than writing them to HBM.\n\nConsider the attention sequence: matmul → softmax → matmul. In the standard implementation, each step is a separate kernel launch:\n- Kernel 1 writes $\\mathbf{S}$ to HBM (expensive)\n- Kernel 2 reads $\\mathbf{S}$, writes $\\mathbf{P}$ to HBM (expensive)\n- Kernel 3 reads $\\mathbf{P}$ from HBM (expensive)\n\nWith fusion, a single kernel does all three steps, keeping $\\mathbf{S}$ and $\\mathbf{P}$ in SRAM:\n- Load tiles of $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$ from HBM once\n- Compute attention within SRAM\n- Write only the final output $\\mathbf{O}$ to HBM\n\nThe challenge is that $\\mathbf{S}$ is $N \\times N$, which doesn't fit in SRAM for any meaningful sequence length. The entire $\\mathbf{S}$ matrix is needed for softmax because softmax normalizes across each row. This seemingly prevents fusion.\n\nFlashAttention's breakthrough is showing that you **can** fuse the entire attention operation by:\n1. **Tiling** — processing blocks of the $N \\times N$ matrix at a time\n2. **Online softmax** — computing softmax incrementally without ever materializing the full row\n\nThe result: HBM accesses drop from $O(N^2)$ to $O(N^2 d / M)$ where $M$ is SRAM size — a significant reduction that translates to real wall-clock speedups of 2–4×, despite performing the same number of FLOPs."
    },
    // Step 10: MC — Kernel fusion benefit
    {
      type: "mc",
      question: "Standard attention executes three separate kernels (matmul, softmax, matmul), writing the $N \\times N$ intermediate matrix to HBM between each step. FlashAttention fuses these into one kernel. What is the primary source of FlashAttention's speedup?",
      options: [
        "FlashAttention uses an approximate softmax that requires fewer FLOPs than exact softmax",
        "FlashAttention runs the three kernels concurrently on different SMs rather than sequentially",
        "FlashAttention reduces the computational complexity from $O(N^2)$ to $O(N \\log N)$ through sparse approximation",
        "FlashAttention eliminates the $N \\times N$ HBM round-trips by keeping intermediates in SRAM through tiling"
      ],
      correct: 3,
      explanation: "FlashAttention computes **exact** attention (not approximate) with the **same** $O(N^2 d)$ FLOPs. Its speedup comes entirely from reducing HBM memory access — by tiling the computation to fit in SRAM and using online softmax, it avoids materializing the $N \\times N$ attention matrix in HBM. The saved memory bandwidth, not saved computation, is what makes it 2–4× faster."
    },
    // Step 11: Info — Implications beyond attention
    {
      type: "info",
      title: "IO-Awareness as a General Principle",
      content: "The IO-awareness insight extends well beyond attention:\n\n**torch.compile and operator fusion**: PyTorch's compiler automatically fuses chains of elementwise operations (e.g., GELU, dropout, residual add) to avoid HBM round-trips between each op. This is the same principle — reduce memory traffic by keeping intermediates in registers/SRAM.\n\n**Triton kernels**: The Triton programming language makes it easier to write custom fused kernels. Many recent papers (e.g., fused cross-entropy, fused RMSNorm) are essentially applying IO-awareness to operations beyond attention.\n\n**Quantization for bandwidth**: INT8/FP8 inference is faster partly because smaller datatypes mean fewer bytes transferred from HBM per element — halving the datatype size nearly doubles the effective bandwidth for memory-bound operations.\n\n**The roofline determines your optimization strategy**: Before optimizing any kernel, check whether it's compute-bound or memory-bound. For memory-bound ops (most elementwise operations, reductions, softmax), reduce data movement. For compute-bound ops (large matmuls), maximize arithmetic throughput.\n\nFlashAttention popularized this way of thinking in the ML community. The paper's lasting contribution isn't just faster attention — it's the principle that **hardware-aware algorithm design** can deliver large speedups without changing the mathematical result."
    },
    // Step 12: MC — Applying IO-awareness
    {
      type: "mc",
      question: "You're profiling a custom transformer layer and find that the RMSNorm operation takes 2ms despite performing very few FLOPs. The norm reads the full hidden state from HBM, computes the norm, then writes the result back. What IO-aware optimization would most likely help?",
      options: [
        "Replace RMSNorm with a mathematically simpler normalization that uses fewer FLOPs",
        "Fuse the RMSNorm with the preceding linear layer's output so the hidden state stays in SRAM between the two operations",
        "Increase the batch size to amortize the kernel launch overhead across more tokens",
        "Switch to FP32 for the normalization to improve numerical precision and reduce rounding operations"
      ],
      correct: 1,
      explanation: "The symptom (high wall-clock time despite low FLOPs) indicates a memory-bound operation. RMSNorm reads the hidden state from HBM and writes it back — two HBM round-trips. By fusing it with the preceding linear layer, the hidden state can stay in SRAM between operations, eliminating one HBM read and one write. Reducing FLOPs won't help a memory-bound op. Increasing batch size doesn't reduce per-element memory traffic. FP32 would actually increase memory traffic."
    },
    // Step 13: MC — Sequence length and memory
    {
      type: "mc",
      question: "If sequence length doubles from 4K to 8K, how does HBM memory traffic scale for standard attention versus an IO-aware fused attention kernel?",
      options: [
        "Both scale as $O(N^2)$ — fusion doesn't change the asymptotic memory access pattern",
        "Standard scales as $O(N^2)$ while fused scales as $O(N)$ by avoiding the attention matrix entirely",
        "Standard scales as $O(N^2)$ due to the materialized attention matrix; fused scales as $O(N^2 d / M)$ which is still quadratic but with a much smaller constant factor",
        "Standard scales as $O(N^2 d)$ while fused scales as $O(N^2 / d)$ because tiling divides work by head dimension"
      ],
      correct: 2,
      explanation: "Standard attention materializes the $N \\times N$ attention matrix in HBM: $O(N^2)$ memory traffic. FlashAttention's fused kernel has HBM access $O(N^2 d / M)$ where $M$ is SRAM size. This is still quadratic in $N$, but the constant factor $d/M$ is small (e.g., $128 / 100{,}000 \\approx 0.001$), giving a large practical reduction. The output $\\mathbf{O}$ itself is $O(Nd)$, but the algorithm still processes all $N^2$ attention scores — it just does so in SRAM-sized tiles rather than materializing the full matrix."
    }
  ]
};
