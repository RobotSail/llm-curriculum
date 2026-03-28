// Focused learning module: FlashAttention-3 Asynchrony — how FA3
// exploits Hopper GPU features (TMA, warp specialization, ping-pong
// scheduling) to overlap data movement with computation.

export const flashAttention3AsyncLearning = {
  id: "G.3-fa3-async-learning-hard",
  sectionId: "G.3",
  title: "FlashAttention-3: Asynchrony and Warp Specialization on Hopper GPUs",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — The remaining gap after FA2
    {
      type: "info",
      title: "FA2's Remaining Performance Gap on Hopper",
      content: "FlashAttention-2 reaches 50–73% of A100 (Ampere) peak FLOPS. On the H100 (Hopper), this gap widens: FA2 achieves only about **35%** of H100's theoretical FP16 peak of 989 TFLOPS.\n\nWhy? The H100 introduced **new hardware features** that FA2 doesn't exploit:\n\n1. **TMA (Tensor Memory Accelerator)**: A dedicated hardware unit that handles data movement between HBM and shared memory **asynchronously** — the compute warps don't have to wait for loads to complete.\n\n2. **WGMMA (Warp Group Matrix Multiply-Accumulate)**: Asynchronous matrix multiply instructions that can overlap with other operations, including memory loads for the next tile.\n\n3. **Larger shared memory**: Hopper SMs have up to 228 KB of shared memory, enabling larger tiles.\n\nFA2's design assumes synchronous execution: load a tile → compute → store → load next tile. On Hopper, this leaves the TMA and compute units alternately idle. FA3 redesigns the algorithm to **pipeline** data movement and computation, keeping both busy simultaneously."
    },
    // Step 2: MC — Why FA2 underperforms on Hopper
    {
      type: "mc",
      question: "FlashAttention-2 achieves ~35% of H100 peak FLOPS despite reaching 50–73% on A100. What is the primary cause of this larger gap?",
      options: [
        "The H100's higher clock speed causes more frequent cache misses, reducing effective bandwidth",
        "H100 peak FLOPS grew much faster than memory bandwidth, widening the compute-to-bandwidth ratio and making non-overlapped data loading more wasteful",
        "FA2's CUDA code is incompatible with Hopper's instruction set and falls back to emulation mode",
        "The H100's tensor cores require FP8 input, and FA2 only supports FP16"
      ],
      correct: 1,
      explanation: "The H100 has ~3× the FLOPS of A100 but only ~2× the bandwidth. The compute-to-bandwidth ratio increased from 156 to ~300 FLOPs/byte. FA2's synchronous load-compute-store pattern wastes a larger fraction of each cycle waiting for memory. The fix is overlapping loads with computation using Hopper's TMA, not changing precision — FA3's FP16 path also benefits."
    },
    // Step 3: Info — Warp specialization
    {
      type: "info",
      title: "Warp Specialization: Producer-Consumer Pipelines",
      content: "The key architectural idea in FA3 is **warp specialization**: dividing warps within a thread block into distinct roles.\n\nInstead of all warps doing the same work (load, compute, store), FA3 assigns:\n- **Producer warps**: responsible for loading K/V tiles from HBM to shared memory via TMA\n- **Consumer warps**: responsible for computing attention (WGMMA matmuls, softmax) using data already in shared memory\n\nThese operate as a **pipeline**:\n\n$$\\text{Producer: load } K_{j+1}, V_{j+1} \\quad || \\quad \\text{Consumer: compute with } K_j, V_j$$\n\nWhile consumers are computing tile $j$'s attention scores and output contribution, producers are simultaneously loading tile $j+1$ into a different shared memory buffer. By the time consumers finish tile $j$, tile $j+1$ is already in shared memory.\n\nThe synchronization between producers and consumers uses **named barriers** in shared memory — a lightweight hardware mechanism on Hopper that avoids expensive global synchronization.\n\nThis producer-consumer pattern is standard in systems programming (it's how CPU pipelines work) but was difficult to implement on pre-Hopper GPUs because loads were synchronous — the loading warp had to stall until data arrived."
    },
    // Step 4: MC — Warp specialization benefit
    {
      type: "mc",
      question: "In FA3's warp specialization, producer warps load data via TMA while consumer warps compute with previously loaded data. What would happen if all warps performed both roles (as in FA2)?",
      options: [
        "Performance would be identical because the total work is the same regardless of how it's distributed among warps",
        "Performance would improve because each warp can optimize its own memory access patterns locally",
        "Compute warps would stall during data loads and load warps would stall during computation, leaving both resources intermittently idle",
        "The GPU would run out of registers because each warp needs both compute and load state simultaneously"
      ],
      correct: 2,
      explanation: "Without specialization, a warp issues a load, then must wait for it to complete before computing. During the wait, compute units are idle. Then during computation, the memory pipeline is idle. Warp specialization eliminates this serialization: producer warps keep the memory pipeline busy while consumer warps keep the compute units busy. The total work is the same, but the two pipelines run in parallel."
    },
    // Step 5: Info — TMA and async copies
    {
      type: "info",
      title: "TMA: Hardware-Accelerated Async Data Movement",
      content: "The **Tensor Memory Accelerator (TMA)** is a dedicated hardware unit on Hopper GPUs that copies multi-dimensional tensors between HBM and shared memory.\n\nWithout TMA (Ampere and earlier):\n1. Each thread computes its source and destination address\n2. Each thread issues its own load instruction\n3. The warp stalls until all threads' loads complete\n4. Threads cooperatively store data to shared memory\n\nWith TMA (Hopper):\n1. One thread issues a single TMA instruction specifying the tensor coordinates\n2. The TMA unit handles the entire copy asynchronously\n3. The issuing warp continues executing other instructions immediately\n4. A barrier signals when the copy is complete\n\nTMA reduces the instruction count for data loading from hundreds of per-thread load instructions to a **single instruction** per tile. This frees up instruction bandwidth for compute.\n\nFor FlashAttention-3, TMA loads the K and V tiles while consumer warps are computing GEMM operations on the current tiles. The coordination happens through a **circular buffer** in shared memory with multiple slots — while one slot is being consumed, another is being filled."
    },
    // Step 6: MC — TMA advantage
    {
      type: "mc",
      question: "On Ampere GPUs, loading a 128×64 tile requires hundreds of individual load instructions across warp threads. On Hopper, TMA reduces this to a single instruction. Beyond reducing instruction count, what other benefit does this provide?",
      options: [
        "TMA loads data at higher bandwidth than individual thread loads, because it uses a dedicated memory bus",
        "TMA eliminates the need for shared memory entirely by placing tiles directly in registers",
        "The issuing warp can continue executing compute instructions immediately rather than stalling, enabling overlap of loads and computation",
        "TMA automatically applies quantization during the copy, converting FP16 data to FP8 on the fly"
      ],
      correct: 2,
      explanation: "TMA's key advantage is asynchrony: after issuing a TMA instruction, the warp doesn't stall. It can immediately proceed with other work (like computing on previously loaded tiles). The bandwidth is similar — TMA uses the same HBM — but the overlap of load latency with computation is what drives FA3's speedup. TMA copies to shared memory, not registers, and doesn't change precision."
    },
    // Step 7: Info — Ping-pong scheduling
    {
      type: "info",
      title: "Ping-Pong Scheduling: Two Thread Blocks per SM",
      content: "FA3 introduces **ping-pong scheduling** to further improve utilization. The idea: run **two thread blocks** on the same SM, alternating (ping-ponging) between them.\n\n**The problem with one thread block per SM**:\nEven with warp specialization, there are brief stalls when:\n- Consumer warps wait for a producer to finish loading\n- Producer warps wait for consumers to free a shared memory buffer slot\n\nDuring these stalls, the SM's compute units and memory pipeline sit idle.\n\n**Ping-pong solution**: Two thread blocks share the SM. When block A's consumers stall waiting for data, block B's consumers can use the compute units. When block B stalls, block A takes over.\n\n$$\\text{SM timeline: } \\underbrace{A_{\\text{compute}}}_{\\text{A active}} \\to \\underbrace{B_{\\text{compute}}}_{\\text{B fills A's gap}} \\to \\underbrace{A_{\\text{compute}}}_{\\text{A data ready}} \\to \\cdots$$\n\nThis is similar to **hyperthreading** on CPUs: multiple execution contexts share a core, and when one stalls, another takes over. The SM's resources (registers, shared memory) are split between the two blocks, but the improved utilization more than compensates.\n\nPing-pong scheduling helps FA3 reach **740+ TFLOPS** on H100, roughly **75% of peak** — a significant improvement over FA2's ~35%."
    },
    // Step 8: MC — Ping-pong scheduling
    {
      type: "mc",
      question: "FA3's ping-pong scheduling runs two thread blocks on each SM, alternating between them to fill idle cycles. What is the main tradeoff?",
      options: [
        "Each block gets half the SM's registers and shared memory, reducing the maximum tile size each can process",
        "Synchronization between the two blocks adds overhead that partially cancels the utilization gain",
        "The second block must use FP8 precision because there isn't enough register space for two FP16 blocks",
        "Ping-pong doubles the total memory bandwidth required, making the approach bandwidth-limited"
      ],
      correct: 0,
      explanation: "Each SM has a fixed pool of registers and shared memory. With two thread blocks, each gets half. This means smaller tile sizes per block, which could mean more tiles and more overhead. The benefit (filling idle cycles when one block stalls) must outweigh this cost. In practice it does — the utilization gain from hiding stall latency more than compensates for the smaller tiles. The blocks operate independently, requiring no inter-block synchronization."
    },
    // Step 9: Info — WGMMA: Async matmuls
    {
      type: "info",
      title: "WGMMA: Asynchronous Matrix Multiplication",
      content: "Hopper introduces **WGMMA (Warp Group Matrix Multiply-Accumulate)** — an asynchronous matrix multiply instruction.\n\nOn Ampere, matrix multiplies via HMMA (tensor core instructions) are **synchronous**: the warp issues the instruction and stalls until the result is ready. If the input data isn't in registers yet, the warp waits.\n\nWGMMA on Hopper is **asynchronous** and can read inputs directly from shared memory:\n1. Consumer warp issues WGMMA instruction, specifying shared memory addresses for inputs\n2. The tensor cores begin computing\n3. The warp can issue **more instructions** (including the next WGMMA, or softmax operations) before the current one finishes\n4. A `wgmma.wait` instruction stalls only when the result is actually needed\n\nThis enables a **software pipeline** within the consumer warps:\n- Issue WGMMA for $\\mathbf{Q}_i \\mathbf{K}_{j+1}^T$ (reading from shared memory)\n- While tensor cores work on that, compute softmax on the previous tile's $\\mathbf{S}_{ij}$\n- Issue WGMMA for $\\mathbf{P}_{ij} \\mathbf{V}_j$ (the output accumulation)\n- Overlap continues...\n\nThe key insight: softmax (a non-matmul operation) can execute **concurrently** with the next tile's matmul on the tensor cores, because WGMMA is asynchronous. This directly addresses FA2's non-matmul FLOP bottleneck."
    },
    // Step 10: MC — WGMMA overlap
    {
      type: "mc",
      question: "In FA2, softmax and matmul operations execute sequentially — softmax runs after $\\mathbf{QK}^T$ completes and before $\\mathbf{PV}$ begins. In FA3 with WGMMA, how does this change?",
      options: [
        "Softmax is replaced with a differentiable approximation that can be expressed as a matmul, eliminating non-matmul ops entirely",
        "Softmax is precomputed for all tiles before any matmuls begin, removing it from the critical path",
        "Softmax for tile $j$ executes concurrently with the matmul $\\mathbf{Q}_i \\mathbf{K}_{j+1}^T$ for the next tile, because WGMMA is asynchronous",
        "Softmax is moved to a separate kernel that runs on different SMs than the matmuls"
      ],
      correct: 2,
      explanation: "WGMMA returns control to the warp immediately after dispatching the matmul to tensor cores. The warp can then execute softmax (scaling, exponentiation, normalization) on previously computed scores using the SM's general-purpose units. The tensor cores and general-purpose units operate simultaneously on different data. This effectively hides the non-matmul cost within the matmul latency."
    },
    // Step 11: Info — Putting it all together
    {
      type: "info",
      title: "The Full FA3 Pipeline",
      content: "FA3 combines three levels of overlap:\n\n**Level 1 — Producer/Consumer warp specialization**:\n- Producer warps: TMA loads of $\\mathbf{K}_{j+1}, \\mathbf{V}_{j+1}$ from HBM → shared memory\n- Consumer warps: compute attention for tile $j$\n- Overlap: memory loads || computation\n\n**Level 2 — Async matmul (WGMMA) within consumer warps**:\n- Tensor cores: compute $\\mathbf{Q}_i \\mathbf{K}_{j+1}^T$ for the next tile\n- General-purpose units: softmax rescaling on the current tile's $\\mathbf{S}_{ij}$\n- Overlap: matmul || softmax\n\n**Level 3 — Ping-pong across thread blocks**:\n- Thread block A stalls → Thread block B fills the gap\n- Overlap: stalled block's wait || active block's compute\n\nWith all three, the execution timeline looks like:\n\n$$\\text{Load}_{j+2} \\quad || \\quad \\text{GEMM}_{j+1} \\quad || \\quad \\text{Softmax}_j$$\n\nNearly every hardware unit stays busy at all times. The result: FA3 achieves **1.5–2.0× speedup over FA2** on H100 for FP16/BF16 attention, reaching up to 740 TFLOPS (75% of peak) compared to FA2's ~350 TFLOPS."
    },
    // Step 12: MC — FA3 speedup sources
    {
      type: "mc",
      question: "FA3 achieves ~2× speedup over FA2 on H100 for FP16 attention, reaching ~75% of peak FLOPS. If you disabled warp specialization but kept WGMMA and ping-pong scheduling, which bottleneck would return?",
      options: [
        "Matmul throughput would drop because WGMMA requires specialized warps to issue instructions",
        "All warps would stall during HBM loads, creating bubbles where neither memory nor compute units are active — the same issue FA2 has",
        "The softmax computation would move to HBM, increasing memory traffic by $O(N^2)$",
        "Ping-pong scheduling would fail because both thread blocks would try to load data simultaneously"
      ],
      correct: 1,
      explanation: "Without warp specialization, every warp does both loading and computation. When a warp issues a TMA load, it can't compute until the load completes (TMA is async, but the warp needs the data in shared memory before it can issue WGMMA). This creates load-compute bubbles. WGMMA helps overlap matmul with softmax, and ping-pong helps at the block level, but neither addresses the fundamental load-compute serialization that warp specialization solves."
    },
    // Step 13: MC — When FA3 matters most
    {
      type: "mc",
      question: "A team is choosing between FA2 and FA3 for their workload. In which scenario does FA3 provide the largest speedup?",
      options: [
        "Training a small model (125M parameters) with batch size 256 on A100 GPUs, where compute utilization is already high",
        "Running long-context inference (32K tokens) on H100 GPUs with small batch sizes, where the compute-to-bandwidth ratio is highest and Hopper features are available",
        "Fine-tuning with LoRA on consumer GPUs (RTX 4090), where the model barely fits in memory",
        "Evaluating a model on short sequences (512 tokens) where attention is a small fraction of total compute"
      ],
      correct: 1,
      explanation: "FA3's advantages are specific to Hopper GPUs (TMA, WGMMA). The speedup is largest when: (1) running on H100/H200, (2) sequence length is long enough that attention dominates compute, and (3) batch sizes are small enough that FA2 can't saturate the GPU. A100s lack TMA/WGMMA; consumer GPUs lack them too. Short sequences make attention a small fraction of total time, limiting FA3's impact."
    }
  ]
};
