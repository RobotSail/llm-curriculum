// Focused learning module: FlashAttention-4 — adapting attention kernels
// for Blackwell GPUs where matmul throughput has outpaced other resources,
// shifting the bottleneck to non-matmul operations and shared memory bandwidth.

export const flashAttention4Learning = {
  id: "G.3-fa4-learning-hard",
  sectionId: "G.3",
  title: "FlashAttention-4: Adapting to Asymmetric Hardware Scaling on Blackwell",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — The asymmetric scaling problem
    {
      type: "info",
      title: "Blackwell's Asymmetric Scaling Problem",
      content: "Each GPU generation increases tensor core throughput faster than other resources. Blackwell (B200/GB200) amplifies this trend to a breaking point:\n\n| Resource | Hopper (H100) | Blackwell (B200) | Scaling |\n|----------|--------------|-----------------|----------|\n| BF16 matmul | 989 TFLOPS | ~2,250 TFLOPS | ~2.3× |\n| Shared memory bandwidth | 128 B/clock | 128 B/clock | **1×** |\n| Exponential unit (MUFU) | 16 ops/clock | 16 ops/clock | **1×** |\n\nOn Hopper, FA3 was limited by the matmul throughput — tensor core compute was the bottleneck. On Blackwell, matmuls are so fast that the bottleneck shifts to **non-matmul operations**: softmax exponentials (via the MUFU unit), shared memory reads for matmul operands, and output rescaling.\n\nSimply porting FA3 to Blackwell and using the new tensor core instructions achieves only **~50%** of peak — the 2× faster tensor cores spend half their time waiting for slower resources. FA4's contribution is a co-design of algorithm and kernel pipeline to rebalance the workload around these new bottlenecks."
    },
    // Step 2: MC — Bottleneck shift
    {
      type: "mc",
      question: "On Hopper, FlashAttention-3 is primarily limited by tensor core throughput. On Blackwell, tensor core throughput doubled but MUFU (exponential) throughput and shared memory bandwidth did not change. What is the new primary bottleneck?",
      options: [
        "HBM bandwidth — Blackwell's larger tiles require more data movement from main memory",
        "Register file capacity — the wider tensor core instructions consume more registers per operation",
        "Non-matmul operations (softmax exponentials via MUFU, shared memory reads for matmul operands) that run at the same speed as Hopper despite 2× faster matmuls",
        "Inter-SM communication — the new 2-CTA MMA mode requires cross-SM synchronization that didn't exist on Hopper"
      ],
      correct: 2,
      explanation: "Blackwell's tensor cores do 2.3× more FLOPS than Hopper, but the MUFU unit (which computes $e^x$ for softmax) and shared memory bandwidth are unchanged. The matmul completes in half the time but then waits for softmax exponentials and shared memory loads at the old speed. The bottleneck has shifted from compute to these auxiliary operations. FA4 addresses this by overlapping and partially replacing these bottlenecked operations."
    },
    // Step 3: Info — Software-emulated exponentials
    {
      type: "info",
      title: "Software-Emulated Exponentials: Bypassing the MUFU Bottleneck",
      content: "The softmax operation requires computing $e^{x}$ for every element of the attention score tile. On both Hopper and Blackwell, the MUFU (Multi-Function Unit) computes exponentials at **16 ops/clock** — a throughput ceiling that hasn't changed.\n\nWith Blackwell's tensor cores finishing matmuls in half the time, the MUFU becomes the critical path. FA4's solution: compute **some** exponentials using general-purpose CUDA cores (FMA units) instead of the MUFU.\n\nThe algorithm uses a **cubic polynomial approximation** based on Cody-Waite range reduction:\n\n1. Decompose $x$ into integer and fractional parts: $2^x = 2^n \\cdot 2^f$\n2. Compute $2^n$ via IEEE 754 exponent manipulation (essentially free)\n3. Approximate $2^f$ with a cubic polynomial: $2^f \\approx p_0 + p_1 f + p_2 f^2 + p_3 f^3$\n4. This requires only **3 FMA (fused multiply-add) instructions**\n\nThe polynomial coefficients are optimized via the Sollya package to minimize relative error over $[0, 1)$. For BF16 operands, the polynomial matches MUFU's output precision.\n\n**Partial emulation**: Only 10–25% of exponentials per tile use the polynomial; the rest use hardware MUFU. This prevents excessive register pressure from the polynomial temporaries while alleviating the MUFU bottleneck."
    },
    // Step 4: MC — Software exp tradeoff
    {
      type: "mc",
      question: "FA4 computes some exponentials via cubic polynomial (3 FMA instructions on CUDA cores) instead of the MUFU hardware unit. Why not compute ALL exponentials this way?",
      options: [
        "The polynomial is less accurate than MUFU for FP32 inputs, causing unacceptable numerical error",
        "CUDA cores are slower than MUFU for exponentials — the polynomial is only used when MUFU is saturated",
        "Each polynomial evaluation requires temporary registers for intermediate values; computing all exponentials this way would cause register spills that slow down the entire kernel",
        "The CUDA cores are already busy with softmax normalization and cannot handle additional work"
      ],
      correct: 2,
      explanation: "The cubic polynomial needs registers for the intermediate Horner-method values ($p_2 + p_3 f$, etc.). Each in-flight polynomial occupies registers. Computing all exponentials via polynomial simultaneously would exhaust the register file, causing spills to local memory (slow). The 10–25% split is empirically tuned: enough polynomial evaluations to keep the MUFU from bottlenecking, but few enough to avoid register pressure. For BF16 precision, the polynomial matches MUFU accuracy."
    },
    // Step 5: Info — Conditional rescaling
    {
      type: "info",
      title: "Conditional Online Softmax Rescaling",
      content: "Standard online softmax rescales the output accumulator $\\mathbf{O}$ every time a new running maximum $m$ changes:\n\n$$\\mathbf{O}_i^{(j+1)} = e^{m_i^{(j)} - m_i^{(j+1)}} \\cdot \\mathbf{O}_i^{(j)} + e^{\\mathbf{S}_{ij} - m_i^{(j+1)}} \\cdot \\mathbf{V}_j$$\n\nThe rescaling factor $e^{m^{\\text{old}} - m^{\\text{new}}}$ requires a non-matmul elementwise multiply across the entire $\\mathbf{O}$ accumulator. On Blackwell, this is expensive because it bottlenecks on shared memory bandwidth and MUFU.\n\nFA4 introduces **conditional rescaling** with a threshold $\\tau$: skip the rescaling when the max change is small.\n\n- If $m^{(j+1)} - m^{(j)} > \\tau$: perform full rescaling (the new max is significantly larger)\n- If $m^{(j+1)} - m^{(j)} \\leq \\tau$: skip rescaling and use $m^{(j)}$ (accept small approximation)\n\nAt the end of all tiles, a **single final correction** is applied using the true maximum, ensuring the output is exact. The intermediate skip doesn't affect the final result because the end-of-loop normalization uses the correct statistics.\n\nIn practice, ~90% of rescaling operations can be skipped because the running maximum stabilizes after the first few tiles. The decision is made at **warp granularity** to avoid thread divergence within a warp."
    },
    // Step 6: MC — Conditional rescaling correctness
    {
      type: "mc",
      question: "FA4 skips ~90% of intermediate rescaling operations in the online softmax. How is the final output still exact?",
      options: [
        "The skipped rescalings are small enough ($< \\tau$) that the accumulated error is within BF16 rounding tolerance",
        "A final correction step applies the true maximum and denominator after all tiles are processed, retroactively fixing any skipped rescalings",
        "The attention output is approximately correct, and the downstream layers are robust to the ~10% error",
        "The warp-level decision ensures that rescaling is only skipped for rows where the attention distribution is flat, minimizing its impact"
      ],
      correct: 1,
      explanation: "The online softmax maintains running statistics (max $m$ and sum $\\ell$) that are always updated, even when the output rescaling is skipped. After all tiles are processed, the final normalization $\\mathbf{O} / \\ell$ uses the correct cumulative statistics. The skipped intermediate rescalings accumulated an error in $\\mathbf{O}$ (using a slightly wrong normalization), but the final division corrects this exactly. The output matches what full rescaling would produce."
    },
    // Step 7: Info — TMEM: Tensor Memory
    {
      type: "info",
      title: "Tensor Memory (TMEM): A New Level of the Hierarchy",
      content: "Blackwell introduces **Tensor Memory (TMEM)** — 256 KB of on-chip memory per SM, tightly coupled to the tensor cores.\n\nThe GPU memory hierarchy is now:\n- **HBM** (~192 GB, ~8 TB/s) — main memory\n- **SMEM** (shared memory, 228 KB/SM, ~128 B/clock) — programmer-controlled scratchpad\n- **TMEM** (256 KB/SM) — tensor core accumulator storage\n- **Registers** (256 KB/SM) — fastest, per-thread\n\nTMEM is special: MMA (matrix multiply) instructions can write their accumulator **directly to TMEM** instead of registers. This is critical because the attention forward pass chains two MMA operations:\n\n$$\\mathbf{S} = \\mathbf{Q} \\mathbf{K}^T \\quad \\to \\quad \\text{softmax} \\quad \\to \\quad \\mathbf{O} = \\mathbf{P} \\mathbf{V}$$\n\nThe $\\mathbf{S}$ accumulator from the first MMA can stay in TMEM. After softmax converts it to $\\mathbf{P}$, the second MMA reads $\\mathbf{P}$ as an operand directly from TMEM. No round-trip through registers or shared memory.\n\nFA4 uses TMEM to store both the $\\mathbf{S}$ accumulator (128×128 tile) and reuse the space for $\\mathbf{P}$ after softmax. In the backward pass, TMEM is even more critical — 5 chained MMAs must share the limited TMEM space through careful column-level reuse."
    },
    // Step 8: MC — TMEM benefit
    {
      type: "mc",
      question: "TMEM allows MMA accumulators to be stored on-chip without consuming general-purpose registers. In FA4's forward pass, the $\\mathbf{S} = \\mathbf{QK}^T$ result stays in TMEM, is converted to $\\mathbf{P}$ via softmax, then used as input to $\\mathbf{O} = \\mathbf{PV}$. What does this eliminate?",
      options: [
        "The need to compute softmax entirely — TMEM can store pre-computed softmax lookup tables",
        "The register pressure from holding the 128×128 accumulator tile, freeing registers for other operations like polynomial exponentials",
        "The HBM round-trip for the attention matrix, which was already eliminated by FlashAttention-1",
        "The shared memory read for the V operand, since TMEM can hold both P and V simultaneously"
      ],
      correct: 1,
      explanation: "Without TMEM, the 128×128 FP32 accumulator ($\\mathbf{S}$) occupies a large chunk of the register file. On Blackwell, FA4 stores it in TMEM instead, freeing registers for other operations — including the polynomial exponential temporaries and the running softmax statistics. TMEM doesn't replace HBM storage (that was FA1's contribution) or softmax computation; it specifically addresses register pressure for MMA accumulators."
    },
    // Step 9: Info — 2-CTA cooperative MMA
    {
      type: "info",
      title: "2-CTA MMA: Cooperative Matmul for the Backward Pass",
      content: "The backward pass is harder to optimize than the forward pass. It chains 5 MMA operations:\n\n$$d\\mathbf{V} = \\mathbf{P}^T d\\mathbf{O}, \\quad d\\mathbf{P} = d\\mathbf{O} \\mathbf{V}^T, \\quad d\\mathbf{S} = f(d\\mathbf{P}, \\mathbf{P}), \\quad d\\mathbf{Q} = d\\mathbf{S} \\cdot \\mathbf{K}, \\quad d\\mathbf{K} = d\\mathbf{S}^T \\cdot \\mathbf{Q}$$\n\nEach MMA needs its operands in shared memory, but SMEM bandwidth hasn't scaled. On Blackwell, **SMEM reads become the backward-pass bottleneck** — the tensor cores finish the matmul before SMEM can deliver the next operand.\n\nFA4 uses Blackwell's **2-CTA MMA mode**: two cooperative thread arrays (CTAs) on the same SM collaborate on a single, larger MMA tile (e.g., 256×128×128). Each CTA loads half of one operand into shared memory. The benefits:\n\n1. **Each CTA stages only half the operand** → SMEM read traffic per CTA is halved\n2. **The MMA tile is 2× larger** → better amortization of softmax and rescaling overhead\n3. **SMEM bandwidth is effectively doubled** for the MMA operand because two CTAs' SMEM loads are interleaved\n\nThis is specific to Blackwell's hardware — the two CTAs are scheduled by the hardware to cooperate on the same tensor core instruction. On older GPUs, CTAs are independent."
    },
    // Step 10: MC — 2-CTA MMA
    {
      type: "mc",
      question: "FA4's backward pass uses 2-CTA MMA mode where two CTAs cooperate on one matrix multiply. The primary motivation is that shared memory bandwidth is the backward-pass bottleneck. How does 2-CTA mode address this?",
      options: [
        "It doubles the physical SMEM bandwidth by activating a second memory port on the SM",
        "Each CTA loads half the operand to SMEM, halving the per-CTA SMEM traffic while the hardware interleaves both CTAs' loads to keep the SMEM pipeline full",
        "It caches frequently-used operands in TMEM, bypassing SMEM entirely for repeated reads",
        "The two CTAs compute independent MMAs on different tiles, doubling throughput without additional SMEM bandwidth"
      ],
      correct: 1,
      explanation: "The physical SMEM bandwidth is unchanged. Instead, 2-CTA mode splits the operand-staging workload: each CTA loads half the data. The hardware schedules both CTAs' loads on alternating cycles, effectively keeping the SMEM pipeline saturated without either CTA needing to provide the full bandwidth alone. The MMAs are cooperative (same tile), not independent — the two CTAs produce complementary halves of the result."
    },
    // Step 11: Info — Performance results
    {
      type: "info",
      title: "FA4 Performance and the Evolving Co-Design Lesson",
      content: "FA4's combined optimizations achieve:\n- **Forward pass**: up to 1.4 PFLOPS on B200 (FP16/BF16), ~62% of peak\n- **Backward pass**: up to 0.8 PFLOPS, ~36% of peak (backward is harder due to 5 chained MMAs)\n- **Overall**: ~1.5–1.8× speedup over FA3 ported to Blackwell\n\nThe broader lesson from the FlashAttention series is about **hardware-software co-design**:\n\n- **FA1 (2022, any GPU)**: The bottleneck was HBM bandwidth → tile to fit in SRAM\n- **FA2 (2023, any GPU)**: The bottleneck was low utilization → better parallelism and work partitioning\n- **FA3 (2024, Hopper)**: The bottleneck was synchronous execution → async pipelines with TMA/WGMMA\n- **FA4 (2026, Blackwell)**: The bottleneck shifted to non-matmul ops → software exp, conditional rescaling, TMEM\n\nEach version solves a **different** bottleneck because hardware evolution shifts where time is spent. The algorithm (computing exact attention) hasn't changed — the same mathematical operation is computed each time. What changes is how the computation maps to hardware.\n\nThis pattern repeats across ML systems: the optimal kernel for one GPU generation is suboptimal for the next, because hardware capabilities evolve unevenly."
    },
    // Step 12: MC — Co-design principle
    {
      type: "mc",
      question: "The FlashAttention series (FA1→FA4) solves different bottlenecks as hardware evolves. A new GPU generation triples matmul throughput but keeps memory bandwidth and MUFU throughput constant. Based on the FA series pattern, what would FA5 likely focus on?",
      options: [
        "Approximating attention to reduce total FLOPs, since the exact computation is becoming too expensive",
        "Moving to a fully asynchronous execution model where every instruction is non-blocking",
        "Further reducing the ratio of non-matmul to matmul work and alleviating memory bandwidth constraints, since these become even more bottlenecked relative to faster matmuls",
        "Redesigning the attention algorithm to use only matmul operations, eliminating softmax entirely"
      ],
      correct: 2,
      explanation: "The FA series maintains exact attention throughout — the mathematical computation never changes. Each version adapts to where the bottleneck has shifted. If matmul throughput triples again while MUFU and memory bandwidth stay constant, the non-matmul bottleneck would worsen further. The likely focus would be more aggressive MUFU avoidance (perhaps fully polynomial softmax), better memory bandwidth utilization, and exploiting whatever new hardware features the next generation provides."
    },
    // Step 13: MC — When to use which FA version
    {
      type: "mc",
      question: "A team has access to both H100 and B200 GPUs. They want the fastest attention for training a 70B model with 32K context. Which FlashAttention configuration should they use on each GPU?",
      options: [
        "FA4 on both — it is strictly better than FA3 regardless of hardware",
        "FA3 on H100 (designed for Hopper features: TMA, WGMMA) and FA4 on B200 (designed for Blackwell features: TMEM, 2-CTA MMA, software exp)",
        "FA2 on both — it has the widest hardware compatibility and the differences are marginal",
        "FA4 on H100 because its algorithmic improvements (conditional rescaling) are hardware-independent, and FA3 on B200 because it better utilizes the faster tensor cores"
      ],
      correct: 1,
      explanation: "Each FA version is co-designed with specific hardware. FA3 uses Hopper-specific features (TMA for async loads, WGMMA for async matmuls, ping-pong scheduling) — these don't exist on Blackwell in the same form. FA4 uses Blackwell-specific features (TMEM, 2-CTA cooperative MMA, UMMA instructions). Using FA4 on H100 would fail to compile because the instructions don't exist. Hardware-specific kernel design is the entire point of the FA series."
    }
  ]
};
