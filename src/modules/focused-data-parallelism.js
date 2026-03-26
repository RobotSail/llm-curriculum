// Focused learning module: Data Parallelism
// Section 1.6: Distributed Training Infrastructure
// Covers: memory wall motivation, data parallelism mechanics, ring all-reduce,
// communication-computation overlap, batch size scaling, gradient accumulation.
// Single-concept module: basic data-parallel training (ZeRO/FSDP covered separately).

export const dataParallelismLearning = {
  id: "1.6-data-parallelism-learning-easy",
  sectionId: "1.6",
  title: "Data Parallelism",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    // Step 1: Info — The Memory Wall
    {
      type: "info",
      title: "The Memory Wall: Why One GPU Isn't Enough",
      content: "Training a large language model requires storing several components in GPU memory simultaneously. For a model with $\\Psi$ parameters trained in mixed precision with Adam:\n\n1. **Parameters** (FP16): $2\\Psi$ bytes\n2. **Gradients** (FP16): $2\\Psi$ bytes\n3. **Optimizer states**: Adam maintains FP32 master weights ($4\\Psi$), first moment $m$ ($4\\Psi$), and second moment $v$ ($4\\Psi$) — totaling $12\\Psi$ bytes\n\nThe combined **model state** is therefore $2\\Psi + 2\\Psi + 12\\Psi = 16\\Psi$ bytes — and this excludes activations entirely.\n\nLet's make this concrete. For a **7B parameter model**:\n\n$$16 \\times 7 \\times 10^9 \\text{ bytes} = 112 \\text{ GB}$$\n\nA single A100 GPU has 80 GB of HBM. The model state alone exceeds that by 40%.\n\nFor a **70B parameter model**: $16 \\times 70 = 1{,}120$ GB — over **1 TB** of model state. No single accelerator comes close.\n\nActivation memory adds further pressure, scaling with batch size and sequence length. The conclusion is inescapable: training frontier models requires **distributing work across multiple GPUs**."
    },
    // Step 2: MC — memory calculation
    {
      type: "mc",
      question: "A 13B parameter model is trained with Adam in mixed precision. The total model state (parameters + gradients + optimizer states) is $16 \\times 13 = 208$ GB. Ignoring activation memory, what is the minimum number of 80 GB A100 GPUs needed to hold this model state?",
      options: [
        "2 GPUs — the 208 GB model state splits evenly across two 80 GB GPUs with memory-efficient partitioning, leaving 80 GB headroom for activations across the pair",
        "3 GPUs — $\\lceil 208 / 80 \\rceil = 3$ GPUs providing 240 GB total, the minimum count where aggregate HBM exceeds 208 GB",
        "4 GPUs — while 3 GPUs provide 240 GB total, each GPU must hold at least a contiguous shard, and alignment overhead pushes the requirement to 4 GPUs in practice",
        "8 GPUs — distributed training frameworks require power-of-two GPU counts as a minimum, so the first feasible configuration above 208 GB is $8 \\times 80 = 640$ GB"
      ],
      correct: 1,
      explanation: "The model state totals $16 \\times 13 = 208$ GB. Each GPU provides 80 GB, so the minimum count is $\\lceil 208 / 80 \\rceil = 3$ GPUs (providing 240 GB total). Note that this is a strict lower bound — it assumes perfect partitioning with no fragmentation. In practice you would want additional headroom for activations, temporary buffers, and memory fragmentation. The dominant cost is the optimizer states: $12 \\times 13 = 156$ GB, accounting for 75% of the total model state."
    },
    // Step 3: Info — Data Parallelism Basics
    {
      type: "info",
      title: "Data Parallelism: Replicate the Model, Split the Data",
      content: "**Data parallelism** (DP) is the simplest distributed training strategy. The recipe has five steps:\n\n1. **Replicate** the entire model (parameters, gradients, optimizer states) on every GPU\n2. **Partition** each global batch into equal shards — with $B$ samples and $N$ GPUs, each GPU gets $B/N$ samples\n3. **Compute** forward and backward passes independently on each GPU using its local data shard\n4. **Synchronize** gradients by averaging across all GPUs\n5. **Update** each GPU applies the identical averaged gradient to its local model copy\n\nBecause all GPUs start with the same parameters and apply the same update, they remain in sync.\n\nThe mathematical equivalence is exact. Let $\\mathcal{D}_i$ be the data shard on GPU $i$, with $|\\mathcal{D}_i| = B/N$. Each GPU computes a local gradient:\n\n$$g_i = \\frac{1}{|\\mathcal{D}_i|} \\sum_{j \\in \\mathcal{D}_i} \\nabla_\\theta \\ell_j$$\n\nThe synchronized gradient is:\n\n$$g_{\\text{sync}} = \\frac{1}{N} \\sum_{i=1}^{N} g_i = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{1}{B/N} \\sum_{j \\in \\mathcal{D}_i} \\nabla_\\theta \\ell_j = \\frac{1}{B} \\sum_{j=1}^{B} \\nabla_\\theta \\ell_j$$\n\nThis is exactly the gradient computed over the full batch. Data parallelism scales **throughput** linearly (each GPU processes $B/N$ samples in parallel) but does **not** reduce per-GPU memory — every GPU still stores the full model state."
    },
    // Step 4: MC — gradient equivalence
    {
      type: "mc",
      question: "A team trains a model with data parallelism across 64 GPUs using a global batch of 1M tokens. Each GPU processes approximately 15,600 tokens. After averaging the 64 local gradients, how does the resulting synchronized gradient relate to computing the gradient on all 1M tokens on a single GPU?",
      options: [
        "It is a stochastic approximation — averaging 64 sub-batch gradients introduces variance that wouldn't exist in the single-GPU full-batch gradient",
        "It is exact in theory but accumulates floating-point errors during the all-reduce, making it unreliable for models larger than 10B parameters",
        "It is mathematically identical — the average of per-shard mean gradients equals the mean gradient over the full batch, since the gradient of a sum is the sum of gradients",
        "It is slightly biased because each GPU normalizes by $B/N$ instead of $B$, introducing a factor-of-$N$ scaling error that must be corrected in the optimizer"
      ],
      correct: 2,
      explanation: "The synchronized gradient is mathematically identical to the full-batch gradient. Since $g_{\\text{sync}} = \\frac{1}{N}\\sum_i g_i = \\frac{1}{N}\\sum_i \\frac{N}{B}\\sum_{j \\in \\mathcal{D}_i} \\nabla_\\theta \\ell_j = \\frac{1}{B}\\sum_{j=1}^{B} \\nabla_\\theta \\ell_j$, the partitioning and re-averaging produces the exact same result. In practice, minor floating-point differences arise from non-deterministic reduction order, but these are negligible and do not constitute bias or approximation in any meaningful sense."
    },
    // Step 5: Info — Ring All-Reduce
    {
      type: "info",
      title: "Ring All-Reduce: Efficient Gradient Synchronization",
      content: "Gradient synchronization requires an **all-reduce** operation: given $N$ GPUs each holding gradient tensor $g_i$, compute $\\bar{g} = \\frac{1}{N}\\sum_i g_i$ and place the result on every GPU.\n\nThe **naive approach** sends all gradients to a single coordinator GPU, averages them, and broadcasts the result. This creates a bottleneck: the coordinator must receive $(N-1)|g|$ bytes and send $N|g|$ bytes, where $|g|$ is the gradient size. As $N$ grows, this single link becomes the limiting factor.\n\n**Ring all-reduce** eliminates this bottleneck. The $N$ GPUs are arranged in a logical ring, and the operation proceeds in two phases:\n\n**Phase 1 — Reduce-Scatter** ($N-1$ steps): Each GPU's gradient is split into $N$ chunks. In each step, every GPU sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating partial sums. After $N-1$ steps, each GPU holds the fully reduced (summed) version of exactly one chunk.\n\n**Phase 2 — All-Gather** ($N-1$ steps): Each GPU broadcasts its fully reduced chunk around the ring. After $N-1$ steps, every GPU has the complete averaged gradient.\n\n**Total per-GPU communication**: Each GPU sends and receives exactly:\n\n$$2 \\cdot \\frac{N-1}{N} \\cdot |g| \\text{ bytes}$$\n\nAs $N \\to \\infty$, this approaches $2|g|$. The per-GPU communication cost is **nearly independent of GPU count** — this is what makes data parallelism scalable."
    },
    // Step 6: MC — ring all-reduce scaling
    {
      type: "mc",
      question: "A training cluster is upgraded from 32 to 64 GPUs, keeping the same model. Using ring all-reduce for gradient synchronization, what happens to the per-GPU communication volume?",
      options: [
        "It doubles from $2|g|$ to $4|g|$ — twice as many GPUs means twice as many ring steps, and each step transfers the same amount of data",
        "It halves from $2|g|$ to $|g|$ — with 64 GPUs, each GPU holds a smaller chunk in the reduce-scatter phase, reducing per-GPU data movement proportionally",
        "It increases by approximately 50% — ring all-reduce cost scales as $O(N \\log N)$ per GPU, so going from 32 to 64 adds roughly $\\log(2) \\approx 50\\%$ overhead",
        "It remains approximately $2|g|$ — the formula $2 \\cdot \\frac{N-1}{N} \\cdot |g|$ yields $1.94|g|$ for $N=32$ and $1.97|g|$ for $N=64$, a negligible difference"
      ],
      correct: 3,
      explanation: "The per-GPU communication volume in ring all-reduce is $2 \\cdot \\frac{N-1}{N} \\cdot |g|$. For $N=32$: $2 \\times 31/32 \\times |g| = 1.9375|g|$. For $N=64$: $2 \\times 63/64 \\times |g| = 1.96875|g|$. The difference is less than 2%. This near-constant per-GPU cost is the fundamental reason ring all-reduce scales well. Note that while bandwidth cost is constant, **latency** grows — $2(N-1)$ sequential steps means more round trips. For very large clusters, hierarchical all-reduce (e.g., reduce within nodes, then across nodes) is used to manage latency."
    },
    // Step 7: Info — Overlapping Communication with Computation
    {
      type: "info",
      title: "Overlapping Communication with Computation",
      content: "In a naive implementation, the training loop is strictly sequential: complete the entire backward pass, then run the all-reduce, then start the next forward pass. This leaves GPUs idle during the all-reduce.\n\n**Bucketed all-reduce** eliminates most of this idle time. The key observation: backpropagation computes gradients **last-to-first** — the gradient for the final layer is available long before the gradient for the first layer.\n\nThe strategy is:\n1. Group parameters into **buckets** (e.g., by layer or by size threshold)\n2. As soon as all gradients in a bucket are computed, **start its all-reduce immediately** — don't wait for the full backward pass to finish\n3. The all-reduce of later-layer buckets runs concurrently with the backward computation of earlier layers\n\nSince the last layers produce gradients first, their all-reduce begins while the backward pass is still computing gradients for earlier layers. By the time the first layer's gradients are ready and its all-reduce begins, the later layers' all-reduce is already complete or nearly so.\n\n**PyTorch DistributedDataParallel (DDP)** implements this automatically. It registers backward hooks on parameter groups that trigger all-reduce as soon as a bucket's gradients are ready. The bucket size (default: 25 MB) controls the granularity of the overlap.\n\nWith good overlap, the all-reduce time is almost entirely hidden behind backward computation, making data parallelism nearly communication-free in practice."
    },
    // Step 8: MC — bucketed all-reduce ordering
    {
      type: "mc",
      question: "In PyTorch DDP's bucketed gradient all-reduce, which layers' gradients begin their all-reduce communication first during the backward pass?",
      options: [
        "The last (output-adjacent) layers, because backpropagation computes gradients from the loss backward through the network, making later layers' gradients available first",
        "The first (input-adjacent) layers, because DDP processes layers in the same order as the forward pass to maintain consistency between forward and backward data flow",
        "All layers simultaneously — DDP launches all-reduce for every bucket at the start of the backward pass and lets the network scheduler interleave them with gradient computation",
        "The layers with the smallest parameter counts, because DDP prioritizes small buckets to minimize per-bucket latency and maximize the number of overlapping transfers"
      ],
      correct: 0,
      explanation: "Backpropagation proceeds from the loss (output) toward the input. The last layers' gradients are computed first, so their buckets are ready for all-reduce first. While the all-reduce for these later-layer buckets is in flight over the network, the GPU continues computing gradients for earlier layers. This overlap is what makes DDP efficient — the communication is hidden behind computation. If the first layers' gradients were sent first, there would be nothing to overlap with since the backward pass would already be complete."
    },
    // Step 9: Info — Batch Size Scaling
    {
      type: "info",
      title: "Batch Size Scaling: The Price of Parallelism",
      content: "Data parallelism inherently increases the **effective batch size**. If each GPU processes a local batch of $b$ samples and there are $N$ GPUs:\n\n$$B_{\\text{effective}} = N \\times b$$\n\nWith 32 GPUs each processing 256 samples, the effective batch is 8,192. This creates challenges.\n\n**The linear scaling rule** (Goyal et al., 2017): When multiplying the batch size by $k$, multiply the learning rate by $k$ as well. The intuition is that a $k\\times$ larger batch means $k\\times$ fewer parameter updates per epoch, so each update must take a proportionally larger step:\n\n$$\\text{lr}_{\\text{scaled}} = k \\times \\text{lr}_{\\text{base}}$$\n\n**Learning rate warmup** is critical when using large batches. Starting with a large learning rate causes instability in the early training phase when the loss landscape is poorly conditioned. The standard approach: linearly ramp the learning rate from a small value to $\\text{lr}_{\\text{scaled}}$ over the first few thousand steps.\n\n**The gradient noise scale** $\\mathcal{B}_{\\text{noise}}$ (McCandlish et al., 2018) provides a principled upper bound on useful batch size. It measures the signal-to-noise ratio of the stochastic gradient:\n\n$$\\mathcal{B}_{\\text{noise}} = \\frac{\\text{tr}(\\Sigma)}{|G|^2}$$\n\nwhere $\\Sigma$ is the gradient covariance and $G$ is the true gradient. When $B \\ll \\mathcal{B}_{\\text{noise}}$, doubling the batch nearly halves training time (linear scaling). When $B \\gg \\mathcal{B}_{\\text{noise}}$, doubling the batch barely helps — the gradient is already accurate, and additional samples provide diminishing returns."
    },
    // Step 10: MC — learning rate scaling
    {
      type: "mc",
      question: "A model trains well on 1 GPU with batch size 256 and learning rate $\\text{lr} = 1 \\times 10^{-4}$. The team scales to 32 GPUs using data parallelism, with each GPU still processing 256 samples (global batch = 8,192). Applying the linear scaling rule, what learning rate and warmup strategy should they use?",
      options: [
        "Use $\\text{lr} = 3.2 \\times 10^{-3}$ ($32 \\times$ base) with a warmup period that linearly ramps from a small learning rate — the larger batch gives a lower-variance gradient that supports larger steps, but early instability requires gradual ramping",
        "Keep $\\text{lr} = 1 \\times 10^{-4}$ unchanged — the per-GPU batch size is the same, so each GPU's gradient has the same variance, and the averaging step reduces noise without requiring a larger step size",
        "Use $\\text{lr} = 3.2 \\times 10^{-3}$ ($32 \\times$ base) from the start without warmup — the linear scaling rule accounts for the batch size increase, and warmup is only needed for batch sizes above 100K",
        "Use $\\text{lr} = \\sqrt{32} \\times 10^{-4} \\approx 5.7 \\times 10^{-4}$ with warmup — the square-root scaling rule is correct because gradient noise decreases as $1/\\sqrt{N}$, not $1/N$"
      ],
      correct: 0,
      explanation: "The linear scaling rule states: multiply lr by the batch size multiplier. The global batch grew from 256 to 8,192 ($32\\times$), so lr scales to $32 \\times 10^{-4} = 3.2 \\times 10^{-3}$. Warmup is essential — without it, the large initial learning rate causes optimization instability before the model has found a reasonable region of the loss landscape. The typical warmup schedule ramps linearly from $\\sim$0 to the target lr over the first 1,000-5,000 steps. Square-root scaling ($\\sqrt{k}$) has been proposed as an alternative but the linear rule ($k$) is the standard from Goyal et al. (2017) for SGD with momentum and remains the default starting point."
    },
    // Step 11: Info — Gradient Accumulation
    {
      type: "info",
      title: "Gradient Accumulation: Large Batches Without More GPUs",
      content: "Sometimes you need a larger effective batch size than $N \\times b_{\\text{max}}$ — either because you have few GPUs or because the target batch size is very large. **Gradient accumulation** solves this without additional hardware.\n\nThe idea: run $K$ forward-backward passes on micro-batches of size $b_{\\text{micro}}$, **accumulating** (summing) the gradients without updating parameters. After $K$ steps, average the accumulated gradient and perform a single optimizer step.\n\n$$g_{\\text{accumulated}} = \\frac{1}{K} \\sum_{k=1}^{K} g_k$$\n\nThe effective batch size becomes:\n\n$$B_{\\text{effective}} = N \\times K \\times b_{\\text{micro}}$$\n\nwhere $N$ is the number of data-parallel GPUs, $K$ is the accumulation steps, and $b_{\\text{micro}}$ is the per-GPU micro-batch size.\n\n**Combining with data parallelism**: When using gradient accumulation with DP, you can defer the all-reduce synchronization until after all $K$ micro-batches are processed. This means gradients are synchronized once per optimizer step rather than once per micro-batch, reducing communication overhead by a factor of $K$.\n\nGradient accumulation is mathematically equivalent to training with the full effective batch in a single pass. The only difference is that activation memory is determined by $b_{\\text{micro}}$ rather than the full batch, making it possible to simulate arbitrarily large batches on limited hardware."
    },
    // Step 12: MC — gradient accumulation calculation
    {
      type: "mc",
      question: "A team wants an effective batch size of 4M tokens. They have 8 GPUs and each GPU can fit a maximum micro-batch of 32K tokens. How many gradient accumulation steps $K$ are needed?",
      options: [
        "$K = 8$ — the 8 GPUs collectively process $8 \\times 32\\text{K} = 256\\text{K}$ tokens per step, so $4\\text{M} / 256\\text{K} = 15.6$, which rounds down to 8 because accumulation steps must be a power of two",
        "$K = 16$ — the per-step throughput is $N \\times b_{\\text{micro}} = 8 \\times 32\\text{K} = 256\\text{K}$ tokens, so $K = 4\\text{M} / 256\\text{K} = 15.625$, rounded up to $K = 16$",
        "$K = 64$ — each GPU accumulates $4\\text{M} / 8 = 512\\text{K}$ tokens, requiring $512\\text{K} / 32\\text{K} = 16$ steps, but each step requires 4 sub-iterations for numerical stability, giving $K = 64$",
        "$K = 128$ — gradient accumulation operates per-GPU without the data-parallel dimension, so $K = 4\\text{M} / 32\\text{K} = 125$, rounded up to the next power of two at $K = 128$"
      ],
      correct: 1,
      explanation: "The effective batch size is $B = N \\times K \\times b_{\\text{micro}}$. Solving: $K = B / (N \\times b_{\\text{micro}}) = 4{,}000{,}000 / (8 \\times 32{,}000) = 4{,}000{,}000 / 256{,}000 = 15.625$. Since $K$ must be an integer, we round up to $K = 16$, giving an actual effective batch of $8 \\times 16 \\times 32\\text{K} = 4{,}096\\text{K} \\approx 4.1\\text{M}$ tokens. The data-parallel dimension absolutely counts — all 8 GPUs contribute their micro-batches to each accumulation step, and synchronization happens once after all $K$ steps."
    }
  ]
};
