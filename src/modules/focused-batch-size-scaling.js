// Focused module: Batch Size Scaling
// Section 0.3: Optimization theory
// ONE concept: the relationship between batch size, learning rate, and training efficiency,
// centered on the critical batch size that separates compute-efficient from communication-bound regimes.

export const batchSizeScalingLearning = {
  id: "0.3-batch-size-scaling-learning",
  sectionId: "0.3",
  title: "Batch Size Scaling",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "The Gradient Noise Problem",
      content: "Stochastic gradient descent uses a **mini-batch** of $B$ training examples to estimate the true gradient:\n\n$$\\hat{g}_B = \\frac{1}{B}\\sum_{i=1}^{B} \\nabla \\mathcal{L}(x_i, \\theta)$$\n\nThis estimate has noise. The variance of the mini-batch gradient scales inversely with batch size:\n\n$$\\text{Var}(\\hat{g}_B) = \\frac{\\sigma^2}{B}$$\n\nwhere $\\sigma^2$ is the per-sample gradient variance.\n\nSmall batches produce noisy gradient estimates that point in approximately the right direction but with substantial random error. Large batches produce cleaner estimates closer to the true gradient. However, each doubling of batch size requires twice the computation per step.\n\nThis sets up the fundamental trade-off: **small batches are compute-efficient** (each FLOP teaches the model something new) but **large batches are time-efficient** (more parallel hardware reduces wall-clock time per step). The question is: at what batch size does the trade-off tip?"
    },
    {
      type: "mc",
      question: "A model is training with batch size $B = 64$ and gradient variance $\\sigma^2 = 100$. The team increases to $B = 256$. How does the gradient noise change, and what is the compute cost?",
      options: [
        "Gradient variance drops by $16\\times$ because variance scales as $1/B^2$, making the $4\\times$ compute increase highly efficient per unit of noise reduction achieved",
        "Gradient variance drops by $4\\times$ to $\\sigma^2/256 \\approx 0.39$, and compute per step increases $4\\times$ — proportional noise reduction for proportional cost",
        "Gradient variance stays the same because each additional sample introduces independent noise that offsets the noise-reduction benefit of averaging across more samples",
        "Gradient variance drops by $2\\times$ (square root scaling) to $\\sigma^2/128 \\approx 0.78$, meaning noise reduction is sublinear relative to the $4\\times$ compute increase"
      ],
      correct: 1,
      explanation: "Gradient variance scales as $\\sigma^2/B$. Going from $B=64$ to $B=256$ (a $4\\times$ increase) reduces variance by $4\\times$: from $100/64 \\approx 1.56$ to $100/256 \\approx 0.39$. The compute cost per step also increases $4\\times$ (4x more forward/backward passes). At this point, the noise reduction is exactly proportional to the compute increase — each additional FLOP reduces noise by the same amount. This proportional regime is exactly where the critical batch size concept becomes important."
    },
    {
      type: "info",
      title: "The Critical Batch Size",
      content: "McCandlish et al. (2018) identified a key quantity: the **critical batch size** $B_{\\text{crit}}$. It divides training into two regimes:\n\n**Small-batch regime** ($B \\ll B_{\\text{crit}}$): The gradient noise dominates the signal. Each step's direction is mostly random, so the optimizer \"wastes\" some compute on noise. But each sample is maximally informative — doubling $B$ here roughly halves the number of steps needed, making training $\\approx 2\\times$ faster in wall-clock time at $2\\times$ the compute per step. The total compute stays roughly constant.\n\n**Large-batch regime** ($B \\gg B_{\\text{crit}}$): The gradient estimate is already clean enough that reducing noise further doesn't speed up convergence. Doubling $B$ still halves the steps, but the total compute now also doubles — you're paying more FLOPs for no additional learning efficiency.\n\nAt $B = B_{\\text{crit}}$, you get the best trade-off: each step uses gradient estimates that are \"clean enough\" to make reliable progress, but not so clean that you're wasting compute on diminishing returns.\n\nFormally, $B_{\\text{crit}}$ can be defined as the batch size where the gradient noise $\\sigma^2/B$ equals the gradient signal (squared norm of the true gradient $\\|g\\|^2$):\n\n$$B_{\\text{crit}} = \\frac{\\sigma^2}{\\|g\\|^2}$$\n\nThis ratio — noise-to-signal — varies throughout training and across model sizes."
    },
    {
      type: "mc",
      question: "A training run has per-sample gradient variance $\\sigma^2 = 10^4$ and true gradient norm $\\|g\\|^2 = 10$. The critical batch size is $B_{\\text{crit}} = 1000$. The team currently trains at $B = 100$. If they double to $B = 200$ (adding more GPUs), what happens?",
      options: [
        "Training uses $2\\times$ the compute per step and still takes roughly the same number of steps — the larger batch provides no speedup at this scale",
        "Training diverges because doubling the batch size requires simultaneously halving the learning rate to maintain stability of the optimizer dynamics",
        "Training uses $2\\times$ compute per step and takes $2\\times$ fewer steps, achieving a $4\\times$ overall speedup in both wall-clock and total compute",
        "Training uses $2\\times$ the compute per step but takes roughly half as many steps, keeping total FLOPs approximately constant — efficient scaling"
      ],
      correct: 3,
      explanation: "At $B = 100 \\ll B_{\\text{crit}} = 1000$, we are in the small-batch (noise-dominated) regime. Here, the gradient noise $\\sigma^2/B = 100$ far exceeds the signal $\\|g\\|^2 = 10$. Doubling $B$ significantly reduces noise relative to signal, so each step makes more reliable progress. The step count drops by roughly $2\\times$, while compute per step doubles — total FLOPs stay approximately the same. This is the efficient scaling regime: you trade wall-clock time for parallel compute without wasting FLOPs."
    },
    {
      type: "info",
      title: "The Linear Scaling Rule",
      content: "When you increase the batch size, you typically need to increase the learning rate to maintain the same effective update magnitude. The **linear scaling rule** (Goyal et al., 2017) states:\n\n$$\\text{If batch size increases by } k\\times, \\text{ increase LR by } k\\times.$$\n\nThe intuition: each training step with batch size $kB$ sees $k\\times$ more data. If the LR stays the same, the model processes $k\\times$ more data per step but makes the same-size update — it effectively learns more slowly per sample. Scaling the LR by $k$ restores the original \"progress per sample\" rate.\n\nMore precisely, the expected parameter update over $k$ steps with batch size $B$ at learning rate $\\alpha$ is approximately:\n\n$$k \\cdot \\alpha \\cdot g \\approx 1 \\cdot (k\\alpha) \\cdot g$$\n\n— which is the same as 1 step with batch size $kB$ at learning rate $k\\alpha$.\n\n**Limitations**: The linear rule is exact only for SGD in the noise-dominated regime. For Adam, the adaptive denominator $\\sqrt{v}$ partially compensates, so the required LR increase is smaller than linear. Empirically, $\\sqrt{k}$ scaling or no scaling works better for Adam in many settings. The linear rule also breaks down for very large batch sizes (above $B_{\\text{crit}}$) where the gradient is already accurate."
    },
    {
      type: "mc",
      question: "A team trains with SGD at batch size 256 and LR $0.1$. They scale to 4 GPUs with batch size 1024. Following the linear scaling rule, they set LR $= 0.4$. However, training immediately diverges with loss spikes in the first 100 steps. What is the most likely fix?",
      options: [
        "Reduce the LR back to $0.1$ — the linear scaling rule is incorrect and batch size changes should never affect the learning rate",
        "Switch from SGD to Adam, which eliminates the need for any learning rate adjustment when batch size changes due to its adaptive per-parameter scaling",
        "Add or extend the learning rate warmup period — the large initial LR combined with uncalibrated optimizer state destabilizes early training",
        "Reduce the batch size to 512 instead — the linear scaling rule only applies for $2\\times$ increases, not $4\\times$ increases in batch size"
      ],
      correct: 2,
      explanation: "The linear scaling rule is correct asymptotically, but the larger LR ($0.4$ vs $0.1$) can be destructive in the first few steps when the model is near random initialization and gradient statistics are unreliable. The standard fix (from Goyal et al., 2017) is **gradual warmup**: start at a lower LR and linearly ramp up to $0.4$ over the first few hundred steps. This gives the optimizer time to accumulate stable gradient statistics before taking large steps. The warmup period is particularly important for larger batch-to-LR ratios."
    },
    {
      type: "info",
      title: "Critical Batch Size Scales with Model Size and Training Progress",
      content: "$B_{\\text{crit}}$ is not a fixed constant — it depends on both the model and the stage of training.\n\n**Model size scaling**: Larger models tend to have larger $B_{\\text{crit}}$. Empirically, $B_{\\text{crit}}$ scales roughly with the loss:\n\n$$B_{\\text{crit}} \\approx \\frac{B_0}{L - L_{\\text{min}}}$$\n\nwhere $L$ is the current loss, $L_{\\text{min}}$ is the irreducible loss, and $B_0$ is a constant. As training progresses and loss decreases, $B_{\\text{crit}}$ increases — meaning larger batches become efficient later in training.\n\nThis has practical implications for frontier LLM training:\n\n**Early training**: $B_{\\text{crit}}$ is relatively small (the loss is high, gradients are large and noisy). Using a massive batch size here wastes compute — you're averaging out noise that wouldn't have hurt convergence anyway.\n\n**Late training**: $B_{\\text{crit}}$ grows as the model approaches its optimum (gradients become smaller and noisier relative to their magnitude). Larger batches become necessary to make reliable progress.\n\nSome training runs exploit this by **increasing the batch size during training** — starting with smaller batches for compute efficiency and ramping up as $B_{\\text{crit}}$ grows. This is more compute-efficient than training with the final batch size throughout."
    },
    {
      type: "mc",
      question: "A 70B model is at 80% of training with loss approaching $L_{\\text{min}}$. The team observes that training progress has slowed despite stable gradients and a well-tuned learning rate. According to the critical batch size framework, what is happening?",
      options: [
        "The learning rate schedule has decayed too aggressively, limiting the step size regardless of batch configuration or gradient quality",
        "$B_{\\text{crit}}$ has grown as loss decreased, so the current batch size is now in the small-batch regime where gradient noise limits per-step progress",
        "The training data has been exhausted after multiple epochs, and the model is memorizing rather than learning generalizable patterns",
        "The model has reached its capacity limit and cannot represent more complex patterns with its current architecture and parameter count"
      ],
      correct: 1,
      explanation: "As loss approaches $L_{\\text{min}}$, the true gradient $\\|g\\|^2$ shrinks (the model is near optimal) while gradient variance $\\sigma^2$ stays large. This drives $B_{\\text{crit}} = \\sigma^2/\\|g\\|^2$ up. If the batch size hasn't increased to match, training enters the small-batch regime where noise dominates — each step makes unreliable progress. The fix is to increase the batch size or, equivalently, accept slower wall-clock convergence. This is why many large-scale runs increase batch size during training."
    },
    {
      type: "info",
      title: "Gradient Accumulation: Simulating Large Batches",
      content: "Not every team has enough GPUs to run large batches in parallel. **Gradient accumulation** lets you simulate a larger batch size with fewer GPUs:\n\n1. Run $k$ forward-backward passes on mini-batches of size $B$\n2. Accumulate (sum) the gradients without updating parameters\n3. After $k$ accumulation steps, divide by $k$ and apply one optimizer step\n\nThe effective batch size is $k \\times B$, identical to running $k$ GPUs in data-parallel mode. The mathematical gradient estimate is the same:\n\n$$\\hat{g}_{kB} = \\frac{1}{k}\\sum_{j=1}^{k}\\hat{g}_B^{(j)}$$\n\nThe key trade-off: gradient accumulation saves on hardware cost (fewer GPUs) but **increases wall-clock time** proportionally. With $k$ accumulation steps, training is $k\\times$ slower in wall time compared to running $k$ GPUs in parallel.\n\nGradient accumulation interacts with batch normalization (which LLMs don't use) and with learning rate warmup (the effective batch size should be considered when setting warmup duration). For LLM training with Adam, accumulation is mathematically equivalent to data parallelism — the updates are identical."
    },
    {
      type: "mc",
      question: "A team has 4 GPUs with per-GPU batch size 8. They need an effective batch size of 256 to match the critical batch size. What gradient accumulation factor $k$ do they need, and what is the wall-clock cost compared to having 32 GPUs?",
      options: [
        "$k = 8$ accumulation steps; training takes $8\\times$ longer in wall time than the 32-GPU setup since each accumulation step is sequential",
        "$k = 32$ accumulation steps; training takes $32\\times$ longer because the 4-GPU compute throughput is $8\\times$ lower than 32 GPUs",
        "$k = 64$ accumulation steps; each GPU processes 64 batches sequentially, making training $64\\times$ slower than the parallel alternative",
        "$k = 8$ accumulation steps; training takes the same wall time as 32 GPUs because the gradient estimates are mathematically identical"
      ],
      correct: 0,
      explanation: "Effective batch size = GPUs $\\times$ per-GPU batch $\\times$ $k$ = $4 \\times 8 \\times k = 256$, so $k = 8$. Each optimizer step requires $k = 8$ sequential forward-backward passes on the 4 GPUs. With 32 GPUs and no accumulation ($32 \\times 8 = 256$), one optimizer step requires just 1 forward-backward pass. So the 4-GPU setup is $8\\times$ slower per step. The gradient estimates are mathematically identical — the only difference is wall-clock time, not convergence behavior or final model quality."
    },
    {
      type: "info",
      title: "Practical Batch Size Selection for LLM Training",
      content: "Putting it all together, here is how batch size is chosen for frontier LLM pretraining:\n\n**Step 1: Estimate $B_{\\text{crit}}$** from small pilot runs. Measure gradient noise $\\sigma^2$ and signal $\\|g\\|^2$ to compute $B_{\\text{crit}} = \\sigma^2/\\|g\\|^2$. For large language models, $B_{\\text{crit}}$ is typically in the range of millions of tokens (corresponding to 1-8 million tokens per batch).\n\n**Step 2: Choose $B$ near $B_{\\text{crit}}$** for the best compute-time trade-off. Going well above $B_{\\text{crit}}$ wastes FLOPs; going well below wastes wall-clock time.\n\n**Step 3: Adjust LR** according to the scaling rule (linear for SGD, sub-linear for Adam). Validate with a short pilot at the target batch size.\n\n**Step 4: Consider batch size ramp-up** — start with a smaller batch in early training (where $B_{\\text{crit}}$ is small) and increase as training progresses. This is more compute-efficient than using the final batch size throughout.\n\nTypical numbers for frontier models:\n- GPT-3 (175B): 3.2M tokens per batch\n- LLaMA-2 (70B): 4M tokens per batch\n- Chinchilla (70B): 1.5-3M tokens per batch with ramp-up\n\nAll of these are in the range of their respective $B_{\\text{crit}}$ values, confirming that practitioners implicitly optimize for this trade-off."
    },
    {
      type: "mc",
      question: "A team estimates $B_{\\text{crit}} \\approx 2M$ tokens for their 13B model. They have enough GPUs to run either $B = 500K$ tokens or $B = 8M$ tokens per step. Which choice is more compute-efficient, and why?",
      options: [
        "$B = 8M$ is more compute-efficient — larger batches always converge faster in total FLOPs because the cleaner gradient estimates make each optimization step more productive overall",
        "$B = 500K$ is more compute-efficient — it uses $4\\times$ fewer FLOPs per step while each step still reduces the loss meaningfully, since $B_{\\text{crit}}$ only governs wall-clock time",
        "Both are equally compute-efficient in total FLOPs — the critical batch size determines only the wall-clock vs hardware utilization trade-off, not the total compute required",
        "$B = 500K$ is more compute-efficient — in the small-batch regime ($B < B_{\\text{crit}}$), total FLOPs to converge is nearly constant, so fewer FLOPs per step wastes less"
      ],
      correct: 3,
      explanation: "At $B = 500K \\ll B_{\\text{crit}} = 2M$, we are in the small-batch regime. Here, doubling the batch size roughly halves the number of steps needed, keeping total FLOPs approximately constant. So $B = 500K$ and $B = 2M$ use similar total FLOPs, but $500K$ has $4\\times$ less compute per step. At $B = 8M \\gg B_{\\text{crit}}$, we are in the large-batch regime where total FLOPs scale linearly with $B$ — the extra FLOPs don't proportionally reduce the step count. The $500K$ option is more compute-efficient (fewer total FLOPs), though it will take more wall-clock time due to more optimizer steps."
    },
    {
      type: "mc",
      question: "Two teams train identical 7B models to the same final loss. Team A uses a fixed batch size of 4M tokens throughout. Team B starts at 500K tokens and ramps to 4M over the first 30% of training. Both use appropriate LR adjustments. Which statement about total training FLOPs is most accurate?",
      options: [
        "Team A uses fewer total FLOPs because their constant batch size avoids the overhead of adjusting the optimizer state during batch size transitions",
        "Both teams use identical total FLOPs because the final model quality depends only on total tokens processed, not on how they were batched during training",
        "Team B uses fewer total FLOPs because early training has small $B_{\\text{crit}}$, so the smaller early batches avoid wasting compute on overly precise gradient estimates",
        "Team B uses more total FLOPs because the smaller early batches produce noisier gradients that require more total steps to reach the same loss value"
      ],
      correct: 2,
      explanation: "Early in training, the loss is high and $B_{\\text{crit}}$ is small — perhaps 500K-1M tokens. Team A's 4M batch is well above $B_{\\text{crit}}$ in this phase, meaning the extra gradient precision is wasted: more FLOPs per step without proportionally fewer steps needed. Team B matches their batch size to the current $B_{\\text{crit}}$, spending fewer FLOPs per step in the compute-efficient regime. As training progresses and $B_{\\text{crit}}$ grows, they ramp up to stay near the optimal trade-off point. This batch ramp strategy can save 10-30% of total training FLOPs while reaching the same final loss."
    }
  ]
};
