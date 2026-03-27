// Focused module: Learning Rate Schedules
// Section 0.3: Optimization theory
// Covers: why schedules matter, warmup, cosine decay, WSD, min LR,
// and scale-dependent LR selection.
// Single-concept: learning rate scheduling as a training control mechanism.

export const lrSchedulesLearning = {
  id: "0.3-lr-schedules-learning-easy",
  sectionId: "0.3",
  title: "Learning Rate Schedules",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Why a Constant Learning Rate Fails",
      content: "The learning rate $\\alpha$ in SGD (and Adam) controls step size: $\\theta_{t+1} = \\theta_t - \\alpha \\, g_t$. A natural question: why not just pick one good value and keep it fixed throughout training?\n\nThe problem is that **different phases of training have different needs**:\n\n**Early training**: The model is far from any good solution. Gradients are large and point in productive directions. A large step size is efficient — it traverses the loss landscape quickly.\n\n**Late training**: The model is near a local minimum. The loss landscape has fine structure — narrow valleys, saddle points, sharp minima nearby. Large steps overshoot, bouncing around the minimum rather than converging into it. A smaller step size is needed for precise convergence.\n\n**The very beginning**: Parameters are randomly initialized and gradient statistics (Adam's $m_t, v_t$) are unreliable. Taking large steps based on noisy, uncalibrated statistics can send the model into bad regions of parameter space (loss spikes).\n\nA **learning rate schedule** $\\alpha(t)$ varies the learning rate over training to address all three phases. The schedule is typically the single most impactful hyperparameter after model architecture and data."
    },
    {
      type: "mc",
      question: "A researcher trains a 1B model with a constant learning rate that was tuned on a small pilot run. Midway through training, loss stops decreasing and oscillates around a plateau. What is the most likely explanation?",
      options: [
        "The learning rate is too large for the current training phase — the optimizer overshoots the minimum, causing oscillation rather than convergence into the loss basin",
        "The model has reached its maximum capacity and cannot represent more complex patterns regardless of the learning rate or training duration",
        "The training data has been exhausted and the model is cycling through memorized examples, which produces the observed oscillation pattern",
        "The gradient norm has grown exponentially due to a lack of gradient clipping, causing the effective step size to increase uncontrollably"
      ],
      correct: 0,
      explanation: "This is the classic symptom of a learning rate that is too high for the current training phase. Early in training, the large LR works well because the loss landscape has broad gradients pointing downhill. As the model approaches a minimum, the same step size is too large to settle into the basin — each step jumps over the minimum. The solution is to decay the learning rate so that late-training steps are smaller and can converge. This is exactly why schedules exist."
    },
    {
      type: "info",
      title: "Warmup: Stabilizing Early Training",
      content: "**Linear warmup** starts the learning rate at zero (or near-zero) and linearly increases it to the peak value over the first $T_w$ steps:\n\n$$\\alpha(t) = \\alpha_{\\text{peak}} \\cdot \\frac{t}{T_w} \\quad \\text{for } t \\leq T_w$$\n\nTypical warmup lengths for LLM pretraining: 500–2,000 steps (often 0.1–1% of total training).\n\nWarmup is critical for Adam-based training because of **second-moment calibration**. Adam divides the gradient by $\\sqrt{v_t + \\epsilon}$, where $v_t$ is the EMA of squared gradients. At step 1, $v_t$ is initialized to zero and has seen only one gradient — it is a terrible estimate of the true variance.\n\nWith a large LR and uncalibrated $v_t$, the effective step size $\\alpha / \\sqrt{v_t + \\epsilon}$ can be enormous for some parameters, causing destructive updates. Warmup gives $v_t$ time to accumulate meaningful statistics before large steps are taken.\n\nNote: Adam's bias correction ($\\hat{v}_t = v_t / (1 - \\beta_2^t)$) helps but is not sufficient — it corrects the expected value of $v_t$ but not its high variance from few samples."
    },
    {
      type: "mc",
      question: "A team skips learning rate warmup and immediately starts training at peak LR with Adam. In the first 100 steps, they observe a massive loss spike that the model only partially recovers from. What is the mechanism behind this spike?",
      options: [
        "The random parameter initialization places the model in a high-loss region that requires thousands of steps to escape, regardless of the learning rate schedule used",
        "Adam's second-moment estimate $v_t$ is poorly calibrated from few samples, causing some parameters to receive enormous effective updates ($\\alpha/\\sqrt{v_t}$) that push the model into a bad loss region",
        "The gradient clipping threshold is not active during the first 100 steps, allowing unbounded gradient norms to produce arbitrarily large updates",
        "The data loader has not yet shuffled the training data, causing the first batches to be from a single domain that produces unrepresentative gradients"
      ],
      correct: 1,
      explanation: "Adam normalizes gradients by $\\sqrt{v_t + \\epsilon}$. After very few steps, $v_t$ estimates are high-variance — some parameters have underestimated $v_t$, producing disproportionately large effective step sizes. These outsized updates can push parameters into extreme regions, causing the loss spike. Warmup prevents this by keeping $\\alpha$ small while $v_t$ calibrates. The model 'partially recovers' because the spike damages some learned structure that must be relearned."
    },
    {
      type: "info",
      title: "Cosine Decay",
      content: "After warmup, the most common schedule for LLM pretraining is **cosine decay** (Loshchilov & Hutter, 2017). It smoothly anneals the learning rate from the peak to a minimum value:\n\n$$\\alpha(t) = \\alpha_{\\text{min}} + \\frac{1}{2}(\\alpha_{\\text{peak}} - \\alpha_{\\text{min}})\\left(1 + \\cos\\left(\\frac{\\pi \\cdot (t - T_w)}{T - T_w}\\right)\\right)$$\n\nwhere $T_w$ is the warmup duration and $T$ is the total training steps.\n\nThe cosine shape has a useful property: it **spends most of its budget at intermediate learning rates**. The LR decreases slowly at first, then quickly through the middle range, then slowly approaches the minimum. This concentrates training compute in the \"productive zone\" where the LR is large enough to make progress but small enough to avoid oscillation.\n\nCompare with **linear decay** ($\\alpha(t) = \\alpha_{\\text{peak}} \\cdot (1 - t/T)$), which decreases uniformly. Linear decay spends relatively more time at very low LRs where learning is slow. Empirically, cosine decay consistently outperforms linear decay by 1-3% on downstream benchmarks for the same compute budget."
    },
    {
      type: "mc",
      question: "Cosine decay spends more training steps at intermediate learning rates compared to linear decay. Why does this improve final model quality?",
      options: [
        "Intermediate learning rates are the optimal regime for gradient noise reduction, and spending more time there produces lower-variance parameter estimates",
        "The cosine function has a unique mathematical property that aligns gradient updates with the curvature of the loss surface, producing naturally adaptive step sizes",
        "Intermediate LRs balance exploration and convergence — large enough to make meaningful progress, small enough to avoid overshooting — and cosine decay concentrates compute in this productive zone",
        "Cosine decay produces a smoother loss curve than linear decay, which makes checkpoint selection easier but does not actually change final model quality"
      ],
      correct: 2,
      explanation: "Very high LRs cause oscillation; very low LRs make negligible progress per step. The intermediate range is where the optimizer makes the most efficient progress — large enough steps to traverse the landscape meaningfully, but controlled enough to avoid wasted oscillation. Cosine decay's shape naturally allocates more training budget to this productive zone. Linear decay, by contrast, spends its compute uniformly across all LR levels, wasting steps at both extremes."
    },
    {
      type: "info",
      title: "WSD: Warmup-Stable-Decay",
      content: "The **WSD (Warmup-Stable-Decay)** schedule (Zhai et al., 2022; adopted in MiniCPM, DeepSeek, and others) splits training into three explicit phases:\n\n1. **Warmup** ($T_w$ steps): Linear increase from 0 to $\\alpha_{\\text{peak}}$\n2. **Stable** (the majority of training): Constant at $\\alpha_{\\text{peak}}$\n3. **Decay** ($T_d$ steps, typically 10–20% of total): Cosine or linear decay to $\\alpha_{\\text{min}}$\n\nWSD has a major practical advantage: **you don't need to know the total training budget in advance**. With cosine decay, the schedule is parameterized by total steps $T$ — if you decide to train longer, you must restart with a new schedule. With WSD, you can extend the stable phase indefinitely and only anneal when you decide to stop.\n\nThis matters for frontier LLM training where:\n- Compute allocations may change mid-run\n- The team may decide to continue training based on intermediate evaluations\n- Multiple checkpoints at different training durations can share a single run\n\nThe decay phase is still essential — it accounts for 1-3% of final benchmark performance. Stopping at the stable phase without decaying leaves quality on the table."
    },
    {
      type: "mc",
      question: "A team is 80% through a cosine-decay training run when they receive additional compute and want to extend training by 50%. What is the core problem with their current schedule?",
      options: [
        "Extending training beyond the original cosine period creates a discontinuity in the LR curve that causes immediate loss spikes and gradient instability",
        "The model has already overfit to the training data by 80% of the run, so additional training regardless of schedule would only degrade generalization",
        "The batch size must be simultaneously increased with the extended schedule, and the original cosine curve cannot accommodate both changes at once",
        "The cosine schedule has already decayed the LR to near-minimum by 80% of the original budget, so the remaining 50% of training would occur at a nearly-zero learning rate with minimal progress"
      ],
      correct: 3,
      explanation: "At 80% of the original total steps, cosine decay has reduced the LR to approximately $\\alpha_{\\text{min}} + 0.1 \\cdot (\\alpha_{\\text{peak}} - \\alpha_{\\text{min}})$ — nearly at the minimum. The additional 50% of training steps would all occur at near-zero LR, wasting the extra compute. This is the inflexibility of cosine decay: the schedule is defined by the total step count, so extending the run provides almost no benefit. WSD avoids this by keeping LR at peak during the stable phase — the team would simply extend the stable phase and anneal when truly ready to stop."
    },
    {
      type: "info",
      title: "Minimum Learning Rate and Numerical Precision",
      content: "Most schedules decay to a **minimum learning rate** $\\alpha_{\\text{min}}$ rather than zero. Typical values: $\\alpha_{\\text{min}} = 0.01 \\cdot \\alpha_{\\text{peak}}$ to $0.1 \\cdot \\alpha_{\\text{peak}}$.\n\nWhy not decay to zero? Two reasons:\n\n**1. Continued learning**: Even late in training, a nonzero LR allows the model to continue adapting to gradient signals. Decaying to exactly zero freezes the parameters permanently, which is suboptimal if training data is not exhausted.\n\n**2. Numerical precision with bfloat16**: Many modern training runs use bfloat16 for parameter storage. BF16 has only 7 bits of mantissa, giving ~2 decimal digits of precision. If the parameter magnitude is $|\\theta| \\approx 1$ and the update $\\alpha \\cdot g$ is smaller than $2^{-7} \\approx 0.008$, the update is lost to **rounding** — it doesn't change the stored parameter at all.\n\nFor $\\alpha_{\\text{peak}} = 3 \\times 10^{-4}$ (typical for LLMs), a minimum LR of $3 \\times 10^{-6}$ combined with gradients of magnitude ~0.01 produces updates of $\\sim 3 \\times 10^{-8}$ — well below the BF16 precision floor. At this point, training effectively stops for BF16-stored parameters even though the optimizer is still running.\n\nThis numerical floor sets a practical lower bound on useful $\\alpha_{\\text{min}}$."
    },
    {
      type: "mc",
      question: "A training run stores parameters in bfloat16 and decays the learning rate to $10^{-7}$. With typical gradient magnitudes around 0.01, the parameter updates are $\\sim 10^{-9}$. What happens to training in this regime?",
      options: [
        "Training continues normally because Adam's adaptive scaling amplifies the small updates to be above the precision floor for all parameter groups",
        "Updates are lost to BF16 rounding for most parameters — the optimizer runs but parameters do not actually change, wasting compute without any model improvement",
        "The model begins to diverge because the tiny updates accumulate rounding errors that push parameters in random directions over thousands of steps",
        "The optimizer automatically switches to FP32 precision when it detects that BF16 updates are being rounded to zero, maintaining training progress"
      ],
      correct: 1,
      explanation: "BF16 has ~7 bits of mantissa precision. For a parameter $\\theta \\approx 1$, updates smaller than $\\sim 2^{-7} \\approx 0.008$ are rounded to zero — the stored value is unchanged. With updates of $\\sim 10^{-9}$, virtually all parameters are frozen. The optimizer computes gradients, updates moments, and applies the learning rate, but the final parameter write is a no-op due to precision limits. This is pure wasted compute. Setting $\\alpha_{\\text{min}}$ above the precision floor (typically $\\geq 10^{-5}$ for BF16) avoids this trap."
    },
    {
      type: "info",
      title: "Peak Learning Rate and Model Scale",
      content: "The optimal peak learning rate depends on model size, and the relationship follows a clear pattern: **larger models need smaller learning rates**.\n\nEmpirical scaling (approximate, varies by architecture):\n\n| Model Size | Typical Peak LR |\n|---|---|\n| 125M | $6 \\times 10^{-4}$ |\n| 1.3B | $2 \\times 10^{-4}$ |\n| 7B | $3 \\times 10^{-4}$ (with careful warmup) |\n| 70B | $1.5 \\times 10^{-4}$ |\n| 400B+ | $\\sim 10^{-4}$ or lower |\n\nThe reason: larger models have more parameters interacting through shared activations. A gradient step that moves one layer's weights affects all downstream layers through the forward pass. In a deep, wide network, even a small per-parameter update creates large changes in the output because the effects compound through the network depth.\n\nMore precisely, the loss change from a parameter update $\\Delta \\theta$ scales with the operator norm of the Jacobian, which grows with width. The **maximal update parameterization ($\\mu$P)** (Yang et al., 2022) provides a principled framework: it identifies a width-dependent LR scaling that ensures the magnitude of hidden-state updates stays constant as width increases. Under $\\mu$P, the optimal LR for Adam scales as $O(1/\\text{width})$ for certain parameter groups."
    },
    {
      type: "mc",
      question: "A team tunes the peak learning rate at 1B scale ($\\alpha = 3 \\times 10^{-4}$) and wants to train a 70B model. Following the empirical trend, they should:",
      options: [
        "Use the same $\\alpha = 3 \\times 10^{-4}$ because modern optimizers like Adam adapt per-parameter step sizes, making the global LR scale-invariant",
        "Increase to $\\alpha = 6 \\times 10^{-4}$ because the 70B model has more parameters to update and needs proportionally larger steps to make progress",
        "Set $\\alpha = 3 \\times 10^{-4} / 70 \\approx 4.3 \\times 10^{-6}$ by scaling inversely with parameter count to keep total update magnitude constant",
        "Reduce to approximately $\\alpha = 1$-$2 \\times 10^{-4}$ because larger models amplify per-parameter updates through deeper and wider computation graphs"
      ],
      correct: 3,
      explanation: "Larger models need moderately smaller peak LRs — typically a 2-3x reduction going from 1B to 70B. The reduction is modest because Adam's per-parameter scaling already handles much of the adaptation; the remaining issue is that width and depth amplify the network-level effect of each update. Scaling inversely with parameter count (option D) would reduce the LR by 70x, far too aggressive — the model would barely learn. The empirical range of $1$-$2 \\times 10^{-4}$ for 70B models is well-established across LLaMA, Chinchilla, and similar model families."
    },
    {
      type: "info",
      title: "Continued Pretraining and Schedule Restarts",
      content: "When **continuing pretraining** from an existing checkpoint — either to extend training, add a new domain (e.g., code), or adapt to a new language — the learning rate schedule must be handled carefully.\n\n**Option 1: Resume the original schedule.** If the original run used cosine decay and you have compute to continue, simply extend from where it left off. But if the LR has already decayed to near-minimum, further training at low LR is inefficient.\n\n**Option 2: Re-warmup to a fraction of peak.** Reset the LR to a moderate value (e.g., $0.3$-$0.5 \\times \\alpha_{\\text{peak}}$) with a short warmup, then decay again. This gives the model enough learning rate to adapt to the new distribution without the full peak LR that could destroy previously learned representations.\n\n**Option 3: WSD continuation.** If the original run used WSD, insert additional stable-phase training at peak LR, then anneal. This is the cleanest approach and is why WSD has gained popularity for training runs that may be extended.\n\nThe choice depends on how different the new data is from the original distribution. Small distribution shifts (more data from the same sources) tolerate lower LRs. Large shifts (new language, new modality) require higher LRs to reorganize internal representations."
    },
    {
      type: "mc",
      question: "A team wants to continue pretraining a model that was trained with cosine decay (LR has already decayed to $\\alpha_{\\text{min}}$). They plan to add 100B tokens of code data. What LR strategy should they use?",
      options: [
        "Continue at $\\alpha_{\\text{min}}$ to preserve the model's existing capabilities — any increase in LR would cause catastrophic forgetting of the original pretraining",
        "Restart at full $\\alpha_{\\text{peak}}$ with the original warmup schedule because the code domain is fundamentally different and the model needs to relearn from scratch",
        "Re-warmup to $0.3$-$0.5 \\times \\alpha_{\\text{peak}}$ and decay again, providing enough LR for the model to adapt to code without the full peak LR that could destabilize learned representations",
        "Use a linearly increasing LR throughout the entire 100B tokens to gradually shift the model's distribution toward code without any sudden changes"
      ],
      correct: 2,
      explanation: "Continuing at $\\alpha_{\\text{min}}$ wastes compute — the LR is too low to meaningfully adapt to a new domain. Full peak LR is too aggressive — it would damage the learned representations before the model can reorganize around code. A moderate re-warmup ($0.3$-$0.5 \\times \\alpha_{\\text{peak}}$) strikes the balance: enough learning rate to integrate code knowledge, but controlled enough to preserve general language capabilities. The short warmup recalibrates Adam's moment estimates for the new gradient statistics."
    }
  ]
};
