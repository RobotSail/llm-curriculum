// Focused learning module: Exponential Moving Average (EMA) of Model Weights
// Section 0.3: Optimization Theory
// Single concept: EMA weight averaging for checkpoint selection and training stabilization.

export const emaAveragingLearning = {
  id: "0.3-ema-averaging-learning-easy",
  sectionId: "0.3",
  title: "EMA Weight Averaging",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Why Individual Checkpoints Are Noisy",
      content: "During training, the model's weights follow a noisy trajectory through the loss landscape. Each SGD or Adam update nudges $\\theta_t$ toward lower loss, but the stochastic gradient introduces noise — the update direction varies from mini-batch to mini-batch.\n\nAt any given step $t$, the checkpoint $\\theta_t$ sits at a noisy instantaneous position along this trajectory. If you evaluate $\\theta_t$ and $\\theta_{t+100}$, their validation losses may differ significantly even though the model hasn't meaningfully changed in capability. This is especially true late in training when the learning rate is small and the model orbits near a minimum rather than descending toward it.\n\nThe key insight: the **average** of multiple nearby checkpoints is often better than any individual checkpoint. Averaging cancels out the per-step noise while preserving the signal (the steady drift toward lower loss). This is the same principle behind why ensemble predictions outperform individual models — but applied to the weight space itself rather than the output space.\n\nThe practical question is **how** to average. Keeping all checkpoints in memory and computing a uniform average is expensive. Exponential moving averaging provides an elegant online solution."
    },
    {
      type: "mc",
      question: "Late in training, a model's validation loss oscillates between 2.31 and 2.35 across consecutive checkpoints despite the learning rate being very small. What best explains this oscillation?",
      options: [
        "The model is overfitting to the training set, with each checkpoint memorizing different subsets of examples and producing different validation losses",
        "The validation set is too small to produce stable loss estimates, so the fluctuation reflects evaluation noise rather than any real change in model weights",
        "Stochastic gradient noise causes the weights to orbit near a minimum rather than converging to it exactly, so each checkpoint samples a slightly different position in the basin",
        "The learning rate schedule has a bug that causes periodic spikes, pushing the model away from the minimum every few steps before it recovers"
      ],
      correct: 2,
      explanation: "With a small learning rate, the model has largely converged but SGD noise prevents it from sitting exactly at the minimum. Each step moves the weights slightly in a random direction around the basin. The checkpoint at any step reflects this noisy position, producing validation loss fluctuations. Averaging over these positions gives a point closer to the basin center, which typically has lower loss than any individual noisy sample."
    },
    {
      type: "info",
      title: "The EMA Update Rule",
      content: "**Exponential Moving Average** (EMA) maintains a shadow copy of the model weights, updated after each training step:\n\n$$\\tilde{\\theta}_t = \\alpha \\, \\tilde{\\theta}_{t-1} + (1 - \\alpha) \\, \\theta_t$$\n\nwhere $\\alpha \\in [0, 1)$ is the **decay rate** (typically 0.999 or 0.9999) and $\\theta_t$ are the current training weights. The EMA weights $\\tilde{\\theta}_t$ are never used for training — only for evaluation and final deployment.\n\nUnrolling the recursion reveals what EMA computes:\n\n$$\\tilde{\\theta}_t = (1-\\alpha) \\sum_{i=0}^{t} \\alpha^{t-i} \\, \\theta_i$$\n\nEach past checkpoint is weighted exponentially: recent checkpoints contribute most, and old checkpoints fade with factor $\\alpha^{\\Delta t}$. The **effective window** — the number of past steps that contribute meaningfully — is approximately $1/(1-\\alpha)$:\n- $\\alpha = 0.999$: window $\\approx 1000$ steps\n- $\\alpha = 0.9999$: window $\\approx 10000$ steps\n\nThe elegance is that this requires only **one extra copy** of the model weights in memory and **one cheap update** per step (a scalar multiply and add). No need to store thousands of checkpoints."
    },
    {
      type: "mc",
      question: "A training run uses EMA with $\\alpha = 0.9999$ and performs 100,000 total steps. A checkpoint from step 90,000 contributes to the final EMA weights $\\tilde{\\theta}_{100000}$ with relative weight proportional to $\\alpha^{10000} = 0.9999^{10000}$. What is this approximately?",
      options: [
        "Approximately $0.37$ — the checkpoint from 10,000 steps ago still contributes about a third of its original weight to the running average",
        "Approximately $0.90$ — with such a high decay rate, checkpoints barely decay even over 10,000 steps, so nearly all history is preserved",
        "Approximately $0.05$ — the checkpoint has decayed to about 5% influence, contributing minimally but not negligibly to the final EMA",
        "Approximately $0.00$ — 10,000 steps of decay at any rate less than 1.0 reduces the weight to effectively zero"
      ],
      correct: 0,
      explanation: "$0.9999^{10000} = e^{10000 \\ln(0.9999)} \\approx e^{10000 \\times (-0.0001)} = e^{-1} \\approx 0.368$. This confirms the effective window formula: with $\\alpha = 0.9999$, the window is $\\approx 1/(1-0.9999) = 10000$ steps. A checkpoint exactly one window-length ago contributes with weight $e^{-1} \\approx 0.37$ — still significant. Checkpoints within the window contribute substantially; those far outside it fade to negligible influence."
    },
    {
      type: "info",
      title: "Choosing the Decay Rate",
      content: "The decay rate $\\alpha$ controls the bias-variance trade-off in weight averaging:\n\n**High decay** ($\\alpha = 0.9999$, window $\\approx 10000$ steps):\n- Averages over many checkpoints, strongly smoothing out SGD noise\n- But may average over weights from different training phases, introducing bias if the model is still improving rapidly\n- Best late in training when the model is near convergence\n\n**Low decay** ($\\alpha = 0.99$, window $\\approx 100$ steps):\n- Averages over fewer checkpoints, tracking the current training state more closely\n- Less smoothing, but adapts faster if the model is still changing significantly\n- Useful early in training or when the learning rate is large\n\n**Practical choices in LLM training:**\n- Most large-scale runs use $\\alpha \\in [0.999, 0.9999]$\n- Some systems use a **warmup schedule** for $\\alpha$: start with a lower decay (e.g., 0.99) and increase to 0.9999 over the first few thousand steps. This prevents the EMA from being dominated by the random initialization\n- A common heuristic: set the EMA window to roughly match the number of steps in one epoch, so the average spans a full pass over the data\n\nThe EMA weights are only used for **evaluation and deployment** — the training optimizer always operates on the raw weights $\\theta_t$. Using EMA weights for training would dampen the gradient signal and slow convergence."
    },
    {
      type: "mc",
      question: "A team trains a 7B model for 500,000 steps. They compare EMA at $\\alpha = 0.99$ vs $\\alpha = 0.9999$ by evaluating both EMA checkpoints at the final step. The $\\alpha = 0.99$ EMA checkpoint has slightly higher validation loss. Why?",
      options: [
        "The $\\alpha = 0.99$ decay is too aggressive, causing the EMA to diverge from the true weights and drift toward the initial random parameter values over 500,000 steps",
        "The $\\alpha = 0.9999$ EMA performs implicit regularization by heavily weighting earlier checkpoints, which prevents overfitting to late-training data distribution",
        "Both EMAs produce identical weights at 500,000 steps because the exponential weighting converges regardless of $\\alpha$ given sufficient training duration",
        "The $\\alpha = 0.99$ window (~100 steps) averages over too few checkpoints to smooth out SGD noise effectively, so the EMA weights are nearly as noisy as a single checkpoint"
      ],
      correct: 3,
      explanation: "With $\\alpha = 0.99$, the effective window is only ~100 steps. The EMA essentially tracks the last 100 training steps with modest smoothing. Late in training when the model orbits near a minimum, this short window doesn't average away enough of the per-step noise. The $\\alpha = 0.9999$ EMA with its ~10,000-step window averages over a much larger neighborhood, better canceling the noise to land closer to the basin center. The short-window EMA is not diverging or drifting — it's simply not averaging enough."
    },
    {
      type: "info",
      title: "EMA vs Polyak Averaging vs Stochastic Weight Averaging",
      content: "EMA is one of several weight averaging strategies. Understanding the differences clarifies when each is appropriate:\n\n**Polyak averaging** (uniform tail averaging): Compute $\\bar{\\theta} = \\frac{1}{T - t_0} \\sum_{t=t_0}^{T} \\theta_t$, averaging all checkpoints from step $t_0$ onward with equal weight. This has optimal convergence rate guarantees for convex problems, but requires choosing $t_0$ carefully. Too early and you include weights from before convergence; too late and you average too few checkpoints.\n\n**EMA** (exponential weighting): No need to choose a start point — old checkpoints naturally fade away. The decay rate $\\alpha$ replaces the start-point decision with a smoother window-size parameter. EMA is more robust to non-stationary dynamics (the loss landscape changing during training).\n\n**Stochastic Weight Averaging** (SWA, Izmailov et al. 2018): Average checkpoints sampled at regular intervals (e.g., every 1000 steps) during a high-constant-learning-rate phase after the main schedule ends. SWA explicitly encourages exploration of the loss surface with a large LR, then averages the diverse points visited. It finds wider, flatter minima than EMA.\n\nIn frontier LLM training, **EMA is dominant** because:\n- It's simple to implement (one extra copy, one update per step)\n- It doesn't require a special LR phase like SWA\n- It doesn't require storing multiple checkpoints like Polyak\n- The improvement over the raw final checkpoint is consistent: typically 0.5-2% better validation loss"
    },
    {
      type: "mc",
      question: "A researcher considers Polyak averaging (uniform average from step $t_0$ to $T$) vs EMA for a training run where the model's loss improves rapidly until step 200K, then plateaus with small oscillations until step 500K. What challenge does Polyak averaging face that EMA avoids?",
      options: [
        "Polyak averaging requires choosing $t_0$: set it too early and suboptimal pre-plateau weights dilute the average, set it too late and too few weights are averaged for effective smoothing",
        "Polyak averaging cannot be computed online and requires storing all checkpoints from $t_0$ to $T$ in memory, making it infeasible for models with billions of parameters",
        "Polyak averaging produces a uniform distribution over weight space rather than concentrating on the minimum, so it always converges to a worse solution than EMA",
        "Polyak averaging is incompatible with Adam because the adaptive learning rates create non-uniform step sizes that invalidate the equal-weighting assumption"
      ],
      correct: 0,
      explanation: "The main practical challenge with Polyak averaging is choosing the start point $t_0$. If $t_0 < 200K$, the average includes rapidly-changing early weights that are far from the converged solution, pulling the average away from the minimum. If $t_0 = 400K$, only 100K steps are averaged — enough but sensitive to the choice. EMA sidesteps this entirely: old weights fade naturally with exponential decay, so pre-convergence weights automatically contribute negligibly regardless of when convergence happened. Note that Polyak averaging can be computed with a running sum (no need to store checkpoints), but the start-point sensitivity remains."
    },
    {
      type: "info",
      title: "EMA in Practice: Implementation Details",
      content: "Implementing EMA in large-scale LLM training involves several practical considerations:\n\n**Memory cost**: The EMA shadow weights require one full copy of the model in the same dtype as the model weights. For a 70B parameter model in float32, this is 280 GB. In practice, EMA weights are often stored in float32 even when training uses mixed precision, because the slow-moving average benefits from higher precision in the accumulation.\n\n**Update frequency**: EMA is updated every step by default, but some implementations update every $k$ steps to reduce overhead. This is mathematically equivalent to using $\\alpha^k$ as the effective per-update decay. With $\\alpha = 0.999$ and $k = 10$, each update uses $\\alpha^{10} \\approx 0.99$.\n\n**Interaction with distributed training**: In data-parallel training, all replicas have identical weights after gradient synchronization, so the EMA can be maintained on a single process or computed identically across all processes. In model-parallel settings (TP/PP), the EMA is maintained per-shard, mirroring the partitioning of the training weights.\n\n**Evaluation protocol**: During training, periodically evaluate both the raw weights $\\theta_t$ and the EMA weights $\\tilde{\\theta}_t$. The gap between them is informative:\n- Large gap (EMA much better): training is noisy, consider reducing the learning rate\n- No gap: the model has converged well, EMA provides little benefit\n- EMA worse: the decay rate is too high (window too long), including stale weights"
    },
    {
      type: "mc",
      question: "A team trains with EMA ($\\alpha = 0.9999$) and observes that their EMA checkpoint consistently has **higher** validation loss than the raw checkpoint throughout training. What is the most likely cause?",
      options: [
        "The model is underfitting — EMA smoothing removes the noise that was helping the model escape suboptimal regions and explore the loss landscape more thoroughly",
        "EMA is incompatible with the optimizer being used — certain adaptive methods like Adam produce weight trajectories where averaging degrades rather than improves performance",
        "The EMA decay is too high for the training dynamics: the ~10,000 step window includes weights from much earlier when the model was significantly worse, pulling the average above the current optimum",
        "The EMA update formula has a sign error — the shadow weights are moving away from rather than toward the training weights, causing systematic divergence"
      ],
      correct: 2,
      explanation: "When EMA is worse than raw weights, the window is too long relative to the rate of improvement. A 10,000-step window averages in weights from a time when the model was meaningfully worse. The fix is to reduce $\\alpha$ (e.g., to 0.999 for a ~1,000-step window) or to restart the EMA shadow weights from the current training weights periodically. This issue is most common early in training or during phases of rapid improvement. EMA works with any optimizer — the issue is purely about the window length relative to the improvement rate."
    },
    {
      type: "info",
      title: "EMA and Generalization: The Flat Minima Connection",
      content: "Why does averaging nearby checkpoints improve generalization? The connection runs through the geometry of the loss landscape.\n\nThe loss surface of a neural network contains many local minima (and saddle points). These minima vary in **sharpness**: some sit in narrow, steep basins while others occupy broad, flat regions. The sharpness of a minimum is related to the eigenvalues of the Hessian $\\nabla^2 \\mathcal{L}$ at that point — large eigenvalues mean steep curvature.\n\n**Flat minima generalize better** (Hochreiter & Schmidhuber 1997, Keskar et al. 2017). Intuitively, a flat minimum is robust to perturbations — if the test distribution shifts the effective weights slightly, a flat minimum still has low loss. A sharp minimum is fragile: tiny perturbations jump to high-loss regions.\n\nWeight averaging acts as an **implicit bias toward flat minima**. Consider averaging two checkpoints $\\theta_a$ and $\\theta_b$ near a minimum:\n- In a **flat** basin: the midpoint $(\\theta_a + \\theta_b)/2$ also has low loss, because the basin is broad\n- In a **sharp** basin: the midpoint may have high loss, because even small deviations from the exact minimum increase the loss rapidly\n\nEMA continuously averages along the trajectory, producing weights that naturally lie in regions where the loss surface is flat enough to tolerate averaging. This geometric selection effect is why EMA checkpoints often generalize better — they preferentially represent flat, robust solutions."
    },
    {
      type: "mc",
      question: "Two training runs converge to different minima: Run A finds a sharp minimum (high Hessian eigenvalues), Run B finds a flat minimum (low Hessian eigenvalues). Both achieve identical training loss. EMA is applied to both. Which run benefits more from EMA, and why?",
      options: [
        "Run A benefits more because EMA's smoothing effect dampens the sharp curvature, effectively reshaping the loss surface to make the minimum appear flatter to the optimizer",
        "Run B benefits more because the flat basin means averaged weights still have low loss, while in Run A's sharp basin the averaged weights may sit at a higher-loss point away from the narrow optimum",
        "Both benefit equally because EMA operates on the weights, not the loss surface, so the curvature of the minimum is irrelevant to the averaging effect",
        "Neither benefits because EMA only helps during the non-convergent phase of training, and both runs have already converged to their respective minima"
      ],
      correct: 1,
      explanation: "Run B benefits more. In a flat basin, nearby points along the training trajectory all have similar loss, so their average (the EMA checkpoint) also has low loss. In Run A's sharp basin, the noisy trajectory visits points on the steep walls of the basin — their average may not sit at the narrow bottom, landing at a higher-loss point. EMA doesn't reshape the loss surface — it selects a point in weight space that is the centroid of the trajectory, and this centroid has low loss only if the basin is wide enough to contain the trajectory's spread."
    },
    {
      type: "mc",
      question: "A team uses EMA with $\\alpha = 0.999$ during training of a 13B model. They want to reduce memory overhead. Someone proposes updating the EMA only every 100 steps instead of every step, using $\\alpha_{\\text{eff}} = 0.999^{100}$ per update. What is the consequence?",
      options: [
        "The effective window shrinks from ~1,000 steps to ~10 updates (1,000 steps), producing identical EMA weights since the mathematical equivalence is exact",
        "The EMA loses all benefit because updating every 100 steps means the shadow weights are always 100 steps stale, introducing a fixed lag that cannot be compensated",
        "The memory is halved because the EMA weights only need to exist during update steps and can be offloaded to CPU between updates, saving GPU memory",
        "The effective window changes from ~1,000 steps to ~100 updates, but the EMA now only sees every 100th checkpoint, potentially missing short-lived weight configurations that occur between updates"
      ],
      correct: 3,
      explanation: "Updating every 100 steps with $\\alpha_{\\text{eff}} = 0.999^{100} \\approx 0.905$ gives an effective window of ~$1/(1-0.905) \\approx 10.5$ updates, or ~1,050 steps — close to the original ~1,000 step window. However, the EMA now samples every 100th checkpoint instead of every checkpoint. It misses the 99 intermediate positions. For slow-moving late training this barely matters, but it means the EMA is a coarser approximation. Memory is NOT saved — the shadow weights must persist in memory between updates. The approach saves compute (fewer EMA updates) but not memory."
    }
  ]
};
