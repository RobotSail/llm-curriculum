// Focused module: Scaling Laws for Language Models
// Covers power-law scaling, Kaplan vs Chinchilla, compute-optimal training,
// irreducible loss, inference-aware scaling, and muP.
// Grounded in Goodfellow et al. (2016) Ch. 5 (capacity, generalization)
// and the Kaplan (2020), Hoffmann/Chinchilla (2022), Yang/muP (2022) literature.

export const scalingLawsLearning = {
  id: "B.1-scaling-laws-learning-easy",
  sectionId: "B.1",
  title: "Scaling Laws for Language Models",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "Why Scaling Laws Matter",
      content: "Training a frontier language model costs millions of dollars. Before committing that budget, you need answers to questions like: **How large should the model be? How much data do we need? What loss can we expect?**\n\nScaling laws give empirical answers. They describe how language model loss depends on three quantities:\n\n- **$N$**: the number of model parameters\n- **$D$**: the number of training tokens\n- **$C$**: the total compute budget (in FLOPs)\n\nThe core finding: loss follows **power laws** in each of these quantities. On a log-log plot, loss vs. compute traces a straight line. This predictability is remarkable — it means you can train small models, measure their loss, and extrapolate to predict the loss of models 100x or 1000x larger.\n\nAs Goodfellow et al. (2016, Ch. 5) note, the relationship between model capacity and generalization error is central to machine learning. Scaling laws make this relationship quantitative and predictive for neural language models."
    },
    {
      type: "mc",
      question: "A lab trains models at 5 different compute budgets spanning $10^{18}$ to $10^{22}$ FLOPs and plots loss vs. compute on a log-log scale. They observe a straight line. This implies the relationship between loss and compute is:",
      options: [
        "Linear — $L \\propto -C$, each doubling of compute gives a constant absolute loss reduction, implying fixed marginal returns regardless of current scale",
        "A power law — $L \\propto C^{-\\alpha}$ for some exponent $\\alpha$, meaning each doubling of compute gives a constant fractional reduction in reducible loss",
        "Exponential — $L \\propto e^{-\\alpha C}$, loss decreases exponentially with compute, meaning returns actually accelerate at larger scales",
        "Logarithmic — $L \\propto -\\log C$, loss decreases as the log of compute, meaning returns diminish significantly faster than a power law"
      ],
      correct: 1,
      explanation: "A straight line on a log-log plot is the signature of a power law: $\\log L = -\\alpha \\log C + \\text{const}$, which gives $L \\propto C^{-\\alpha}$. Each doubling of compute multiplies the reducible loss by $2^{-\\alpha}$ — a constant fractional (not absolute) reduction. This means diminishing absolute returns but smooth, predictable improvement."
    },
    {
      type: "info",
      title: "The Power-Law Form",
      content: "The standard functional form for scaling laws is:\n\n$$L(C) = aC^{-\\alpha} + L_\\infty$$\n\nwhere:\n- $a$ is a constant that depends on architecture, data quality, and other fixed choices\n- $\\alpha$ is the **scaling exponent** — typically around 0.05 for compute-optimal training\n- $L_\\infty$ is the **irreducible loss**: the loss you would achieve with infinite compute\n\nThe irreducible loss $L_\\infty$ corresponds to the **entropy of the data distribution** $H(P)$. This is the theoretical minimum — even a perfect model cannot predict inherently ambiguous next tokens. In natural language, there are genuinely many valid continuations of any sentence.\n\nThe quantity $L(C) - L_\\infty$ is the **reducible loss**: the gap between your model and perfection. Scaling laws tell you how this gap shrinks with compute. Crucially, it never reaches zero — you get diminishing returns, not a sudden leap to perfection.\n\nThis connects to a deep insight from Goodfellow et al. (Ch. 5): generalization error has an irreducible component (Bayes error) that no model can eliminate, regardless of capacity or data."
    },
    {
      type: "mc",
      question: "The irreducible loss $L_\\infty$ in the scaling law $L = aC^{-\\alpha} + L_\\infty$ represents:",
      options: [
        "The loss of the largest model currently in existence, representing a practical ceiling on performance",
        "The loss at initialization before any training, representing the random baseline",
        "Zero, because a sufficiently large model can memorize the training data and achieve perfect prediction",
        "The entropy of the data distribution — the minimum achievable loss reflecting genuine ambiguity in the data"
      ],
      correct: 3,
      explanation: "The irreducible loss is the entropy $H(P)$ of the true data distribution. Natural language is inherently stochastic — \"The cat sat on the ___\" has many valid completions, and no model can predict the specific one that was written. This entropy sets a floor that no amount of compute can breach. Estimating $L_\\infty$ is important for understanding how much headroom remains."
    },
    {
      type: "info",
      title: "Kaplan et al. (2020): The First Scaling Laws",
      content: "Kaplan et al. at OpenAI established the first systematic scaling laws for Transformers. They varied $N$ (60M to 1.5B parameters) and $D$ (up to 300B tokens) and found:\n\n$$L(N) \\approx \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}, \\quad L(D) \\approx \\left(\\frac{D_c}{D}\\right)^{\\alpha_D}$$\n\nwith $\\alpha_N \\approx 0.076$ and $\\alpha_D \\approx 0.095$. The critical conclusion: **loss decreases more steeply with model size than with data size** ($\\alpha_N$ is not too far from $\\alpha_D$, but their compute allocation recommendation amplified the difference).\n\nTheir compute-optimal prescription was roughly:\n$$N^* \\propto C^{0.73}, \\quad D^* \\propto C^{0.27}$$\n\nThis says: when you get more compute, spend most of it making the model bigger, not training on more data. Larger models are more \"sample-efficient\" — they extract more per training token.\n\nThis recommendation shaped an era of model development. GPT-3 (175B parameters, 300B tokens, ~1.7 tokens/parameter) followed this philosophy: very large model, relatively limited data."
    },
    {
      type: "mc",
      question: "Under Kaplan's scaling recommendations ($N^* \\propto C^{0.73}$, $D^* \\propto C^{0.27}$), a team with 10x more compute than their previous training run should primarily:",
      options: [
        "Train the same model for 10x longer on 10x more data, keeping model size fixed",
        "Scale both model size and data equally — roughly $\\sqrt{10} \\approx 3.2$x each",
        "Make the model roughly 5.4x larger and use about 1.9x more data, heavily favoring model size",
        "Use the extra compute for hyperparameter search on the original model, since tuning yields larger gains than scaling"
      ],
      correct: 2,
      explanation: "With $N^* \\propto C^{0.73}$: $10^{0.73} \\approx 5.4$x more parameters. With $D^* \\propto C^{0.27}$: $10^{0.27} \\approx 1.9$x more data. Kaplan's prescription heavily favors scaling parameters over data. This led to the \"big model, less data\" paradigm that Chinchilla later overturned."
    },
    {
      type: "info",
      title: "Chinchilla (Hoffmann et al., 2022): A Correction",
      content: "Two years later, DeepMind's Chinchilla paper fundamentally revised Kaplan's recommendations. They trained over 400 models from 70M to 16B parameters and found a different compute-optimal split:\n\n$$N^* \\propto C^{0.50}, \\quad D^* \\propto C^{0.50}$$\n\nModel size and data should scale **equally** with compute. The practical rule of thumb:\n\n$$\\text{Optimal tokens} \\approx 20 \\times \\text{parameters}$$\n\nSo a 10B model should train on ~200B tokens, and a 70B model on ~1.4T tokens.\n\n**Why did Kaplan get it wrong?** A subtle but critical methodological issue: Kaplan used the **same learning rate schedule** (long cosine decay) for all runs, regardless of how many tokens they trained on. Short runs with less data were effectively undertrained — not because they lacked data, but because their learning rate hadn't decayed enough. This made data look less valuable than it actually is.\n\nWhen Chinchilla properly tuned the learning rate schedule for each configuration, data turned out to be just as important as model size."
    },
    {
      type: "mc",
      question: "GPT-3 was trained with 175B parameters on 300B tokens (~1.7 tokens per parameter). Under the Chinchilla-optimal ratio of ~20 tokens per parameter, how should the compute have been allocated?",
      options: [
        "Approximately 40B parameters on 800B tokens — smaller model, much more data, achieving similar or better loss at lower total compute",
        "175B parameters on 3.5T tokens — same model, far more data",
        "175B parameters on 300B tokens — GPT-3 was already compute-optimal since more data wasn't available at the time",
        "700B parameters on 14T tokens — both the model and data were too small by a factor of 4"
      ],
      correct: 0,
      explanation: "GPT-3's compute of roughly $C \\approx 6 \\times 175\\text{B} \\times 300\\text{B} = 3.15 \\times 10^{23}$ FLOPs, allocated Chinchilla-optimally ($N^* \\approx D^* \\approx \\sqrt{C/6}$), gives approximately $N^* \\approx 40\\text{B}$ and $D^* \\approx 800\\text{B}$. A 40B model on 800B tokens would achieve better loss than 175B on 300B tokens at similar compute. GPT-3 was massively \"overparameterized\" relative to its data."
    },
    {
      type: "info",
      title: "Compute, Parameters, and Data: The FLOPs Connection",
      content: "A useful approximation connects compute to parameters and data:\n\n$$C \\approx 6ND$$\n\nwhere $C$ is FLOPs, $N$ is parameters, and $D$ is training tokens. The factor of 6 comes from: each token requires a forward pass (~$2N$ FLOPs for matrix multiplications) and a backward pass (~$4N$ FLOPs).\n\nGiven a fixed compute budget $C$, you must choose how to split it between $N$ and $D$. From $C = 6ND$:\n$$D = \\frac{C}{6N}$$\n\nMaking the model bigger ($N$ up) forces you to train on less data ($D$ down) at fixed compute, and vice versa. The scaling law tells you which split minimizes loss.\n\nUnder Chinchilla's equal-exponent finding ($N^* \\propto C^{0.5}$, $D^* \\propto C^{0.5}$), the optimal split keeps the ratio $D/N \\approx 20$ constant. This means:\n- Double your compute → make the model $\\sqrt{2} \\approx 1.4$x bigger AND use $1.4$x more data\n- 10x compute → $\\sqrt{10} \\approx 3.2$x bigger model AND $3.2$x more data"
    },
    {
      type: "mc",
      question: "A team has a compute budget of $C = 6 \\times 10^{22}$ FLOPs. Using the approximation $C \\approx 6ND$ and the Chinchilla-optimal ratio of 20 tokens per parameter, what is the optimal model size?",
      options: [
        "$N^* = 10^{22} / 20 = 5 \\times 10^{20}$ — simply divide the FLOP count by the token ratio",
        "$N^* \\approx \\sqrt{C/120} \\approx \\sqrt{5 \\times 10^{20}} \\approx 22\\text{B}$ — derived from $C = 6N \\cdot 20N = 120N^2$",
        "$N^* = C / 6 = 10^{22}$ — all compute goes to parameters under Chinchilla",
        "$N^* = 20 \\times \\sqrt{C} \\approx 20 \\times 7.7 \\times 10^{10} \\approx 1.5\\text{T}$ — scale parameters by the square root of FLOPs times the token ratio"
      ],
      correct: 1,
      explanation: "From $C = 6ND$ and $D = 20N$: $C = 6N(20N) = 120N^2$, so $N^* = \\sqrt{C/120}$. Plugging in: $N^* = \\sqrt{6 \\times 10^{22}/120} = \\sqrt{5 \\times 10^{20}} \\approx 22\\text{B}$ parameters, trained on $D^* = 20 \\times 22\\text{B} = 440\\text{B}$ tokens. This is a concrete recipe: given a FLOP budget, compute the optimal model size directly."
    },
    {
      type: "info",
      title: "Beyond Chinchilla: Inference-Aware Scaling",
      content: "Chinchilla optimizes for **training cost** alone. But in production, **inference cost** dominates: a model serves millions of queries after training. Inference cost scales with model size $N$ but is independent of training data $D$.\n\nThis changes the optimal allocation. Consider two models with the same training compute:\n- Model A: 70B parameters, 1.4T tokens (Chinchilla-optimal)\n- Model B: 20B parameters, 4.9T tokens (\"overtrained\")\n\nModel A has slightly lower loss. But Model B is 3.5x cheaper to serve per query. If you serve billions of queries, Model B has a vastly lower **total cost of ownership**.\n\nInference-aware scaling laws (Sardana & Frankle, 2024) formalize this: when you account for inference cost, the optimal strategy shifts to **smaller models trained on more data**. This is why modern models like Llama 3 (8B on 15T tokens, ~1875 tokens/parameter) far exceed the 20 tokens/parameter Chinchilla ratio.\n\nThe principle: **overtrain relative to Chinchilla** when inference cost matters. You pay a small penalty in training efficiency to get a much cheaper model to deploy."
    },
    {
      type: "mc",
      question: "Llama 3 8B was trained on 15T tokens (~1875 tokens per parameter), far exceeding the Chinchilla-optimal ~20 tokens per parameter. Why is this a good strategy?",
      options: [
        "The Chinchilla ratio was wrong — newer research shows the optimal ratio is actually ~2000 tokens per parameter for all models",
        "Llama 3 uses a different architecture that benefits more from data than from parameters, shifting the optimal ratio",
        "Meta had excess data but limited GPU hours, so they had no choice but to train a small model on more data",
        "Inference cost scales with model size, so training a smaller model on more data gives better loss per inference FLOP, even though training is less compute-efficient"
      ],
      correct: 3,
      explanation: "Chinchilla-optimal minimizes training cost for a given loss. But Meta needs to serve Llama to millions of users. A 70B Chinchilla-optimal model would achieve similar training loss at the same compute but cost 8.75x more per inference query. By overtraining 8B on 15T tokens, they get a model that is slightly worse than compute-optimal but dramatically cheaper to deploy. Training compute is a one-time cost; inference is ongoing."
    },
    {
      type: "info",
      title: "Transferring Hyperparameters: Maximal Update Parameterization (muP)",
      content: "A practical challenge in scaling: the optimal **learning rate** changes with model size. Training a 7B model requires a different LR than training a 125M model. Running hyperparameter sweeps at 7B scale is prohibitively expensive.\n\n**Maximal Update Parameterization** ($\\mu$P, Yang et al., 2022) solves this by defining a parameterization where the optimal learning rate **stays constant** across model widths.\n\nThe key idea: in standard parameterization (SP), as you increase model width $d$, the scale of activations, gradients, and updates all change in complicated ways. The optimal LR must compensate for these shifts. In $\\mu$P, initialization scales and per-layer learning rate multipliers are set so that the **magnitudes of activations, gradients, and parameter updates all remain $\\Theta(1)$** as width varies.\n\nDifferent layer types need different rules:\n- **Input embeddings**: scale as $1/\\sqrt{d}$ (standard) but with a $1/d$ multiplier on updates\n- **Hidden layers**: scale as $1/\\sqrt{d}$ with a $1/\\sqrt{d}$ multiplier on updates\n- **Output layer**: scale as $1/d$ with standard updates\n\nThe practical payoff: tune hyperparameters on a small proxy model (e.g., 40M parameters) and transfer them directly to the target scale (e.g., 7B). This saves enormous amounts of compute in the hyperparameter search phase."
    },
    {
      type: "mc",
      question: "A team tunes the learning rate for a 125M parameter proxy model using $\\mu$P and finds the optimal value is $\\eta = 3 \\times 10^{-3}$. They want to train a 7B model. Under $\\mu$P, what learning rate should they use?",
      options: [
        "$\\eta = 3 \\times 10^{-3}$ — the same value, because $\\mu$P ensures optimal LR transfers directly across scales",
        "$\\eta = 3 \\times 10^{-3} \\times \\sqrt{125\\text{M}/7\\text{B}} \\approx 4 \\times 10^{-4}$ — scale down by the square root of the width ratio",
        "$\\eta = 3 \\times 10^{-3} \\times (125\\text{M}/7\\text{B}) \\approx 5 \\times 10^{-5}$ — scale down linearly with model size",
        "$\\eta = 3 \\times 10^{-3} \\times (7\\text{B}/125\\text{M}) \\approx 0.17$ — scale up since larger models need larger updates per step"
      ],
      correct: 0,
      explanation: "The entire point of $\\mu$P is that the optimal learning rate transfers without modification. The parameterization handles the width-dependent scaling internally through initialization and per-layer multipliers. You tune once at small scale, then use the exact same LR at large scale. Without $\\mu$P, the optimal LR typically decreases with model size, requiring expensive sweeps at every scale."
    },
    {
      type: "info",
      title: "From Loss to Capabilities: The Emergence Debate",
      content: "Scaling laws predict **pretraining loss** with remarkable accuracy. But practitioners care about **downstream capabilities**: can the model solve math problems, write code, follow instructions?\n\nThe relationship between loss and capabilities is more complex:\n\n**Smooth view**: Average performance across many benchmarks improves smoothly with scale. Each reduction in loss translates to gradual improvement in the probability of generating correct tokens.\n\n**Emergent view**: Individual tasks sometimes show sharp transitions — a model \"can't\" do a task below some scale, then suddenly \"can\" above it. Wei et al. (2022) documented several such cases.\n\n**Resolution** (Schaeffer et al., 2023): Much of the apparent emergence is an artifact of **nonlinear evaluation metrics**. Exact-match accuracy is a hard threshold — the model either gets the entire answer right or scores 0. Underlying token-level probabilities improve smoothly, but the step function of exact-match creates a sharp visible transition.\n\nWhen you use smoother metrics (token-level probability, Brier score), the transitions are generally smooth. Some tasks may genuinely require threshold capabilities, but pure metric artifacts explain most reported emergence."
    },
    {
      type: "mc",
      question: "A 3B model scores 2% on a math benchmark (exact-match accuracy), while a 30B model scores 45%. This looks like a sharp \"emergent\" capability. Schaeffer et al. (2023) would most likely explain this as:",
      options: [
        "A real discontinuity — mathematical reasoning requires a critical mass of parameters to represent multi-step logical chains, and the 3B model falls below this threshold",
        "The 30B model was trained on math-specific data that the 3B model didn't see, making this a data composition effect rather than a genuine scaling effect",
        "The metric artifact explanation: the 3B model's per-token probabilities for math are only slightly worse, but exact-match magnifies this into 2% vs 45% because partial credit is impossible",
        "Random variation — with different random seeds and initialization, the 3B model might also reach 45% since small models have high variance across training runs"
      ],
      correct: 2,
      explanation: "Exact-match accuracy is a hard threshold: the model must get every token in the answer correct. If the per-token accuracy is 90%, a 10-token answer has only $0.9^{10} \\approx 35\\%$ exact-match probability. At 80% per-token: $0.8^{10} \\approx 10\\%$. At 70%: $0.7^{10} \\approx 3\\%$. The difference between 70% and 90% per-token accuracy is modest, but exact-match amplifies it from 3% to 35%. The improvement is smooth at the token level; the metric creates the sharp transition."
    }
  ]
};
