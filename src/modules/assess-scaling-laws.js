// B.1 Scaling Laws — per-section test (split from assess-branch-b.js)

export const scalingLawsAssessment = {
  id: "B.1-assess",
  sectionId: "B.1",
  title: "Assessment: Scaling Laws",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The original Kaplan et al. (2020) scaling laws suggested that language model loss scales as a power law in model size $N$, dataset size $D$, and compute $C$. A key (later-revised) recommendation was:",
      options: ["Train all models for exactly one epoch regardless of size, since repeated passes over the data yield diminishing returns at scale", "Use a fixed learning rate across all model scales, since the optimal rate is determined by batch size rather than parameter count", "Keep model size fixed and only increase the dataset size, since data is cheaper to acquire than additional GPU compute", "Scale model size faster than dataset size — larger models are more sample-efficient, so allocate most additional compute to parameters"],
      correct: 3,
      explanation: "Kaplan et al. found that loss decreases more steeply with model size than with data. Their recommendation was to scale $N$ faster than $D$ when compute grows — roughly $N \\propto C^{0.73}$ and $D \\propto C^{0.27}$. The Chinchilla paper later overturned this by showing the exponents were closer to equal."
    },
    {
      type: "mc",
      question: "The Chinchilla (Hoffmann et al., 2022) scaling law fundamentally revised Kaplan's recommendations. The compute-optimal prescription is approximately:",
      options: ["1 token per parameter — keep models large and data small to maximize model capacity per FLOP", "200 tokens per parameter — use vastly more data than parameters to ensure thorough coverage of the distribution", "20 tokens per parameter — scale model size and data equally with compute for balanced optimization", "The ratio does not matter as long as total FLOPs are fixed, since loss depends only on aggregate compute"],
      correct: 2,
      explanation: "Chinchilla showed the compute-optimal ratio is roughly 20 tokens per parameter. A 10B parameter model should train on ~200B tokens. This means Kaplan-era models like the original GPT-3 (175B parameters, 300B tokens, ~1.7 tokens/param) were significantly undertrained relative to their size."
    },
    {
      type: "mc",
      question: "Why did the Kaplan and Chinchilla scaling laws arrive at different compute-optimal allocations between model size $N$ and data $D$?",
      options: ["Kaplan did not tune the learning rate schedule for smaller token budgets, biasing results toward larger models appearing more efficient", "They used different model architectures (RNNs vs Transformers), making their scaling exponents incomparable across studies", "Chinchilla used a larger vocabulary, making each token more informative and thus shifting the optimal data-to-parameter ratio", "The two studies measured different loss functions, with Kaplan using token-level cross-entropy and Chinchilla using bits-per-byte"],
      correct: 0,
      explanation: "A critical methodological issue: Kaplan used a fixed learning rate schedule (cosine decay over a long horizon) even for short runs with less data. This meant small-data runs were undertrained — not because they had too little data, but because their LR schedule was suboptimal. When Chinchilla properly tuned the schedule for each configuration, data turned out to be as valuable as model size."
    },
    {
      type: "mc",
      question: "Maximal Update Parameterization ($\\mu$P) addresses a practical problem in scaling research. What problem does it solve?",
      options: [
        "It eliminates the need for warmup in the learning rate schedule by normalizing gradient magnitudes at initialization, allowing training to start at the peak rate",
        "It enables hyperparameters (especially learning rate) tuned on a small model to transfer directly to larger models without re-tuning",
        "It ensures all layers have equal gradient norms regardless of depth by rescaling the backward pass, preventing vanishing and exploding gradient issues",
        "It replaces Adam with a scale-invariant optimizer that adjusts its update rule based on model width, removing the need for per-scale tuning"
      ],
      correct: 1,
      explanation: "$\\mu$P (Yang et al., 2022) defines a parameterization where the optimal learning rate remains stable across model widths. You tune hyperparameters on a small \"proxy\" model (e.g., 40M params) and transfer them to the full-scale model (e.g., 6.7B params). Without $\\mu$P, the optimal LR shifts with scale, making large-scale HP sweeps prohibitively expensive."
    },
    {
      type: "mc",
      question: "Inference-aware scaling laws (e.g., Sardana & Frankle, 2024) modify the Chinchilla-optimal strategy by accounting for deployment costs. Their key recommendation is:",
      options: ["Use the largest possible model to minimize total training cost, amortizing the higher inference expense over the training savings", "Use quantization to make the Chinchilla-optimal model cheaper at inference without changing the training recipe itself", "Distill the Chinchilla-optimal large model into a smaller one after training to recover inference efficiency", "Train a smaller-than-Chinchilla-optimal model on significantly more data, because inference cost scales with model size, not training data"],
      correct: 3,
      explanation: "When you account for inference cost (which depends on model size but not training data), the optimal strategy shifts: train a smaller model on more data than Chinchilla prescribes. A model that is \"overtrained\" relative to Chinchilla is slightly worse in loss but much cheaper to serve. This is why Llama models train on far more tokens per parameter than Chinchilla suggests."
    },
    {
      type: "mc",
      question: "Power-law scaling of loss $L(C) = aC^{-\\alpha} + L_\\infty$ implies which of the following about the returns from increasing compute?",
      options: ["Each doubling of compute yields a constant absolute improvement in loss regardless of the current loss level", "There is a critical compute threshold beyond which loss drops suddenly, marking the onset of emergent capabilities", "Each doubling of compute yields diminishing absolute improvements — you get a fixed multiplicative reduction in reducible loss $L - L_\\infty$", "Returns are increasing — larger models improve faster per FLOP, making each successive doubling more valuable than the last"],
      correct: 2,
      explanation: "A power law $L - L_\\infty \\propto C^{-\\alpha}$ means doubling compute multiplies the reducible loss by $2^{-\\alpha}$ — a constant fractional reduction. In absolute terms, each doubling gives less improvement because $L - L_\\infty$ is shrinking. There are no sudden thresholds or increasing returns; the curve is smooth and concave on a log-log plot."
    },
    {
      type: "mc",
      question: "Scaling laws predict pretraining loss, but practitioners care about downstream task performance. Research on predicting downstream capabilities from loss has found:",
      options: ["Average downstream accuracy often improves smoothly with loss, but individual tasks can show sharp \"emergent\" transitions when measured with nonlinear metrics like exact-match accuracy", "There is no reliable relationship between pretraining loss and downstream tasks, since each task depends on entirely different model capabilities", "Downstream task accuracy is a simple linear function of pretraining loss, making it straightforward to predict benchmark scores from loss alone", "Downstream performance follows the same power law exponent as training loss, with identical scaling coefficients across all tasks and benchmarks"],
      correct: 0,
      explanation: "The relationship between loss and downstream performance is nuanced. Schaeffer et al. (2023) showed that \"emergence\" often arises from the choice of metric: exact-match accuracy is a nonlinear, threshold-like function of per-token probabilities. When you switch to smoother metrics (like Brier score or token-level probability), performance improves continuously. But some tasks genuinely require a threshold level of capability."
    },
    {
      type: "mc",
      question: "In the Chinchilla scaling framework, suppose you have a compute budget of $C$ FLOPs and want to minimize loss. The approximate relationship between optimal model size $N^*$, optimal data $D^*$, and compute is:",
      options: [
        "$N^* \\propto C$ and $D^*$ is constant — all additional compute goes to model size with fixed data",
        "$N^* \\propto C^{0.5}$ and $D^* \\propto C^{0.5}$ — both scale as the square root of compute",
        "$N^* \\propto C^{0.73}$ and $D^* \\propto C^{0.27}$ — parameters scale much faster than data",
        "$N^* \\propto C^{0.3}$ and $D^* \\propto C^{0.7}$ — data scales much faster than parameters"
      ],
      correct: 1,
      explanation: "Under Chinchilla, both $N^*$ and $D^*$ scale approximately as $C^{0.5}$ (since $C \\approx 6ND$, equal exponents imply $N^* \\propto \\sqrt{C/6}$ and $D^* \\propto \\sqrt{C/6}$). The exponents are close to equal — roughly $a \\approx b \\approx 0.5$ — in contrast to Kaplan's $0.73/0.27$ split. This equal scaling is what leads to the stable ~20 tokens/parameter ratio."
    },
    {
      type: "mc",
      question: "The irreducible loss $L_\\infty$ in scaling laws $L = aC^{-\\alpha} + L_\\infty$ corresponds to:",
      options: ["The loss achieved by the largest model ever trained, representing the current practical lower bound of achievable performance", "Zero, since a sufficiently large model can memorize any dataset and achieve perfect next-token prediction on every sequence", "The loss of a randomly initialized model before any gradient updates, representing the uninformed prediction baseline", "The entropy of the data distribution — the theoretical minimum loss achievable by any model, reflecting inherent noise and ambiguity in the data"],
      correct: 3,
      explanation: "The irreducible loss represents the Bayes-optimal loss: the entropy $H(P)$ of the true data distribution. No model can beat this because the data itself contains inherent randomness (e.g., multiple valid next tokens in natural language). Estimating $L_\\infty$ is important for understanding how much room for improvement remains, but it's difficult to measure precisely in practice."
    },
    {
      type: "mc",
      question: "When using $\\mu$P to transfer hyperparameters from a small proxy model to a large target model, which of the following must change with model width $d$?",
      options: ["The learning rate — it must decrease as $1/d$ to prevent divergence caused by the accumulated contribution of more neurons per layer", "The batch size — it must scale linearly with $d$ to maintain a consistent signal-to-noise ratio in the gradient estimates across scales", "The initialization scale of weight matrices — input and output layers use different scaling than hidden layers, with specific $1/\\sqrt{d}$ and $1/d$ prescriptions", "Nothing changes — that is the entire point of $\\mu$P, which guarantees all hyperparameters transfer without any modification whatsoever"],
      correct: 2,
      explanation: "In $\\mu$P, the initialization scales and per-layer learning rate multipliers are set so that activations, gradients, and updates remain $\\Theta(1)$ as width varies. Different layer types (input embeddings, hidden layers, output layer) require different scaling rules. The point is not that nothing changes — it is that the *optimal learning rate* stays the same, while the parameterization itself adapts via prescribed initialization and multiplier rules."
    }
  ]
};
