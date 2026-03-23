// Branch B Assessments: Scaling, Architecture, Data, Training Dynamics, Pretraining Objectives
// Sections B.1–B.5: Pure assessment modules (no info steps), 10 MC questions each

// ────────────────────────────────────────────────────────────────────────────
// B.1  Scaling Laws
// ────────────────────────────────────────────────────────────────────────────
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
        "It eliminates the need for warmup in the learning rate schedule",
        "It enables hyperparameters (especially learning rate) tuned on a small model to transfer directly to larger models without re-tuning",
        "It ensures all layers have equal gradient norms regardless of depth",
        "It replaces Adam with a scale-invariant optimizer"
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
        "$N^* \\propto C$ and $D^*$ is constant",
        "$N^* \\propto C^{0.5}$ and $D^* \\propto C^{0.5}$ — both scale as the square root of compute",
        "$N^* \\propto C^{0.73}$ and $D^* \\propto C^{0.27}$ — parameters scale much faster",
        "$N^* \\propto C^{0.3}$ and $D^* \\propto C^{0.7}$ — data scales much faster"
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

// ────────────────────────────────────────────────────────────────────────────
// B.2  Architecture Innovations
// ────────────────────────────────────────────────────────────────────────────
export const architectureAssessment = {
  id: "B.2-assess",
  sectionId: "B.2",
  title: "Assessment: Architecture Innovations",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In a Mixture-of-Experts (MoE) Transformer, the gating network selects a subset of expert FFN layers for each token. The primary computational advantage is:",
      options: ["Total parameters increase but FLOPs per token stay constant — only $k$ of $E$ experts activate per token, decoupling parameter count from compute cost", "The attention layers become cheaper because experts handle most of the computational work that attention would otherwise perform", "Experts share weights with each other, reducing total memory requirements compared to having independent fully-connected layers", "The vocabulary size can be reduced because experts specialize in different token types, enabling each expert to use a smaller subvocabulary"],
      correct: 0,
      explanation: "MoE decouples model capacity (total parameters) from per-token compute. A model with $E$ experts but top-$k$ routing uses $k/E$ of the FFN FLOPs per token. For example, Mixtral 8x7B has ~47B total parameters but uses only ~13B per token (top-2 of 8 experts). The attention layers are unchanged."
    },
    {
      type: "mc",
      question: "Routing collapse in MoE models is a failure mode where:",
      options: [
        "All experts converge to identical weights, wasting capacity",
        "The router assigns nearly all tokens to a small subset of experts, leaving most experts undertrained while overloaded experts become bottlenecks",
        "The gating network outputs uniform probabilities, ignoring input features",
        "Experts become too specialized and cannot generalize to new domains"
      ],
      correct: 1,
      explanation: "Routing collapse is a rich-get-richer dynamic: experts that receive more tokens learn faster, causing the router to send even more tokens their way. This leaves most experts undertrained. Load-balancing losses (auxiliary losses that penalize uneven expert utilization) are the standard mitigation, but they introduce a tension between routing quality and load balance."
    },
    {
      type: "mc",
      question: "Standard self-attention computes $\\text{Softmax}(QK^\\top / \\sqrt{d})V$, which is $O(n^2 d)$ in sequence length $n$. Linear attention replaces this with $\\phi(Q)(\\phi(K)^\\top V)$, achieving $O(nd^2)$. Why does linear attention underperform standard attention on retrieval-intensive tasks?",
      options: ["Linear attention cannot represent positional information because the kernel map destroys the ordering of tokens in the sequence", "Linear attention cannot be parallelized during training, making it slower than standard attention despite the lower asymptotic complexity", "Linear attention uses fewer parameters than standard attention, reducing model capacity below what is needed for retrieval tasks", "The kernel feature map $\\phi$ compresses the key-query interaction into a fixed-size matrix $\\phi(K)^\\top V \\in \\mathbb{R}^{d \\times d}$, which cannot store and retrieve arbitrary token-level associations from a long context"],
      correct: 3,
      explanation: "By associating right-to-left, linear attention maintains a $d \\times d$ state matrix that accumulates key-value associations. This is a fixed-size bottleneck regardless of sequence length — it cannot perfectly store $n$ distinct key-value pairs when $n > d$. Standard attention computes each query against all keys explicitly, enabling precise retrieval. This is why hybrid architectures pair linear attention (for efficiency) with some standard attention layers (for retrieval)."
    },
    {
      type: "mc",
      question: "Mamba and S4 are examples of state-space models (SSMs) for sequence modeling. Their key structural property, compared to Transformers, is:",
      options: ["They use convolutions instead of any recurrence, making them purely feedforward with fixed-length receptive fields determined by kernel size", "They replace attention with graph neural networks over token dependency trees, leveraging syntactic structure for efficient context modeling", "They model sequences through a latent continuous-time dynamical system $\\dot{h}(t) = Ah(t) + Bx(t)$, discretized for efficiency, enabling linear-time sequence processing with a fixed-size hidden state", "They use external memory modules to store long-range dependencies, reading and writing to a differentiable memory bank at each processing step"],
      correct: 2,
      explanation: "SSMs are grounded in continuous-time state-space equations: $\\dot{h} = Ah + Bx$, $y = Ch + Dx$. After discretization, these become linear recurrences that can be computed as convolutions during training (parallelizable) or step-by-step during inference (efficient autoregressive generation). Mamba adds input-dependent (selective) gating to the $A$, $B$, $C$ matrices, which is crucial for content-based reasoning."
    },
    {
      type: "mc",
      question: "What distinguishes Mamba's \"selective\" state-space mechanism from the original S4 model?",
      options: ["Mamba makes the SSM parameters ($B$, $C$, and $\\Delta$) input-dependent, allowing the model to selectively filter or retain information based on content rather than using fixed dynamics", "Mamba uses a larger state dimension to store more information per step, compensating for the lack of explicit attention over previous tokens", "Mamba replaces the HiPPO initialization with random initialization, relying on gradient descent to learn the optimal state transition dynamics", "Mamba adds self-attention layers between SSM layers to combine the linear-time efficiency of SSMs with the retrieval capabilities of attention"],
      correct: 0,
      explanation: "S4's $A$, $B$, $C$ matrices are fixed (input-independent) — the same dynamics apply to every input. Mamba makes $B$, $C$, and the discretization step $\\Delta$ functions of the input, enabling content-aware filtering. This selectivity is analogous to gating in LSTMs and is essential for tasks that require ignoring irrelevant tokens. The cost is losing the convolution-mode parallelism of S4, which Mamba compensates with a hardware-aware scan algorithm."
    },
    {
      type: "mc",
      question: "Mixture-of-Depths (MoD) applies a routing mechanism across layers rather than across experts. The core idea is:",
      options: [
        "Different layers use different hidden dimensions to save compute, with narrower layers for simpler transformations and wider layers for complex reasoning",
        "A learned router decides, per token, whether to apply the full Transformer block or skip it (via a residual bypass), allowing easy tokens to exit early and hard tokens to receive more computation",
        "The model has a variable number of layers that changes dynamically during training based on the current loss landscape and gradient statistics",
        "Attention heads at deeper layers are pruned to reduce compute, since empirically the deeper layers contribute less to the final token predictions"
      ],
      correct: 1,
      explanation: "MoD (Raposo et al., 2024) routes each token at each layer: the token either passes through the full block (attention + FFN) or takes a residual skip. This means \"easy\" tokens (e.g., deterministic function words) consume fewer FLOPs than \"hard\" tokens (e.g., content words requiring reasoning). Unlike early exit (which stops processing entirely), MoD allows tokens to re-enter processing at later layers."
    },
    {
      type: "mc",
      question: "The load-balancing auxiliary loss used in MoE training typically takes the form $\\mathcal{L}_{\\text{aux}} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot p_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $p_i$ is the average router probability for expert $i$. Why is this formulation used instead of directly penalizing the variance of $f_i$?",
      options: ["Variance penalization would be too expensive to compute at scale since it requires tracking running statistics across all experts per batch", "The $f_i \\cdot p_i$ formulation also regularizes the expert weights directly, preventing any single expert from developing disproportionately large parameters", "Variance penalization would force all experts to learn identical representations, eliminating the specialization that makes MoE models effective", "The $f_i \\cdot p_i$ product is differentiable with respect to the router's logits (through $p_i$), whereas $f_i$ alone involves a non-differentiable argmax/top-k selection"],
      correct: 3,
      explanation: "The token-to-expert assignment $f_i$ involves discrete top-$k$ selection, which is not differentiable. But the router probability $p_i$ (the softmax output before the discrete decision) is differentiable. By multiplying $f_i \\cdot p_i$, the gradient flows through $p_i$ to update the router, encouraging it to spread probability mass more evenly. Minimizing $\\sum f_i p_i$ is minimized when both fractions and probabilities are uniform ($1/E$ each)."
    },
    {
      type: "mc",
      question: "Hybrid architectures that combine attention with linear recurrences (e.g., Jamba, Griffin) typically interleave the two layer types. The design rationale is:",
      options: ["Attention layers are used only in the first few layers to establish position encoding, while recurrent layers handle the remaining computation", "Recurrent layers are only needed during inference for efficient KV-cache management, while training uses pure attention throughout the network", "Attention provides precise in-context retrieval for tasks that need it, while recurrent layers provide efficient long-range context compression — the hybrid gets both capabilities at lower total cost than pure attention", "The combination enables the model to process images and text simultaneously, with attention handling visual tokens and recurrence handling language"],
      correct: 2,
      explanation: "Pure recurrent models (Mamba, RWKV) struggle with tasks requiring precise recall from long contexts (e.g., \"what was the 3rd item in the list?\") because their fixed-size state compresses information. Pure attention is $O(n^2)$. Hybrids use a few attention layers (often every 4th or 8th layer) to handle retrieval while the recurrent layers efficiently process the majority of the context at $O(n)$ cost."
    },
    {
      type: "mc",
      question: "Early exit strategies in Transformer inference allow the model to produce an output token from an intermediate layer rather than processing through all $L$ layers. A key practical challenge is:",
      options: ["The hidden representations at early layers live in a different space than what the final output head expects, requiring either separate output heads per layer or representation alignment techniques that add training complexity", "Early layers do not have enough parameters to make predictions because the parameter count scales linearly with layer index in standard Transformer designs", "Early exit requires a different tokenizer for each exit point because intermediate representations encode tokens at different granularity levels", "Gradient computation is impossible with early exit during training because the computational graph is truncated at the exit point, blocking backpropagation"],
      correct: 0,
      explanation: "The output projection (unembedding) is trained against the final layer's representations. Intermediate representations may not yet encode the information needed for prediction, or may encode it in a different subspace. Solutions include: training separate lightweight classifiers at each potential exit layer, using shared output heads with representation alignment losses, or \"overthinking\" classifiers that decide when additional layers would not change the prediction."
    },
    {
      type: "mc",
      question: "In an MoE model using top-2 routing with 8 experts, what happens during training if one expert's router logits are consistently 10x larger than the others?",
      options: [
        "The model automatically normalizes the logits to prevent this",
        "That expert and one other will be selected for nearly all tokens, the remaining 6 experts will receive almost no gradient signal, and model capacity will be severely underutilized",
        "Training will diverge immediately due to gradient explosion",
        "The other experts will quickly catch up because they receive cleaner gradients"
      ],
      correct: 1,
      explanation: "With top-2 routing, the dominant expert gets selected for almost every token. The second slot gets competed for among the remaining experts, but with much smaller logits, one or two runners-up will also dominate. The result: 5-6 experts are effectively dead. This is routing collapse. The load-balancing loss combats this by penalizing uneven utilization, but if the coefficient $\\alpha$ is too small, collapse still occurs. If $\\alpha$ is too large, routing quality degrades because load balance overrides content-based routing."
    }
  ]
};

// ────────────────────────────────────────────────────────────────────────────
// B.3  Data-Centric Pretraining
// ────────────────────────────────────────────────────────────────────────────
export const dataCentricAssessment = {
  id: "B.3-assess",
  sectionId: "B.3",
  title: "Assessment: Data-Centric Pretraining",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Influence functions estimate how a model's prediction would change if a specific training example were removed (or upweighted). The classic formula involves $\\mathcal{I}(z, z_{\\text{test}}) = -\\nabla_\\theta \\ell(z_{\\text{test}})^\\top H_\\theta^{-1} \\nabla_\\theta \\ell(z)$. Why do influence functions not scale to modern LLMs?",
      options: ["The loss function of LLMs is not twice-differentiable due to the discrete argmax in token selection, making the Hessian undefined", "The gradient $\\nabla_\\theta \\ell(z)$ is always zero at the optimum, so the influence function evaluates to zero for any training example", "Influence functions require the model to be trained to full convergence on the training set, which is deliberately avoided in LLM pretraining", "Computing or approximating the inverse Hessian $H_\\theta^{-1}$ is intractable for billions of parameters, and the quadratic approximation breaks down in the non-convex, overparameterized regime where LLMs operate"],
      correct: 3,
      explanation: "The Hessian $H_\\theta$ is an $N \\times N$ matrix where $N$ is the parameter count — storing it is impossible for LLMs (e.g., $70\\text{B}^2$ entries). Even Hessian-vector product approximations (like LiSSA) are noisy and expensive. Furthermore, influence functions assume a convex loss landscape near the optimum, which does not hold for deep networks. Recent work (TRAK, datamodels) uses random projection-based approximations that trade fidelity for scalability."
    },
    {
      type: "mc",
      question: "Data attribution methods like TRAK (Tracing with Randomly-projected After Kernel) address the scalability limitations of influence functions by:",
      options: ["Using only the first-order gradient without any Hessian information, computing a simple dot product between training and test gradients as the influence proxy", "Training a separate neural network to predict influence scores from input features, bypassing the need to differentiate through the original model entirely", "Projecting per-example gradients into a low-dimensional random subspace, then computing attribution scores via a linear model in that projected space — trading exact inverse-Hessian computation for tractable random projections", "Computing influence only for the last layer of the model, where gradients are largest and most informative about the mapping from representations to predictions"],
      correct: 2,
      explanation: "TRAK projects the high-dimensional gradient vectors $\\nabla_\\theta \\ell(z) \\in \\mathbb{R}^N$ down to $\\mathbb{R}^k$ (with $k \\ll N$) using random matrices. In this compressed space, it fits a linear model that predicts test loss from training example features. This is motivated by the neural tangent kernel (NTK) perspective: near convergence, the model behaves approximately linearly in the projected gradient space. TRAK is orders of magnitude cheaper than exact influence functions."
    },
    {
      type: "mc",
      question: "DSIR (Data Selection with Importance Resampling) selects pretraining data that resembles a target distribution. The core mechanism is:",
      options: ["Computing importance weights $w(x) = p_{\\text{target}}(x) / p_{\\text{source}}(x)$ using n-gram language model ratios, then resampling the source corpus according to these weights", "Training a binary classifier to label each document as \"good\" or \"bad\" and keeping only those predicted as positive for pretraining inclusion", "Clustering the data into semantic groups and selecting the clusters whose centroids are closest to the target distribution's centroid in embedding space", "Using perplexity under a target-domain language model as the sole selection criterion, discarding documents above a fixed perplexity threshold"],
      correct: 0,
      explanation: "DSIR fits lightweight n-gram models to both the target domain and the source corpus, then computes importance weights as the density ratio. Data points that are more likely under the target distribution (relative to the source) get upweighted. Resampling according to these weights yields a subset whose distribution approximates the target. This is much cheaper than training a neural classifier, and importance resampling has well-understood statistical properties."
    },
    {
      type: "mc",
      question: "DoReMi (Xie et al., 2023) optimizes the domain mixing proportions for pretraining data (e.g., how much web text vs. code vs. Wikipedia). How does it determine the optimal mixture?",
      options: [
        "It uses the proportion of each domain in the raw crawl as the optimal mixture",
        "It trains a small proxy model using distributionally robust optimization (DRO) to upweight domains where the model struggles most, then uses those optimized proportions to train the large model",
        "It computes the KL divergence between each domain and the target, selecting domains with lowest divergence",
        "It alternates between domains in round-robin fashion"
      ],
      correct: 1,
      explanation: "DoReMi uses a two-stage process: (1) train a small reference model on the default mixture, (2) train another small model with group DRO, which dynamically upweights domains with higher excess loss (current loss minus reference loss). The domain weights learned by the small DRO model transfer to the large-scale training run. This avoids expensive large-scale ablations over mixture proportions."
    },
    {
      type: "mc",
      question: "Catastrophic forgetting in continual pretraining occurs when a model fine-tuned on domain-specific data loses its general capabilities. Which of the following is NOT a standard mitigation strategy?",
      options: ["Mixing domain-specific data with a fraction of the original pretraining distribution during continued training to maintain general capabilities", "Using elastic weight consolidation (EWC) or similar regularization that penalizes changes to parameters important for previous tasks", "Replaying a small buffer of original pretraining data alongside the new domain data to maintain the model's prior knowledge", "Training on the new domain for exactly one epoch to prevent overfitting, relying on the single-pass constraint to limit forgetting"],
      correct: 3,
      explanation: "Training for exactly one epoch is not a principled forgetting mitigation — forgetting depends on the degree of distribution shift, not epochs. The other three are well-established approaches: data mixing (most common in practice), EWC-style regularization (penalizes parameter drift weighted by Fisher information), and replay buffers (store and interleave old examples). In practice, simple data mixing (e.g., 90% domain + 10% general) is the most widely used because it is effective and easy to implement."
    },
    {
      type: "mc",
      question: "Learning rate rewarming is a technique used when continuing pretraining on a new data distribution. The practice involves:",
      options: ["Resetting the learning rate to its initial maximum and repeating the full warmup + decay schedule from scratch as if starting pretraining over", "Using a constant learning rate throughout continual pretraining to maintain a steady adaptation rate across the entire new data distribution", "Briefly increasing the learning rate back to a moderate value before decaying again, which helps the model escape the loss basin of the original training distribution and adapt to the new data", "Reducing the learning rate to near zero to prevent catastrophic forgetting by ensuring the model's weights change as little as possible during adaptation"],
      correct: 2,
      explanation: "After pretraining, the LR has decayed to a very small value. If you continue training at this low LR on new data, the model adapts very slowly. Rewarming briefly raises the LR (typically not to the original maximum, but to a meaningful fraction) and then decays again. This lets the model move away from its current minimum to better accommodate the new distribution. The Gupta et al. (2023) work on continual pretraining found rewarming essential for efficient adaptation."
    },
    {
      type: "mc",
      question: "When building a domain-specific LLM (e.g., for biomedicine), you can either (A) pretrain from scratch on domain data, or (B) continue pretraining a general-purpose LLM on domain data. Which statement is most accurate?",
      options: ["Continued pretraining is almost always more compute-efficient: general LLMs have already learned syntax, reasoning, and world knowledge that transfers to the domain, so domain adaptation requires far fewer tokens than learning everything from scratch", "From-scratch pretraining always produces superior domain models because the tokenizer can be optimized for domain-specific vocabulary and subword patterns", "The two approaches yield identical results given the same total compute, since the final loss depends only on aggregate FLOPs regardless of training trajectory", "Continued pretraining cannot work because the general tokenizer lacks domain-specific tokens, causing excessive fragmentation of specialized terminology"],
      correct: 0,
      explanation: "Continued pretraining leverages transfer learning: a 7B model pretrained on 2T tokens has learned language structure, reasoning patterns, and broad knowledge. Adapting it to biomedicine with 50-100B domain tokens is far cheaper than training a biomedical model from scratch on hundreds of billions of tokens. The tokenizer concern is real but secondary — subword tokenizers handle unseen terms by decomposition, and domain terms can be added. Models like BioMedLM, PMC-LLaMA, and SciLLM all use continued pretraining."
    },
    {
      type: "mc",
      question: "Data deduplication before pretraining is considered essential. What is the primary failure mode if near-duplicate documents are not removed?",
      options: [
        "The model's vocabulary size becomes too large",
        "Training loss decreases artificially without improving generalization — the model memorizes duplicated sequences, inflating training metrics while wasting compute on redundant updates and increasing verbatim memorization risks",
        "The optimizer diverges due to repeated gradient directions",
        "Attention heads become specialized for the duplicated content, reducing capacity for other patterns"
      ],
      correct: 1,
      explanation: "Lee et al. (2022) showed that deduplication improves both training efficiency and downstream performance. Duplicated data means the model sees certain patterns disproportionately often, leading to memorization rather than generalization. It also wastes compute — tokens spent on duplicates could have been spent on diverse examples. MinHash-based near-deduplication is standard practice. Carlini et al. showed that memorization rates correlate strongly with duplication frequency."
    },
    {
      type: "mc",
      question: "In the context of data quality filtering for pretraining, a perplexity-based filter uses a reference language model to score each document. What is a known failure mode of naive perplexity filtering?",
      options: ["It cannot process documents longer than the reference model's context window, causing systematic exclusion of long-form content such as technical papers and books", "It is too slow to apply to web-scale corpora because computing perplexity requires a full forward pass of the reference model over every candidate document", "It removes all non-English text regardless of quality, since the reference language model assigns high perplexity to any text in an unfamiliar language", "It systematically biases the pretraining data toward the style and domain of the reference model's training data — e.g., a Wikipedia-trained reference model will favor Wikipedia-like text and discard informal but informative content"],
      correct: 3,
      explanation: "A reference LM assigns low perplexity to text similar to its own training distribution. A Wikipedia-trained filter will favor encyclopedic prose and penalize code, dialogue, informal writing, and domain-specific jargon — all of which may be high-quality and valuable for a general-purpose LLM. The C4 dataset used a Wikipedia perplexity filter, which is now recognized as having been too aggressive. Modern pipelines use classifier-based quality scoring with more diverse positive examples."
    },
    {
      type: "mc",
      question: "When selecting data for continued pretraining of an LLM on a specialized domain, the optimal strategy with respect to data mixing is:",
      options: [
        "Use only domain-specific data to maximize specialization",
        "Use only general data but increase the learning rate to capture domain knowledge from the few relevant examples",
        "Mix domain-specific data with general-purpose data, tuning the ratio empirically — too much domain data causes forgetting of general capabilities, too little yields insufficient specialization",
        "Alternate between pure domain and pure general data in separate phases"
      ],
      correct: 2,
      explanation: "Data mixing is a Pareto optimization between domain performance and general capability retention. The optimal ratio depends on (1) how different the domain is from general text, (2) how much domain data is available, and (3) which general capabilities matter for the application. Typical ratios range from 50-90% domain data. Pure domain training causes rapid forgetting; phase alternation creates oscillation in capabilities. Continuous mixing provides the smoothest learning dynamics."
    }
  ]
};

// ────────────────────────────────────────────────────────────────────────────
// B.4  Training Stability & Dynamics
// ────────────────────────────────────────────────────────────────────────────
export const trainingDynamicsAssessment = {
  id: "B.4-assess",
  sectionId: "B.4",
  title: "Assessment: Training Stability & Dynamics",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The \"edge of stability\" phenomenon (Cohen et al., 2021) in gradient descent training describes a regime where:",
      options: ["The sharpness (largest eigenvalue of the Hessian) rises until it reaches $\\approx 2/\\eta$ (where $\\eta$ is the learning rate), then oscillates around this threshold while loss continues to decrease non-monotonically", "Training loss oscillates wildly between high and low values but validation loss remains stable and smoothly decreasing throughout the entire optimization", "The model parameters reach a critical point where any perturbation causes immediate divergence, requiring careful checkpointing and restart protocols", "Batch normalization causes gradient norms to hover at a fixed value, creating an artificial stability boundary that prevents the loss from decreasing further"],
      correct: 0,
      explanation: "Classical optimization theory predicts divergence when sharpness exceeds $2/\\eta$. Instead, Cohen et al. observed that full-batch GD on neural networks enters a regime where sharpness self-stabilizes at $\\approx 2/\\eta$: when it exceeds this threshold, the loss temporarily increases (the optimizer takes steps that are \"too large\"), which modifies the landscape to reduce sharpness back below the threshold. This is not predicted by convex optimization theory and suggests GD implicitly regularizes toward flatter minima."
    },
    {
      type: "mc",
      question: "The distinction between the \"feature learning\" regime and the \"kernel (lazy)\" regime in neural network training refers to:",
      options: [
        "Whether the model uses convolutional or attention-based features, since each architecture type operates in a distinct optimization regime",
        "Whether the model's internal representations (features) change substantially during training, or whether the network behaves approximately like a linear model around initialization (kernel regime), only adjusting output-layer-like combinations of fixed random features",
        "Whether features are learned in supervised or unsupervised fashion, since self-supervised objectives produce qualitatively different internal representations",
        "Whether the kernel function is Gaussian (RBF) or polynomial, which determines the implicit bias of the neural network toward smooth or piecewise solutions"
      ],
      correct: 1,
      explanation: "In the kernel (lazy/NTK) regime — which can arise with very large width or very small learning rate — the network's internal representations barely move from their random initialization. Learning happens only by adjusting output weights over essentially fixed features. In the feature learning regime, representations transform substantially, enabling the model to discover task-relevant abstractions. Practical LLMs operate firmly in the feature learning regime. The $\\mu$P parameterization is designed to keep models in this regime across scales."
    },
    {
      type: "mc",
      question: "Induction heads are a specific attention pattern discovered in Transformer language models. They perform the operation of:",
      options: [
        "Attending to the first token in the sequence to establish a global context vector used to ground all subsequent token predictions",
        "Identifying a previous occurrence of the current token and copying the token that followed it — implementing a simple in-context bigram lookup like [$A$][$B$] ... [$A$] $\\rightarrow$ [$B$]",
        "Computing the average of all previous token embeddings to form a compressed context representation that captures the overall sequence meaning",
        "Attending to the most semantically similar token in the context based on cosine similarity between query and key representations"
      ],
      correct: 1,
      explanation: "Induction heads (Olsson et al., 2022) implement a two-step copying mechanism: (1) a \"previous token\" head identifies where the current token last appeared, (2) the induction head attends to the position after that previous occurrence and copies its value. This implements the pattern: if [$A$][$B$] appeared before and we now see [$A$], predict [$B$]. This is a fundamental circuit for in-context learning and is one of the clearest examples of interpretable algorithmic behavior in Transformers."
    },
    {
      type: "mc",
      question: "The formation of induction heads during training exhibits a phase transition. What does this mean concretely?",
      options: ["Induction heads form instantly at initialization due to the random weight configuration already containing the required circuit structure in expectation", "Induction heads form only if the model has more than 12 layers, since the two-head composition circuit requires sufficient depth to develop", "The model alternates between having and not having induction heads as training progresses, oscillating with the learning rate schedule", "There is a sudden, discrete jump in in-context learning ability at a specific point during training, with the loss on repeated-pattern tasks dropping sharply over a narrow window of training steps rather than improving gradually"],
      correct: 3,
      explanation: "Olsson et al. observed that in-context learning ability (measured by how much loss decreases from the first to the second occurrence of a pattern) remains near zero for many training steps, then rapidly improves over a narrow window. This coincides with the formation of the induction head circuit. This is a genuine phase transition — a qualitative change in capability emerging suddenly from continuous optimization. It's one of the clearest examples of emergent capability in a controlled setting."
    },
    {
      type: "mc",
      question: "Loss landscape mode connectivity refers to the finding that:",
      options: ["All local minima have the same loss value, meaning there is no benefit to searching for better solutions beyond the first minimum found", "The loss landscape is convex near any local minimum, ensuring that gradient descent in the local neighborhood always improves the objective", "Different trained models (from different initializations) can often be connected by simple low-loss paths (e.g., linear or piecewise-linear) in weight space, suggesting they lie in the same broad basin or on the same loss-level set", "Gradient descent always converges to the global minimum in overparameterized networks, so all trained models end up at the same point in weight space"],
      correct: 2,
      explanation: "Mode connectivity (Garipov et al., 2018; Draxler et al., 2018) showed that independently trained models often lie in connected low-loss regions. While the straight line between two models in weight space may cross a loss barrier, a slightly curved path (found by optimization) often connects them with negligible loss increase. This suggests the loss landscape of overparameterized networks has a simpler structure than previously thought — most good minima are connected."
    },
    {
      type: "mc",
      question: "Mode connectivity has direct implications for model merging. When we average the weights of two fine-tuned models (linear interpolation $\\theta_{\\text{merged}} = \\alpha \\theta_1 + (1 - \\alpha) \\theta_2$), the merged model performs well only when:",
      options: ["The two models share a common pretrained initialization — this ensures they lie in the same basin of the loss landscape, making the linear interpolation path stay in a low-loss region", "Both models have the same number of parameters, since mismatched architectures create dimensional incompatibilities that prevent meaningful weight interpolation", "Both models were trained on identical data distributions, because different training data pushes models into incompatible regions of the loss landscape", "The models use different optimizers to ensure diversity, since optimizer disagreement creates complementary solutions that average well together"],
      correct: 0,
      explanation: "Models fine-tuned from the same pretrained checkpoint tend to remain in the same loss basin (the pretrained model acts as an \"anchor\"). Linear interpolation between them stays in the low-loss region. Models trained from different random initializations typically do NOT mode-connect linearly — there are loss barriers between their basins. This is why weight averaging works well for merging LoRA adapters or task-specific fine-tunes of the same base model, but fails for independently pretrained models."
    },
    {
      type: "mc",
      question: "Training instabilities (loss spikes) in large language model training are often attributed to:",
      options: [
        "Hardware failures causing corrupted gradients that propagate through the distributed training pipeline before being detected by checksum validation",
        "Outlier activations and attention logit growth — as training progresses, a few hidden dimensions develop very large magnitudes, which can cause softmax saturation, gradient explosion, and sudden loss spikes",
        "The training data containing adversarial examples specifically crafted to maximize gradient magnitudes and destabilize the optimization trajectory",
        "Running out of unique training data partway through training, causing the model to memorize repeated examples and diverge from generalizable solutions"
      ],
      correct: 1,
      explanation: "Dettmers et al. (2022) and Zhai et al. (2023) documented how outlier features (hidden dimensions with magnitudes 10-100x larger than typical) emerge during training. These cause numerical issues: attention logits grow large, softmax saturates, and gradients spike. Mitigations include QK-norm (normalizing query and key vectors before the dot product), logit capping, and careful initialization. PaLM and other large models reported loss spikes that required manual intervention (learning rate reduction or data skipping)."
    },
    {
      type: "mc",
      question: "In the context of $\\mu$P, what happens to the gradient dynamics of a standard (non-$\\mu$P) Transformer as you increase width $d$ while keeping learning rate fixed?",
      options: ["Gradients vanish because each individual weight contributes less to the output, causing the effective learning signal per parameter to shrink toward zero", "The model becomes more robust to learning rate choices because the wider layers average out noise in the gradient estimates across more parameters", "Training speed doubles with each doubling of width due to increased parallelism in the matrix operations, making wider models strictly more efficient", "The model enters the kernel (lazy) regime: weight updates become infinitesimally small relative to the random initialization, so internal representations stop learning meaningful features"],
      correct: 3,
      explanation: "Under standard parameterization (SP), if you keep the learning rate fixed and increase width, each weight's update contributes less to the output (because activations are averaged over more dimensions). In the infinite-width limit, this gives the Neural Tangent Kernel regime where the network is effectively linear around initialization. $\\mu$P rescales learning rates and initialization so that the contribution of each weight update to the output remains $\\Theta(1)$, preserving feature learning dynamics regardless of width."
    },
    {
      type: "mc",
      question: "The phenomenon of \"grokking\" in neural network training refers to:",
      options: ["The model failing to learn despite sufficient capacity, where the loss plateaus at a high value regardless of training duration or hyperparameter tuning", "Rapid learning in the first few training steps followed by a sustained plateau where neither training nor test metrics show measurable improvement", "A delayed generalization pattern where the model first memorizes training data (achieving zero training loss with high test loss), then — much later in training — suddenly generalizes (test loss drops sharply), despite no change in training loss", "The model learning multiple tasks simultaneously without interference, where multi-task training achieves the same loss as single-task training on each individual objective"],
      correct: 2,
      explanation: "Grokking (Power et al., 2022) is a striking phenomenon where generalization occurs long after memorization. On modular arithmetic tasks, models achieve perfect training accuracy quickly, but test accuracy remains at chance for many more steps before suddenly jumping to near-perfect. This suggests the model transitions from a memorization solution to an algorithmic (generalizing) solution. Weight decay and regularization accelerate grokking, supporting the interpretation that regularization pressure eventually pushes the model toward the simpler, generalizing solution."
    },
    {
      type: "mc",
      question: "When training a large Transformer, practitioners often observe that the effective learning rate must be adjusted for different parts of the model. Which statement about per-layer learning rate dynamics is correct?",
      options: ["In standard training, earlier layers tend to have smaller gradients (and thus effectively lower learning rates under Adam), while attention logits and embedding layers require special handling (e.g., lower LR or normalization) to prevent instability", "All layers should use exactly the same learning rate for optimal training, since Adam's adaptive rates already account for differences in gradient magnitude across layers", "Later layers should always use a smaller learning rate because they are closer to the loss and receive larger gradients, requiring dampening to maintain stable updates", "Per-layer learning rates are only needed for models with more than 100B parameters, since smaller models have sufficiently uniform gradient norms across all layers"],
      correct: 0,
      explanation: "The gradient magnitudes vary systematically across a Transformer: embedding layers and attention logits tend to grow disproportionately, contributing to instability. Adam's adaptive rates help but don't fully resolve this. Techniques like QK-LayerNorm (normalizing queries and keys), embedding scaling, and logit capping address specific problematic components. $\\mu$P provides a principled framework by prescribing per-layer multipliers that maintain consistent update scales. In practice, many large-scale training runs use lower learning rates for embeddings."
    }
  ]
};

// ────────────────────────────────────────────────────────────────────────────
// B.5  Novel Pretraining Objectives
// ────────────────────────────────────────────────────────────────────────────
export const novelObjectivesAssessment = {
  id: "B.5-assess",
  sectionId: "B.5",
  title: "Assessment: Novel Pretraining Objectives",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Masked language modeling (MLM, as in BERT) and autoregressive language modeling (as in GPT) differ in a fundamental way regarding the joint distribution $P(x_1, \\dots, x_T)$. Which statement is correct?",
      options: [
        "Both model the exact same joint distribution, just with different factorizations that are mathematically equivalent under the chain rule of probability",
        "Autoregressive models factorize the exact joint via the chain rule $P(x) = \\prod_t P(x_t \\mid x_{<t})$, while MLM does not define a consistent joint distribution — it models conditional distributions $P(x_t \\mid x_{\\setminus t})$ that may not correspond to any valid joint",
        "MLM defines the joint distribution more efficiently because it processes all tokens in parallel, capturing bidirectional dependencies in a single forward pass",
        "Autoregressive models can only generate text, while MLM can both generate and understand because bidirectional conditioning enables both directions of inference"
      ],
      correct: 1,
      explanation: "The chain rule factorization used by autoregressive models is exact: $\\prod_t P(x_t \\mid x_{<t})$ is guaranteed to be a valid joint distribution. MLM trains conditional distributions $P(x_t \\mid x_{\\setminus t})$ (each token given all others). But a set of conditional distributions may be inconsistent — there may be no joint distribution that produces all of them. This makes MLM models unsuitable for generation without additional techniques (e.g., iterative refinement as in BERT-based generation)."
    },
    {
      type: "mc",
      question: "A key practical advantage of MLM over autoregressive LM during pretraining is:",
      options: ["MLM can use a smaller vocabulary because the masking procedure naturally clusters rare tokens into shared prediction targets, reducing the effective vocabulary size", "MLM requires less training data because it uses each token more efficiently by training on multiple masked positions per sequence rather than a single next-token prediction", "MLM is faster at inference because it generates all tokens in parallel, avoiding the sequential bottleneck of autoregressive decoding that limits tokens per second", "MLM processes all tokens bidirectionally — each masked position attends to both left and right context — which produces richer contextualized representations for downstream tasks that require understanding (e.g., classification, NER), and it achieves this with $T$ prediction tasks per sequence rather than requiring left-to-right factorization"],
      correct: 3,
      explanation: "MLM's bidirectional context is its main strength for representation learning. When predicting a masked token, the model can use information from both sides, producing representations that capture the full context. Autoregressive models only see left context at each position. However, MLM only trains on the ~15% of tokens that are masked (the rest don't contribute to the loss), while autoregressive models get a gradient signal from every token. This makes autoregressive pretraining more compute-efficient per token."
    },
    {
      type: "mc",
      question: "UL2 (Unifying Language Learning Paradigms) proposes training a single model with multiple denoising objectives. Its core insight is:",
      options: ["UL2 eliminates the need for fine-tuning by training on all possible task formats, making the pretrained model directly usable for any downstream task", "A single denoising objective is always optimal if tuned properly, since mixing objectives introduces conflicting gradient signals that hurt performance", "Different downstream tasks benefit from different pretraining objectives (short spans for understanding, long spans for generation), so mixing multiple denoising tasks with mode-switching tokens creates a model that handles both regimes", "UL2 uses reinforcement learning instead of maximum likelihood to optimize the denoising objective, enabling the model to learn more complex reconstruction strategies"],
      correct: 2,
      explanation: "UL2 defines three denoising modes: R-denoiser (short spans, like BERT), S-denoiser (sequential/prefix LM), and X-denoiser (extreme/long spans). A special sentinel token tells the model which mode is active. The key insight is that no single denoising objective dominates across all downstream tasks — short-span denoising helps classification and NLU, while long-span and prefix modes help generation. By mixing modes, UL2 produces a single model competitive on both understanding and generation benchmarks."
    },
    {
      type: "mc",
      question: "Diffusion models have been highly successful for continuous data (images, audio). Why is applying diffusion to discrete text fundamentally harder?",
      options: ["Discrete data cannot be interpolated smoothly — there is no natural continuous noise process for tokens. Adding Gaussian noise to token embeddings destroys the discrete structure, and discrete corruption processes (e.g., random token replacement) lack the mathematical properties (e.g., known reverse process) that make continuous diffusion tractable", "Text sequences are too short for the diffusion process to work effectively, since diffusion models need long inputs to amortize the cost of the multi-step denoising procedure", "The vocabulary size is too large for the denoising network to predict over, since each denoising step must output a probability vector over the entire vocabulary at every position", "Diffusion requires 2D spatial structure that text does not naturally have, since the denoising U-Net architecture relies on spatial convolutions that cannot be applied to sequential data"],
      correct: 0,
      explanation: "Continuous diffusion relies on gradually adding Gaussian noise and learning to reverse this process. For discrete tokens, there is no natural analog: you cannot \"slightly noise\" a token. Approaches include: (1) embedding tokens in continuous space and applying continuous diffusion (D3PM, Diffusion-LM), (2) using discrete corruption (token masking/replacement) as forward process (multinomial diffusion), or (3) score-matching on the simplex (MDLM). Each has trade-offs: continuous embeddings disconnect from the discrete structure; discrete corruption requires custom transition matrices."
    },
    {
      type: "mc",
      question: "Non-autoregressive generation (NAG) methods aim to generate all tokens in parallel rather than sequentially. The fundamental challenge they face is:",
      options: [
        "They require more parameters than autoregressive models to achieve equivalent quality, making them impractical for large-scale deployment",
        "They must model the joint distribution without the chain rule's conditional independence structure — tokens generated in parallel cannot condition on each other, leading to repetition, omission, and incoherence when the true distribution has strong inter-token dependencies",
        "They cannot use the Transformer architecture because the causal attention mask is incompatible with simultaneous token generation",
        "They are slower than autoregressive models in practice because the parallel decoding overhead exceeds the sequential generation cost"
      ],
      correct: 1,
      explanation: "Autoregressive models factor $P(x_1, \\dots, x_T)$ into conditionals, each depending on all previous tokens. NAG must model $P(x_1, \\dots, x_T)$ without this sequential structure — often assuming conditional independence given some latent $z$: $P(x \\mid z) = \\prod_t P(x_t \\mid z)$. This \"conditional independence\" assumption is violated when strong dependencies exist between adjacent tokens (e.g., \"New York\" — generating \"New\" and \"York\" independently risks producing \"New London\" or duplicating tokens). Knowledge distillation from AR models, iterative refinement, and CTC losses are common mitigations."
    },
    {
      type: "mc",
      question: "Energy-based models (EBMs) for text define an unnormalized density $p_\\theta(x) \\propto \\exp(-E_\\theta(x))$ over sequences. The central computational challenge of EBMs is:",
      options: ["The energy function $E_\\theta(x)$ is difficult to parameterize for text because variable-length sequences require architecture-specific pooling strategies", "The energy function must be non-negative by construction, which limits the expressiveness of the model class to distributions with bounded support", "EBMs cannot assign meaningfully different probabilities to different sequences because the softmax normalization collapses the energy differences", "Computing the normalizing constant $Z_\\theta = \\sum_x \\exp(-E_\\theta(x))$ requires summing over all possible sequences (exponential in length and vocabulary), making exact likelihood evaluation and gradient computation intractable"],
      correct: 3,
      explanation: "The partition function $Z_\\theta$ sums over all possible token sequences — $|V|^T$ terms for vocabulary $V$ and length $T$. This is astronomically intractable. Training EBMs requires approximations: contrastive divergence (MCMC sampling for negative examples), noise contrastive estimation (NCE), or score matching. For text specifically, MCMC sampling is difficult because the discrete space makes gradient-based sampling (Langevin dynamics) inapplicable. These challenges are why EBMs remain niche for text despite their theoretical elegance."
    },
    {
      type: "mc",
      question: "The prefix language modeling objective (used in T5 and UL2) treats part of the input as a bidirectional prefix and the rest as an autoregressive target. Compared to pure causal LM, this means:",
      options: ["The model has fewer parameters because the prefix encoder shares weights with the decoder, eliminating the need for separate encoder and decoder parameter sets", "The prefix must always be exactly half the sequence length to maintain a balanced ratio between bidirectional context encoding and autoregressive generation", "Tokens in the prefix attend to each other bidirectionally (full self-attention), while target tokens attend causally — this unifies the benefits of bidirectional encoding for the input context with autoregressive generation for the output", "Prefix LM cannot perform zero-shot generation because it requires a non-empty prefix to condition on, making it unsuitable for open-ended text generation tasks"],
      correct: 2,
      explanation: "Prefix LM uses a single Transformer with a hybrid attention mask: prefix tokens see each other fully (bidirectional), target tokens see all prefix tokens plus previous target tokens (causal). This is strictly more expressive than causal LM for the prefix portion (which benefits from bidirectional context) while maintaining valid autoregressive generation for the target. It is a natural fit for conditional generation tasks (question$\\rightarrow$answer, document$\\rightarrow$summary) where the input benefits from bidirectional encoding."
    },
    {
      type: "mc",
      question: "Noise contrastive estimation (NCE) has been proposed as an alternative to maximum likelihood for training language models. NCE trains the model to distinguish real data from noise samples. Why has NCE not replaced cross-entropy for large-scale LM pretraining?",
      options: ["NCE requires the noise distribution to be close to the data distribution for efficient learning, but designing such a noise distribution for natural language is itself a hard problem — and NCE's statistical efficiency degrades with vocabulary size, requiring many noise samples per data point", "NCE produces a discriminator rather than a generator, so it cannot be used for text generation since the model only learns to classify real versus fake tokens", "NCE cannot be combined with Transformer architectures because the contrastive objective requires a fundamentally different computational graph than autoregressive attention", "NCE requires labeled data with explicit positive and negative categories, which is unavailable in the unsupervised pretraining setting where models learn from raw text"],
      correct: 0,
      explanation: "NCE converts density estimation into binary classification: real vs. noise. The quality of the noise distribution matters enormously — if noise is too different from data, the classification is trivial and uninformative; if too similar, training is slow. For LLMs with vocabulary sizes of 30K-100K, NCE needs $k$ noise samples per real token (where $k$ should ideally grow with $|V|$), making it less efficient than the softmax cross-entropy loss which processes the entire vocabulary in one shot via the log-sum-exp. Modern hardware makes full-vocabulary softmax feasible."
    },
    {
      type: "mc",
      question: "Discrete diffusion models like D3PM and MDLM define a forward corruption process that gradually replaces tokens with random tokens or a [MASK] symbol. The number of denoising steps $T$ at inference time presents a trade-off:",
      options: [
        "More steps always produces worse results due to error accumulation across the denoising chain, where small per-step mistakes compound into large final errors",
        "Fewer steps are faster but each step must correct more corruption at once, requiring the model to make larger and less accurate jumps — more steps allow smaller, more accurate denoising increments but multiply the inference latency by $T$",
        "The number of steps does not affect output quality at all and only impacts generation speed, since the model converges to the same output regardless of stride",
        "Discrete diffusion requires exactly 1000 steps to work correctly because the token transition matrices are calibrated for that specific diffusion schedule length"
      ],
      correct: 1,
      explanation: "This is the fundamental speed-quality trade-off in all diffusion models. With $T = 1$ step, the model must denoise from pure noise to clean text in one shot (essentially non-autoregressive generation with all its problems). With $T = 1000$ steps, each step only slightly adjusts the sequence, making each denoising prediction easier but inference very slow. Practical discrete diffusion models use 10-100 steps with techniques like stride scheduling to concentrate steps where they matter most. This is still slower than autoregressive generation for short sequences."
    },
    {
      type: "mc",
      question: "The \"exposure bias\" problem in autoregressive language models refers to the discrepancy between training and inference. Specifically:",
      options: ["The model is exposed to too much data during training, causing it to memorize surface patterns rather than learning generalizable generation strategies", "Longer sequences receive more gradient updates during training, biasing the model toward generating verbose outputs that maximize the number of tokens produced", "The model is biased toward frequent tokens in the training data, causing it to underrepresent rare tokens and produce repetitive, high-frequency outputs", "During training the model conditions on ground-truth previous tokens (teacher forcing), but during inference it conditions on its own predictions — errors compound because the model never learns to recover from its own mistakes"],
      correct: 3,
      explanation: "Teacher forcing provides ground-truth context during training: $P(x_t \\mid x_1^*, \\dots, x_{t-1}^*)$. At inference, the model generates $P(x_t \\mid \\hat{x}_1, \\dots, \\hat{x}_{t-1})$ where $\\hat{x}$ are its own (potentially erroneous) predictions. The distribution of contexts at inference differs from training, causing errors to accumulate. Scheduled sampling (mixing ground-truth and model predictions during training) partially addresses this, and it is one motivation for non-autoregressive and diffusion-based alternatives. In practice, exposure bias is less damaging for very large LLMs because their per-token error rate is low."
    }
  ]
};
