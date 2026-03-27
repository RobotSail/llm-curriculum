// Assessment module for C.4: Compression & Distillation
// Split from assess-branch-cd.js — per-section test (10 questions)

export const compressionAssessment = {
  id: "C.4-assess",
  sectionId: "C.4",
  title: "Assessment: Compression & Distillation",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In knowledge distillation, a student model is trained to match the teacher's **soft targets** (softmax outputs with temperature $\\tau > 1$) rather than just the hard labels. The key insight behind using soft targets is:",
      options: [
        "Soft targets are easier to compute because the temperature-scaled softmax requires fewer floating-point operations than the standard argmax, reducing overall training cost per batch significantly",
        "The teacher's distribution over **incorrect classes** encodes similarity structure (\"dark knowledge\") — e.g., assigning 0.05 to 'cat' and 0.001 to 'car' for a dog image reveals inter-class relationships",
        "Soft targets prevent the student from overfitting by smoothing the label distribution, acting as label regularization that reduces the effective learning rate on confident examples",
        "Temperature scaling makes the loss function convex, guaranteeing convergence to the global optimum during student training by eliminating all non-convex saddle points"
      ],
      correct: 1,
      explanation: "Hinton et al. (2015) showed that the teacher's full probability vector, especially the relative probabilities of incorrect classes, contains far more information per training example than a one-hot label. With temperature $\\tau$, the softmax becomes $p_i = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}$, which smooths the distribution to expose these inter-class relationships. The student loss combines the soft target KL divergence (weighted by $\\tau^2$) with the hard label cross-entropy."
    },
    {
      type: "mc",
      question: "Logit matching distillation minimizes $\\text{KL}(p_{\\text{teacher}} \\| p_{\\text{student}})$ over the output distribution. Feature matching distillation instead:",
      options: ["Uses reinforcement learning to train the student, with the teacher's per-token predictions serving as the reward signal that shapes the student's generation policy", "Matches only the final prediction layer but uses squared error loss instead of KL divergence, altering gradient dynamics to favor high-confidence outputs over calibrated ones", "Aligns **intermediate layer representations** between teacher and student — minimizing $\\|f_{\\text{teacher}}^{(l)} - g(f_{\\text{student}}^{(k)})\\|^2$ with a learned projection $g$ for dimension alignment", "Matches the gradient norms of teacher and student at each layer during training, ensuring the student's optimization trajectory mirrors the teacher's parameter update dynamics"],
      correct: 2,
      explanation: "Feature matching (FitNets, PKD) adds losses that align intermediate representations. Since teacher and student may have different hidden dimensions, a learned linear projection $g$ maps student features to teacher feature space. This provides richer supervision than output-only matching: the student learns not just what to predict but how to represent. For LLM distillation, this can include matching attention patterns, hidden states at specific layers, or the output of feed-forward blocks."
    },
    {
      type: "mc",
      question: "On-policy distillation (used in models like Gemma and some LLaMA variants) differs from standard offline distillation by:",
      options: ["Using a smaller teacher model that closely matches the student's capacity, providing a more achievable learning target that reduces the capacity gap and improves transfer efficiency across tasks", "Having the **student generate its own outputs**, then using the teacher to score/correct them — avoiding the distribution mismatch where the student trains on teacher text but generates its own at inference", "Training without any teacher signal during the generation phase, relying solely on the student's self-supervised next-token prediction objective to learn useful representations from generated sequences", "Using only hard labels from the teacher instead of soft probability distributions, removing the temperature hyperparameter entirely and simplifying the distillation loss to standard cross-entropy"],
      correct: 1,
      explanation: "In offline distillation, the student trains on teacher-generated sequences. At inference, the student generates from its own distribution, creating exposure bias — errors compound because the student never learned to recover from its own mistakes. On-policy distillation lets the student generate sequences, then uses the teacher's per-token probabilities as training signal. This is analogous to DAgger in imitation learning. The GKD (Generalized Knowledge Distillation) framework formalizes the spectrum between on-policy and off-policy distillation."
    },
    {
      type: "mc",
      question: "Structured pruning removes entire **structures** (attention heads, neurons, layers), while unstructured pruning removes individual weights. The practical advantage of structured pruning is:",
      options: ["The resulting model has **regular, dense tensor shapes** that map efficiently to standard GPU/TPU hardware without specialized sparse kernels — unstructured sparsity creates irregular patterns that are hard to accelerate", "It can be applied during training at no additional cost because the optimizer's weight decay naturally drives entire structures to zero magnitude, handling their removal from the graph automatically", "It preserves more model quality at the same sparsity level because structure-level redundancy in attention heads and FFN neurons is inherently more common than individual weight redundancy in those layers", "It achieves higher compression ratios than unstructured pruning because removing entire structures eliminates more parameters per pruning decision while preserving the remaining gradient flow paths"],
      correct: 0,
      explanation: "Removing 50% of individual weights (unstructured) leaves an irregular sparse matrix requiring specialized sparse GEMM kernels to achieve speedup — and these kernels often underperform dense GEMMs until >90% sparsity on GPUs. Removing 50% of neurons (structured) simply halves the matrix dimension, yielding dense smaller matrices that run at full hardware efficiency. The trade-off: unstructured pruning preserves more quality at the same compression ratio, but structured pruning gives predictable, hardware-friendly speedups."
    },
    {
      type: "mc",
      question: "When merging two LoRA adapters trained on different tasks, the simplest approach is weight-space averaging: $\\Delta W_{\\text{merged}} = \\alpha \\Delta W_A + (1 - \\alpha) \\Delta W_B$. The fundamental limitation of this approach is:",
      options: [
        "The merged weights are always larger in magnitude than the originals, causing activation magnitudes to grow unboundedly and potentially overflow during inference on long sequences with many tokens",
        "The merged adapter has a different effective rank than either original, causing the merged model to systematically underfit or overfit relative to each individual adapter's optimal task performance",
        "LoRA rank prevents merging because two rank-$r$ adapters combined produce a rank-$2r$ update that exceeds the stable numerical rank supported by the base model's pretrained weight matrices",
        "Weight-space interpolation assumes a **linear loss landscape** between solutions — if loss barriers exist between adapter basins, the merged point performs poorly on both tasks despite each being strong individually"
      ],
      correct: 3,
      explanation: "Linear interpolation in weight space only works well when the two solutions lie in the same loss basin (connected by a low-loss path). If training dynamics led the adapters to different basins, the midpoint can sit on a high-loss ridge. This connects to mode connectivity research: models trained from the same pre-trained checkpoint tend to be linearly connected in the loss landscape, but fine-tuning on very different tasks can break this. Techniques like TIES-Merging and DARE address this by resolving sign conflicts and pruning redundant parameters before merging."
    },
    {
      type: "mc",
      question: "A teacher model has 70B parameters and a student has 7B. After distillation, the student achieves 95% of the teacher's accuracy on benchmarks. The student's inference cost is approximately:",
      options: ["The same as the teacher's cost because the student must internally simulate the teacher's full computation graph to reproduce the distilled knowledge at inference time", "95% of the teacher's cost, since the distillation process transfers most of the teacher's computational requirements into the student's learned weight representations", "50% of the teacher's cost due to the shared architecture design, where the student reuses half of the teacher's transformer layers directly during inference", "~10% of the teacher's cost — inference cost scales roughly linearly with parameter count (dominated by memory bandwidth for decode), so a 10x smaller model is ~10x cheaper"],
      correct: 3,
      explanation: "Inference compute and memory scale with parameter count, not training method. The 7B student needs ~10x fewer FLOPs per forward pass, ~10x less memory for weights, and ~10x less KV-cache memory (assuming proportionally smaller hidden dimensions). Distillation improves the student's quality for its size class but doesn't change its computational cost. This is precisely the value proposition of distillation: getting a model that punches above its weight class computationally."
    },
    {
      type: "mc",
      question: "Neural Architecture Search (NAS) for LLMs differs from NAS for vision models primarily because:",
      options: ["The **training cost of each candidate architecture** is enormous (millions of dollars for a full pre-training run), making exhaustive search infeasible — NAS for LLMs relies on proxy metrics or constrained subspaces", "LLM architectures are already optimal due to extensive manual tuning over the past decade, leaving no room for automated improvement through search-based methods", "LLMs don't have meaningful hyperparameters to search over since the Transformer architecture has converged to a standard design with fixed depth, width, and head count ratios", "NAS requires supervised labels to evaluate candidate architectures, which LLMs don't use since they are trained with self-supervised next-token prediction objectives"],
      correct: 0,
      explanation: "Vision NAS can evaluate a candidate architecture by training it to convergence in hours on a single GPU. LLM NAS cannot — training a 7B model costs ~\\$100K, making brute-force search over architecture variants prohibitive. Practical LLM NAS uses: (1) scaling law extrapolation from small proxy models, (2) zero-shot proxies based on gradient statistics, (3) constrained search over specific dimensions (depth, width, FFN ratio, head count) with other choices fixed. Results like the Chinchilla scaling laws are a form of two-variable NAS over model size and data quantity."
    },
    {
      type: "mc",
      question: "Layer pruning (removing entire transformer layers) from a 32-layer model shows an interesting pattern: removing layers from the **middle** of the network causes less degradation than removing early or late layers. This suggests:",
      options: ["Middle layers are redundant and should always be removed to create more efficient models, since they contribute minimal unique computation and require no recovery fine-tuning to restore capacity", "Layer ordering doesn't matter in transformers, so removing any arbitrary set of layers — early, middle, or late — produces equivalent degradation regardless of their position in the network", "The model was trained incorrectly, with insufficient regularization causing middle layers to learn redundant features rather than distinct complementary representations that justify their compute cost", "Middle layers exhibit more **representational redundancy** — adjacent middle layers compute similar transformations (high cosine similarity between inputs and outputs), while early and late layers perform harder-to-replace computation"],
      correct: 3,
      explanation: "Empirical studies (e.g., ShortGPT, LaCo) show that the cosine similarity between a middle layer's input and output is often >0.99, meaning the layer makes only a small residual update. Early layers show lower similarity (larger transformations building representations), and final layers show specialized computation for the output distribution. This pattern motivates depth pruning strategies that remove middle layers and connects to the broader observation that over-parameterized networks have redundant capacity concentrated in particular regions."
    },
    {
      type: "mc",
      question: "When distilling an LLM for a specific task (e.g., code generation), which data strategy typically yields the best student?",
      options: [
        "Training on the same pre-training corpus as the teacher, relying on the distillation loss to implicitly extract task-relevant knowledge from the general-purpose data distribution",
        "Using only human-labeled examples collected specifically for the target task, since expert annotations provide a stronger per-example learning signal than teacher-generated outputs",
        "Using the teacher to generate **synthetic task-specific data** with chain-of-thought reasoning, then training the student on this curated dataset that is both task-relevant and teacher-distribution-aligned",
        "Random data from the internet filtered by keyword and embedding similarity to the target task, maximizing distributional diversity while maintaining topical coverage"
      ],
      correct: 2,
      explanation: "Synthetic data generation from the teacher provides several advantages: (1) unlimited data at the cost of teacher inference, (2) data that is distributionally aligned with the teacher's capabilities, (3) ability to include reasoning traces that teach the student the process, not just the answer. This is the approach behind Phi-1 (textbook-quality synthetic data), Orca (explanation-augmented distillation), and WizardLM (evolved instructions). The key insight is that the teacher's ability to generate informative training data can be more valuable than its raw predictions."
    },
    {
      type: "mc",
      question: "Pruning at initialization (before training) versus pruning after training represents a fundamental debate. The **lottery ticket hypothesis** states:",
      options: ["Pruned networks always outperform dense networks because removing redundant parameters acts as implicit regularization that consistently improves generalization on held-out evaluation data and reduces overfitting", "All neural network architectures are equally expressive regardless of depth, width, or connectivity pattern, meaning pruning reduces computational efficiency but cannot reduce the function class the model can represent", "Dense randomly-initialized networks contain sparse subnetworks (winning tickets) that, trained in isolation from their original initialization, match the full network's accuracy — most parameters exist to help find them", "Random pruning is as effective as magnitude-based or gradient-informed methods because the specific identity of removed weights matters less than the overall sparsity level for determining final model quality"],
      correct: 2,
      explanation: "Frankle & Carlin (2019) showed that within a randomly initialized dense network, there exist sparse subnetworks that can be trained from their original initialization to match the dense network's performance. Finding these \"winning tickets\" requires train-prune-reset cycles. For LLMs, directly finding winning tickets is computationally prohibitive, but the hypothesis motivates the intuition behind successful post-training pruning: well-trained networks contain many low-importance parameters whose removal minimally impacts function. Subsequent work (especially on supermasks) has refined these findings significantly."
    }
  ]
};
