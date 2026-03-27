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
        "Soft targets are easier to compute because the temperature-scaled softmax requires fewer floating-point operations than the standard argmax over logits",
        "The teacher's probability distribution over **incorrect classes** encodes rich similarity structure (\"dark knowledge\") — e.g., a teacher assigning 0.05 to 'cat' and 0.001 to 'car' for a dog image reveals that dogs are more like cats than cars",
        "Soft targets prevent the student from overfitting to the training data by smoothing the label distribution, acting as a form of label regularization",
        "Temperature scaling makes the loss function convex, guaranteeing that gradient descent converges to the global optimum during student training"
      ],
      correct: 1,
      explanation: "Hinton et al. (2015) showed that the teacher's full probability vector, especially the relative probabilities of incorrect classes, contains far more information per training example than a one-hot label. With temperature $\\tau$, the softmax becomes $p_i = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}$, which smooths the distribution to expose these inter-class relationships. The student loss combines the soft target KL divergence (weighted by $\\tau^2$) with the hard label cross-entropy."
    },
    {
      type: "mc",
      question: "Logit matching distillation minimizes $\\text{KL}(p_{\\text{teacher}} \\| p_{\\text{student}})$ over the output distribution. Feature matching distillation instead:",
      options: ["Uses reinforcement learning to train the student, with the teacher's predictions serving as the reward signal that guides the student's policy optimization", "Matches only the final prediction but uses a squared error loss instead of KL divergence, which changes the gradient dynamics to favor high-confidence predictions", "Aligns **intermediate layer representations** between teacher and student — e.g., minimizing $\\|f_{\\text{teacher}}^{(l)} - g(f_{\\text{student}}^{(k)})\\|^2$ where $g$ is a learned projection to handle dimension mismatches", "Matches the gradient norms of teacher and student at each layer, ensuring that the student's optimization dynamics mirror the teacher's training trajectory"],
      correct: 2,
      explanation: "Feature matching (FitNets, PKD) adds losses that align intermediate representations. Since teacher and student may have different hidden dimensions, a learned linear projection $g$ maps student features to teacher feature space. This provides richer supervision than output-only matching: the student learns not just what to predict but how to represent. For LLM distillation, this can include matching attention patterns, hidden states at specific layers, or the output of feed-forward blocks."
    },
    {
      type: "mc",
      question: "On-policy distillation (used in models like Gemma and some LLaMA variants) differs from standard offline distillation by:",
      options: ["Using a smaller teacher model that more closely matches the student's capacity, providing a more achievable learning target than a large teacher", "Having the **student generate its own outputs**, then using the teacher to score/correct them — this avoids the train-test distribution mismatch where the student is trained on teacher-generated text but must generate its own text at inference", "Training without any teacher signal during the generation phase, relying solely on the student's own self-supervised objective for the output sequences", "Using only hard labels from the teacher instead of soft probability distributions, which removes the temperature hyperparameter and simplifies the loss computation"],
      correct: 1,
      explanation: "In offline distillation, the student trains on teacher-generated sequences. At inference, the student generates from its own distribution, creating exposure bias — errors compound because the student never learned to recover from its own mistakes. On-policy distillation lets the student generate sequences, then uses the teacher's per-token probabilities as training signal. This is analogous to DAgger in imitation learning. The GKD (Generalized Knowledge Distillation) framework formalizes the spectrum between on-policy and off-policy distillation."
    },
    {
      type: "mc",
      question: "Structured pruning removes entire **structures** (attention heads, neurons, layers), while unstructured pruning removes individual weights. The practical advantage of structured pruning is:",
      options: ["The resulting model has **regular, dense tensor shapes** that run efficiently on standard hardware (GPUs/TPUs) without specialized sparse kernels — unstructured pruning creates irregular sparsity patterns that standard hardware cannot accelerate", "It can be applied during training at no additional cost because the structure removal is handled automatically by the optimizer's weight decay", "It preserves more model quality at the same sparsity level because structure-level redundancy is more common than individual weight redundancy", "It achieves higher sparsity levels than unstructured pruning because removing entire structures eliminates more parameters per pruning decision"],
      correct: 0,
      explanation: "Removing 50% of individual weights (unstructured) leaves an irregular sparse matrix requiring specialized sparse GEMM kernels to achieve speedup — and these kernels often underperform dense GEMMs until >90% sparsity on GPUs. Removing 50% of neurons (structured) simply halves the matrix dimension, yielding dense smaller matrices that run at full hardware efficiency. The trade-off: unstructured pruning preserves more quality at the same compression ratio, but structured pruning gives predictable, hardware-friendly speedups."
    },
    {
      type: "mc",
      question: "When merging two LoRA adapters trained on different tasks, the simplest approach is weight-space averaging: $\\Delta W_{\\text{merged}} = \\alpha \\Delta W_A + (1 - \\alpha) \\Delta W_B$. The fundamental limitation of this approach is:",
      options: [
        "The merged weights are always larger in magnitude than the originals, causing activation magnitudes to grow and potentially overflow at inference",
        "The merged adapter has a different effective rank than the originals, causing the merged model to either underfit or overfit relative to each individual adapter",
        "LoRA rank prevents merging because two rank-$r$ adapters combined exceed the maximum rank supported by the base model architecture",
        "Weight-space interpolation assumes a **linear loss landscape** between the two solutions — if the loss landscape has barriers between the adapter basins, the merged point may perform poorly on both tasks despite each adapter being individually excellent"
      ],
      correct: 3,
      explanation: "Linear interpolation in weight space only works well when the two solutions lie in the same loss basin (connected by a low-loss path). If training dynamics led the adapters to different basins, the midpoint can sit on a high-loss ridge. This connects to mode connectivity research: models trained from the same pre-trained checkpoint tend to be linearly connected in the loss landscape, but fine-tuning on very different tasks can break this. Techniques like TIES-Merging and DARE address this by resolving sign conflicts and pruning redundant parameters before merging."
    },
    {
      type: "mc",
      question: "A teacher model has 70B parameters and a student has 7B. After distillation, the student achieves 95% of the teacher's accuracy on benchmarks. The student's inference cost is approximately:",
      options: ["The same as the teacher's cost because the student must internally simulate the teacher's computation to reproduce the distilled knowledge at inference time", "95% of the teacher's cost, since the distillation process transfers most of the teacher's computational requirements into the student's learned representations", "50% of the teacher's cost due to the shared architecture design, where the student reuses half of the teacher's layers during inference", "~10% of the teacher's cost — inference cost scales roughly linearly with parameter count (dominated by weight-loading for decode), so a 10x smaller model is ~10x cheaper regardless of how well it was trained"],
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
      options: ["Middle layers are redundant and should always be removed to create more efficient models without any fine-tuning or recovery training needed to restore the lost capacity", "Layer ordering doesn't matter in transformers, so removing any arbitrary set of layers — early, middle, or late — has the same effect regardless of position in the network", "The model was trained incorrectly, with insufficient regularization causing the middle layers to learn redundant features rather than distinct, complementary representations", "Middle layers exhibit more **representational redundancy** — adjacent middle layers compute similar transformations (high cosine similarity between inputs and outputs), while early and late layers are harder to compensate for"],
      correct: 3,
      explanation: "Empirical studies (e.g., ShortGPT, LaCo) show that the cosine similarity between a middle layer's input and output is often >0.99, meaning the layer makes only a small residual update. Early layers show lower similarity (larger transformations building representations), and final layers show specialized computation for the output distribution. This pattern motivates depth pruning strategies that remove middle layers and connects to the broader observation that over-parameterized networks have redundant capacity concentrated in particular regions."
    },
    {
      type: "mc",
      question: "When distilling an LLM for a specific task (e.g., code generation), which data strategy typically yields the best student?",
      options: [
        "Training on the same pre-training corpus as the teacher, relying on the distillation loss to extract task-relevant knowledge from the general-purpose training data",
        "Using only human-labeled examples collected specifically for the target task, since human annotations provide a stronger learning signal than teacher-generated outputs",
        "Using the teacher to generate **synthetic task-specific data** — having the teacher produce many high-quality examples with chain-of-thought reasoning, then training the student on this curated dataset, which is both task-relevant and teacher-distribution-aligned",
        "Random data from the internet filtered by keyword relevance to the target task, maximizing data diversity while maintaining topical coverage"
      ],
      correct: 2,
      explanation: "Synthetic data generation from the teacher provides several advantages: (1) unlimited data at the cost of teacher inference, (2) data that is distributionally aligned with the teacher's capabilities, (3) ability to include reasoning traces that teach the student the process, not just the answer. This is the approach behind Phi-1 (textbook-quality synthetic data), Orca (explanation-augmented distillation), and WizardLM (evolved instructions). The key insight is that the teacher's ability to generate informative training data can be more valuable than its raw predictions."
    },
    {
      type: "mc",
      question: "Pruning at initialization (before training) versus pruning after training represents a fundamental debate. The **lottery ticket hypothesis** states:",
      options: ["Pruned networks always outperform dense networks because removing redundant parameters acts as regularization that improves generalization on held-out data", "All neural network architectures are equally expressive regardless of depth, width, or connectivity pattern, so pruning cannot reduce capability", "Dense randomly-initialized networks contain **sparse subnetworks** (winning tickets) that, when trained in isolation from their original initialization, match the full dense network's accuracy — suggesting that most parameters exist to help find these subnetworks during training", "Random pruning is as effective as informed pruning methods, since the specific identity of removed weights matters less than the overall sparsity level"],
      correct: 2,
      explanation: "Frankle & Carlin (2019) showed that within a randomly initialized dense network, there exist sparse subnetworks that can be trained from their original initialization to match the dense network's performance. Finding these \"winning tickets\" requires train-prune-reset cycles. For LLMs, directly finding winning tickets is computationally prohibitive, but the hypothesis motivates the intuition behind successful post-training pruning: well-trained networks contain many low-importance parameters whose removal minimally impacts function. Subsequent work (especially on supermasks) has refined these findings significantly."
    }
  ]
};
