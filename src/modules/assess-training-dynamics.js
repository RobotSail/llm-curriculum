// B.4 Training Stability & Dynamics — per-section test (split from assess-branch-b.js)

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
