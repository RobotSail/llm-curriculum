// Section B.4: Training Stability & Dynamics Assessment

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
      options: [
        "Sharpness (largest Hessian eigenvalue) rises to $\\approx 2/\\eta$, then oscillates around this threshold while loss continues to decrease non-monotonically",
        "The model parameters reach a critical point where any perturbation causes immediate divergence, requiring careful checkpointing and frequent restart protocols",
        "Training loss oscillates wildly between high and low values, but validation loss remains stable and smoothly decreasing throughout the entire optimization process",
        "Batch normalization causes gradient norms to hover at a fixed value, creating an artificial stability boundary that prevents the loss from decreasing any further"
      ],
      correct: 0,
      explanation: "Classical optimization theory predicts divergence when sharpness exceeds $2/\\eta$. Instead, Cohen et al. observed that full-batch GD on neural networks enters a regime where sharpness self-stabilizes at $\\approx 2/\\eta$: when it exceeds this threshold, the loss temporarily increases (the optimizer takes steps that are \"too large\"), which modifies the landscape to reduce sharpness back below the threshold. This is not predicted by convex optimization theory and suggests GD implicitly regularizes toward flatter minima."
    },
    {
      type: "mc",
      question: "The distinction between the \"feature learning\" regime and the \"kernel (lazy)\" regime in neural network training refers to:",
      options: [
        "Whether the model uses convolutional or attention-based features, since each architecture type operates in a fundamentally distinct optimization regime",
        "Whether features are learned in supervised or unsupervised fashion, since self-supervised objectives produce qualitatively different internal representation dynamics",
        "Whether the network's internal representations change substantially during training, or it behaves like a linear model around initialization over fixed random features",
        "Whether the kernel function is Gaussian (RBF) or polynomial, which determines the implicit bias of the neural network toward smooth or piecewise solutions"
      ],
      correct: 2,
      explanation: "In the kernel (lazy/NTK) regime — which can arise with very large width or very small learning rate — the network's internal representations barely move from their random initialization. Learning happens only by adjusting output weights over essentially fixed features. In the feature learning regime, representations transform substantially, enabling the model to discover task-relevant abstractions. Practical LLMs operate firmly in the feature learning regime. The $\\mu$P parameterization is designed to keep models in this regime across scales."
    },
    {
      type: "mc",
      question: "Induction heads are a specific attention pattern discovered in Transformer language models. They perform the operation of:",
      options: [
        "Attending to the first token in the sequence to establish a global context vector that grounds all subsequent token predictions throughout the forward pass",
        "Computing the average of all previous token embeddings to form a compressed context representation that captures the overall sequence-level meaning",
        "Attending to the most semantically similar token in context based on cosine similarity between the query and key representations at each position",
        "Identifying a previous occurrence of the current token and copying the token that followed it — implementing an in-context bigram lookup pattern"
      ],
      correct: 3,
      explanation: "Induction heads (Olsson et al., 2022) implement a two-step copying mechanism: (1) a \"previous token\" head identifies where the current token last appeared, (2) the induction head attends to the position after that previous occurrence and copies its value. This implements the pattern: if [$A$][$B$] appeared before and we now see [$A$], predict [$B$]. This is a fundamental circuit for in-context learning and is one of the clearest examples of interpretable algorithmic behavior in Transformers."
    },
    {
      type: "mc",
      question: "The formation of induction heads during training exhibits a phase transition. What does this mean concretely?",
      options: [
        "Induction heads form instantly at initialization because the random weight configuration already contains the required circuit structure in statistical expectation",
        "A sudden, discrete jump in in-context learning ability occurs at a specific training step, with loss on repeated patterns dropping sharply over a narrow window",
        "Induction heads form only if the model has more than 12 layers, since the two-head composition circuit requires sufficient network depth to develop properly",
        "The model alternates between having and not having induction heads as training progresses, oscillating in sync with the learning rate schedule cycles"
      ],
      correct: 1,
      explanation: "Olsson et al. observed that in-context learning ability (measured by how much loss decreases from the first to the second occurrence of a pattern) remains near zero for many training steps, then rapidly improves over a narrow window. This coincides with the formation of the induction head circuit. This is a genuine phase transition — a qualitative change in capability emerging suddenly from continuous optimization. It's one of the clearest examples of emergent capability in a controlled setting."
    },
    {
      type: "mc",
      question: "Loss landscape mode connectivity refers to the finding that:",
      options: [
        "All local minima have the same loss value, meaning there is no benefit to searching for better solutions beyond the first local minimum found by gradient descent",
        "Different trained models (from different initializations) can often be connected by low-loss paths in weight space, suggesting they share the same broad basin",
        "The loss landscape is convex near any local minimum, ensuring that gradient descent in the local neighborhood always monotonically improves the training objective",
        "Gradient descent always converges to the global minimum in overparameterized networks, so all independently trained models end up at the same point in weight space"
      ],
      correct: 1,
      explanation: "Mode connectivity (Garipov et al., 2018; Draxler et al., 2018) showed that independently trained models often lie in connected low-loss regions. While the straight line between two models in weight space may cross a loss barrier, a slightly curved path (found by optimization) often connects them with negligible loss increase. This suggests the loss landscape of overparameterized networks has a simpler structure than previously thought — most good minima are connected."
    },
    {
      type: "mc",
      question: "Mode connectivity has direct implications for model merging. When we average the weights of two fine-tuned models (linear interpolation $\\theta_{\\text{merged}} = \\alpha \\theta_1 + (1 - \\alpha) \\theta_2$), the merged model performs well only when:",
      options: [
        "Both models have the same parameter count, since mismatched architectures create dimensional incompatibilities that prevent any meaningful weight interpolation",
        "Both models were trained on identical data distributions, because different training data always pushes models into fundamentally incompatible loss landscape regions",
        "The two models share a common pretrained initialization — this keeps them in the same loss basin, so the linear interpolation path stays in a low-loss region",
        "The models use different optimizers to ensure diversity, since optimizer disagreement creates complementary solutions that average together more effectively"
      ],
      correct: 2,
      explanation: "Models fine-tuned from the same pretrained checkpoint tend to remain in the same loss basin (the pretrained model acts as an \"anchor\"). Linear interpolation between them stays in the low-loss region. Models trained from different random initializations typically do NOT mode-connect linearly — there are loss barriers between their basins. This is why weight averaging works well for merging LoRA adapters or task-specific fine-tunes of the same base model, but fails for independently pretrained models."
    },
    {
      type: "mc",
      question: "Training instabilities (loss spikes) in large language model training are often attributed to:",
      options: [
        "Hardware failures causing corrupted gradients that propagate through the distributed training pipeline before being detected by checksum-based validation protocols",
        "The training data containing adversarial examples specifically crafted to maximize gradient magnitudes and intentionally destabilize the optimization trajectory early in training",
        "Running out of unique training data partway through the run, causing the model to memorize repeated examples and diverge from generalizable learned representations",
        "Outlier activations and attention logit growth — a few hidden dimensions develop very large magnitudes, causing softmax saturation and sudden gradient explosion"
      ],
      correct: 3,
      explanation: "Dettmers et al. (2022) and Zhai et al. (2023) documented how outlier features (hidden dimensions with magnitudes 10-100x larger than typical) emerge during training. These cause numerical issues: attention logits grow large, softmax saturates, and gradients spike. Mitigations include QK-norm (normalizing query and key vectors before the dot product), logit capping, and careful initialization. PaLM and other large models reported loss spikes that required manual intervention (learning rate reduction or data skipping)."
    },
    {
      type: "mc",
      question: "In the context of $\\mu$P, what happens to the gradient dynamics of a standard (non-$\\mu$P) Transformer as you increase width $d$ while keeping learning rate fixed?",
      options: [
        "Gradients vanish because each individual weight contributes less to the output, causing the effective learning signal per parameter to shrink asymptotically toward zero",
        "The model becomes more robust to learning rate choices because the wider layers average out gradient noise across their larger number of parameters",
        "The model enters the kernel (lazy) regime: weight updates become infinitesimal relative to initialization, so internal representations stop changing meaningfully",
        "Training speed doubles with each doubling of width due to increased parallelism in the matrix operations, making wider models strictly more compute-efficient"
      ],
      correct: 2,
      explanation: "Under standard parameterization (SP), if you keep the learning rate fixed and increase width, each weight's update contributes less to the output (because activations are averaged over more dimensions). In the infinite-width limit, this gives the Neural Tangent Kernel regime where the network is effectively linear around initialization. $\\mu$P rescales learning rates and initialization so that the contribution of each weight update to the output remains $\\Theta(1)$, preserving feature learning dynamics regardless of width."
    },
    {
      type: "mc",
      question: "The phenomenon of \"grokking\" in neural network training refers to:",
      options: [
        "The model failing to learn despite sufficient capacity, where the loss plateaus at a high value regardless of training duration or any hyperparameter tuning attempts",
        "Delayed generalization: the model first memorizes training data (zero training loss, high test loss), then much later suddenly generalizes (test loss drops sharply)",
        "Rapid learning in the first few training steps followed by a sustained plateau where neither training nor test metrics show any measurable further improvement",
        "The model learning multiple tasks simultaneously without interference, where multi-task training matches single-task performance on each individual learning objective"
      ],
      correct: 1,
      explanation: "Grokking (Power et al., 2022) is a striking phenomenon where generalization occurs long after memorization. On modular arithmetic tasks, models achieve perfect training accuracy quickly, but test accuracy remains at chance for many more steps before suddenly jumping to near-perfect. This suggests the model transitions from a memorization solution to an algorithmic (generalizing) solution. Weight decay and regularization accelerate grokking, supporting the interpretation that regularization pressure eventually pushes the model toward the simpler, generalizing solution."
    },
    {
      type: "mc",
      question: "When training a large Transformer, practitioners often observe that the effective learning rate must be adjusted for different parts of the model. Which statement about per-layer learning rate dynamics is correct?",
      options: [
        "Earlier layers tend to have smaller gradients (effectively lower learning rates under Adam), while attention logits and embeddings require special handling to prevent instability",
        "All layers should use exactly the same learning rate for optimal training, since Adam's adaptive rates already fully account for gradient magnitude differences across layers",
        "Later layers should always use a smaller learning rate because they are closer to the loss function and receive correspondingly larger gradients requiring dampening",
        "Per-layer learning rates are only necessary for models exceeding 100B parameters, since smaller models have sufficiently uniform gradient norms across all their layers"
      ],
      correct: 0,
      explanation: "The gradient magnitudes vary systematically across a Transformer: embedding layers and attention logits tend to grow disproportionately, contributing to instability. Adam's adaptive rates help but don't fully resolve this. Techniques like QK-LayerNorm (normalizing queries and keys), embedding scaling, and logit capping address specific problematic components. $\\mu$P provides a principled framework by prescribing per-layer multipliers that maintain consistent update scales. In practice, many large-scale training runs use lower learning rates for embeddings."
    }
  ]
};
