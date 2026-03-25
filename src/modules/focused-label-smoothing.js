// Focused module: Label Smoothing
// Covers one-hot targets vs smoothed targets, the finite optimal logit gap,
// label smoothing as entropy regularization, and practical implications.

export const labelSmoothingLearning = {
  id: "0.2-label-smoothing-learning-hard",
  sectionId: "0.2",
  title: "Label Smoothing",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Label Smoothing: Softening the Target",
      content: "Standard classification training uses **one-hot** targets: $P(y) = \\mathbf{1}[y = y^*]$. The cross-entropy loss drives the model to output $Q(y^*) \\to 1$, which requires the logit for the correct class to go to $+\\infty$ relative to all others.\n\n**Label smoothing** (Szegedy et al., 2016) replaces the one-hot target with a mixture of the one-hot and the uniform distribution:\n\n$$P'(y) = (1 - \\alpha) \\cdot \\mathbf{1}[y = y^*] + \\frac{\\alpha}{K}$$\n\nwhere $\\alpha \\in (0, 1)$ is the smoothing parameter and $K$ is the number of classes. This assigns probability $(1 - \\alpha + \\alpha/K)$ to the correct class and $\\alpha/K$ to each incorrect class.\n\nThe cross-entropy with smoothed targets decomposes as:\n\n$$H(P', Q) = (1 - \\alpha) \\cdot H(\\text{one-hot}, Q) + \\alpha \\cdot H(\\text{uniform}, Q)$$\n\n$$= -(1 - \\alpha) \\log Q(y^*) - \\frac{\\alpha}{K} \\sum_k \\log Q(k)$$\n\nThe second term $-\\frac{\\alpha}{K} \\sum_k \\log Q(k)$ penalizes the model for being too confident — it's an **entropy regularizer** that prevents the output distribution from collapsing to a point mass."
    },
    {
      type: "mc",
      question: "Without label smoothing ($\\alpha = 0$), what are the optimal logits for the correct class to minimize cross-entropy with a one-hot target?",
      options: [
        "The logit should converge to exactly $1.0$ to match the one-hot probability target",
        "The logit should converge to $\\log K$ to balance the softmax normalizer over $K$ classes",
        "The logit should go to $+\\infty$ since driving $Q(y^*) \\to 1$ requires unbounded logits",
        "The logit should converge to the log-prior $\\log P(y^*)$ for Bayesian consistency"
      ],
      correct: 2,
      explanation: "With a one-hot target, the loss is $-\\log Q(y^*)$, which is minimized by $Q(y^*) \\to 1$. Since $Q(y^*) = \\text{softmax}(z_{y^*})$, achieving $Q(y^*) = 1$ requires $z_{y^*} - z_k \\to \\infty$ for all $k \\neq y^*$. The optimal logits are unbounded — they grow without limit during training. This drives the model toward extreme confidence and encourages memorization, as the gradients never vanish regardless of how confident the model already is."
    },
    {
      type: "info",
      title: "The Finite Optimal Logit Gap",
      content: "With label smoothing, the optimal logits are **finite**. The smoothed target assigns:\n- Correct class: $p^* = 1 - \\alpha + \\alpha/K = 1 - \\alpha(K-1)/K$\n- Each incorrect class: $p_{\\text{wrong}} = \\alpha/K$\n\nThe optimal model output must match this target distribution. For a softmax with logits $z$, the optimal solution has all incorrect logits equal (by symmetry) and the gap between the correct logit $z^*$ and any incorrect logit $z_{\\text{wrong}}$ is:\n\n$$z^* - z_{\\text{wrong}} = \\log \\frac{p^*}{p_{\\text{wrong}}} = \\log \\frac{1 - \\alpha(K-1)/K}{\\alpha/K}$$\n\nFor small $\\alpha$ and large $K$, this simplifies to approximately:\n\n$$\\Delta z \\approx \\log \\frac{(1-\\alpha) \\cdot K}{\\alpha}$$\n\nThis is finite and well-defined. The model converges to a confident-but-not-infinitely-confident prediction. The output distribution retains nonzero entropy — the model maintains a \"soft\" probability over alternatives, which acts as a form of **built-in uncertainty quantification**."
    },
    {
      type: "mc",
      question: "With label smoothing $\\alpha = 0.1$ and vocabulary size $K = 50000$, what is the approximate optimal logit gap between the correct and incorrect classes?",
      options: [
        "$\\log(\\alpha \\cdot K) = \\log(0.1 \\times 50000) = \\log(5000) \\approx 8.5$",
        "$\\log((1-\\alpha) \\cdot K / \\alpha) = \\log(0.9 \\times 50000 / 0.1) = \\log(450000) \\approx 13.0$",
        "$\\log(K) = \\log(50000) \\approx 10.8$, independent of $\\alpha$",
        "$\\log((1-\\alpha)/\\alpha) = \\log(0.9 / 0.1) = \\log(9) \\approx 2.2$"
      ],
      correct: 1,
      explanation: "The optimal gap is $\\log\\frac{(1-\\alpha)K}{\\alpha} = \\log\\frac{0.9 \\times 50000}{0.1} = \\log(450000) \\approx 13.0$ (using natural log). Without label smoothing, this gap would be $+\\infty$. With $\\alpha = 0.1$, it is a large but finite number. The logits stabilize rather than growing without bound, leading to better-conditioned gradients in the final layers."
    },
    {
      type: "mc",
      question: "You increase the label smoothing parameter from $\\alpha = 0.1$ to $\\alpha = 0.3$. How does this change the optimal logit gap?",
      options: [
        "The gap increases — stronger smoothing forces the model to separate classes with more confident predictions",
        "The gap stays the same — $\\alpha$ only affects the loss magnitude during training, not the optimal logit values",
        "The gap decreases — stronger smoothing moves the target closer to uniform, so optimal logits are less extreme",
        "The gap becomes negative — stronger smoothing reverses the ordering so the model favors wrong answers"
      ],
      correct: 2,
      explanation: "The optimal logit gap is $\\log\\frac{(1-\\alpha)K}{\\alpha}$. Increasing $\\alpha$ from 0.1 to 0.3 means the numerator decreases ($0.7K$ vs $0.9K$) and the denominator increases ($0.3$ vs $0.1$), so the gap shrinks substantially. At $\\alpha = 0.3$: $\\log(0.7 \\times 50000 / 0.3) \\approx \\log(116667) \\approx 11.7$, vs $\\approx 13.0$ at $\\alpha = 0.1$. Stronger smoothing pulls the optimal output closer to uniform (smaller logit gaps)."
    },
    {
      type: "info",
      title: "Label Smoothing as KL Regularization",
      content: "The label smoothing loss can be decomposed in a revealing way:\n\n$$\\mathcal{L}_{\\text{LS}} = (1 - \\alpha) \\cdot H(\\text{one-hot}, Q) + \\alpha \\cdot H(\\text{uniform}, Q)$$\n\nThe second term $H(\\text{uniform}, Q) = \\log K - H(Q)$ is (up to a constant) the **negative entropy** of the model's output. Minimizing this term maximizes $H(Q)$.\n\nEquivalently, the label smoothing objective includes a term proportional to:\n\n$$-\\text{KL}(\\text{uniform} \\| Q) + \\text{const}$$\n\nThis penalizes the model for being far from the uniform distribution in the KL sense. The model is pulled toward two competing goals:\n1. **Accuracy**: put most mass on the correct class (from the one-hot term)\n2. **Entropy**: don't collapse to a point mass (from the uniform term)\n\nThe parameter $\\alpha$ controls the balance. At $\\alpha = 0$: pure accuracy (one-hot cross-entropy). As $\\alpha \\to 1$: pure entropy maximization (output approaches uniform). Typical values in practice are $\\alpha \\in [0.05, 0.2]$."
    },
    {
      type: "mc",
      question: "A team trains a language model with label smoothing $\\alpha = 0.1$. Compared to training without label smoothing, what happens to the gradient magnitude from the final linear layer during late training?",
      options: [
        "Gradients become larger because the smoothed loss has a steeper curvature near the optimum",
        "Gradients remain the same size because label smoothing only affects the target, not the loss landscape shape",
        "Gradients become smaller and more stable because the finite logit gap prevents the loss from driving logits to extremes",
        "Gradients oscillate between large and small as the model alternates between fitting the one-hot and uniform components"
      ],
      correct: 2,
      explanation: "Without label smoothing, the loss $-\\log Q(y^*)$ produces gradients proportional to $(Q(y^*) - 1)$, which only vanishes as $Q(y^*) \\to 1$ (requiring $z \\to \\infty$). The model keeps pushing logits larger indefinitely. With label smoothing, the optimal logits are finite — once the model reaches the optimal gap, gradients from the correct-class and uniform terms approximately cancel. This stabilizes the final-layer gradients and prevents the logit-growth pathology."
    },
    {
      type: "info",
      title: "Label Smoothing vs Temperature Scaling",
      content: "Label smoothing and temperature scaling both affect model confidence, but at different stages and in different ways:\n\n**Label smoothing** (training time):\n- Modifies the *target distribution* the model fits\n- Changes what the model learns — representations differ\n- The model internalizes the \"don't be too confident\" signal\n- Cannot be undone after training\n\n**Temperature scaling** (post-hoc):\n- Rescales logits *after* training is complete\n- Does not change representations — only adjusts confidence\n- A single scalar parameter fit on a validation set\n- Can be adjusted or removed at any time\n\nCrucially, label smoothing affects representation quality. Müller et al. (2019) showed that label smoothing produces **more clustered penultimate-layer representations** — examples from the same class are closer together, and classes are more equidistant from each other. Temperature scaling has no such effect on representations.\n\nIn LLM pretraining, label smoothing is less common than in vision. Most LLMs train with standard cross-entropy loss and rely on other regularization (dropout, weight decay) to control overconfidence. However, label smoothing is standard in some machine translation and speech recognition pipelines."
    },
    {
      type: "mc",
      question: "A vision model trained with label smoothing $\\alpha = 0.1$ is evaluated. Its penultimate-layer representations show tight, equidistant class clusters. A second model trained without label smoothing on the same data shows more diffuse, overlapping clusters. Which model likely has better **transfer learning** features?",
      options: [
        "The model without label smoothing — diffuse representations preserve more fine-grained information for downstream tasks",
        "The label-smoothed model — tighter clusters indicate better class separation, which always transfers well",
        "It depends on the downstream task — tighter clusters help classification but may hurt tasks requiring within-class discrimination",
        "Neither — representation geometry has no empirical relationship with transfer learning performance"
      ],
      correct: 2,
      explanation: "This is nuanced. Label smoothing's equidistant clusters are great for classification transfer (classes are well-separated). But the tight clustering means within-class variation is compressed — fine-grained features (distinguishing breeds of dogs, subtypes of cells) may be lost. For retrieval or fine-grained recognition, the more diffuse representations might actually transfer better. Müller et al. (2019) noted that label smoothing can hurt knowledge distillation for exactly this reason: the teacher's dark knowledge (relative probabilities among incorrect classes) is suppressed."
    },
    {
      type: "info",
      title: "When NOT to Use Label Smoothing",
      content: "Label smoothing is not universally beneficial. There are specific scenarios where it hurts:\n\n**1. Knowledge distillation**: When using a teacher model's soft predictions as targets for a student, the teacher's relative confidence across incorrect classes carries information about class similarities. Label smoothing suppresses this \"dark knowledge\" by making the teacher's output more uniform across wrong classes.\n\n**2. Calibration-sensitive applications**: Label smoothing makes the model less confident, but not necessarily better calibrated. The model may become *underconfident* on easy examples while still being overconfident on hard ones. Post-hoc calibration (temperature scaling) is often more principled for calibration.\n\n**3. Extreme class imbalance**: With rare classes, label smoothing steals probability from the correct class and redistributes it uniformly. For a class that appears in 0.01% of data, the model already struggles to assign it high probability — smoothing makes this harder.\n\n**4. When $\\alpha$ is too large**: Values above 0.2–0.3 can significantly hurt accuracy. The model becomes too uncertain about the correct class, and the uniform pressure dominates the learning signal. Standard practice: $\\alpha = 0.1$ for most tasks, lower (0.05) for fine-grained classification."
    },
    {
      type: "mc",
      question: "A team uses knowledge distillation: a large teacher model's soft outputs train a smaller student. They also apply label smoothing ($\\alpha = 0.1$) to the teacher. What effect does this have on distillation quality?",
      options: [
        "Distillation improves because the teacher's outputs are already smoothed, giving the student better soft targets",
        "Distillation degrades because the teacher's dark knowledge — relative probabilities among wrong classes — is suppressed by label smoothing",
        "No effect because the student only uses the teacher's top-1 prediction, not the full probability distribution",
        "Distillation improves for hard examples but degrades for easy examples due to the confidence redistribution"
      ],
      correct: 1,
      explanation: "Knowledge distillation works by transferring the teacher's soft predictions, which encode class similarities (\"this 7 looks a bit like a 1\"). Label smoothing pushes the teacher's output toward uniform over incorrect classes, destroying this structure — all wrong classes look equally wrong. The student receives less informative targets. Müller et al. (2019) demonstrated this empirically: label-smoothed teachers produce worse distillation students than teachers trained with standard cross-entropy."
    }
  ]
};
