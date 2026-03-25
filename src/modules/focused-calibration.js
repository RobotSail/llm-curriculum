// Module: Model Calibration
// Section 0.2: Probability, statistics & information theory
// Focused learning module covering calibration: what it means, how it breaks,
// how to measure it (ECE, reliability diagrams), how to fix it (temperature
// scaling, Platt scaling), and calibration in LLMs specifically.

export const calibrationLearning = {
  id: "0.2-calibration-learning-hard",
  sectionId: "0.2",
  title: "Model Calibration",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Calibration: When Confidence Matches Accuracy",
      content: "A model is **calibrated** if its confidence estimates match its actual accuracy:\n\n$$P(\\text{correct} \\mid \\text{confidence} = p) = p$$\n\nIf a calibrated model says \"90% confident\" on 1000 predictions, approximately 900 should be correct. If it says \"60% confident\" on 500 predictions, approximately 300 should be correct. This must hold across *all* confidence levels simultaneously.\n\nCross-entropy is a **proper scoring rule**: the expected loss is uniquely minimized when $Q = P_{\\text{true}}$. A model trained to the global optimum on infinite data with cross-entropy loss would therefore be perfectly calibrated.\n\nBut in practice, three factors break calibration:\n\n**1. Finite data**: The model overfits, learning to be confident on training examples without proportional accuracy on new data.\n\n**2. Overparameterization**: Modern networks can fit the training data with room to spare, and the excess capacity drives predictions toward extreme confidence.\n\n**3. Batch normalization and weight decay**: These interact with softmax in subtle ways that shift the confidence distribution upward.\n\nThe result: modern neural networks are systematically **overconfident** — their predicted probabilities exceed their actual accuracy. Guo et al. (2017) demonstrated this across architectures: deeper and wider networks are more miscalibrated despite being more accurate."
    },
    {
      type: "mc",
      question: "A model outputs \"90% confident\" on 100 predictions. Of these, 72 are correct. Is the model well-calibrated, overconfident, or underconfident for this confidence level?",
      options: [
        "Underconfident — the model should have stated even higher confidence given the base rate of the dataset",
        "Well-calibrated — 72% accuracy is within acceptable tolerance of the stated 90% confidence",
        "Cannot determine — calibration requires examining the full joint distribution of labels and predictions",
        "Overconfident — the model claimed 90% confidence but only achieved 72% accuracy on those predictions"
      ],
      correct: 3,
      explanation: "The model said \"90% confident\" but only 72/100 = 72% were correct. It is **overconfident**: its stated confidence (90%) significantly exceeds its actual accuracy (72%) for this confidence bin. A well-calibrated model at 90% confidence should get approximately 90 out of 100 correct. This 18-percentage-point gap is far too large to attribute to sampling noise (a binomial test rejects calibration at this level with $p < 0.001$)."
    },
    {
      type: "mc",
      question: "Why are modern deep networks typically overconfident rather than underconfident, even when well-trained with cross-entropy loss?",
      options: [
        "The softmax function itself is inherently biased toward assigning extreme probabilities regardless of the underlying logit magnitudes",
        "Overparameterized models perfectly fit training data, then logit magnitudes keep growing and concentrating softmax mass — finite data cannot prevent this",
        "Cross-entropy loss is an asymmetric scoring rule that penalizes underconfident predictions more heavily than overconfident ones",
        "Batch normalization rescales activations in a way that systematically pushes softmax outputs above their calibrated probabilities"
      ],
      correct: 1,
      explanation: "Cross-entropy is a proper scoring rule, so at the infinite-data optimum the model would be calibrated. But in practice, overparameterized networks fit training data perfectly and then continue pushing logits apart (increasing $\\|z\\|$), which concentrates the softmax output toward a point mass. Without enough held-out data during training to penalize this, overconfidence grows unchecked. This is exacerbated by residual connections (which allow logit magnitudes to grow freely) and other architectural choices."
    },
    {
      type: "info",
      title: "Reliability Diagrams: Visualizing Calibration",
      content: "A **reliability diagram** is the standard tool for visually assessing calibration. It plots predicted confidence against observed accuracy, bin by bin.\n\n**Construction**: Partition all predictions into $M$ bins by their predicted confidence (e.g., $[0, 0.1), [0.1, 0.2), \\ldots, [0.9, 1.0]$). For each bin $B_m$, compute:\n- **Average predicted confidence**: $\\bar{p}_m = \\frac{1}{|B_m|} \\sum_{i \\in B_m} \\hat{p}_i$\n- **Observed accuracy**: $\\bar{a}_m = \\frac{1}{|B_m|} \\sum_{i \\in B_m} \\mathbf{1}[\\hat{y}_i = y_i]$\n\nPlot $(\\bar{p}_m, \\bar{a}_m)$ for each bin. A perfectly calibrated model produces points along the **diagonal** $y = x$.\n\n**Interpreting deviations**:\n- Points **below** the diagonal: the model is overconfident in that range (confidence exceeds accuracy).\n- Points **above** the diagonal: the model is underconfident (accuracy exceeds confidence).\n\nFor modern deep networks, the reliability diagram typically shows points well below the diagonal at high confidence levels (0.8–1.0), confirming systematic overconfidence. The gap between the diagonal and the curve directly shows how much the model's stated probabilities deviate from reality."
    },
    {
      type: "mc",
      question: "On a reliability diagram, a model's curve lies consistently below the diagonal for all confidence bins above 0.5. Which statement best describes this model?",
      options: [
        "The model is overconfident for high-confidence predictions — it claims more certainty than its accuracy warrants",
        "The model has perfect calibration in the low-confidence regime but miscalibrated high-confidence predictions",
        "The model is underconfident across the board — it achieves higher accuracy than its stated confidence suggests",
        "The model's classification accuracy is poor because its curve is far from the diagonal"
      ],
      correct: 0,
      explanation: "Points below the diagonal mean observed accuracy is lower than predicted confidence — the model is overconfident. Being below the diagonal specifically for bins above 0.5 means the model's high-confidence predictions are not as reliable as it claims. This is the typical failure mode for deep networks: they assign high softmax probabilities (0.9+) to many predictions, but a substantial fraction of those are incorrect. Note that the distance from the diagonal measures calibration error, not classification accuracy — a model can be highly accurate overall but still poorly calibrated."
    },
    {
      type: "info",
      title: "Expected Calibration Error (ECE)",
      content: "The **Expected Calibration Error** (ECE) summarizes the reliability diagram into a single scalar. It is the weighted average of per-bin calibration gaps:\n\n$$\\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} \\left| \\bar{a}_m - \\bar{p}_m \\right|$$\n\nwhere $|B_m|$ is the number of samples in bin $m$, $N$ is the total sample count, $\\bar{a}_m$ is the observed accuracy in bin $m$, and $\\bar{p}_m$ is the average predicted confidence in bin $m$.\n\nECE = 0 means perfect calibration. Values around 0.01–0.03 are considered well-calibrated for practical purposes. Modern uncalibrated deep networks often show ECE in the range 0.05–0.15.\n\n**Limitations of ECE**:\n- **Bin sensitivity**: The number of bins $M$ (typically 10 or 15) affects the value. More bins capture finer-grained miscalibration but suffer from small sample sizes per bin.\n- **Top-label only**: Standard ECE only evaluates the confidence assigned to the predicted class, ignoring how probability is distributed among the remaining classes.\n- **Debiased ECE**: The finite-sample estimate of ECE is positively biased (it overestimates miscalibration). Kumar et al. (2019) proposed a bias-corrected variant.\n\nDespite these issues, ECE remains the most widely reported calibration metric because of its simplicity and interpretability."
    },
    {
      type: "mc",
      question: "A 10-bin ECE computation yields the following: bins 0.0–0.8 each contain 5% of the data with near-zero calibration gap, while bins 0.8–0.9 and 0.9–1.0 each contain 30% of the data with calibration gaps of 0.10 and 0.20 respectively. What is the approximate ECE?",
      options: [
        "$(0.10 + 0.20) / 10 = 0.03$ — the average gap across all 10 bins",
        "$0.30 \\times 0.10 + 0.30 \\times 0.20 = 0.09$ — the sample-weighted average of the non-negligible gaps",
        "$\\max(0.10, 0.20) = 0.20$ — the maximum calibration error across bins",
        "$(0.10 + 0.20) / 2 = 0.15$ — the unweighted average of the two miscalibrated bins"
      ],
      correct: 1,
      explanation: "ECE weights each bin's calibration gap by the fraction of samples in that bin. The 8 low-confidence bins contribute roughly $8 \\times 0.05 \\times 0 \\approx 0$. The two high-confidence bins contribute $0.30 \\times 0.10 + 0.30 \\times 0.20 = 0.03 + 0.06 = 0.09$. This illustrates a key point: ECE is dominated by bins where most predictions fall. Since overconfident models tend to cluster predictions in the 0.8–1.0 range, those high-confidence bins drive the ECE — exactly where miscalibration is most concerning."
    },
    {
      type: "info",
      title: "Temperature Scaling and Platt Scaling",
      content: "**Temperature scaling** (Guo et al., 2017) is a remarkably simple calibration fix: after training is complete, learn a single scalar $T > 0$ that rescales all logits:\n\n$$Q_{\\text{cal}}(y \\mid x) = \\text{softmax}(z(x) / T)$$\n\nThe temperature $T$ is chosen to minimize the negative log-likelihood on a **held-out validation set**. Temperature scaling is a **monotonic transformation** — it preserves the $\\arg\\max$ (accuracy unchanged) while adjusting confidence. For overconfident models, the optimal $T > 1$: logits are compressed toward zero, spreading probability mass and raising entropy.\n\n**Platt scaling** (Platt, 1999) is more expressive, fitting $Q_{\\text{cal}}(y = 1 \\mid x) = \\sigma(a \\cdot z(x) + b)$ with two parameters (scale $a$ and bias $b$). Temperature scaling is the special case $a = 1/T, b = 0$ — it assumes only the magnitude needs fixing, not the offset.\n\n**Multi-class extensions**:\n- Temperature scaling: divide all logits by $T$. One parameter regardless of class count.\n- Platt scaling extends to **matrix scaling** ($W \\cdot z + b$) or **vector scaling** (diagonal $W$), learning per-class parameters. More expressive but prone to overfitting on small validation sets.\n\n**In practice**, temperature scaling is almost always preferred. Guo et al. (2017) showed it matches or beats richer methods while being far less prone to overfitting — the one-parameter constraint acts as a strong regularizer, and overconfidence is typically a global phenomenon (the same $T$ works across all classes)."
    },
    {
      type: "mc",
      question: "Post-hoc temperature scaling with $T > 1$ does what to a trained model's predictions?",
      options: [
        "Makes predictions less confident (higher entropy) without changing which class is predicted",
        "Retrains the weights of the final layer to directly improve the model's calibration and accuracy jointly",
        "Changes which class is predicted (different $\\arg\\max$) to improve classification accuracy",
        "Makes predictions more confident (lower entropy) to correct the model's underconfidence problem"
      ],
      correct: 0,
      explanation: "Dividing logits by $T > 1$ compresses them toward zero, making the softmax output more uniform — higher entropy, lower confidence. Since dividing all logits by the same positive constant preserves their ordering, the $\\arg\\max$ is unchanged: accuracy is identical. Only the *confidence* of the prediction changes. This is why temperature scaling is so appealing: it fixes calibration (confidence matches accuracy) with zero cost to accuracy and only one parameter to tune."
    },
    {
      type: "mc",
      question: "A practitioner has a 1000-class image classifier and a validation set of 5000 samples. They want to calibrate it post-hoc. Which approach is most appropriate?",
      options: [
        "Matrix scaling ($W \\cdot z + b$) with a $1000 \\times 1000$ weight matrix — this provides the most flexible calibration",
        "Vector scaling with 1000 per-class temperature parameters — each class needs its own calibration",
        "Platt scaling with 2000 parameters (scale and bias per class) — the binary method extended to multi-class",
        "Temperature scaling with a single parameter $T$ — it is robust to overfitting and typically matches richer methods"
      ],
      correct: 3,
      explanation: "With 5000 validation samples and 1000 classes, there are only 5 samples per class on average. Matrix scaling would have $10^6$ parameters — massive overfitting is guaranteed. Vector scaling (1000 params) and extended Platt scaling (2000 params) are also problematic with so few samples per class. Temperature scaling uses exactly one parameter, making overfitting virtually impossible. Empirically, it performs as well as more complex methods because overconfidence in deep networks is a global phenomenon: logit magnitudes are systematically too large across all classes, and a single divisor corrects this."
    },
    {
      type: "info",
      title: "Calibration in LLMs: Token-Level vs Sequence-Level",
      content: "Calibration takes on **two distinct meanings** in large language models:\n\n**Token-level calibration**: Does the model's softmax distribution over the next token match the true conditional distribution? When the model assigns probability 0.3 to a particular next token, is that token actually correct 30% of the time in that context? This is the direct analog of classification calibration, applied at each autoregressive step.\n\nModern LLMs are often **reasonably well-calibrated at the token level**, partly because they are trained on massive datasets that reduce the finite-data miscalibration problem. However, they can still be overconfident on rare tokens and underconfident on common ones.\n\n**Sequence-level calibration**: When an LLM says \"I am 85% confident the answer is Paris,\" does it get such questions right 85% of the time? This is a much harder problem because:\n\n1. **Probability collapse**: Token-level probabilities multiply across a sequence. Even well-calibrated token distributions can produce poorly calibrated sequence probabilities because small per-token errors compound.\n2. **Verbalized vs implicit confidence**: The model's *stated* confidence (in natural language) may not match its *internal* confidence (the actual probability assigned to the generated sequence).\n3. **No direct training signal**: LLMs are trained to predict next tokens, not to output calibrated confidence statements about factual accuracy.\n\nThis gap between token-level and sequence-level calibration is an active research area, with methods like verbalized confidence elicitation and consistency-based calibration being explored."
    },
    {
      type: "mc",
      question: "An LLM assigns well-calibrated token-level probabilities (each next-token softmax is calibrated). A user asks it a factual question and it generates a 20-token answer with high per-token confidence. What can we conclude about the sequence-level calibration of this answer?",
      options: [
        "The sequence is also well-calibrated — token-level calibration guarantees sequence-level calibration by the chain rule of probability",
        "The sequence is necessarily overconfident — multiplying 20 well-calibrated token probabilities always produces sequence probabilities that are too high",
        "Token-level calibration does not guarantee sequence-level calibration — small per-token errors compound and the model lacks a training signal for factual confidence",
        "The sequence is necessarily underconfident — the product of 20 probabilities less than 1 will always be much lower than the true probability of the full sequence"
      ],
      correct: 2,
      explanation: "Token-level calibration does not transfer to sequence-level calibration for multiple reasons. First, small per-token calibration errors compound multiplicatively across 20 tokens. Second, even if each $P(w_t \\mid w_{<t})$ is calibrated, the model's probability of generating the *correct answer* depends on whether it generates the right sequence of tokens, which involves a combinatorial space of possible phrasings. Third, the model's verbalized confidence (\"I'm 85% sure\") is generated by the same next-token process, not by a separate calibrated confidence mechanism. These issues mean token-level and sequence-level calibration are related but distinct properties."
    },
    {
      type: "info",
      title: "The Entropy-Accuracy Tradeoff",
      content: "There is a fundamental tension between entropy and accuracy that underlies all calibration methods:\n\n**Maximum entropy** (uniform distribution): worst possible accuracy ($1/K$ for $K$ classes) but maximum coverage — every class gets equal probability. The model \"hedges\" completely, never committing to any prediction. Calibration is trivially perfect (it says $1/K$ confident and is right $1/K$ of the time).\n\n**Minimum entropy** (deterministic): best accuracy on seen patterns (always picks the most likely class with probability 1) but no uncertainty quantification and poor generalization. The model is maximally brittle — it provides no signal about when it might be wrong.\n\nGood models navigate between these extremes:\n- **High confidence** where training data is dense and patterns are clear.\n- **High entropy** where data is sparse, ambiguous, or out-of-distribution.\n\nLabel smoothing, temperature scaling, and the KL penalty in RLHF ($\\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$) all push models away from the minimum-entropy extreme. The KL penalty prevents the policy from collapsing to a low-entropy distribution that always produces the same high-reward response, maintaining the diversity of the reference model. All three are fundamentally **calibration tools** — they trade off peak confidence for better-calibrated uncertainty."
    },
    {
      type: "mc",
      question: "In RLHF, the KL penalty $\\beta \\cdot \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ prevents the policy from becoming too low-entropy. This is conceptually most similar to which training technique?",
      options: [
        "Dropout — both inject noise to prevent the model from concentrating on a narrow set of features",
        "Gradient clipping — both bound the magnitude of updates to prevent sharp distributional shifts",
        "Label smoothing — both prevent overconfidence by maintaining minimum entropy in the output distribution",
        "Weight decay — both add a penalty term to the loss that regularizes the model toward simpler solutions"
      ],
      correct: 2,
      explanation: "Label smoothing and the RLHF KL penalty are both **entropy regularizers** that prevent distributional collapse:\n\n- **Label smoothing** mixes the target with a uniform distribution, ensuring the model's output retains nonzero entropy. The optimal output is confident but not infinitely so.\n- **RLHF KL penalty** penalizes deviation from $\\pi_{\\text{ref}}$, preventing the policy from collapsing to a low-entropy distribution that always generates the single highest-reward response.\n\nBoth mechanisms maintain diversity/uncertainty by penalizing overconfidence, and both are controlled by a scalar hyperparameter ($\\alpha$ or $\\beta$) that trades off task performance against distributional smoothness. Weight decay regularizes *parameters*; dropout regularizes *features*; these two regularize *output distributions* — a more targeted form of regularization directly relevant to calibration."
    }
  ]
};
