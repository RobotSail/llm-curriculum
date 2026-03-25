// Focused learning module for Cross-Entropy.
// Covers cross-entropy as a loss function for language models and classification.
// Assumes the student already knows entropy and KL divergence basics.

export const crossEntropyLearning = {
  id: "0.2-cross-entropy-learning-easy",
  sectionId: "0.2",
  title: "Cross-Entropy Loss for Language Models",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "From Entropy to Cross-Entropy",
      content: "Recall that **entropy** $H(P) = -\\sum_x P(x) \\log P(x)$ measures the average surprise under a distribution $P$. It tells you the minimum number of bits (or nats) needed to encode samples from $P$.\n\nBut in practice, we rarely know the true distribution $P$. Instead, we have a **model** $Q$ that approximates $P$. When we use $Q$ to encode data that actually follows $P$, we pay more than the optimal $H(P)$ bits. The total cost is the **cross-entropy**:\n\n$$H(P, Q) = -\\sum_x P(x) \\log Q(x) = \\mathbb{E}_P[-\\log Q(X)]$$\n\nCross-entropy is always at least as large as entropy: $H(P, Q) \\geq H(P)$, with equality only when $Q = P$. The gap between them is exactly the KL divergence: $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$."
    },
    {
      type: "mc",
      question: "If $P$ is the true next-token distribution and $Q$ is your language model, what does the gap $H(P, Q) - H(P)$ equal?",
      options: [
        "The mutual information $I(P; Q)$",
        "The Jensen-Shannon divergence $\\text{JS}(P \\| Q)$",
        "The reverse KL divergence $\\text{KL}(Q \\| P)$",
        "The forward KL divergence $\\text{KL}(P \\| Q)$"
      ],
      correct: 3,
      explanation: "By definition: $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$, so $H(P, Q) - H(P) = \\text{KL}(P \\| Q)$. This is the *forward* KL — the penalty your model pays for deviating from the true distribution. It is always $\\geq 0$, with equality iff $Q = P$. This decomposition is why cross-entropy loss is a sound training objective: minimizing it drives $Q$ toward $P$."
    },
    {
      type: "info",
      title: "Cross-Entropy as Negative Log-Likelihood",
      content: "In language modeling, the training loss over a sequence $w_1, w_2, \\ldots, w_T$ is:\n\n$$\\mathcal{L} = -\\frac{1}{T} \\sum_{t=1}^{T} \\log Q_\\theta(w_t \\mid w_{<t})$$\n\nThis is the **negative log-likelihood (NLL)** of the data under the model — and it is exactly the cross-entropy $H(P_{\\text{data}}, Q_\\theta)$ estimated from the training corpus.\n\nThe connection is direct: each training token $w_t$ is a sample from the empirical data distribution $P_{\\text{data}}$, and $-\\log Q_\\theta(w_t \\mid w_{<t})$ is the surprise the model experiences at that token. Averaging over all tokens gives the expected surprise — which is cross-entropy.\n\nSince $H(P_{\\text{data}})$ is a constant that does not depend on model parameters $\\theta$, **minimizing cross-entropy is equivalent to minimizing KL divergence** between the data distribution and the model, which is equivalent to **maximizing the likelihood** of the training data."
    },
    {
      type: "mc",
      question: "A researcher says \"we trained the model by maximizing log-likelihood.\" A colleague says \"we trained by minimizing cross-entropy loss.\" Which statement is correct?",
      options: [
        "Both are describing the same optimization, since minimizing $-\\frac{1}{T}\\sum_t \\log Q_\\theta(w_t)$ is simultaneously minimizing cross-entropy and maximizing likelihood",
        "Only the first — log-likelihood and cross-entropy are fundamentally different objectives with different optima",
        "Neither — modern LLMs use contrastive losses, not cross-entropy or likelihood",
        "Only the second — maximizing likelihood would cause the model to overfit, unlike cross-entropy minimization"
      ],
      correct: 0,
      explanation: "Minimizing $-\\sum_t \\log Q_\\theta(w_t \\mid w_{<t})$ is exactly minimizing cross-entropy. Equivalently, maximizing $\\sum_t \\log Q_\\theta(w_t \\mid w_{<t})$ is maximizing log-likelihood. These are the same objective with a sign flip. The distinction between \"minimize NLL\" and \"minimize cross-entropy\" is purely terminological — both describe the standard LM training loss."
    },
    {
      type: "info",
      title: "Cross-Entropy Is Not Symmetric",
      content: "A critical property: cross-entropy is **not symmetric**. That is, $H(P, Q) \\neq H(Q, P)$ in general.\n\nWhy? Because the two expressions differ in which distribution does the weighting and which appears inside the log:\n\n$$H(P, Q) = -\\sum_x P(x) \\log Q(x) \\qquad H(Q, P) = -\\sum_x Q(x) \\log P(x)$$\n\nUsing the decomposition: $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$ while $H(Q, P) = H(Q) + \\text{KL}(Q \\| P)$. Since both the entropy terms and the KL terms generally differ, the two cross-entropies are unequal.\n\nIn LM training, the convention is $H(P_{\\text{data}}, Q_\\theta)$ — we sample from $P_{\\text{data}}$ (the training corpus) and evaluate the model $Q_\\theta$'s log-probabilities. This choice matters: it means the training loss inherits the **mode-covering** behavior of forward KL."
    },
    {
      type: "mc",
      question: "Is cross-entropy symmetric? That is, does $H(P, Q) = H(Q, P)$ in general?",
      options: [
        "Yes — since $\\sum_x P(x) \\log Q(x) = \\sum_x Q(x) \\log P(x)$ by commutativity of multiplication inside the log",
        "It depends on whether $P$ and $Q$ have the same support, since mismatched supports break the symmetry",
        "Yes — cross-entropy is a proper distance metric and all distance metrics satisfy the symmetry axiom",
        "No — $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$ while $H(Q, P) = H(Q) + \\text{KL}(Q \\| P)$, and these differ when $P \\neq Q$"
      ],
      correct: 3,
      explanation: "Cross-entropy is NOT symmetric. $H(P, Q) = H(P) + \\text{KL}(P \\| Q)$ and $H(Q, P) = H(Q) + \\text{KL}(Q \\| P)$. Both the entropy terms and KL terms differ between $P$ and $Q$ in general, so the two cross-entropies are unequal. Cross-entropy also fails the other metric axioms (triangle inequality, identity of indiscernibles) — it is not a distance metric at all."
    },
    {
      type: "info",
      title: "Cross-Entropy for Multi-Class Classification",
      content: "Cross-entropy is not limited to language modeling. For **multi-class classification** with $K$ classes, the label for sample $i$ is a one-hot vector $y_i$ and the model outputs a probability vector $\\hat{y}_i$ (typically via softmax). The cross-entropy loss is:\n\n$$\\mathcal{L} = -\\sum_{k=1}^{K} y_{i,k} \\log \\hat{y}_{i,k}$$\n\nSince $y_i$ is one-hot with $y_{i,c} = 1$ for the correct class $c$, this simplifies to:\n\n$$\\mathcal{L} = -\\log \\hat{y}_{i,c}$$\n\nThis is just the negative log-probability the model assigns to the correct class — identical in form to the per-token LM loss. Language modeling is literally next-token classification over a vocabulary of $K$ tokens, with cross-entropy as the loss at each position."
    },
    {
      type: "mc",
      question: "You have two language models. Model X assigns probability 0.8 to the correct next token on average. Model Y assigns probability 0.2. Roughly how do their per-token cross-entropies compare?",
      options: [
        "They have comparable cross-entropy because both models are far from perfect and log is slowly varying near 0",
        "Model X's cross-entropy is about $-\\ln(0.8) \\approx 0.22$ nats vs Model Y's $-\\ln(0.2) \\approx 1.61$ nats — a ratio of exactly 4x",
        "Model X's cross-entropy is about $-\\ln(0.8) \\approx 0.22$ nats vs Model Y's $-\\ln(0.2) \\approx 1.61$ nats — a ratio of roughly 7x",
        "Model Y's cross-entropy is lower because assigning 0.2 spreads probability more evenly across the vocabulary"
      ],
      correct: 2,
      explanation: "Cross-entropy involves $-\\log Q(w_{\\text{correct}})$. For Model X: $-\\ln(0.8) \\approx 0.22$ nats. For Model Y: $-\\ln(0.2) \\approx 1.61$ nats. The ratio is about $1.61/0.22 \\approx 7\\text{x}$, not 4x, because cross-entropy is logarithmic in probability — small changes in probability near 1.0 matter less than small changes near 0. This is why the last few percentage points of accuracy are disproportionately hard to achieve in terms of loss reduction."
    },
    {
      type: "info",
      title: "Why Cross-Entropy Beats MSE for Classification",
      content: "Why not use mean squared error (MSE) for classification? Consider a model predicting $\\hat{y} = \\text{softmax}(z)$. The gradients tell the story.\n\nWith **MSE loss** $(y - \\hat{y})^2$, the gradient with respect to the logit $z_c$ involves the derivative of the softmax, which includes terms like $\\hat{y}_c(1 - \\hat{y}_c)$. When the model's prediction is confidently wrong ($\\hat{y}_c \\approx 0$ for the correct class), this derivative is near zero — the gradient **vanishes** precisely when the model most needs to learn.\n\nWith **cross-entropy loss** $-\\log \\hat{y}_c$, the gradient with respect to the logits is remarkably clean:\n\n$$\\frac{\\partial \\mathcal{L}}{\\partial z_k} = \\hat{y}_k - y_k$$\n\nThis is simply **prediction minus target**. When the model is confidently wrong ($\\hat{y}_c \\approx 0$), the gradient magnitude is close to 1 — large and corrective. There is no vanishing gradient problem. This clean gradient is a major practical reason cross-entropy dominates classification tasks."
    },
    {
      type: "mc",
      question: "A classifier outputs softmax probabilities and uses cross-entropy loss. For a sample with true class $c$, the gradient of the loss with respect to logit $z_k$ is:",
      options: [
        "$-y_k \\cdot \\hat{y}_k \\cdot (1 - \\hat{y}_k)$, which vanishes when $\\hat{y}_k$ is near 0 or 1",
        "$\\hat{y}_k - y_k$, which is large when the prediction is wrong and zero when correct",
        "$-\\frac{y_k}{\\hat{y}_k^2}$, which explodes when the model assigns low probability to the correct class",
        "$2(\\hat{y}_k - y_k) \\cdot \\hat{y}_k(1 - \\hat{y}_k)$, which is the standard MSE-softmax gradient"
      ],
      correct: 1,
      explanation: "The gradient of cross-entropy loss with softmax outputs is $\\hat{y}_k - y_k$. For the correct class ($y_c = 1$), this is $\\hat{y}_c - 1$, which has magnitude close to 1 when the model is wrong ($\\hat{y}_c \\approx 0$). For incorrect classes ($y_k = 0$), the gradient is simply $\\hat{y}_k$. This clean, well-behaved gradient is a key advantage of cross-entropy over MSE for classification."
    },
    {
      type: "info",
      title: "Binary Cross-Entropy",
      content: "For **binary classification** (two classes), the cross-entropy loss simplifies to the **binary cross-entropy** (BCE), also called log loss:\n\n$$\\mathcal{L} = -\\big[y \\log \\hat{y} + (1 - y) \\log(1 - \\hat{y})\\big]$$\n\nwhere $y \\in \\{0, 1\\}$ is the true label and $\\hat{y} \\in (0, 1)$ is the predicted probability (typically from a sigmoid: $\\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}$).\n\nWhen $y = 1$, only the first term survives: $\\mathcal{L} = -\\log \\hat{y}$. When $y = 0$, only the second: $\\mathcal{L} = -\\log(1 - \\hat{y})$. In both cases, the loss is the negative log-probability assigned to the correct outcome.\n\nThe gradient with respect to the logit $z$ is equally clean: $\\frac{\\partial \\mathcal{L}}{\\partial z} = \\hat{y} - y = \\sigma(z) - y$. This is the binary analog of the softmax gradient — prediction minus target."
    },
    {
      type: "mc",
      question: "A binary classifier outputs $\\hat{y} = \\sigma(z) = 0.95$ for a sample with true label $y = 0$. What is the binary cross-entropy loss for this sample?",
      options: [
        "$-\\log(0.05) \\approx 3.0$, since the model assigns only 5% probability to the correct class ($y=0$)",
        "$-\\log(0.95) \\approx 0.05$, since the model is confident and confidence is penalized logarithmically",
        "$(0.95 - 0)^2 = 0.9025$, which is the squared error between prediction and target",
        "$-0.95 \\log(0.95) \\approx 0.049$, since cross-entropy weights the log by the predicted probability"
      ],
      correct: 0,
      explanation: "Since $y = 0$, the correct class probability is $1 - \\hat{y} = 1 - 0.95 = 0.05$. The BCE loss is $-\\log(1 - \\hat{y}) = -\\log(0.05) \\approx 3.0$ nats. The model is confidently wrong (assigns 95% to the wrong class), so the loss is large. This is how cross-entropy penalizes confident incorrect predictions much more harshly than uncertain ones."
    },
    {
      type: "info",
      title: "Cross-Entropy Gradients Drive Learning",
      content: "Let us summarize why cross-entropy with softmax/sigmoid activations produces such effective training dynamics.\n\nThe gradient $\\frac{\\partial \\mathcal{L}}{\\partial z_k} = \\hat{y}_k - y_k$ has several desirable properties:\n\n**1. No vanishing gradients from the activation.** Unlike MSE paired with sigmoid/softmax, the softmax derivative cancels terms in the cross-entropy gradient, leaving a simple difference. The model receives strong learning signals even when predictions are far from the target.\n\n**2. Linear error signal.** The gradient is proportional to the size of the error. A prediction that is 0.9 off from the target produces a gradient 9x larger than one that is 0.1 off.\n\n**3. Correct direction.** For the true class, $\\hat{y}_c - 1 < 0$ pushes the logit $z_c$ upward. For incorrect classes, $\\hat{y}_k - 0 > 0$ pushes their logits downward. The gradient automatically increases the correct class probability and decreases the others.\n\nThis is not a coincidence — it is a consequence of cross-entropy being the **natural loss function** for the exponential family (softmax is a member). The pairing of cross-entropy loss with softmax output is a canonical example of matching a loss to an activation for clean optimization."
    },
    {
      type: "mc",
      question: "A model outputs softmax probabilities $\\hat{y} = (0.1, 0.6, 0.05, 0.25)$ over 4 classes, and the true class is $k=0$. What is the cross-entropy gradient $\\frac{\\partial \\mathcal{L}}{\\partial z_k}$ for each logit?",
      options: [
        "$(-10, 1.67, 20, 4)$ — the gradient is $\\frac{-y_k}{\\hat{y}_k}$, which explodes for small predictions",
        "$(-0.81, 0.36, -0.9025, -0.5625)$ — the gradient involves squaring each term",
        "$(-0.9, 0.6, 0.05, 0.25)$ — the gradient is $\\hat{y}_k - y_k$ for each class",
        "$(0.9, -0.6, -0.05, -0.25)$ — the gradient is $y_k - \\hat{y}_k$ for each class"
      ],
      correct: 2,
      explanation: "The gradient is $\\hat{y}_k - y_k$. With one-hot $y = (1, 0, 0, 0)$: for $k=0$: $0.1 - 1 = -0.9$; for $k=1$: $0.6 - 0 = 0.6$; for $k=2$: $0.05 - 0 = 0.05$; for $k=3$: $0.25 - 0 = 0.25$. The negative gradient at $k=0$ pushes that logit up (increasing correct class probability), while positive gradients at other classes push their logits down."
    }
  ]
};
