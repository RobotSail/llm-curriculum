// Assessment: Information Theory in Practice — LLM Applications
// Section 0.2: Diagnostic test — applying probability & info theory to LLM training, RLHF, generation
// Pure assessment to gauge ability to connect theory to practice

export const appliedInfoTheoryAssessment = {
  id: "0.2-assess-applied",
  sectionId: "0.2",
  title: "Assessment: Information Theory in LLM Practice",
  difficulty: "hard",
  estimatedMinutes: 14,
  assessmentOnly: true,
  steps: [
    {
      type: "info",
      title: "Diagnostic: Information Theory in LLM Practice",
      content: "This is a **diagnostic assessment** testing your ability to connect probability and information theory to concrete LLM phenomena.\n\nEach question describes a real situation in LLM training, fine-tuning, or generation and asks you to identify the underlying theoretical principle.\n\nIf you score below 70%, work through the other assessments first, then return to this one — it integrates everything."
    },
    {
      type: "mc",
      question: "During LLM pre-training, the loss is $-\\frac{1}{T} \\sum_{t=1}^{T} \\log P_\\theta(w_t \\mid w_{<t})$. As training progresses, this loss decreases but eventually plateaus. The **irreducible** component of this loss corresponds to:",
      options: [
        "The numerical precision of float16 arithmetic",
        "The **entropy of natural language** $H(W_t \\mid W_{<t})$ — the inherent unpredictability that no model can eliminate because language itself is stochastic",
        "The size of the vocabulary",
        "The number of parameters in the model"
      ],
      correct: 1,
      explanation: "Cross-entropy loss $= H(P) + \\text{KL}(P \\| Q_\\theta)$. Even a perfect model ($Q = P$) achieves loss $H(P)$, the conditional entropy of the next token. This is the irreducible noise in language — given perfect context, many continuations are still plausible. The gap between the model's loss and $H(P)$ is the KL divergence, which measures model imperfection. Scaling laws describe how this KL component decreases with compute."
    },
    {
      type: "mc",
      question: "When you increase the **temperature** $\\tau$ in sampling ($P(w) \\propto \\exp(z_w / \\tau)$ where $z_w$ are logits), you are:",
      options: [
        "Changing the model's parameters",
        "Interpolating between the model's learned distribution ($\\tau = 1$) and uniform ($\\tau \\to \\infty$), which increases entropy and diversity. At $\\tau \\to 0$, you get argmax (greedy) decoding",
        "Applying dropout at inference time",
        "Changing the loss function"
      ],
      correct: 1,
      explanation: "Dividing logits by $\\tau > 1$ flattens the distribution (higher entropy, more random). Dividing by $\\tau < 1$ sharpens it (lower entropy, more deterministic). At $\\tau \\to 0$, all mass concentrates on the highest-logit token (argmax). At $\\tau \\to \\infty$, the distribution approaches uniform over the vocabulary. Temperature is a post-hoc entropy control that doesn't change the model — it's a monotonic transformation of the entropy of the output distribution."
    },
    {
      type: "mc",
      question: "In RLHF, increasing the KL penalty coefficient $\\beta$ in $\\max_\\pi \\mathbb{E}[r(x)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ causes the optimal policy to:",
      options: [
        "Maximize reward regardless of the reference policy",
        "Stay closer to $\\pi_{\\text{ref}}$, trading off reward for distributional similarity — in the limit $\\beta \\to \\infty$, $\\pi^* = \\pi_{\\text{ref}}$",
        "Become a uniform distribution",
        "Increase its entropy independently of $\\pi_{\\text{ref}}$"
      ],
      correct: 1,
      explanation: "The optimal policy is $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. As $\\beta \\to \\infty$, $\\exp(r/\\beta) \\to 1$, so $\\pi^* \\to \\pi_{\\text{ref}}$. As $\\beta \\to 0$, the reward dominates, and the policy concentrates on the highest-reward outputs regardless of naturalness. $\\beta$ controls the **rate-distortion trade-off**: how much \"distortion\" (reward sacrifice) you accept to stay within a certain \"rate\" (KL budget) from the reference."
    },
    {
      type: "mc",
      question: "A model trained on code achieves cross-entropy 0.5 nats/token on Python but 1.8 nats/token on Haskell. Assuming equal vocabulary, the **perplexity ratio** tells us:",
      options: [
        "Python is a better language than Haskell",
        "The model has learned far more predictable structure in Python — perplexity is $e^{0.5} \\approx 1.65$ for Python vs $e^{1.8} \\approx 6.05$ for Haskell, meaning the model is ~3.7× more uncertain per token on Haskell",
        "The Haskell tokenizer is broken",
        "The Python test set is smaller"
      ],
      correct: 1,
      explanation: "Perplexity $= e^{H(P,Q)}$. Lower cross-entropy → lower perplexity → more predictable. The ratio $e^{1.8}/e^{0.5} = e^{1.3} \\approx 3.67$ means the model's uncertainty per token is 3.7× higher on Haskell. This could reflect: less Haskell in training data, Haskell's inherently different structure, or poor tokenization. The information-theoretic view lets you decompose the loss into data entropy + model deficiency."
    },
    {
      type: "mc",
      question: "**Reward hacking** in RLHF occurs when the policy finds outputs that score high reward but are low-quality. From an information-theoretic perspective, this happens because:",
      options: [
        "The reward model has high entropy",
        "The reward model is a learned approximation — as $\\pi$ moves far from $\\pi_{\\text{ref}}$ (high KL), it enters out-of-distribution regions where the reward model's predictions are unreliable, effectively exploiting the reward model's epistemic uncertainty",
        "The KL penalty is too large",
        "The vocabulary is too small"
      ],
      correct: 1,
      explanation: "The reward model $r_\\phi$ was trained on data from $\\pi_{\\text{ref}}$. When $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ is large, $\\pi$ generates text unlike anything the reward model saw during training. The reward model's predictions become meaningless — it may assign high scores to adversarial outputs. This is why the KL penalty is crucial: it keeps $\\pi$ in the region where $r_\\phi$ is calibrated. This is a distributional shift problem quantified by KL divergence."
    },
    {
      type: "mc",
      question: "During fine-tuning, you notice the model's **token-level entropy** $H(P_\\theta(\\cdot \\mid x))$ dropping rapidly. The attention entropy (entropy of attention weights) is also collapsing. This likely indicates:",
      options: [
        "The model is learning perfectly",
        "The model is **overfitting and losing diversity** — it's becoming overconfident on training patterns, potentially memorizing rather than generalizing. Entropy collapse in attention means it's attending to fewer positions, losing contextual flexibility",
        "The learning rate is too low",
        "The model needs a larger vocabulary"
      ],
      correct: 1,
      explanation: "Entropy collapse is a warning sign: the model assigns near-deterministic predictions and attends to very few positions. Low output entropy means the model \"thinks\" it's very certain — but on new inputs, this overconfidence leads to poor calibration and repetitive/degenerate outputs. Monitoring entropy during training is an information-theoretic diagnostic: healthy training should reduce entropy gradually (learning structure) without collapsing it (losing flexibility)."
    },
    {
      type: "mc",
      question: "The **bits-per-byte** (BPB) metric normalizes perplexity across different tokenizers. If model A has perplexity 15.2 with a BPE tokenizer averaging 3.8 bytes/token, and model B has perplexity 8.1 with a character-level tokenizer (1 byte/token), which is better?",
      options: [
        "Model B — it has lower perplexity",
        "Must compare BPB: model A's BPB $= \\frac{\\log_2(15.2)}{3.8} \\approx 1.03$, model B's BPB $= \\frac{\\log_2(8.1)}{1.0} \\approx 3.02$. Model A is much better despite higher perplexity — it compresses information more efficiently per byte",
        "Model A — it has a better tokenizer",
        "They are equivalent"
      ],
      correct: 1,
      explanation: "BPB = (bits per token) / (bytes per token) = $\\log_2(\\text{PPL}) / \\text{avg\\_bytes\\_per\\_token}$. Model A: $\\log_2(15.2)/3.8 \\approx 1.03$ BPB. Model B: $\\log_2(8.1)/1.0 \\approx 3.02$ BPB. Model A achieves much better compression per byte of text. Raw perplexity is misleading across tokenizers because a character-level model predicts many more (easier) tokens. BPB is the fair comparison metric."
    },
    {
      type: "mc",
      question: "In the **information bottleneck** framework, a representation $Z$ of input $X$ is optimized to maximize $I(Z; Y)$ (task-relevant information) while minimizing $I(Z; X)$ (total information retained). In the context of LLM representations, this means:",
      options: [
        "Layers should memorize all input details",
        "Good representations should **compress away irrelevant details of the input** while retaining information needed for prediction — the $\\beta$ parameter controls this trade-off, analogous to the KL penalty in RLHF",
        "All layers should have the same mutual information with the input",
        "The output should be independent of the input"
      ],
      correct: 1,
      explanation: "The IB objective $\\max_{p(z|x)} I(Z; Y) - \\beta I(Z; X)$ formalizes the compression/prediction trade-off. Small $\\beta$ keeps all information (overfitting); large $\\beta$ compresses aggressively (underfitting). This framework connects to: dropout (random compression), bottleneck layers (architectural compression), and the KL penalty in RLHF (behavioral compression). The parallel to $\\max \\mathbb{E}[r] - \\beta \\text{KL}$ is exact in structure."
    },
    {
      type: "mc",
      question: "When computing $\\log P(y \\mid x)$ for a long sequence $y = (y_1, \\dots, y_T)$, numerical issues arise because the log-probability is a sum of $T$ negative terms that can become very negative. The standard solution is:",
      options: [
        "Clipping the log-probabilities to a minimum value",
        "Working in **log-space throughout** — using the log-sum-exp trick for normalization and accumulating log-probabilities additively, which is numerically stable and avoids underflow from multiplying many small probabilities",
        "Using float64 instead of float32",
        "Truncating sequences to a maximum length"
      ],
      correct: 1,
      explanation: "Direct computation of $P(y|x) = \\prod_t P(y_t | y_{<t}, x)$ would underflow to zero for long sequences (multiplying many numbers < 1). Instead, we compute $\\log P(y|x) = \\sum_t \\log P(y_t | y_{<t}, x)$, keeping everything in log-space. The log-sum-exp trick $\\log \\sum_i e^{a_i} = A + \\log \\sum_i e^{a_i - A}$ (where $A = \\max_i a_i$) handles the softmax normalization. This is a universal pattern: always work in log-probability space."
    }
  ]
};
