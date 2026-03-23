// Assessment: Information Theory in Practice — LLM Applications
// Section 0.2: Diagnostic test — applying probability & info theory to LLM training, RLHF, generation
// Pure assessment to gauge ability to connect theory to practice

export const appliedInfoTheoryAssessment = {
  id: "0.2-assess-applied",
  sectionId: "0.2",
  title: "Assessment: Information Theory in LLM Practice",
  difficulty: "hard",
  estimatedMinutes: 14,
  moduleType: "test",
  steps: [
    {
      type: "info",
      title: "Diagnostic: Information Theory in LLM Practice",
      content: "This is a **diagnostic assessment** testing your ability to connect probability and information theory to concrete LLM phenomena.\n\nEach question describes a real situation in LLM training, fine-tuning, or generation and asks you to identify the underlying theoretical principle.\n\nIf you score below 70%, work through the other assessments first, then return to this one — it integrates everything."
    },
    {
      type: "mc",
      question: "During LLM pre-training, the loss is $-\\frac{1}{T} \\sum_{t=1}^{T} \\log P_\\theta(w_t \\mid w_{<t})$. As training progresses, this loss decreases but eventually plateaus. The **irreducible** component of this loss corresponds to:",
      options: ["The numerical precision limits of float16 arithmetic, which introduce quantization noise that prevents the loss from reaching its true minimum", "The total size of the vocabulary, which sets a lower bound on cross-entropy because the model must distribute probability across all possible tokens", "The **entropy of natural language** $H(W_t \\mid W_{<t})$ — the inherent unpredictability that no model can eliminate because language itself is stochastic", "The number of parameters in the model, which determines the expressiveness ceiling and therefore the minimum achievable loss value"],
      correct: 2,
      explanation: "Cross-entropy loss $= H(P) + \\text{KL}(P \\| Q_\\theta)$. Even a perfect model ($Q = P$) achieves loss $H(P)$, the conditional entropy of the next token. This is the irreducible noise in language — given perfect context, many continuations are still plausible. The gap between the model's loss and $H(P)$ is the KL divergence, which measures model imperfection. Scaling laws describe how this KL component decreases with compute."
    },
    {
      type: "mc",
      question: "When you increase the **temperature** $\\tau$ in sampling ($P(w) \\propto \\exp(z_w / \\tau)$ where $z_w$ are logits), you are:",
      options: ["Interpolating between the model's learned distribution ($\\tau = 1$) and uniform ($\\tau \\to \\infty$), which increases entropy and diversity — at $\\tau \\to 0$ you get argmax (greedy) decoding", "Modifying the model's internal weight parameters by scaling the final layer's weight matrix, which permanently changes the learned token probability estimates", "Applying a form of inference-time dropout that randomly masks logits before the softmax, introducing controlled stochasticity into the generation process", "Changing the cross-entropy loss function used during the forward pass, which alters the gradient signal and causes the model to reweight its learned distributions"],
      correct: 0,
      explanation: "Dividing logits by $\\tau > 1$ flattens the distribution (higher entropy, more random). Dividing by $\\tau < 1$ sharpens it (lower entropy, more deterministic). At $\\tau \\to 0$, all mass concentrates on the highest-logit token (argmax). At $\\tau \\to \\infty$, the distribution approaches uniform over the vocabulary. Temperature is a post-hoc entropy control that doesn't change the model — it's a monotonic transformation of the entropy of the output distribution."
    },
    {
      type: "mc",
      question: "In RLHF, increasing the KL penalty coefficient $\\beta$ in $\\max_\\pi \\mathbb{E}[r(x)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ causes the optimal policy to:",
      options: [
        "Maximize reward without any constraint from the reference policy, since larger $\\beta$ amplifies the reward signal relative to the penalty term",
        "Stay closer to $\\pi_{\\text{ref}}$, trading off reward for distributional similarity — in the limit $\\beta \\to \\infty$, $\\pi^* = \\pi_{\\text{ref}}$",
        "Converge toward a uniform distribution over all possible outputs, since the KL penalty increasingly dominates and favors maximum entropy",
        "Increase its entropy independently of $\\pi_{\\text{ref}}$, producing more diverse outputs without regard for the reference distribution's shape"
      ],
      correct: 1,
      explanation: "The optimal policy is $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. As $\\beta \\to \\infty$, $\\exp(r/\\beta) \\to 1$, so $\\pi^* \\to \\pi_{\\text{ref}}$. As $\\beta \\to 0$, the reward dominates, and the policy concentrates on the highest-reward outputs regardless of naturalness. $\\beta$ controls the **rate-distortion trade-off**: how much \"distortion\" (reward sacrifice) you accept to stay within a certain \"rate\" (KL budget) from the reference."
    },
    {
      type: "mc",
      question: "A model trained on code achieves cross-entropy 0.5 nats/token on Python but 1.8 nats/token on Haskell. Assuming equal vocabulary, the **perplexity ratio** tells us:",
      options: ["Python is inherently a better-designed language than Haskell, which is why the model finds it fundamentally easier to predict and compress", "The Python test set contains fewer tokens overall, so the model achieves lower average cross-entropy simply due to the reduced evaluation sample size", "The Haskell tokenizer is producing suboptimal token boundaries, artificially inflating the per-token cross-entropy through poor segmentation choices", "The model has learned far more predictable structure in Python — perplexity is $e^{0.5} \\approx 1.65$ for Python vs $e^{1.8} \\approx 6.05$ for Haskell, meaning ~3.7× more uncertainty per token"],
      correct: 3,
      explanation: "Perplexity $= e^{H(P,Q)}$. Lower cross-entropy → lower perplexity → more predictable. The ratio $e^{1.8}/e^{0.5} = e^{1.3} \\approx 3.67$ means the model's uncertainty per token is 3.7× higher on Haskell. This could reflect: less Haskell in training data, Haskell's inherently different structure, or poor tokenization. The information-theoretic view lets you decompose the loss into data entropy + model deficiency."
    },
    {
      type: "mc",
      question: "**Reward hacking** in RLHF occurs when the policy finds outputs that score high reward but are low-quality. From an information-theoretic perspective, this happens because:",
      options: ["The reward model has excessively high entropy in its output scores, assigning near-uniform reward across all possible outputs regardless of their actual quality", "The KL penalty coefficient is too large, forcing the policy to stay so close to the reference that it cannot explore the reward landscape meaningfully", "The reward model is a learned approximation — as $\\pi$ moves far from $\\pi_{\\text{ref}}$ (high KL), it enters out-of-distribution regions where the reward model's predictions are unreliable", "The vocabulary is too small to express the nuanced outputs the reward model expects, causing a systematic mismatch between generation capacity and reward criteria"],
      correct: 2,
      explanation: "The reward model $r_\\phi$ was trained on data from $\\pi_{\\text{ref}}$. When $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ is large, $\\pi$ generates text unlike anything the reward model saw during training. The reward model's predictions become meaningless — it may assign high scores to adversarial outputs. This is why the KL penalty is crucial: it keeps $\\pi$ in the region where $r_\\phi$ is calibrated. This is a distributional shift problem quantified by KL divergence."
    },
    {
      type: "mc",
      question: "During fine-tuning, you notice the model's **token-level entropy** $H(P_\\theta(\\cdot \\mid x))$ dropping rapidly. The attention entropy (entropy of attention weights) is also collapsing. This likely indicates:",
      options: ["The model is **overfitting and losing diversity** — it's becoming overconfident on training patterns, potentially memorizing rather than generalizing, with attention collapsing onto fewer positions", "The model is learning perfectly and converging to the true data distribution, which naturally has lower entropy than the pretrained model's initial predictions", "The learning rate is set too low, causing the model to make only incremental updates that gradually concentrate probability mass without meaningful learning", "The model needs a substantially larger vocabulary to express the fine-tuning distribution, and the current vocabulary bottleneck is forcing artificial certainty"],
      correct: 0,
      explanation: "Entropy collapse is a warning sign: the model assigns near-deterministic predictions and attends to very few positions. Low output entropy means the model \"thinks\" it's very certain — but on new inputs, this overconfidence leads to poor calibration and repetitive/degenerate outputs. Monitoring entropy during training is an information-theoretic diagnostic: healthy training should reduce entropy gradually (learning structure) without collapsing it (losing flexibility)."
    },
    {
      type: "mc",
      question: "The **bits-per-byte** (BPB) metric normalizes perplexity across different tokenizers. If model A has perplexity 15.2 with a BPE tokenizer averaging 3.8 bytes/token, and model B has perplexity 8.1 with a character-level tokenizer (1 byte/token), which is better?",
      options: [
        "Model B — it has lower perplexity, which is the standard metric for language model quality regardless of tokenization differences",
        "Must compare BPB: model A's BPB $= \\frac{\\log_2(15.2)}{3.8} \\approx 1.03$, model B's BPB $= \\frac{\\log_2(8.1)}{1.0} \\approx 3.02$ — model A compresses more efficiently per byte",
        "Model A — it uses a better tokenizer with higher bytes-per-token, which inherently leads to more efficient compression of the input text",
        "They are equivalent — the perplexity difference is exactly offset by the tokenizer granularity, so both models capture identical amounts of structure"
      ],
      correct: 1,
      explanation: "BPB = (bits per token) / (bytes per token) = $\\log_2(\\text{PPL}) / \\text{avg\\_bytes\\_per\\_token}$. Model A: $\\log_2(15.2)/3.8 \\approx 1.03$ BPB. Model B: $\\log_2(8.1)/1.0 \\approx 3.02$ BPB. Model A achieves much better compression per byte of text. Raw perplexity is misleading across tokenizers because a character-level model predicts many more (easier) tokens. BPB is the fair comparison metric."
    },
    {
      type: "mc",
      question: "In the **information bottleneck** framework, a representation $Z$ of input $X$ is optimized to maximize $I(Z; Y)$ (task-relevant information) while minimizing $I(Z; X)$ (total information retained). In the context of LLM representations, this means:",
      options: ["Layers should memorize all input details to preserve maximum information, since any compression risks discarding task-relevant features that cannot be recovered", "The output representation should be statistically independent of the input, retaining no mutual information with the original token embeddings", "All layers in the network should maintain exactly the same mutual information with the input, ensuring uniform information flow throughout the architecture", "Good representations should **compress away irrelevant details of the input** while retaining information needed for prediction — the $\\beta$ parameter controls this trade-off"],
      correct: 3,
      explanation: "The IB objective $\\max_{p(z|x)} I(Z; Y) - \\beta I(Z; X)$ formalizes the compression/prediction trade-off. Small $\\beta$ keeps all information (overfitting); large $\\beta$ compresses aggressively (underfitting). This framework connects to: dropout (random compression), bottleneck layers (architectural compression), and the KL penalty in RLHF (behavioral compression). The parallel to $\\max \\mathbb{E}[r] - \\beta \\text{KL}$ is exact in structure."
    },
    {
      type: "mc",
      question: "When computing $\\log P(y \\mid x)$ for a long sequence $y = (y_1, \\dots, y_T)$, numerical issues arise because the log-probability is a sum of $T$ negative terms that can become very negative. The standard solution is:",
      options: ["Clipping the log-probabilities to a minimum threshold value, preventing any single token from contributing more than a bounded amount to the total", "Using float64 instead of float32 for all intermediate computations, which provides sufficient dynamic range to avoid underflow in most practical cases", "Working in **log-space throughout** — using the log-sum-exp trick for normalization and accumulating log-probabilities additively, avoiding underflow from multiplying small values", "Truncating sequences to a fixed maximum length, ensuring the accumulated log-probability sum never exceeds the representable range of the floating-point format"],
      correct: 2,
      explanation: "Direct computation of $P(y|x) = \\prod_t P(y_t | y_{<t}, x)$ would underflow to zero for long sequences (multiplying many numbers < 1). Instead, we compute $\\log P(y|x) = \\sum_t \\log P(y_t | y_{<t}, x)$, keeping everything in log-space. The log-sum-exp trick $\\log \\sum_i e^{a_i} = A + \\log \\sum_i e^{a_i - A}$ (where $A = \\max_i a_i$) handles the softmax normalization. This is a universal pattern: always work in log-probability space."
    }
  ]
};
