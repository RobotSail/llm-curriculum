// Focused learning module: Reward Modeling
// Section A.2: Reward Modeling
// Covers: why reward models are needed, pairwise preference data, the Bradley-Terry
// model, the RM training objective, reward hacking and overoptimization, and the
// KL penalty as a trust budget.
// Single-concept module building from first principles.

export const rewardModelingLearning = {
  id: "A.2-reward-modeling-learning-easy",
  sectionId: "A.2",
  title: "Reward Modeling for Language Models",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why reward models?
    {
      type: "info",
      title: "The Problem: Specifying What We Want",
      content: "After SFT, a language model can follow instructions and produce coherent responses. But how do we make it produce **better** responses? \"Better\" is subjective — it depends on helpfulness, accuracy, safety, conciseness, and many other factors that are hard to formalize as a loss function.\n\nWe can't write a mathematical formula for \"helpful and harmless.\" But humans can easily compare two responses and say which one is better. This asymmetry — judgment is easier than specification — is the key insight behind reward modeling.\n\nA **reward model** (RM) converts these pairwise human judgments into a scalar score $r(x, y) \\in \\mathbb{R}$ that rates any prompt-response pair. Once trained, it serves as a differentiable proxy for human preferences that can be optimized with gradient-based methods (like PPO).\n\nThe pipeline:\n1. Collect **preference data**: for each prompt $x$, generate two responses $y_w$ (preferred/winning) and $y_l$ (rejected/losing), judged by a human\n2. Train the reward model on this data to predict which response humans would prefer\n3. Use the reward model's scores to optimize the policy via RL"
    },
    // Step 2: MC — why RM
    {
      type: "mc",
      question: "Why is human preference data collected as pairwise comparisons (\"response A is better than response B\") rather than absolute ratings (\"response A gets 4 out of 5\")?",
      options: [
        "Pairwise comparisons are more reliable — humans show much higher inter-annotator agreement when comparing two responses than when assigning absolute scores on a scale",
        "Absolute ratings are too computationally expensive to convert into a loss function that neural networks can optimize",
        "Pairwise comparisons produce more training data per annotation because each comparison generates two data points instead of one",
        "Neural network reward models can only accept binary labels, so absolute ratings would need to be discretized anyway"
      ],
      correct: 0,
      explanation: "Calibration is the key problem with absolute ratings: one annotator's 4/5 is another's 3/5. Different annotators use scales differently, and even the same annotator's scale drifts over time. Pairwise comparisons are much more consistent — \"which is better?\" is a simpler judgment that produces higher inter-annotator agreement. This reliability is crucial because the reward model is only as good as its training signal. The Bradley-Terry model converts these reliable pairwise judgments into learned scalar scores."
    },
    // Step 3: Bradley-Terry model
    {
      type: "info",
      title: "The Bradley-Terry Preference Model",
      content: "The **Bradley-Terry model** (1952) provides the mathematical link between pairwise comparisons and scalar scores. It assumes each item has a latent \"strength\" and that the probability of one item being preferred over another depends only on their strength difference.\n\nFor two responses $y_w$ (preferred) and $y_l$ (rejected) to the same prompt $x$, the model assumes:\n\n$$P(y_w \\succ y_l | x) = \\sigma\\big(r(x, y_w) - r(x, y_l)\\big)$$\n\nwhere $\\sigma(z) = \\frac{1}{1+e^{-z}}$ is the sigmoid function, and $r(x, y)$ is the reward model's scalar output.\n\nKey properties of this model:\n\n1. **Only differences matter**: If you add a constant to all rewards (shifting $r(x,y) \\to r(x,y) + c$ for all $y$), the preference probability is unchanged. The reward model learns a **relative** scale, not an absolute one.\n\n2. **Transitivity**: If $r(x, y_a) > r(x, y_b) > r(x, y_c)$, then $P(y_a \\succ y_c) > P(y_a \\succ y_b)$ and $P(y_a \\succ y_c) > P(y_b \\succ y_c)$. Preferences are consistent with a total ordering.\n\n3. **Calibrated confidence**: When $r(x, y_w) - r(x, y_l)$ is large, the model is very confident in its preference. When the difference is near zero, the model assigns near-50/50 probability — appropriate for responses of similar quality."
    },
    // Step 4: MC — Bradley-Terry
    {
      type: "mc",
      question: "A reward model outputs $r(x, y_A) = 3.0$ and $r(x, y_B) = 1.0$ for two responses. Under the Bradley-Terry model, $P(y_A \\succ y_B) = \\sigma(2.0) \\approx 0.88$. Now suppose we add 10 to both rewards: $r(x, y_A) = 13.0$, $r(x, y_B) = 11.0$. What is $P(y_A \\succ y_B)$?",
      options: [
        "Much higher (close to 1.0), because the larger absolute reward values indicate stronger confidence",
        "Still $\\sigma(2.0) \\approx 0.88$, because Bradley-Terry depends only on the reward difference, which is still $2.0$",
        "Lower (close to 0.5), because the proportional difference $13/11$ is smaller than $3/1$",
        "Undefined, because the Bradley-Terry model requires rewards to be normalized to $[0, 1]$"
      ],
      correct: 1,
      explanation: "The Bradley-Terry model computes $P(y_A \\succ y_B) = \\sigma(r_A - r_B)$. Adding a constant $c$ to both rewards: $\\sigma((r_A + c) - (r_B + c)) = \\sigma(r_A - r_B)$. The constant cancels. Only the difference matters — here it remains $2.0$ regardless of the offset. This shift invariance means the reward model learns a relative ranking, not absolute quality scores. It's the spacing between rewards that carries information."
    },
    // Step 5: The RM training loss
    {
      type: "info",
      title: "Training the Reward Model",
      content: "Given a dataset of preferences $\\mathcal{D} = \\{(x_i, y_{w,i}, y_{l,i})\\}_{i=1}^N$, we train the reward model by maximizing the log-likelihood under the Bradley-Terry model:\n\n$$\\mathcal{L}_{\\text{RM}}(\\phi) = -\\frac{1}{N}\\sum_{i=1}^{N} \\log \\sigma\\big(r_\\phi(x_i, y_{w,i}) - r_\\phi(x_i, y_{l,i})\\big)$$\n\nThis is a **binary cross-entropy loss** — the same loss used for logistic regression, but applied to reward differences. The gradient has an elegant form:\n\n$$\\nabla_\\phi \\mathcal{L} = -\\frac{1}{N}\\sum_i \\underbrace{\\sigma\\big(r_\\phi(y_l) - r_\\phi(y_w)\\big)}_{\\text{error signal}} \\big(\\nabla_\\phi r_\\phi(y_w) - \\nabla_\\phi r_\\phi(y_l)\\big)$$\n\nThe error signal $\\sigma(r_l - r_w)$ is the probability that the model gets the comparison **wrong**. When the model already ranks the pair correctly with high confidence, this term is near zero and the gradient is small — the model focuses its updates on pairs it finds difficult. When the model gets a comparison wrong (assigns higher reward to the rejected response), the error signal is large and the model receives a strong corrective gradient.\n\nThis adaptive weighting is a natural property of the loss — difficult comparisons receive more gradient, which is exactly what we want."
    },
    // Step 6: MC — RM training
    {
      type: "mc",
      question: "During RM training, a pair $(y_w, y_l)$ has current rewards $r(y_w) = 0.5$ and $r(y_l) = 2.0$ — the model incorrectly ranks the rejected response higher. What is the magnitude of the gradient error signal $\\sigma(r_l - r_w) = \\sigma(1.5)$?",
      options: [
        "Near zero ($\\approx 0.02$), because the sigmoid saturates for large inputs and produces vanishing gradients",
        "Exactly $0.5$, because the sigmoid always outputs $0.5$ when the model is uncertain",
        "Exactly $1.0$, because any incorrect ranking produces the maximum possible gradient signal",
        "Near $0.82$, because $\\sigma(1.5) \\approx 0.82$ — the model gets a strong corrective gradient since it has the ranking wrong"
      ],
      correct: 3,
      explanation: "The error signal is $\\sigma(r_l - r_w) = \\sigma(2.0 - 0.5) = \\sigma(1.5) \\approx 0.82$. Since the model ranks the pair incorrectly (rejected response has higher reward), the error signal is large — close to 1 — producing a strong gradient that pushes $r(y_w)$ up and $r(y_l)$ down. If the model had the ranking correct with $r(y_w) \\gg r(y_l)$, then $\\sigma(r_l - r_w)$ would be close to 0 and the gradient would be small. This natural curriculum effect is a key property of the loss."
    },
    // Step 7: RM architecture
    {
      type: "info",
      title: "Reward Model Architecture",
      content: "The reward model is typically a **pretrained language model** with a modification: the language modeling head (which predicts next tokens) is replaced with a **scalar head** — a single linear layer that maps the final hidden state to a scalar value $r \\in \\mathbb{R}$.\n\nConcretely, for a transformer-based RM:\n1. The prompt and response are concatenated: $[x, y]$\n2. The full sequence is processed through the transformer backbone\n3. The hidden state at the **last token** $h_T \\in \\mathbb{R}^d$ is extracted\n4. A linear projection $r = w^\\top h_T + b$ produces the scalar reward\n\nWhy start from a pretrained model? The backbone already understands language, reasoning, and factual knowledge. The reward model needs all of these to assess response quality — it must understand the question to judge whether the answer is correct, and it must understand nuance to judge which answer is more helpful.\n\nIn practice, the RM is often initialized from the **same SFT checkpoint** used as the reference policy. This ensures the reward model and policy share a common understanding of language. The RM is typically the same size or smaller than the policy — Anthropic and OpenAI have used RMs ranging from 6B to 175B parameters."
    },
    // Step 8: MC — RM architecture
    {
      type: "mc",
      question: "A reward model extracts the hidden state $h_T \\in \\mathbb{R}^{4096}$ at the last token position and applies a linear head $r = w^\\top h_T + b$. How many learnable parameters does the scalar head itself have?",
      options: [
        "$4096^2 + 4096 = 16,781,312$ — a full $d \\times d$ matrix is needed to transform the hidden state before the final projection",
        "$4096 + 1 = 4097$ — one weight per hidden dimension plus a bias term, since the head is a single linear layer to a scalar",
        "$2 \\times 4096 = 8192$ — separate weight vectors for the preferred and rejected responses, since the model must score both",
        "$4096 \\times V$ where $V$ is the vocabulary size — the head must project to token probabilities before converting to a scalar"
      ],
      correct: 1,
      explanation: "The scalar head is a linear map from $\\mathbb{R}^{4096}$ to $\\mathbb{R}^1$: $r = w^\\top h_T + b$ with $w \\in \\mathbb{R}^{4096}$ and $b \\in \\mathbb{R}$. That's $4096 + 1 = 4097$ parameters. The same head is used for both $y_w$ and $y_l$ — each gets scored independently in a separate forward pass through the same model with the same parameters. The vast majority of the RM's parameters are in the transformer backbone, not the head."
    },
    // Step 9: Reward hacking
    {
      type: "info",
      title: "Reward Hacking and Overoptimization",
      content: "The reward model is a **proxy** for human preferences — it's trained on a finite sample of comparisons and can only approximate the true preference function. This creates a fundamental problem: if the policy optimizes the reward model too aggressively, it will find inputs that score highly according to the RM but are actually low-quality. This is **reward hacking** (also called reward overoptimization).\n\nGao et al. (2023) characterized this with a striking empirical finding. As the policy diverges from the reference (measured by KL divergence), two things happen:\n\n1. The **proxy reward** (RM score) increases monotonically — the policy keeps finding ways to score higher\n2. The **true reward** (human judgment) first increases, then **peaks and decreases** — past a certain point, the policy is exploiting RM artifacts rather than improving quality\n\nThis is a manifestation of **Goodhart's Law**: \"When a measure becomes a target, it ceases to be a good measure.\" The RM score is a measure of quality; when we optimize it as a target, it breaks.\n\nExamples of reward hacking in practice:\n- Generating longer responses (if the RM has a length bias)\n- Excessive hedging and caveats (\"I'm not sure, but...\" followed by a correct answer)\n- Sycophantic agreement with the user\n- Repetitive phrasing that the RM associates with high quality"
    },
    // Step 10: MC — reward hacking
    {
      type: "mc",
      question: "Gao et al. (2023) found that as KL divergence from the reference policy increases during RLHF training, the proxy reward (RM score) increases monotonically but the true reward (human judgment) eventually decreases. What does this imply about the optimal training strategy?",
      options: [
        "Train until the proxy reward plateaus, since the plateau indicates the reward model has been fully optimized and the policy has converged to optimal behavior",
        "This divergence is an artifact of the evaluation methodology — in practice, higher RM scores always correspond to better human judgments",
        "The reward model should be periodically retrained on the current policy's outputs to prevent the proxy and true rewards from diverging",
        "There exists an optimal KL budget beyond which further optimization degrades true quality — the KL penalty $\\beta$ should be set to stop optimization near this point"
      ],
      correct: 3,
      explanation: "The proxy-true reward divergence means there's a sweet spot: enough optimization to improve quality, but not so much that the policy starts exploiting RM artifacts. The KL penalty $\\beta$ in the RLHF objective directly controls how far the policy can stray from the reference. Setting $\\beta$ appropriately stops optimization near the peak of true reward. Too small a $\\beta$ allows overoptimization past the peak; too large a $\\beta$ stops optimization before reaching the peak. This is why $\\beta$ is one of the most important hyperparameters in RLHF."
    },
    // Step 11: KL penalty as optimization budget
    {
      type: "info",
      title: "The KL Penalty as an Optimization Budget",
      content: "The RLHF objective includes a KL penalty: $\\max_\\pi \\mathbb{E}[r(x,y)] - \\beta \\, D_{\\text{KL}}(\\pi \\| \\pi_{\\text{ref}})$. Understanding the reward hacking problem gives us a clear interpretation of $\\beta$.\n\nThink of the KL divergence as a **budget**: the total amount of distributional change you can \"spend\" before the reward model becomes unreliable. The $\\beta$ parameter sets the exchange rate between reward improvement and distributional change.\n\n- **Large $\\beta$** (tight budget): The policy stays close to the reference. Reward improvement is modest but reliable — the RM's predictions are accurate in this neighborhood.\n- **Small $\\beta$** (loose budget): The policy can deviate further. It may achieve higher RM scores, but those scores become unreliable as the policy moves to parts of the output space where the RM has little training data.\n\nThe Gao et al. result can be restated: the true reward is a concave function of KL divergence. There exists a $D^*$ such that for $D_{\\text{KL}} < D^*$, optimization helps; for $D_{\\text{KL}} > D^*$, it hurts. The optimal $\\beta$ implicitly sets the policy's KL at approximately $D^*$.\n\nThis framing also explains why **reward model quality matters**: a better RM pushes $D^*$ further out, allowing more optimization before overoptimization kicks in. Investing in higher-quality preference data directly translates to more room for policy improvement."
    },
    // Step 12: MC — KL budget
    {
      type: "mc",
      question: "Two teams train reward models on the same task. Team A's RM is trained on 10K high-quality expert comparisons. Team B's RM is trained on 50K noisy crowdsourced comparisons. Both teams then run RLHF with the same $\\beta$. Which outcome is most likely?",
      options: [
        "Team B achieves higher true reward because their RM has seen more data and therefore has better coverage of the output space",
        "Both teams achieve identical results because the KL penalty $\\beta$ is the same, so the policies diverge by the same amount regardless of RM quality",
        "Team A likely achieves higher true reward because their RM is more accurate, which pushes the overoptimization threshold further out — the same KL budget buys more genuine improvement",
        "Team A's policy diverges less from the reference because a higher-quality RM produces smaller gradients, resulting in less policy change"
      ],
      correct: 2,
      explanation: "RM quality determines how far you can optimize before the proxy diverges from true quality. A more accurate RM (Team A's, trained on expert data) maintains alignment between proxy and true reward over a larger KL range — the overoptimization threshold $D^*$ is further out. With the same $\\beta$, both policies reach roughly the same KL, but Team A's RM is still reliable at that KL while Team B's noisier RM has already started to diverge. The same optimization budget buys more genuine improvement with a better reward model."
    },
    // Step 13: Annotator disagreement
    {
      type: "info",
      title: "Handling Annotator Disagreement",
      content: "A fundamental challenge in reward modeling is that **human preferences are not fully consistent**. Different annotators may prefer different responses to the same prompt, and even the same annotator may be inconsistent across sessions.\n\nSources of disagreement:\n\n1. **Genuine ambiguity**: For many prompts, there is no objectively \"better\" response. One annotator may prefer a concise answer; another may prefer a detailed one. Both are valid preferences.\n\n2. **Annotator expertise**: Domain experts and non-experts may have systematically different preferences, especially for technical content. An expert may value precision; a non-expert may value accessibility.\n\n3. **Annotation noise**: Fatigue, misunderstanding the prompt, or accidental clicks introduce random noise into the labels.\n\nPractical mitigations:\n- **Multiple annotators per comparison**: Collecting 3-5 judgments per pair and using majority vote reduces noise. When annotators strongly disagree, the pair can be discarded or down-weighted — forcing a binary label on genuinely ambiguous pairs teaches the RM a spurious preference.\n- **Annotator-specific models**: Some approaches train per-annotator reward models and combine them, acknowledging that \"preferred\" is not a single concept.\n- **Filtering by agreement**: Pairs where annotators agree are more informative than pairs where they disagree. Down-weighting or removing high-disagreement pairs improves RM reliability.\n\nThe fundamental limit: the reward model can't be more consistent than the training signal. Noisy or contradictory preferences create a ceiling on RM accuracy that no architecture improvement can overcome."
    },
    // Step 14: MC — annotator disagreement
    {
      type: "mc",
      question: "A preference dataset contains 10,000 comparison pairs. On 2,000 of these, the 5 annotators split 3-2 (narrow majority). On the remaining 8,000, annotators agree 4-1 or 5-0. How should the narrow-majority pairs be handled during RM training?",
      options: [
        "Down-weight or remove them — the narrow margin suggests genuine ambiguity, and forcing a confident binary label on ambiguous pairs teaches the RM spurious preferences",
        "Use them exclusively for training, since disagreement indicates these are the hardest and most informative examples",
        "Include them with full weight — any label from a majority vote is equally valid, and removing data always hurts model performance",
        "Assign them the opposite label from the majority vote, since narrow majorities are statistically likely to be wrong due to sampling noise"
      ],
      correct: 0,
      explanation: "When annotators split 3-2, the \"preferred\" response was barely preferred — this often reflects genuine ambiguity rather than clear quality difference. Training the RM to confidently distinguish these near-equivalent responses teaches it arbitrary preferences that don't generalize. Common approaches: remove these pairs entirely, or down-weight them so they contribute less to the loss. The high-agreement pairs (4-1 or 5-0) provide clearer signal and should dominate training. This improves RM reliability at the cost of a smaller effective dataset — a worthwhile tradeoff."
    }
  ]
};
