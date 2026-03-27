// Section A.2: Reward Modeling Assessment

export const rewardModelingAssessment = {
  id: "A.2-assess",
  sectionId: "A.2",
  title: "Assessment: Reward Modeling",
  difficulty: "medium",
  estimatedMinutes: 14,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The Bradley-Terry model for pairwise preferences defines $P(y_w \\succ y_l \\mid x) = \\sigma(r(x, y_w) - r(x, y_l))$ where $\\sigma$ is the sigmoid. What assumption does this encode about human preferences?",
      options: [
        "Humans always prefer longer responses, so the model uses response length as a proxy for quality in its scalar reward predictions",
        "Preference probability depends only on the difference in scalar rewards — preferences are transitive and follow a logistic noise model",
        "Preferences are uniformly random and independent of content, serving only as a regularization signal for the reward model's outputs",
        "The reward function must be linear in response length, constraining the model to additive per-token scoring across the full sequence"
      ],
      correct: 1,
      explanation: "Bradley-Terry assumes a latent scalar \"quality\" score $r(x, y)$ for each response, and preference probability is a function of the *difference* $r(x, y_w) - r(x, y_l)$ passed through a sigmoid. This implies transitivity (if $A \\succ B$ and $B \\succ C$, then $A \\succ C$) and a specific noise model (logistic). These are strong assumptions — real human preferences can be intransitive and context-dependent — but the model is tractable and works surprisingly well in practice."
    },
    {
      type: "mc",
      question: "The reward model training loss is $\\mathcal{L} = -\\mathbb{E}_{(x, y_w, y_l)}\\left[\\log \\sigma(r_\\theta(x, y_w) - r_\\theta(x, y_l))\\right]$. Why does this loss only depend on the *difference* $r(x, y_w) - r(x, y_l)$ rather than individual reward values?",
      options: [
        "Individual rewards are easier to learn but produce less accurate rankings for fine-grained preference distinctions between similar responses",
        "Individual rewards require reinforcement learning to train, which is incompatible with the supervised pairwise comparison training framework",
        "The Bradley-Terry likelihood is invariant to adding a constant to all rewards — only differences are identifiable from pairwise data",
        "The sigmoid function requires exactly two inputs to compute, so the loss must be a pairwise difference to match the activation function's domain"
      ],
      correct: 2,
      explanation: "If we replace $r(x, y)$ with $r(x, y) + c$ for any constant $c$, the difference $r(x, y_w) - r(x, y_l)$ is unchanged, so the likelihood is identical. This means the absolute scale of rewards is unidentifiable from pairwise data — only relative differences matter. This is why reward models need careful normalization and why reward values can drift during training. It also means comparing rewards across different prompts $x$ is not inherently meaningful."
    },
    {
      type: "mc",
      question: "\"Reward hacking\" (or reward overoptimization) occurs when the policy exploits the reward model. Which of the following best describes the phenomenon observed by Gao et al. (2023)?",
      options: [
        "As policy optimization intensifies (increasing KL from reference), proxy reward rises but true (gold) reward eventually falls — the policy finds adversarial RM inputs",
        "The reward model assigns perfect scores to all outputs once the policy has been optimized for more than a few hundred gradient steps of reinforcement learning",
        "The reward model diverges due to numerical instability caused by extreme reward values at the boundaries of the learned output distribution during optimization",
        "Reward hacking only occurs with models smaller than 1B parameters, since larger models have sufficient capacity to accurately represent the true reward function"
      ],
      correct: 0,
      explanation: "Gao et al. showed a clear pattern: proxy reward (from the learned RM) increases monotonically with KL budget, but gold reward (from a much larger or human-evaluated RM) follows an inverted-U shape — it improves initially, peaks, then *degrades*. The policy discovers outputs that score high on the proxy RM but are actually low quality. This is a form of Goodhart's law: \"when a measure becomes a target, it ceases to be a good measure.\" The KL penalty $\\beta$ controls where on this curve you operate."
    },
    {
      type: "mc",
      question: "What is the key difference between a **Process Reward Model (PRM)** and an **Outcome Reward Model (ORM)**?",
      options: [
        "PRMs are faster to train because they evaluate shorter subsequences, while ORMs achieve higher accuracy through full-sequence reward aggregation",
        "ORMs use reinforcement learning for credit assignment across the full trace, while PRMs use supervised learning on step-level correctness annotations",
        "PRMs can only be used with chain-of-thought prompting, since they require explicit reasoning steps to evaluate against the learned process reward",
        "PRMs assign rewards to each intermediate reasoning step, while ORMs assign a single reward to the final answer — PRMs provide denser signal"
      ],
      correct: 3,
      explanation: "ORMs give a single scalar reward for the complete response (correct/incorrect final answer). PRMs evaluate each reasoning step individually ($r(x, s_1, \\ldots, s_t)$ at each step $t$). Lightman et al. (2023) showed PRMs significantly outperform ORMs for mathematical reasoning because: (1) they provide credit assignment — identifying *where* reasoning went wrong, (2) denser reward signal reduces variance, and (3) they can prune bad reasoning paths early. The cost is that PRM training data requires per-step human annotations."
    },
    {
      type: "mc",
      question: "In RLHF, an \"implicit\" reward can be extracted from any language model via $r_{\\text{implicit}}(x, y) = \\beta \\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}$. What does this represent?",
      options: [
        "The perplexity ratio of the response under both models, serving as a proxy for the combined fluency and coherence quality assessment",
        "The entropy of the policy distribution over next tokens, measuring the model's predictive uncertainty about the optimal continuation path",
        "The log-likelihood ratio measuring how much the policy has shifted from the reference — upweighted responses have higher implicit reward",
        "The gradient of the loss with respect to policy parameters, indicating the direction of steepest reward improvement in parameter space"
      ],
      correct: 2,
      explanation: "This comes from the closed-form solution of the KL-constrained RL objective: $\\pi^*(y \\mid x) \\propto \\pi_{\\text{ref}}(y \\mid x) \\exp(r(y, x) / \\beta)$. Rearranging: $r(y, x) = \\beta \\log \\frac{\\pi^*(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$. So the log-ratio is the implicit reward (up to a prompt-dependent constant). This is the foundation of DPO — instead of learning an explicit reward model, use the policy itself as an implicit reward."
    },
    {
      type: "mc",
      question: "Reward model accuracy typically scales with model size, but Anthropic's work showed an important nuance. What did they find about the relationship between RM size and policy size?",
      options: [
        "The RM should always match the policy size exactly to ensure paired representational capacity during the optimization procedure",
        "Smaller RMs always produce better policies because their limited capacity acts as implicit regularization against overoptimization",
        "RM size has no measurable effect on final policy quality because the reward signal is a single scalar regardless of model capacity",
        "An RM should be at least as large as the policy — otherwise the stronger policy can easily exploit the RM's blind spots"
      ],
      correct: 3,
      explanation: "If the policy model is larger and more capable than the reward model, the policy can find adversarial outputs that fool the RM — it's effectively a stronger adversary than the RM can handle. This creates an asymmetry: the RM needs to be at least as capable as the policy to provide reliable signal. In practice, teams often use an RM that is the same size or larger than the policy. This has implications for scalable oversight — as policies get stronger, reward models must keep pace."
    },
    {
      type: "mc",
      question: "When training a reward model from human pairwise comparisons, annotator disagreement is common. What is the standard approach to handling this?",
      options: [
        "Use majority vote, optionally weighting the loss by agreement level — high-agreement pairs provide a stronger, more reliable training signal",
        "Discard all examples where annotators disagree to ensure only unambiguous, high-confidence preference pairs enter the training data pipeline",
        "Train a separate reward model per annotator and ensemble their scalar predictions at inference time to capture the full preference diversity",
        "Randomly assign preference labels when annotators disagree, treating the resulting ambiguity as a form of data augmentation for regularization"
      ],
      correct: 0,
      explanation: "Majority voting is the most common approach: if 3 out of 5 annotators prefer $y_w$, it's labeled as preferred. Some approaches further weight the Bradley-Terry loss by agreement level — a 5/5 agreement pair gets full weight, while a 3/2 split gets reduced weight, reflecting genuine ambiguity. More sophisticated methods model annotator-level preferences or treat disagreement as inherent noise in the generative process. Discarding disagreements wastes data and biases toward \"easy\" examples."
    },
    {
      type: "mc",
      question: "The reward model is initialized from a pretrained language model with the final unembedding layer replaced by a scalar head. Why is this initialization important?",
      options: [
        "It allows the model to generate text during reward scoring, enabling richer evaluation through autoregressive analysis of each candidate response",
        "The pretrained representations encode the semantic understanding needed to evaluate quality — training from scratch would require far more comparison data",
        "It is purely a computational optimization that speeds up convergence with no measurable impact on the reward model's final prediction quality",
        "The unembedding layer is directly reused as the reward head after applying a simple dimensionality reduction to produce the scalar output"
      ],
      correct: 1,
      explanation: "The RM needs to *understand* text to evaluate it — reading comprehension, factual knowledge, reasoning ability, style detection. These capabilities come from pretraining. The scalar reward head is the only new component; the pretrained backbone provides the representation. Without this transfer, the RM would need to learn language understanding from comparison data alone, which is far too sparse. This is why RM performance scales with the quality of the pretrained backbone."
    },
    {
      type: "mc",
      question: "Consider the reward model loss gradient: $\\nabla_\\theta \\mathcal{L} = -\\mathbb{E}\\left[(1 - \\sigma(\\Delta r)) \\cdot (\\nabla_\\theta r_\\theta(x, y_w) - \\nabla_\\theta r_\\theta(x, y_l))\\right]$ where $\\Delta r = r_\\theta(x, y_w) - r_\\theta(x, y_l)$. The factor $(1 - \\sigma(\\Delta r))$ implies:",
      options: [
        "All training examples contribute equally to the gradient regardless of the current reward margin between the preferred and dispreferred responses",
        "When the model already ranks the preferred response much higher ($\\Delta r \\gg 0$), the gradient vanishes — it focuses on misranked pairs",
        "Negative rewards produce larger gradients than positive rewards, so the model learns disproportionately from dispreferred response examples",
        "The gradient magnitude is always exactly 1 due to the sigmoid's bounded derivative, so only the learning rate controls update size"
      ],
      correct: 1,
      explanation: "When $\\Delta r \\gg 0$, $\\sigma(\\Delta r) \\approx 1$, so $(1 - \\sigma(\\Delta r)) \\approx 0$ and the gradient vanishes. This is the standard logistic regression property: well-classified examples contribute little gradient. The model automatically upweights pairs where it's currently wrong ($\\Delta r < 0$) or uncertain ($\\Delta r \\approx 0$). This implicit curriculum means early training focuses on obvious preferences, then shifts to harder, more ambiguous pairs."
    },
    {
      type: "mc",
      question: "A reward model trained on helpfulness comparisons is used to optimize a policy. After optimization, the policy produces very long, verbose responses that score high reward but are rated lower by humans. This is an example of:",
      options: [
        "Underfitting of the reward model, where insufficient capacity prevents it from distinguishing genuinely helpful responses from merely verbose ones",
        "The policy model being too small to express high-quality responses, forcing it to compensate with verbosity as a substitute for genuine depth",
        "A length bias spurious correlation — the RM learned that longer responses tend to be preferred in training data, so the policy exploits this shortcut",
        "A bug in the KL penalty implementation that fails to penalize the policy for deviating from the reference model's typical response length distribution"
      ],
      correct: 2,
      explanation: "Length bias is one of the most common and well-documented reward hacking modes. In human comparison data, longer responses are often preferred (they tend to be more detailed and thorough), so the RM learns a spurious correlation between length and quality. The policy then exploits this by padding responses with redundant content. Mitigations include: (1) length-conditioned reward normalization, (2) adding length as an explicit feature and regressing it out, (3) including length-controlled comparison pairs in RM training data."
    }
  ]
};
