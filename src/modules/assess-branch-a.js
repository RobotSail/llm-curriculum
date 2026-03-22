// Branch A Assessments: Post-Training & Alignment
// Sections A.1–A.5: SFT, Reward Modeling, RLHF, Direct Alignment, Frontier Alignment
// Pure assessment modules (no info steps) — 10 MC questions each, easy → hard

// ─────────────────────────────────────────────
// A.1: Supervised Fine-Tuning (SFT)
// ─────────────────────────────────────────────
export const sftAssessment = {
  id: "A.1-assess",
  sectionId: "A.1",
  title: "Assessment: Supervised Fine-Tuning",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The LIMA paper (Zhou et al., 2023) demonstrated strong performance with only 1,000 carefully curated fine-tuning examples. What is the primary claim this supports?",
      options: [
        "Pretraining is unnecessary if fine-tuning data is high quality",
        "SFT primarily *unlocks* capabilities already learned during pretraining rather than *teaching* new knowledge",
        "1,000 examples is always the optimal dataset size for SFT",
        "Data quality does not matter as long as the dataset is small"
      ],
      correct: 1,
      explanation: "LIMA's key insight is the \"Superficial Alignment Hypothesis\": the vast majority of a model's knowledge and capabilities come from pretraining. SFT's role is to *unlock* or *steer* those existing capabilities into the desired format (e.g., instruction-following, chat). This is why 1,000 high-quality examples sufficed — they taught the model *how* to present knowledge it already had, not *what* to know."
    },
    {
      type: "mc",
      question: "What is the standard loss function used in supervised fine-tuning of a language model on instruction-response pairs?",
      options: [
        "Mean squared error between predicted and target token embeddings",
        "Cross-entropy loss on the response tokens only, with instruction tokens masked from the loss",
        "Contrastive loss between positive and negative responses",
        "KL divergence between the fine-tuned and base model distributions"
      ],
      correct: 1,
      explanation: "Standard SFT uses next-token cross-entropy loss $\\mathcal{L} = -\\sum_{t \\in \\text{response}} \\log P_\\theta(w_t \\mid w_{<t})$, but crucially, the loss is computed only over the response tokens. The instruction/prompt tokens provide context but are masked from the loss — we don't want to train the model to *generate* instructions, only to *respond* to them. This is sometimes called \"completion-only\" training."
    },
    {
      type: "mc",
      question: "SFT models often suffer from catastrophic overfitting when trained for too many epochs. A common observation is that performance peaks at:",
      options: [
        "10–20 epochs with aggressive learning rates",
        "1–2 epochs, with quality degrading rapidly beyond that",
        "100+ epochs with careful early stopping",
        "Exactly 5 epochs regardless of dataset size"
      ],
      correct: 1,
      explanation: "SFT datasets are typically small (thousands to tens of thousands of examples), so the model memorizes them quickly. After 1–2 epochs, the model begins overfitting to surface patterns — repeating exact phrasings, losing diversity, and degrading on out-of-distribution prompts. This is why SFT is often described as a \"light touch\" — a brief adaptation, not extended training."
    },
    {
      type: "mc",
      question: "Why is learning rate selection particularly sensitive during SFT compared to pretraining?",
      options: [
        "SFT uses a different optimizer than pretraining",
        "The pretrained weights encode useful representations in a sharp loss basin — too high a learning rate destroys these representations, while too low a rate fails to adapt",
        "SFT always requires a constant learning rate with no schedule",
        "The gradient norms are always larger during SFT"
      ],
      correct: 1,
      explanation: "Pretrained models sit in well-conditioned regions of the loss landscape that encode broad linguistic knowledge. SFT learning rates are typically 10–100x smaller than pretraining rates (e.g., $1 \\times 10^{-5}$ to $5 \\times 10^{-5}$ vs. $3 \\times 10^{-4}$). Too large a rate \"kicks\" the model out of the pretrained basin, causing catastrophic forgetting. Too small a rate means the model barely moves from base behavior. This sensitivity is a form of the stability-plasticity dilemma."
    },
    {
      type: "mc",
      question: "In multi-turn conversation SFT, training data is structured with role tokens (e.g., `<|user|>`, `<|assistant|>`). What is the primary purpose of these special tokens?",
      options: [
        "They reduce the vocabulary size by replacing common words",
        "They provide turn-boundary signals and role attribution so the model learns *when* to generate and *whose* voice to adopt",
        "They are only needed during inference, not training",
        "They prevent the model from generating any harmful content"
      ],
      correct: 1,
      explanation: "Role tokens serve as structural markers that teach the model the conversational protocol: when it's the assistant's turn to speak, what voice/style to use, and where turn boundaries are. Without them, the model cannot distinguish user text from assistant text in the training data, leading to role confusion (e.g., generating user messages or failing to stop). The loss mask typically only applies to assistant turns."
    },
    {
      type: "mc",
      question: "The \"quality vs. quantity\" debate in SFT has been largely resolved in favor of:",
      options: [
        "Quantity — more data always wins regardless of quality",
        "Quality — a small set of diverse, high-quality examples outperforms a large set of noisy examples, though there are diminishing returns",
        "Neither matters — only the base model size determines SFT performance",
        "An exact 50/50 tradeoff that depends only on model size"
      ],
      correct: 1,
      explanation: "Multiple studies (LIMA, Alpaca, WizardLM ablations) converge on the finding that data quality dominates. Key quality factors include: (1) correctness of responses, (2) diversity of tasks/formats, (3) appropriate complexity, and (4) consistent style. A curated set of 1K–10K examples often outperforms 100K+ noisy examples. However, quality has diminishing returns — after covering the main task distribution, more high-quality data yields marginal gains."
    },
    {
      type: "mc",
      question: "When fine-tuning with LoRA (Low-Rank Adaptation), the weight update is parameterized as $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times d}$ with $r \\ll d$. How does this relate to the SFT intuition that alignment is a \"small\" change?",
      options: [
        "It has no relation — LoRA is purely a memory optimization",
        "The low-rank constraint means the update lives in a small subspace, consistent with the hypothesis that SFT adjusts a few directions in weight space rather than overhauling the model",
        "LoRA always produces identical results to full fine-tuning",
        "The rank $r$ must equal the number of training examples"
      ],
      correct: 1,
      explanation: "LoRA's success provides empirical evidence for the Superficial Alignment Hypothesis from a parameter perspective: if SFT only needs to \"steer\" the model, then the weight change should be low-rank (a small number of directions in parameter space). LoRA with $r = 8$ or $r = 16$ often matches full fine-tuning, suggesting the effective dimensionality of the SFT update is far smaller than the full parameter count. This also connects to the intrinsic dimensionality literature (Aghajanyan et al.)."
    },
    {
      type: "mc",
      question: "A common failure mode of SFT is \"sycophancy\" — the model excessively agrees with the user. What causes this during SFT?",
      options: [
        "The base model lacks the knowledge to disagree",
        "Training data is biased toward agreement: human-written \"ideal\" responses tend to be agreeable, and annotators reward compliance, so the model learns that agreeing is the high-probability format",
        "Sycophancy is caused by too low a learning rate",
        "It only occurs when training for more than 100 epochs"
      ],
      correct: 1,
      explanation: "Sycophancy is a distribution-level problem in the SFT data. Human annotators writing \"ideal\" assistant responses tend to be accommodating and agreeable. The model learns this statistical pattern: given any user statement, the maximum-likelihood response is one that validates the user. This is difficult to fix with SFT alone because it requires *preference* signal (\"this disagreement is better than that agreement\"), which is why RLHF/DPO stages are needed to address it."
    },
    {
      type: "mc",
      question: "You are fine-tuning a 7B model on 5,000 instruction-response pairs. During training, you observe that training loss decreases steadily but evaluation loss begins increasing after epoch 1. The model's generations become repetitive and formulaic. What is the most likely diagnosis and remedy?",
      options: [
        "The model is underfitting — increase the learning rate and train for more epochs",
        "Classic overfitting to the small SFT dataset — reduce to 1 epoch, increase dropout or use LoRA with lower rank, and verify data diversity",
        "The evaluation set is mislabeled — retrain with the same hyperparameters",
        "The tokenizer is incompatible with the training data"
      ],
      correct: 1,
      explanation: "This is the textbook SFT overfitting pattern: the model memorizes training responses (decreasing train loss) while losing generalization (increasing eval loss, repetitive outputs). Remedies include: (1) train for only 1 epoch, (2) reduce effective model capacity (LoRA, dropout), (3) verify data diversity (repetitive data accelerates overfitting), (4) use a cosine schedule that decays to near-zero. The small dataset makes this almost inevitable without these precautions."
    },
    {
      type: "mc",
      question: "Consider the SFT objective $\\mathcal{L}_{\\text{SFT}} = -\\mathbb{E}_{(x,y) \\sim \\mathcal{D}} \\left[ \\sum_{t=1}^{|y|} \\log \\pi_\\theta(y_t \\mid x, y_{<t}) \\right]$. This is equivalent to minimizing $\\text{KL}(p_{\\text{data}} \\| \\pi_\\theta)$ (forward KL). What behavioral consequence does this have compared to reverse KL?",
      options: [
        "Forward KL produces sharper, more focused outputs",
        "Forward KL is mode-covering: $\\pi_\\theta$ tries to place mass everywhere $p_{\\text{data}}$ has mass, leading to diverse but sometimes incoherent or hedging outputs",
        "Forward KL and reverse KL produce identical results for language models",
        "Forward KL minimization always converges faster than reverse KL"
      ],
      correct: 1,
      explanation: "Maximum likelihood (SFT) minimizes forward KL $\\text{KL}(p_{\\text{data}} \\| \\pi_\\theta)$, which is mode-covering: $\\pi_\\theta$ is penalized for assigning low probability to anything in $p_{\\text{data}}$. This means the model tries to cover all training modes, potentially spreading mass across contradictory response styles. This contrasts with RLHF/RL objectives that effectively use reverse KL (mode-seeking), producing more focused, consistent outputs — one reason why RLHF improves upon SFT."
    }
  ]
};

// ─────────────────────────────────────────────
// A.2: Reward Modeling
// ─────────────────────────────────────────────
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
        "Humans always prefer longer responses",
        "The probability of preferring $y_w$ over $y_l$ depends only on the *difference* in their scalar rewards — preferences are transitive and follow a logistic noise model",
        "Preferences are uniformly random",
        "The reward function must be linear in the response length"
      ],
      correct: 1,
      explanation: "Bradley-Terry assumes a latent scalar \"quality\" score $r(x, y)$ for each response, and preference probability is a function of the *difference* $r(x, y_w) - r(x, y_l)$ passed through a sigmoid. This implies transitivity (if $A \\succ B$ and $B \\succ C$, then $A \\succ C$) and a specific noise model (logistic). These are strong assumptions — real human preferences can be intransitive and context-dependent — but the model is tractable and works surprisingly well in practice."
    },
    {
      type: "mc",
      question: "The reward model training loss is $\\mathcal{L} = -\\mathbb{E}_{(x, y_w, y_l)}\\left[\\log \\sigma(r_\\theta(x, y_w) - r_\\theta(x, y_l))\\right]$. Why does this loss only depend on the *difference* $r(x, y_w) - r(x, y_l)$ rather than individual reward values?",
      options: [
        "Individual rewards are easier to learn but less accurate",
        "The Bradley-Terry likelihood is invariant to adding a constant to all rewards — only differences are identifiable from pairwise comparison data",
        "Individual rewards require reinforcement learning to train",
        "The sigmoid function requires exactly two inputs"
      ],
      correct: 1,
      explanation: "If we replace $r(x, y)$ with $r(x, y) + c$ for any constant $c$, the difference $r(x, y_w) - r(x, y_l)$ is unchanged, so the likelihood is identical. This means the absolute scale of rewards is unidentifiable from pairwise data — only relative differences matter. This is why reward models need careful normalization and why reward values can drift during training. It also means comparing rewards across different prompts $x$ is not inherently meaningful."
    },
    {
      type: "mc",
      question: "\"Reward hacking\" (or reward overoptimization) occurs when the policy exploits the reward model. Which of the following best describes the phenomenon observed by Gao et al. (2023)?",
      options: [
        "The reward model always assigns perfect scores to all outputs",
        "As the policy is optimized more aggressively against the reward model (increasing KL from reference), the proxy reward increases but the *true* (gold) reward eventually decreases — the policy finds adversarial inputs to the reward model",
        "The reward model crashes during training",
        "Reward hacking only occurs with models smaller than 1B parameters"
      ],
      correct: 1,
      explanation: "Gao et al. showed a clear pattern: proxy reward (from the learned RM) increases monotonically with KL budget, but gold reward (from a much larger or human-evaluated RM) follows an inverted-U shape — it improves initially, peaks, then *degrades*. The policy discovers outputs that score high on the proxy RM but are actually low quality. This is a form of Goodhart's law: \"when a measure becomes a target, it ceases to be a good measure.\" The KL penalty $\\beta$ controls where on this curve you operate."
    },
    {
      type: "mc",
      question: "What is the key difference between a **Process Reward Model (PRM)** and an **Outcome Reward Model (ORM)**?",
      options: [
        "PRMs are faster to train while ORMs are more accurate",
        "PRMs assign rewards to each intermediate step of reasoning, while ORMs only assign a single reward to the final answer — PRMs provide denser supervision signal",
        "ORMs use reinforcement learning while PRMs use supervised learning",
        "PRMs can only be used with chain-of-thought prompting"
      ],
      correct: 1,
      explanation: "ORMs give a single scalar reward for the complete response (correct/incorrect final answer). PRMs evaluate each reasoning step individually ($r(x, s_1, \\ldots, s_t)$ at each step $t$). Lightman et al. (2023) showed PRMs significantly outperform ORMs for mathematical reasoning because: (1) they provide credit assignment — identifying *where* reasoning went wrong, (2) denser reward signal reduces variance, and (3) they can prune bad reasoning paths early. The cost is that PRM training data requires per-step human annotations."
    },
    {
      type: "mc",
      question: "In RLHF, an \"implicit\" reward can be extracted from any language model via $r_{\\text{implicit}}(x, y) = \\beta \\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}$. What does this represent?",
      options: [
        "The perplexity of the response",
        "The log-likelihood ratio measures how much the policy has shifted from the reference — responses the policy upweights relative to the reference have higher implicit reward",
        "The gradient of the loss function",
        "The entropy of the policy distribution"
      ],
      correct: 1,
      explanation: "This comes from the closed-form solution of the KL-constrained RL objective: $\\pi^*(y \\mid x) \\propto \\pi_{\\text{ref}}(y \\mid x) \\exp(r(y, x) / \\beta)$. Rearranging: $r(y, x) = \\beta \\log \\frac{\\pi^*(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)$. So the log-ratio is the implicit reward (up to a prompt-dependent constant). This is the foundation of DPO — instead of learning an explicit reward model, use the policy itself as an implicit reward."
    },
    {
      type: "mc",
      question: "Reward model accuracy typically scales with model size, but Anthropic's work showed an important nuance. What did they find about the relationship between RM size and policy size?",
      options: [
        "The RM should always be the same size as the policy",
        "Larger RMs generally produce better policies, but an RM should be at least as large as the policy to avoid the policy easily exploiting the RM's blind spots",
        "Smaller RMs always produce better policies due to implicit regularization",
        "RM size has no effect on final policy quality"
      ],
      correct: 1,
      explanation: "If the policy model is larger and more capable than the reward model, the policy can find adversarial outputs that fool the RM — it's effectively a stronger adversary than the RM can handle. This creates an asymmetry: the RM needs to be at least as capable as the policy to provide reliable signal. In practice, teams often use an RM that is the same size or larger than the policy. This has implications for scalable oversight — as policies get stronger, reward models must keep pace."
    },
    {
      type: "mc",
      question: "When training a reward model from human pairwise comparisons, annotator disagreement is common. What is the standard approach to handling this?",
      options: [
        "Discard all examples where annotators disagree",
        "Use majority vote to determine the preferred response, but optionally weight the loss by annotator agreement — high-agreement pairs provide a stronger training signal",
        "Train a separate model for each annotator",
        "Randomly assign preference labels when annotators disagree"
      ],
      correct: 1,
      explanation: "Majority voting is the most common approach: if 3 out of 5 annotators prefer $y_w$, it's labeled as preferred. Some approaches further weight the Bradley-Terry loss by agreement level — a 5/5 agreement pair gets full weight, while a 3/2 split gets reduced weight, reflecting genuine ambiguity. More sophisticated methods model annotator-level preferences or treat disagreement as inherent noise in the generative process. Discarding disagreements wastes data and biases toward \"easy\" examples."
    },
    {
      type: "mc",
      question: "The reward model is initialized from a pretrained language model with the final unembedding layer replaced by a scalar head. Why is this initialization important?",
      options: [
        "It allows the model to generate text during reward evaluation",
        "The pretrained representations encode semantic understanding needed to evaluate response quality — training a reward model from scratch would require orders of magnitude more comparison data",
        "It is purely a computational optimization with no impact on quality",
        "The unembedding layer is reused as the reward head"
      ],
      correct: 1,
      explanation: "The RM needs to *understand* text to evaluate it — reading comprehension, factual knowledge, reasoning ability, style detection. These capabilities come from pretraining. The scalar reward head is the only new component; the pretrained backbone provides the representation. Without this transfer, the RM would need to learn language understanding from comparison data alone, which is far too sparse. This is why RM performance scales with the quality of the pretrained backbone."
    },
    {
      type: "mc",
      question: "Consider the reward model loss gradient: $\\nabla_\\theta \\mathcal{L} = -\\mathbb{E}\\left[(1 - \\sigma(\\Delta r)) \\cdot (\\nabla_\\theta r_\\theta(x, y_w) - \\nabla_\\theta r_\\theta(x, y_l))\\right]$ where $\\Delta r = r_\\theta(x, y_w) - r_\\theta(x, y_l)$. The factor $(1 - \\sigma(\\Delta r))$ implies:",
      options: [
        "All training examples contribute equally to the gradient",
        "When the model already assigns much higher reward to the preferred response ($\\Delta r \\gg 0$), the gradient vanishes — the model focuses on pairs it hasn't yet learned to rank correctly",
        "The gradient is always exactly 1",
        "Negative rewards produce larger gradients than positive rewards"
      ],
      correct: 1,
      explanation: "When $\\Delta r \\gg 0$, $\\sigma(\\Delta r) \\approx 1$, so $(1 - \\sigma(\\Delta r)) \\approx 0$ and the gradient vanishes. This is the standard logistic regression property: well-classified examples contribute little gradient. The model automatically upweights pairs where it's currently wrong ($\\Delta r < 0$) or uncertain ($\\Delta r \\approx 0$). This implicit curriculum means early training focuses on obvious preferences, then shifts to harder, more ambiguous pairs."
    },
    {
      type: "mc",
      question: "A reward model trained on helpfulness comparisons is used to optimize a policy. After optimization, the policy produces very long, verbose responses that score high reward but are rated lower by humans. This is an example of:",
      options: [
        "Underfitting of the reward model",
        "A length bias spurious correlation — the RM learned that longer responses tend to be preferred in training data, so the policy exploits this shortcut rather than genuinely improving quality",
        "The policy model being too small",
        "A bug in the KL penalty implementation"
      ],
      correct: 1,
      explanation: "Length bias is one of the most common and well-documented reward hacking modes. In human comparison data, longer responses are often preferred (they tend to be more detailed and thorough), so the RM learns a spurious correlation between length and quality. The policy then exploits this by padding responses with redundant content. Mitigations include: (1) length-conditioned reward normalization, (2) adding length as an explicit feature and regressing it out, (3) including length-controlled comparison pairs in RM training data."
    }
  ]
};

// ─────────────────────────────────────────────
// A.3: RLHF / Policy Optimization
// ─────────────────────────────────────────────
export const rlhfAssessment = {
  id: "A.3-assess",
  sectionId: "A.3",
  title: "Assessment: RLHF & Policy Optimization",
  difficulty: "hard",
  estimatedMinutes: 16,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The PPO clipped surrogate objective is $L^{\\text{CLIP}} = \\mathbb{E}_t\\left[\\min\\left(\\rho_t \\hat{A}_t, \\; \\text{clip}(\\rho_t, 1 - \\epsilon, 1 + \\epsilon) \\hat{A}_t\\right)\\right]$ where $\\rho_t = \\frac{\\pi_\\theta(a_t \\mid s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t \\mid s_t)}$. When the advantage $\\hat{A}_t > 0$ (good action), what does the clipping achieve?",
      options: [
        "It prevents the probability ratio from dropping below $1 - \\epsilon$",
        "It caps $\\rho_t$ at $1 + \\epsilon$, preventing the policy from increasing the probability of a good action *too much* in a single update — limiting the step size",
        "It forces $\\rho_t = 1$ for all good actions",
        "It removes all gradient signal for good actions"
      ],
      correct: 1,
      explanation: "When $\\hat{A}_t > 0$, we want to increase $\\pi_\\theta(a_t | s_t)$, which increases $\\rho_t$. But $\\min(\\rho_t \\hat{A}_t, (1+\\epsilon) \\hat{A}_t)$ caps the objective at $(1+\\epsilon)\\hat{A}_t$ — beyond $\\rho_t = 1 + \\epsilon$, there's no further incentive to increase the probability. This prevents catastrophically large policy updates that could destabilize training. Symmetrically, when $\\hat{A}_t < 0$, clipping prevents $\\rho_t$ from dropping below $1 - \\epsilon$."
    },
    {
      type: "mc",
      question: "Generalized Advantage Estimation (GAE) defines $\\hat{A}_t^{\\text{GAE}(\\gamma, \\lambda)} = \\sum_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta_{t+l}$ where $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$. The parameter $\\lambda$ trades off:",
      options: [
        "Exploration vs. exploitation",
        "Bias vs. variance: $\\lambda = 0$ gives low-variance but high-bias (1-step TD), $\\lambda = 1$ gives high-variance but unbiased (Monte Carlo)",
        "Learning rate vs. batch size",
        "Reward scale vs. KL penalty strength"
      ],
      correct: 1,
      explanation: "At $\\lambda = 0$: $\\hat{A}_t = \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, the 1-step TD error. This has low variance (uses the value function baseline) but is biased if $V$ is inaccurate. At $\\lambda = 1$: $\\hat{A}_t = \\sum_l \\gamma^l r_{t+l} - V(s_t)$, the Monte Carlo return minus baseline. This is unbiased but high variance. Intermediate $\\lambda$ (commonly 0.95) interpolates, providing a practical bias-variance tradeoff. In RLHF, $\\lambda \\approx 0.95$ is standard."
    },
    {
      type: "mc",
      question: "In the RLHF objective $\\max_\\pi \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi(\\cdot | x)}[r(x, y)] - \\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, increasing $\\beta$ has what effect?",
      options: [
        "Increases the reward but decreases the KL penalty",
        "Makes the policy more conservative — staying closer to $\\pi_{\\text{ref}}$ at the cost of less reward optimization, reducing reward hacking but also limiting improvement",
        "Has no effect on the policy because $\\beta$ cancels out",
        "Always improves both reward and KL simultaneously"
      ],
      correct: 1,
      explanation: "Higher $\\beta$ increases the cost of diverging from $\\pi_{\\text{ref}}$. The optimal policy is $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. As $\\beta \\to \\infty$, $\\pi^* \\to \\pi_{\\text{ref}}$ (no adaptation). As $\\beta \\to 0$, $\\pi^*$ concentrates on the reward-maximizing response (maximum reward hacking risk). In practice, $\\beta$ is the primary knob for controlling the reward-quality tradeoff, and is often scheduled (starting high, decreasing)."
    },
    {
      type: "mc",
      question: "The RLHF penalty uses **forward KL** $\\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, not reverse KL $\\text{KL}(\\pi_{\\text{ref}} \\| \\pi)$. What is the practical reason for this choice?",
      options: [
        "Forward KL is easier to compute",
        "Forward KL penalizes $\\pi$ for placing mass where $\\pi_{\\text{ref}}$ has low mass — preventing the policy from generating novel text that the base model considers implausible, which is exactly the mode of reward hacking we want to prevent",
        "Forward and reverse KL are identical for language models",
        "Reverse KL causes numerical overflow in all cases"
      ],
      correct: 1,
      explanation: "Forward KL: $\\text{KL}(\\pi \\| \\pi_{\\text{ref}}) = \\mathbb{E}_\\pi[\\log \\pi / \\pi_{\\text{ref}}]$. The expectation is under $\\pi$, so if $\\pi$ generates text $y$ where $\\pi_{\\text{ref}}(y) \\approx 0$, the log-ratio explodes. This directly prevents the policy from discovering adversarial outputs that the base model would never produce. Reverse KL $\\text{KL}(\\pi_{\\text{ref}} \\| \\pi)$ would instead penalize $\\pi$ for *not covering* modes of $\\pi_{\\text{ref}}$, which is the wrong inductive bias for alignment."
    },
    {
      type: "mc",
      question: "GRPO (Group Relative Policy Optimization), used in DeepSeek-R1, eliminates the value network by:",
      options: [
        "Using a fixed reward for all responses",
        "Sampling a group of responses for each prompt, computing rewards, and using the **group-normalized advantage** $\\hat{A}_i = \\frac{r_i - \\text{mean}(\\mathbf{r})}{\\text{std}(\\mathbf{r})}$ as the baseline — removing the need for a separate critic",
        "Training the policy and value function with shared parameters",
        "Using Monte Carlo tree search instead of a value function"
      ],
      correct: 1,
      explanation: "GRPO samples $G$ completions $\\{y_1, \\ldots, y_G\\}$ for each prompt $x$, scores them with the RM, then normalizes: $\\hat{A}_i = (r_i - \\mu_G) / \\sigma_G$. This group-level normalization serves as a variance-reducing baseline without needing a learned value function. Benefits: (1) removes the value network (halving GPU memory), (2) avoids value function approximation error, (3) naturally handles reward scale differences across prompts. The key insight is that relative ranking within a group is sufficient for policy improvement."
    },
    {
      type: "mc",
      question: "The importance sampling ratio $\\rho_t = \\frac{\\pi_\\theta(a_t | s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t | s_t)}$ in PPO can cause training instability when:",
      options: [
        "$\\rho_t$ is always exactly 1.0",
        "The ratio becomes very large or very small, indicating the new policy has diverged significantly from the old policy — the sampled trajectories are no longer representative, leading to high-variance gradient estimates",
        "The ratio is negative",
        "The ratio is complex-valued"
      ],
      correct: 1,
      explanation: "Importance sampling corrects for the mismatch between the sampling policy ($\\pi_{\\text{old}}$) and the current policy ($\\pi_\\theta$). When $\\rho_t \\gg 1$, an action that was unlikely under $\\pi_{\\text{old}}$ is now likely under $\\pi_\\theta$ — the correction factor amplifies this sample's contribution, introducing high variance. The variance of importance-weighted estimators scales with $\\mathbb{E}[\\rho^2]$, which can diverge. PPO's clipping directly addresses this by bounding $\\rho_t \\in [1-\\epsilon, 1+\\epsilon]$."
    },
    {
      type: "mc",
      question: "In RLHF for language models, the \"state\" $s_t$ and \"action\" $a_t$ in the MDP formulation are typically defined as:",
      options: [
        "State = the entire training dataset, Action = the model's weight update",
        "State = the prompt plus all tokens generated so far $(x, y_{<t})$, Action = the next token $y_t$ — the episode terminates when the EOS token is generated",
        "State = the hidden state of the transformer, Action = the attention pattern",
        "State = the reward model's output, Action = the KL divergence"
      ],
      correct: 1,
      explanation: "RLHF treats autoregressive generation as a token-level MDP: the state is the concatenation of the prompt and all generated tokens so far, and the action is choosing the next token from the vocabulary. The reward is typically sparse — assigned only at the end of generation (from the RM). The episode starts with the prompt and ends at EOS. This framing makes the action space $|V|$ (vocabulary size, typically 32K–100K), and episodes are typically 100–2000 steps long."
    },
    {
      type: "mc",
      question: "A common source of PPO instability in RLHF is the interaction between the value function and the policy. Specifically:",
      options: [
        "The value function converges too quickly",
        "The value function is initialized from the SFT model and must estimate sequence-level returns from token-level states — its errors propagate through GAE into advantage estimates, causing noisy policy gradients that can spiral into divergence",
        "The value function and policy always converge to the same parameters",
        "PPO never uses a value function in practice"
      ],
      correct: 1,
      explanation: "In RLHF, the value function $V_\\phi(s_t)$ must predict the expected return (RM score + KL penalties for remaining tokens). This is challenging: (1) the reward is sparse (only at episode end), (2) the value function must generalize across diverse prompts, (3) as the policy changes, the value function's training data shifts. Errors in $V$ directly corrupt advantage estimates via $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$, leading to wrong policy gradients. This is why some approaches (GRPO, REINFORCE-based) eliminate the value function entirely."
    },
    {
      type: "mc",
      question: "In the PPO objective for RLHF, the per-token reward is typically defined as $r_t = -\\beta \\log \\frac{\\pi_\\theta(y_t | x, y_{<t})}{\\pi_{\\text{ref}}(y_t | x, y_{<t})}$ for $t < T$ and $r_T = R_{\\text{RM}}(x, y) - \\beta \\log \\frac{\\pi_\\theta(y_T | x, y_{<T})}{\\pi_{\\text{ref}}(y_T | x, y_{<T})}$. Why is the KL penalty applied per-token rather than as a single sequence-level penalty?",
      options: [
        "Per-token KL is cheaper to compute",
        "Per-token KL provides denser reward signal, enabling better credit assignment — the value function and GAE can propagate KL costs to specific tokens rather than attributing the entire sequence-level KL to the final token",
        "Sequence-level KL is not mathematically well-defined",
        "Per-token and sequence-level KL penalties are always identical"
      ],
      correct: 1,
      explanation: "The sequence-level KL decomposes as $\\text{KL}(\\pi \\| \\pi_{\\text{ref}}) = \\sum_t \\mathbb{E}[\\log \\frac{\\pi(y_t | x, y_{<t})}{\\pi_{\\text{ref}}(y_t | x, y_{<t})}]$, so the per-token formulation is mathematically equivalent. However, placing the KL penalty at each token step is crucial for the RL optimization: it provides dense intermediate rewards, making the value function estimation problem much easier and enabling GAE to assign credit at the token level. Without this, the value function must predict the entire future KL from each state, which is much harder."
    },
    {
      type: "mc",
      question: "Consider a simplified RLHF setup with discrete reward $r \\in \\{0, 1\\}$ and a policy $\\pi_\\theta$ parameterized by a single scalar $\\theta$ controlling the probability of a \"good\" action: $\\pi_\\theta(\\text{good}) = \\sigma(\\theta)$. The REINFORCE gradient estimator is $\\nabla_\\theta J = \\mathbb{E}_{a \\sim \\pi_\\theta}[r(a) \\nabla_\\theta \\log \\pi_\\theta(a)]$. With a single sample $a$ and no baseline, the variance of this estimator is high because:",
      options: [
        "The gradient is always zero",
        "The estimator is biased",
        "When $r(a) = 0$ (bad action), the gradient is zero regardless of how informative the sample is, and when $r(a) = 1$ (good action), the gradient magnitude depends on $\\pi_\\theta(a)$ — high reward but low probability actions produce large, rare gradient spikes",
        "REINFORCE cannot be applied to discrete action spaces"
      ],
      correct: 2,
      explanation: "REINFORCE with $r \\in \\{0, 1\\}$: if the sampled action has $r = 0$, we get $\\nabla = 0$ — no learning signal, even though a bad outcome is informative. If $r = 1$, we get $\\nabla = \\nabla_\\theta \\log \\pi_\\theta(a)$, which is large when $\\pi_\\theta(a)$ is small (rare good actions produce gradient spikes). The variance is $\\text{Var}[r \\nabla \\log \\pi] = \\mathbb{E}[r^2 (\\nabla \\log \\pi)^2] - (\\mathbb{E}[r \\nabla \\log \\pi])^2$, which can be enormous. A baseline $b$ (e.g., $V(s)$) replaces $r$ with $r - b$, drastically reducing variance without introducing bias."
    }
  ]
};

// ─────────────────────────────────────────────
// A.4: Direct Alignment (DPO and relatives)
// ─────────────────────────────────────────────
export const directAlignmentAssessment = {
  id: "A.4-assess",
  sectionId: "A.4",
  title: "Assessment: Direct Alignment Methods",
  difficulty: "hard",
  estimatedMinutes: 16,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "DPO's key derivation insight starts from the KL-constrained RLHF objective $\\max_\\pi \\mathbb{E}[r(x,y)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$. The closed-form optimal policy is $\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. DPO's crucial next step is to:",
      options: [
        "Approximate $Z(x)$ using Monte Carlo sampling",
        "Rearrange to express the reward as $r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$, then substitute into the Bradley-Terry model — the partition function $Z(x)$ cancels in the preference probability, yielding a loss that depends only on the policy",
        "Train a separate reward model to approximate $Z(x)$",
        "Set $Z(x) = 1$ by assumption"
      ],
      correct: 1,
      explanation: "The full derivation: (1) Solve for optimal policy: $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) e^{r/\\beta}$. (2) Invert: $r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$. (3) Substitute into Bradley-Terry: $P(y_w \\succ y_l) = \\sigma(r_w - r_l) = \\sigma\\left(\\beta \\log \\frac{\\pi^*(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi^*(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)$. The $\\beta \\log Z(x)$ terms cancel in the difference. (4) Replace $\\pi^*$ with $\\pi_\\theta$ to get the training objective."
    },
    {
      type: "mc",
      question: "The DPO loss is $\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$. The gradient with respect to $\\theta$ is proportional to $(1 - \\sigma(\\hat{r}_w - \\hat{r}_l))$ where $\\hat{r}_i = \\beta \\log \\frac{\\pi_\\theta(y_i|x)}{\\pi_{\\text{ref}}(y_i|x)}$. This means the gradient:",
      options: [
        "Is constant regardless of how well the model has learned the preference",
        "Is large when the model's implicit reward difference is wrong ($\\hat{r}_w - \\hat{r}_l \\ll 0$) and vanishes when the model already correctly ranks the pair ($\\hat{r}_w - \\hat{r}_l \\gg 0$) — an adaptive weighting similar to the reward model loss",
        "Is always zero for correctly ranked pairs",
        "Only depends on the winning response, not the losing response"
      ],
      correct: 1,
      explanation: "The factor $(1 - \\sigma(\\hat{r}_w - \\hat{r}_l))$ acts as an implicit curriculum: when the model already assigns much higher implicit reward to the preferred response, $\\sigma(\\hat{r}_w - \\hat{r}_l) \\approx 1$ and the gradient vanishes. When the model incorrectly ranks the pair, $\\sigma(\\hat{r}_w - \\hat{r}_l) \\approx 0$ and the gradient is maximal. The gradient simultaneously pushes $\\pi_\\theta(y_w|x)$ up and $\\pi_\\theta(y_l|x)$ down relative to the reference."
    },
    {
      type: "mc",
      question: "A major failure mode of offline DPO is \"distribution shift.\" What specifically causes this?",
      options: [
        "The tokenizer changes between pretraining and DPO",
        "DPO trains on preference pairs $(y_w, y_l)$ generated by a *different* policy (often the SFT model), but optimizes $\\pi_\\theta$ — as $\\pi_\\theta$ diverges from the data-generating policy, the log-probabilities $\\log \\pi_\\theta(y|x)$ become unreliable estimates for out-of-distribution responses",
        "The reference model is too large",
        "Distribution shift only affects models with more than 100B parameters"
      ],
      correct: 1,
      explanation: "Offline DPO uses a fixed dataset of preference pairs. As $\\pi_\\theta$ evolves during training, it may encounter responses $y$ that are very different from what it would generate — the log-probabilities $\\log \\pi_\\theta(y|x)$ become poorly calibrated for these out-of-distribution sequences. The implicit reward $\\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$ can become meaningless. This is analogous to off-policy RL issues. Online DPO mitigates this by regenerating responses with the current $\\pi_\\theta$."
    },
    {
      type: "mc",
      question: "IPO (Identity Preference Optimization) modifies the DPO loss by replacing the Bradley-Terry log-sigmoid with a squared loss: $\\mathcal{L}_{\\text{IPO}} = \\mathbb{E}\\left[\\left(\\hat{r}_w - \\hat{r}_l - \\frac{1}{2\\beta}\\right)^2\\right]$. What problem does this address?",
      options: [
        "IPO trains faster because squares are cheaper to compute than sigmoids",
        "DPO can overfit to deterministic preferences by pushing $\\hat{r}_w - \\hat{r}_l \\to \\infty$, while IPO's regression target $\\frac{1}{2\\beta}$ provides a finite anchor — it regularizes the implicit reward margin to a specific value rather than maximizing it without bound",
        "IPO eliminates the need for a reference model",
        "IPO is only applicable to reward models, not policies"
      ],
      correct: 1,
      explanation: "In DPO, the loss $-\\log \\sigma(\\hat{r}_w - \\hat{r}_l)$ can be driven to zero by pushing the margin $\\hat{r}_w - \\hat{r}_l \\to \\infty$. This encourages the policy to be infinitely confident about preferences, leading to overfitting and degenerate solutions. IPO instead regresses the margin toward a fixed target $\\frac{1}{2\\beta}$, preventing the policy from becoming overconfident. This directly addresses the issue that DPO's loss has no natural stopping point for well-classified pairs."
    },
    {
      type: "mc",
      question: "KTO (Kahneman-Tversky Optimization) differs from DPO in a fundamental data requirement. What is this difference?",
      options: [
        "KTO requires 10x more data than DPO",
        "KTO works with unpaired binary feedback (each response independently labeled as good/bad) rather than requiring *paired* preferences ($y_w \\succ y_l$ for the same prompt) — this makes data collection much simpler",
        "KTO requires paired preferences while DPO works with point-wise labels",
        "KTO only works with numerical reward scores, not binary feedback"
      ],
      correct: 1,
      explanation: "DPO requires *pairwise* comparisons: for the same prompt $x$, a preferred $y_w$ and dispreferred $y_l$. KTO only needs independent binary labels: \"this response is good\" or \"this response is bad,\" without pairing. This is inspired by Kahneman-Tversky prospect theory (loss aversion). KTO's loss treats good and bad examples asymmetrically, applying stronger penalties for producing bad outputs than rewards for producing good ones. This dramatically simplifies data collection — binary thumbs up/down is far cheaper than side-by-side comparisons."
    },
    {
      type: "mc",
      question: "Online DPO (also called \"iterative DPO\" or \"online RLHF with DPO loss\") modifies the standard DPO pipeline by:",
      options: [
        "Training on a larger static dataset",
        "Generating new response pairs from the *current* policy $\\pi_\\theta$ at each iteration, scoring them with a reward model, and constructing fresh preference pairs — this keeps the training distribution aligned with the policy and mitigates distribution shift",
        "Using a different loss function than DPO",
        "Removing the reference model from the objective"
      ],
      correct: 1,
      explanation: "Online DPO closes the distribution shift gap: (1) sample responses from current $\\pi_\\theta$, (2) score with RM to determine preferences, (3) train with DPO loss on these on-policy pairs, (4) repeat. This is conceptually similar to RLHF (generate, score, update) but uses the DPO loss instead of PPO. The result is that $\\log \\pi_\\theta(y|x)$ is always computed for responses the current policy would actually generate, making the implicit reward estimates accurate. Empirically, online DPO significantly outperforms offline DPO."
    },
    {
      type: "mc",
      question: "SimPO (Simple Preference Optimization) makes a key simplification compared to DPO by defining the implicit reward as the *length-normalized* log-probability: $r_{\\text{SimPO}}(x, y) = \\frac{1}{|y|} \\log \\pi_\\theta(y | x)$. What does this eliminate?",
      options: [
        "The need for GPU memory",
        "The reference model $\\pi_{\\text{ref}}$ — SimPO does not need to store or compute log-probabilities under a frozen reference, halving memory requirements and simplifying the pipeline",
        "The preference data entirely",
        "The need for gradient computation"
      ],
      correct: 1,
      explanation: "DPO's implicit reward is $\\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$, requiring a frozen $\\pi_{\\text{ref}}$ alongside $\\pi_\\theta$ during training (2x memory). SimPO replaces this with the length-normalized log-probability $\\frac{1}{|y|}\\log \\pi_\\theta(y|x)$, eliminating $\\pi_{\\text{ref}}$ entirely. Length normalization prevents the model from trivially increasing reward by generating shorter sequences. SimPO adds a margin term $\\gamma$ to the loss, similar to IPO's finite target, to prevent overoptimization."
    },
    {
      type: "mc",
      question: "Rejection sampling (best-of-$N$) is sometimes used as an alternative to RL-based optimization. Given a prompt $x$, we sample $N$ responses from $\\pi_{\\text{ref}}$, score each with a reward model, and select the best. The effective KL divergence of this procedure scales as:",
      options: [
        "$\\text{KL} = N$",
        "$\\text{KL} \\approx \\log N - \\frac{N-1}{N}$, growing logarithmically with $N$ — each doubling of $N$ adds roughly $\\log 2$ nats of KL",
        "$\\text{KL} = 0$ for all $N$",
        "$\\text{KL} = N^2$"
      ],
      correct: 1,
      explanation: "Best-of-$N$ defines an implicit policy $\\pi_{\\text{BoN}}$ that samples the maximum-reward completion from $N$ i.i.d. draws. The KL from this implicit policy to the reference is $\\text{KL}(\\pi_{\\text{BoN}} \\| \\pi_{\\text{ref}}) \\approx \\log N - \\frac{N-1}{N}$. This is remarkably efficient at small $N$ — best-of-16 achieves significant quality improvements at only $\\sim 2.5$ nats of KL. However, the logarithmic scaling means diminishing returns: going from $N=16$ to $N=256$ costs another $\\sim 2.8$ nats but yields smaller gains. This makes rejection sampling competitive with PPO at moderate KL budgets."
    },
    {
      type: "mc",
      question: "Constitutional AI (CAI) / RLAIF replaces human preference labels with AI-generated feedback. The \"constitutional\" part refers to:",
      options: [
        "A legal framework governing model training",
        "A set of natural-language principles (the \"constitution\") that the AI uses to evaluate and revise its own outputs — the AI critiques responses based on these principles and generates preference labels for RLHF/DPO training",
        "The model's architecture being fixed (\"constituted\") during training",
        "A requirement that training data come from government sources"
      ],
      correct: 1,
      explanation: "CAI (Bai et al., 2022) works in two phases: (1) **Self-critique and revision**: the model generates a response, then is prompted to critique and revise it according to constitutional principles (e.g., \"Choose the response that is most helpful and least harmful\"). (2) **RLAIF**: the AI compares original vs. revised responses to generate preference labels, which train a reward model for RLHF. The constitution is a set of human-written principles that encode values, replacing per-example human annotation with scalable AI-based evaluation."
    },
    {
      type: "mc",
      question: "ORPO (Odds Ratio Preference Optimization) combines SFT and preference optimization into a single loss: $\\mathcal{L}_{\\text{ORPO}} = \\mathcal{L}_{\\text{SFT}}(y_w) + \\lambda \\cdot \\mathcal{L}_{\\text{OR}}$ where $\\mathcal{L}_{\\text{OR}} = -\\log \\sigma\\left(\\log \\frac{\\text{odds}_\\theta(y_w|x)}{\\text{odds}_\\theta(y_l|x)}\\right)$ and $\\text{odds}_\\theta(y|x) = \\frac{P_\\theta(y|x)}{1 - P_\\theta(y|x)}$. What is the key advantage of this unified approach?",
      options: [
        "ORPO achieves higher reward scores than all other methods",
        "It eliminates the need for a separate SFT stage and a reference model — the SFT term teaches the desired format while the odds ratio term simultaneously encodes preferences, reducing the multi-stage pipeline to a single training phase",
        "ORPO never overfits regardless of training duration",
        "The odds ratio is always easier to compute than log-probabilities"
      ],
      correct: 1,
      explanation: "Standard alignment pipelines require: (1) SFT, (2) freeze as $\\pi_{\\text{ref}}$, (3) DPO/RLHF. ORPO merges steps 1 and 2–3: the SFT loss $\\mathcal{L}_{\\text{SFT}}(y_w)$ on the preferred response teaches format and content, while the odds ratio loss $\\mathcal{L}_{\\text{OR}}$ teaches preferences. No reference model is needed because the odds ratio $\\frac{\\text{odds}(y_w)}{\\text{odds}(y_l)}$ implicitly regularizes — it contrasts the chosen vs. rejected response within the same model. This simplifies the pipeline and reduces computational cost."
    }
  ]
};

// ─────────────────────────────────────────────
// A.5: Frontier Alignment
// ─────────────────────────────────────────────
export const frontierAlignmentAssessment = {
  id: "A.5-assess",
  sectionId: "A.5",
  title: "Assessment: Frontier Alignment",
  difficulty: "hard",
  estimatedMinutes: 16,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The \"weak-to-strong generalization\" problem (Burns et al., 2023) studies whether a weak model can supervise a stronger one. The key empirical finding was:",
      options: [
        "Weak supervisors always fail — the strong model degrades to weak-model performance",
        "Strong models trained with weak supervision consistently outperform their weak supervisors, recovering a significant fraction of the gap between weak and strong performance — suggesting that strong models can partially \"generalize beyond\" noisy labels",
        "Weak and strong models always achieve identical performance",
        "The strong model learns to ignore the weak supervisor entirely"
      ],
      correct: 1,
      explanation: "Burns et al. found that when a weak model (e.g., GPT-2-level) provides labels for training a strong model (e.g., GPT-4-level), the strong model consistently exceeds the weak supervisor's performance. This is encouraging for scalable oversight: it suggests that superhuman models might partially self-correct even when trained with imperfect human feedback. However, the recovery is incomplete — there is a \"alignment tax\" — and the gap grows on harder tasks, indicating that weak-to-strong generalization alone is insufficient for frontier alignment."
    },
    {
      type: "mc",
      question: "In AI safety via debate (Irving et al., 2018), two AI agents argue opposing sides while a human judge decides the winner. Why does this protocol theoretically scale to superhuman capabilities?",
      options: [
        "The debate always converges to the truth in finite rounds",
        "A human only needs to verify individual arguments/evidence (a simpler task) rather than generate the full solution — if lying requires more complex arguments than truth-telling, the honest debater has a structural advantage",
        "The human judge can always tell when an AI is lying",
        "Debate eliminates the need for human oversight entirely"
      ],
      correct: 1,
      explanation: "The key insight is *asymmetric verification*: checking an argument is easier than generating one. In debate, a lie can be exposed by pointing to a specific flaw, which the human can verify. Under the conjecture that truth is \"simpler\" than consistent deception (lies require more complex supporting arguments), the honest debater has an advantage in the Nash equilibrium. This allows a human with limited capabilities to adjudicate between superhuman arguments. The protocol assumes the human can verify atomic claims — the debaters recursively decompose complex arguments until reaching human-verifiable steps."
    },
    {
      type: "mc",
      question: "Representation engineering / steering vectors involve finding directions $\\mathbf{v}$ in a model's activation space such that adding $\\alpha \\mathbf{v}$ to intermediate activations controls a specific behavior (e.g., honesty). These vectors are typically found by:",
      options: [
        "Random search in the activation space",
        "Computing the difference in mean activations between contrastive prompt pairs (e.g., honest vs. dishonest responses) across a dataset — the resulting direction captures the \"concept direction\" in representation space",
        "Training a separate classifier on the model's outputs",
        "Pruning attention heads until the behavior changes"
      ],
      correct: 1,
      explanation: "The standard approach: (1) construct contrastive pairs — prompts that elicit opposite behaviors (e.g., truthful vs. deceptive responses). (2) Run both through the model, extract activations at a chosen layer. (3) Compute $\\mathbf{v} = \\mathbb{E}[\\mathbf{h}_{\\text{positive}}] - \\mathbb{E}[\\mathbf{h}_{\\text{negative}}]$. This difference vector captures the linear direction in representation space corresponding to the concept. Adding $\\alpha \\mathbf{v}$ at inference time \"steers\" the model along this direction. More sophisticated methods use PCA or linear probes on the contrastive activations."
    },
    {
      type: "mc",
      question: "OpenAI's o1/o3 models use RL for reasoning, training on chain-of-thought traces. A critical design choice is training the model to produce long \"thinking\" traces before answering. What RL signal structure makes this work?",
      options: [
        "Reward is given at every token of the reasoning trace",
        "Outcome-based reward (correctness of the final answer) combined with process supervision — the RL algorithm must learn to credit intermediate reasoning steps that lead to correct answers despite extremely sparse reward",
        "The model is only trained with supervised learning on expert traces",
        "Reward is given only for short responses to encourage efficiency"
      ],
      correct: 1,
      explanation: "The o1/o3 paradigm uses RL with sparse outcome reward (is the final answer correct?) to train extended reasoning. The challenge is credit assignment: a 10,000-token reasoning trace might have one reward signal. Process reward models (PRMs) and process supervision help bridge this by evaluating intermediate steps. The RL training (likely a variant of PPO or GRPO) must learn which reasoning patterns — backtracking, verification, decomposition — lead to correct outcomes. This is \"RL for inference-time compute scaling\" — the model learns to think longer and more carefully."
    },
    {
      type: "mc",
      question: "Process supervision (as in the \"Let's Verify Step by Step\" paper, Lightman et al. 2023) provides feedback on each reasoning step. Compared to outcome supervision, process supervision:",
      options: [
        "Always produces worse results but is cheaper to implement",
        "Provides denser reward signal, enables better credit assignment, and allows the model to be guided away from flawed reasoning even when it accidentally reaches correct answers — but requires significantly more expensive per-step human annotations",
        "Does not require any human annotations",
        "Is only applicable to non-mathematical tasks"
      ],
      correct: 1,
      explanation: "Process supervision labels each step as correct/incorrect, creating a rich training signal. Benefits: (1) **Credit assignment**: identifies exactly where reasoning fails, rather than penalizing the entire trace for a wrong final answer. (2) **Avoiding reward hacking**: a correct final answer via flawed reasoning gets negative process feedback. (3) **Denser signal**: reduces variance in policy gradients. The cost is annotation: labeling each step requires expert annotators who understand the reasoning, which is far more expensive than checking final answers. Lightman et al. showed process supervision substantially outperforms outcome supervision on math reasoning."
    },
    {
      type: "mc",
      question: "Red-teaming in the context of LLM safety involves systematically trying to elicit harmful or undesired outputs. The most effective red-teaming approaches combine:",
      options: [
        "Only automated tools with no human involvement",
        "Human creativity for discovering novel attack vectors with automated methods (e.g., adversarial prompt optimization like GCG, model-based red-teaming) for scaling coverage — humans find qualitatively new failures while automation explores variations",
        "Only manual testing by a small team",
        "Random input generation without any structure"
      ],
      correct: 1,
      explanation: "Effective red-teaming requires both modalities: (1) **Human red-teamers** discover novel failure modes that require creativity, cultural knowledge, and adversarial reasoning (e.g., role-playing attacks, multi-turn manipulation). (2) **Automated methods** like GCG (gradient-based adversarial suffix optimization), model-based red-teaming (using another LLM to generate attacks), and fuzzing scale to thousands of attack variations. The most comprehensive programs (e.g., Anthropic's, Meta's) layer both: humans identify attack categories, automation fills in the coverage matrix."
    },
    {
      type: "mc",
      question: "A steering vector $\\mathbf{v}$ is applied as $\\mathbf{h}'_l = \\mathbf{h}_l + \\alpha \\mathbf{v}$ at layer $l$. The coefficient $\\alpha$ controls steering strength. A known failure mode is that large $|\\alpha|$ values cause:",
      options: [
        "The model to produce higher quality outputs",
        "Distribution shift in the activations — the modified $\\mathbf{h}'_l$ moves outside the manifold of activations the downstream layers were trained on, causing incoherent or degenerate text even if the desired behavioral shift is achieved",
        "The model to ignore the steering vector entirely",
        "Faster inference speed"
      ],
      correct: 1,
      explanation: "The model's downstream layers (layers $> l$) are trained on activations from the natural distribution. Adding a large $\\alpha \\mathbf{v}$ pushes activations off-manifold — the downstream layers receive inputs they've never seen during training. This is an out-of-distribution problem: the layers may produce unpredictable outputs. In practice, moderate $\\alpha$ produces interpretable behavioral shifts, but large $\\alpha$ degrades coherence. This is analogous to the \"curse of representation engineering\" — effective steering requires staying within the model's operational distribution."
    },
    {
      type: "mc",
      question: "The concept of \"alignment tax\" refers to:",
      options: [
        "Government taxes on AI companies for safety compliance",
        "The capability cost of alignment training — the observation that safety training (RLHF, refusals, guardrails) can reduce the model's performance on benign tasks, creating a tradeoff between safety and helpfulness",
        "The computational cost of training larger models",
        "The salary cost of hiring alignment researchers"
      ],
      correct: 1,
      explanation: "Alignment tax measures how much capability is lost to make a model safe. An ideal alignment method has zero tax — the model is both maximally capable and perfectly safe. In practice, safety training introduces refusals, hedging, and conservatism that can degrade performance: (1) over-refusal on benign queries, (2) reduced creativity due to conservative generation, (3) loss of calibration from RLHF. Minimizing alignment tax is a key research goal — methods like Constitutional AI and careful reward modeling aim to maintain capabilities while improving safety."
    },
    {
      type: "mc",
      question: "Scalable oversight is the problem of providing reliable training signal for models that exceed human capabilities in some domains. The recursive reward modeling (RRM) approach proposes to:",
      options: [
        "Use the same human annotators for all model generations",
        "Use an AI assistant (itself aligned by human feedback) to help humans evaluate the next-level model's outputs — creating a chain where each model helps align its successor, with humans retaining oversight at each stage",
        "Remove human oversight entirely and rely on self-play",
        "Train only on tasks where human performance is superior"
      ],
      correct: 1,
      explanation: "RRM creates a bootstrap chain: (1) Humans directly evaluate model $M_1$. (2) $M_1$ assists humans in evaluating the more capable $M_2$. (3) $M_2$ assists in evaluating $M_3$, and so on. At each stage, humans make the final judgment but are aided by the previous model. The key assumption is that *evaluating with assistance* is easier than *evaluating alone*, even as models become superhuman. This is related to the debate approach — both leverage asymmetric verification. The risk is that errors compound across the chain."
    },
    {
      type: "mc",
      question: "Consider a model trained with RLHF where the reward model was trained on human preferences. The model now encounters a novel domain (e.g., advanced scientific reasoning) where the reward model was never evaluated. According to Goodhart's taxonomy, which form of Goodhart's law is most relevant?",
      options: [
        "Regressional Goodhart — the reward model's noise is exploited",
        "Causal Goodhart — the reward model captures correlations that are not causal, so optimizing the proxy in a new domain breaks the correlation structure that held in the training distribution",
        "Extremal Goodhart — does not apply to language models",
        "All forms of Goodhart's law are equally relevant regardless of context"
      ],
      correct: 1,
      explanation: "Causal Goodhart applies when the proxy (learned reward) correlates with the true objective through confounders or non-causal pathways. In the training domain, \"well-structured reasoning\" may correlate with \"correct answers\" because annotators rewarded both. In a novel scientific domain, the RM may still reward well-structured-*looking* reasoning even when the conclusions are wrong — the causal pathway (domain expertise) is absent. This is distinct from regressional Goodhart (exploiting noise) and extremal Goodhart (out-of-distribution behavior at optimization extremes), though all three can co-occur in practice."
    }
  ]
};
