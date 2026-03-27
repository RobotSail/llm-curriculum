// Section A.1: Supervised Fine-Tuning Assessment

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
      options: ["SFT primarily *unlocks* capabilities already learned during pretraining rather than *teaching* new knowledge", "Pretraining is unnecessary if the fine-tuning data is of sufficiently high quality and domain coverage", "1,000 examples is always the compute-optimal dataset size for SFT regardless of the target task distribution", "Data quality does not matter as long as the dataset is kept small enough to avoid catastrophic forgetting"],
      correct: 0,
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
      options: ["10–20 epochs with aggressive learning rates and periodic restarts", "Exactly 5 epochs regardless of dataset size or domain complexity", "100+ epochs with careful early stopping on a held-out set", "1–2 epochs, with quality degrading rapidly beyond that point"],
      correct: 3,
      explanation: "SFT datasets are typically small (thousands to tens of thousands of examples), so the model memorizes them quickly. After 1–2 epochs, the model begins overfitting to surface patterns — repeating exact phrasings, losing diversity, and degrading on out-of-distribution prompts. This is why SFT is often described as a \"light touch\" — a brief adaptation, not extended training."
    },
    {
      type: "mc",
      question: "Why is learning rate selection particularly sensitive during SFT compared to pretraining?",
      options: ["SFT uses a fundamentally different optimizer than pretraining, requiring separate hyperparameter tuning from scratch", "SFT always requires a constant learning rate with no schedule, unlike the cosine decay used in pretraining", "The pretrained weights encode useful representations in a sharp loss basin — too high a learning rate destroys these representations, while too low a rate fails to adapt", "The gradient norms are consistently larger during SFT, requiring gradient clipping thresholds that differ from pretraining"],
      correct: 2,
      explanation: "Pretrained models sit in well-conditioned regions of the loss landscape that encode broad linguistic knowledge. SFT learning rates are typically 10–100x smaller than pretraining rates (e.g., $1 \\times 10^{-5}$ to $5 \\times 10^{-5}$ vs. $3 \\times 10^{-4}$). Too large a rate \"kicks\" the model out of the pretrained basin, causing catastrophic forgetting. Too small a rate means the model barely moves from base behavior. This sensitivity is a form of the stability-plasticity dilemma."
    },
    {
      type: "mc",
      question: "In multi-turn conversation SFT, training data is structured with role tokens (e.g., `<|user|>`, `<|assistant|>`). What is the primary purpose of these special tokens?",
      options: ["They provide turn-boundary signals and role attribution so the model learns *when* to generate and *whose* voice to adopt", "They reduce the effective vocabulary size by replacing common conversational words with compressed single-token representations", "They are only needed during inference for proper formatting, not during the actual training process of the model", "They serve as safety markers that prevent the model from generating harmful content during multi-turn conversations"],
      correct: 0,
      explanation: "Role tokens serve as structural markers that teach the model the conversational protocol: when it's the assistant's turn to speak, what voice/style to use, and where turn boundaries are. Without them, the model cannot distinguish user text from assistant text in the training data, leading to role confusion (e.g., generating user messages or failing to stop). The loss mask typically only applies to assistant turns."
    },
    {
      type: "mc",
      question: "The \"quality vs. quantity\" debate in SFT has been largely resolved in favor of:",
      options: [
        "Quantity — scaling the number of examples always wins regardless of individual example quality, since more data reduces variance in the gradient estimates",
        "Quality — a small set of diverse, high-quality examples outperforms a large set of noisy examples, though there are diminishing returns beyond core task coverage",
        "Neither quality nor quantity matters significantly — the base model size and pretraining corpus are the dominant factors determining post-SFT performance",
        "An exact 50/50 tradeoff between quality and quantity that shifts predictably based on model size, with larger models favoring quantity over curation"
      ],
      correct: 1,
      explanation: "Multiple studies (LIMA, Alpaca, WizardLM ablations) converge on the finding that data quality dominates. Key quality factors include: (1) correctness of responses, (2) diversity of tasks/formats, (3) appropriate complexity, and (4) consistent style. A curated set of 1K–10K examples often outperforms 100K+ noisy examples. However, quality has diminishing returns — after covering the main task distribution, more high-quality data yields marginal gains."
    },
    {
      type: "mc",
      question: "When fine-tuning with LoRA (Low-Rank Adaptation), the weight update is parameterized as $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times d}$ with $r \\ll d$. How does this relate to the SFT intuition that alignment is a \"small\" change?",
      options: ["It has no relation — LoRA is purely a memory optimization that does not reflect the nature of the underlying weight change", "The rank $r$ must equal the number of training examples for LoRA to properly capture the alignment signal", "LoRA always produces identical results to full fine-tuning, meaning the low-rank structure is merely a compression artifact", "The low-rank constraint means the update lives in a small subspace, consistent with the hypothesis that SFT adjusts a few directions in weight space rather than overhauling the model"],
      correct: 3,
      explanation: "LoRA's success provides empirical evidence for the Superficial Alignment Hypothesis from a parameter perspective: if SFT only needs to \"steer\" the model, then the weight change should be low-rank (a small number of directions in parameter space). LoRA with $r = 8$ or $r = 16$ often matches full fine-tuning, suggesting the effective dimensionality of the SFT update is far smaller than the full parameter count. This also connects to the intrinsic dimensionality literature (Aghajanyan et al.)."
    },
    {
      type: "mc",
      question: "A common failure mode of SFT is \"sycophancy\" — the model excessively agrees with the user. What causes this during SFT?",
      options: ["The base model lacks the factual knowledge needed to formulate well-reasoned disagreements with user claims", "Sycophancy is caused by too low a learning rate, which prevents the model from learning contrastive response patterns", "Training data is biased toward agreement: human-written \"ideal\" responses tend to be agreeable, and annotators reward compliance, so the model learns that agreeing is the high-probability format", "It only occurs when training for more than 100 epochs, at which point the model has overfit to agreeable response templates"],
      correct: 2,
      explanation: "Sycophancy is a distribution-level problem in the SFT data. Human annotators writing \"ideal\" assistant responses tend to be accommodating and agreeable. The model learns this statistical pattern: given any user statement, the maximum-likelihood response is one that validates the user. This is difficult to fix with SFT alone because it requires *preference* signal (\"this disagreement is better than that agreement\"), which is why RLHF/DPO stages are needed to address it."
    },
    {
      type: "mc",
      question: "You are fine-tuning a 7B model on 5,000 instruction-response pairs. During training, you observe that training loss decreases steadily but evaluation loss begins increasing after epoch 1. The model's generations become repetitive and formulaic. What is the most likely diagnosis and remedy?",
      options: ["Classic overfitting to the small SFT dataset — reduce to 1 epoch, increase dropout or use LoRA with lower rank, and verify data diversity", "The model is underfitting — increase the learning rate and train for more epochs to allow the model to capture the full task distribution", "The evaluation set is mislabeled — retrain with the same hyperparameters after re-curating the held-out data with better quality controls", "The tokenizer is incompatible with the training data — retokenize the dataset using the base model's original tokenizer and vocabulary mapping"],
      correct: 0,
      explanation: "This is the textbook SFT overfitting pattern: the model memorizes training responses (decreasing train loss) while losing generalization (increasing eval loss, repetitive outputs). Remedies include: (1) train for only 1 epoch, (2) reduce effective model capacity (LoRA, dropout), (3) verify data diversity (repetitive data accelerates overfitting), (4) use a cosine schedule that decays to near-zero. The small dataset makes this almost inevitable without these precautions."
    },
    {
      type: "mc",
      question: "Consider the SFT objective $\\mathcal{L}_{\\text{SFT}} = -\\mathbb{E}_{(x,y) \\sim \\mathcal{D}} \\left[ \\sum_{t=1}^{|y|} \\log \\pi_\\theta(y_t \\mid x, y_{<t}) \\right]$. This is equivalent to minimizing $\\text{KL}(p_{\\text{data}} \\| \\pi_\\theta)$ (forward KL). What behavioral consequence does this have compared to reverse KL?",
      options: [
        "Forward KL is mode-seeking: $\\pi_\\theta$ concentrates mass on a single high-probability mode of $p_{\\text{data}}$, producing sharper, more focused but less diverse outputs",
        "Forward KL is mode-covering: $\\pi_\\theta$ tries to place mass everywhere $p_{\\text{data}}$ has mass, leading to diverse but sometimes incoherent or hedging outputs",
        "Forward KL and reverse KL produce identical gradient updates for autoregressive language models, so the behavioral consequences are indistinguishable in practice",
        "Forward KL minimization converges faster than reverse KL in all cases, producing equivalent outputs with fewer gradient steps and lower overall compute cost"
      ],
      correct: 1,
      explanation: "Maximum likelihood (SFT) minimizes forward KL $\\text{KL}(p_{\\text{data}} \\| \\pi_\\theta)$, which is mode-covering: $\\pi_\\theta$ is penalized for assigning low probability to anything in $p_{\\text{data}}$. This means the model tries to cover all training modes, potentially spreading mass across contradictory response styles. This contrasts with RLHF/RL objectives that effectively use reverse KL (mode-seeking), producing more focused, consistent outputs — one reason why RLHF improves upon SFT."
    }
  ]
};
