// Module: Supervised Fine-Tuning (SFT) Mechanics
// Section A.1: Supervised Fine-Tuning
// Teaches SFT from first principles: what it is, why it works,
// how the loss function works, and common failure modes.
// Grounded in Goodfellow et al. (2016) Ch. 5 (MLE), Ch. 8 (Optimization).

export const sftMechanicsLearning = {
  id: "A.1-learning-easy",
  sectionId: "A.1",
  title: "Supervised Fine-Tuning Mechanics",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "From Pretrained Model to Assistant",
      content: "A pretrained language model can predict the next token with remarkable accuracy — it has absorbed grammar, facts, reasoning patterns, and even some code from trillions of tokens. But it has no concept of a \"conversation.\" If you type a question, it's equally likely to continue with another question, a Wikipedia paragraph, or random web text.\n\n**Supervised fine-tuning (SFT)** bridges this gap. The idea is deceptively simple: show the model examples of (instruction, desired response) pairs and train it to produce the response given the instruction.\n\nThe key insight — formalized as the **Superficial Alignment Hypothesis** (Zhou et al., 2023, LIMA) — is that SFT does not teach the model new knowledge. The model already *knows* how to answer questions, write code, and reason. SFT teaches it *when* and *how* to deploy those capabilities: adopt an assistant persona, follow instructions, and stop when done.\n\nThis is why SFT is sometimes called \"alignment\" in a narrow sense: aligning the model's output format with what users expect."
    },
    {
      type: "mc",
      question: "A 70B pretrained model scores poorly on an instruction-following benchmark but has strong perplexity on a broad text corpus. After SFT on 1,000 high-quality instruction-response pairs, it scores competitively. What does this suggest about the role of SFT?",
      options: [
        "The 1,000 examples provided enough new factual knowledge to cover the benchmark's question distribution",
        "SFT primarily unlocks and steers capabilities the model already acquired during pretraining, rather than teaching new knowledge",
        "The benchmark is too easy — any fine-tuning signal, regardless of quality, would produce competitive scores",
        "SFT replaced the model's pretrained representations with task-specific ones that are more efficient for instruction following"
      ],
      correct: 1,
      explanation: "This is the core finding of the LIMA paper. The pretrained model already has the knowledge and reasoning ability — it just doesn't know it should present that knowledge in a helpful, instruction-following format. 1,000 high-quality examples are enough to teach the *format*, not the *content*. This is the Superficial Alignment Hypothesis: alignment is a shallow, surface-level change to the model's behavior."
    },
    {
      type: "info",
      title: "The SFT Training Objective",
      content: "SFT uses the same loss function as pretraining — **next-token cross-entropy** — but with a critical modification.\n\nRecall from Goodfellow et al. (2016, §5.5) that maximum likelihood estimation (MLE) for a conditional model $P_\\theta(y \\mid x)$ is equivalent to minimizing the cross-entropy between the data distribution and the model:\n\n$$\\mathcal{L}_{\\text{MLE}} = -\\mathbb{E}_{(x,y) \\sim \\mathcal{D}}\\left[\\log P_\\theta(y \\mid x)\\right]$$\n\nFor an autoregressive language model, the response $y = (y_1, y_2, \\ldots, y_T)$ is factored token-by-token:\n\n$$\\mathcal{L}_{\\text{SFT}} = -\\mathbb{E}_{(x,y) \\sim \\mathcal{D}}\\left[\\sum_{t=1}^{T} \\log P_\\theta(y_t \\mid x, y_{<t})\\right]$$\n\nThe critical modification: **loss masking**. The instruction tokens $x$ are fed through the model to build context, but the loss is computed **only on the response tokens** $y$. We don't want to train the model to *generate* instructions — only to *respond* to them."
    },
    {
      type: "mc",
      question: "During SFT, a training example has an instruction of 200 tokens and a response of 50 tokens. All 250 tokens are fed through the model in a single forward pass. The loss is computed on:",
      options: [
        "All 250 tokens, weighted equally — the model must learn both the instruction distribution and response distribution",
        "Only the 50 response tokens — the 200 instruction tokens provide context but are masked from the loss computation",
        "Only the 200 instruction tokens — the model needs to learn what good instructions look like to respond properly",
        "A random 50% subset of all tokens, selected uniformly, to regularize against overfitting to any specific position"
      ],
      correct: 1,
      explanation: "This is \"completion-only\" training. The instruction tokens participate in the forward pass (building up the key-value cache and providing context), but the loss gradient flows only through the response token predictions. Computing loss on instruction tokens would waste model capacity learning to generate user prompts rather than assistant responses."
    },
    {
      type: "info",
      title: "SFT as Forward KL Minimization",
      content: "The SFT objective has a precise information-theoretic interpretation. Minimizing the cross-entropy loss is equivalent to minimizing the **forward KL divergence** from the data distribution to the model:\n\n$$\\mathcal{L}_{\\text{SFT}} = \\text{KL}(p_{\\text{data}} \\| \\pi_\\theta) + H(p_{\\text{data}})$$\n\nSince $H(p_{\\text{data}})$ is constant w.r.t. $\\theta$, minimizing $\\mathcal{L}_{\\text{SFT}}$ minimizes $\\text{KL}(p_{\\text{data}} \\| \\pi_\\theta)$.\n\nForward KL is **mode-covering**: the model is penalized whenever it assigns low probability to something the data distribution covers. If $p_{\\text{data}}$ contains diverse response styles (formal and casual, verbose and terse, agreeable and critical), $\\pi_\\theta$ must spread its probability mass to cover all of them.\n\nThis has a concrete behavioral consequence: SFT models tend to produce **averaged**, sometimes inconsistent outputs — hedging between the multiple modes present in the training data. This is one reason RLHF/DPO (which effectively use reverse KL) produce more focused, coherent responses."
    },
    {
      type: "mc",
      question: "An SFT dataset contains two styles of responses to the same prompt: 60% are detailed and formal, 40% are concise and casual. After training with standard cross-entropy (forward KL), the model's outputs for that prompt will most likely:",
      options: [
        "Produce exclusively the formal style since it has higher frequency, completely ignoring the casual mode",
        "Alternate randomly between the two styles with each generation, maintaining both modes but never mixing them",
        "Blend features of both styles, sometimes producing awkwardly mixed outputs that hedge between formal and casual",
        "Collapse to producing a single token repeated, since the conflicting gradients from two modes cancel out"
      ],
      correct: 2,
      explanation: "Forward KL is mode-covering — the model must assign probability mass everywhere the data distribution has mass. With two modes in the data, the model doesn't cleanly separate them. Instead, it learns a distribution that covers both, which in practice means generated samples often blend features of both modes. This \"mode averaging\" produces outputs that are neither cleanly formal nor cleanly casual, but an awkward hybrid. This is a fundamental limitation of MLE-based training."
    },
    {
      type: "info",
      title: "Chat Templates and Role Tokens",
      content: "Real SFT training uses structured formats to encode multi-turn conversations. A typical **chat template** wraps each turn with special tokens that mark role boundaries:\n\n```\n<|user|>\nExplain gradient descent in simple terms.\n<|assistant|>\nGradient descent is an optimization algorithm...\n<|end|>\n```\n\nThese role tokens serve several functions:\n\n**1. Turn boundaries**: The model learns that `<|assistant|>` means \"start generating\" and `<|end|>` (or an end-of-sequence token) means \"stop generating.\"\n\n**2. Role attribution**: The model learns different behavioral patterns for different roles. It should never generate text that looks like a user message during inference.\n\n**3. Loss masking boundaries**: The training code uses role tokens to determine which tokens to include in the loss. Typically, only tokens between `<|assistant|>` and `<|end|>` contribute to the loss.\n\nFor multi-turn conversations, the entire conversation history is concatenated into a single sequence. The model sees all previous turns as context, with loss computed only on assistant turns. This teaches the model to condition on conversational history."
    },
    {
      type: "mc",
      question: "A multi-turn SFT example has: User turn 1 (30 tokens) → Assistant turn 1 (80 tokens) → User turn 2 (25 tokens) → Assistant turn 2 (60 tokens). Role tokens add 10 tokens overhead. The total sequence is 215 tokens. How many tokens contribute to the training loss?",
      options: [
        "215 — all tokens in the sequence contribute equally to maximize data efficiency",
        "140 — only the assistant turns (80 + 60 tokens) contribute, as user turns and role tokens are masked",
        "55 — only the user turns (30 + 25 tokens) contribute, since the model needs to understand instructions",
        "205 — all tokens except the role token overhead contribute to the training loss"
      ],
      correct: 1,
      explanation: "Only assistant response tokens contribute to the loss: 80 (turn 1) + 60 (turn 2) = 140 tokens. The user turns, role tokens, and any system prompts provide context during the forward pass but are masked from the loss. This ensures the model learns to *generate* assistant responses, not to reproduce user prompts or formatting tokens."
    },
    {
      type: "info",
      title: "Why SFT Overfits Fast",
      content: "SFT datasets are tiny compared to pretraining corpora. A typical pretraining run uses 1–15 trillion tokens; a typical SFT dataset has 10K–100K examples, totaling perhaps 50M–500M tokens — a factor of **1,000–10,000x smaller**.\n\nThis creates a severe overfitting risk. From the perspective of optimization theory (Goodfellow et al., 2016, §8.1), the model has far more parameters than training examples. A 7B model fine-tuned on 10K examples has ~700K parameters per example — the system is massively overparameterized.\n\nIn practice, SFT typically peaks at **1–2 epochs**:\n- **Epoch 1**: The model learns the formatting and behavioral patterns (instruction-following, stopping, role adherence)\n- **Epoch 2**: Marginal improvements on harder examples, but memorization starts\n- **Epoch 3+**: Training loss continues decreasing, but evaluation loss increases — the model memorizes exact phrasings from the training set, loses output diversity, and produces formulaic, repetitive text\n\nThis is why SFT is described as a \"light touch\" — it's a brief adaptation, not an extended training run."
    },
    {
      type: "mc",
      question: "During SFT of a 13B model on 5,000 examples, training loss steadily decreases but eval loss starts rising after epoch 1. Generated outputs become repetitive and formulaic. What is the most appropriate remedy?",
      options: [
        "Increase the learning rate to help the model escape the local minimum that is causing repetitive outputs",
        "Train for more epochs with a larger dataset, since the model clearly hasn't converged to the optimal solution yet",
        "Reduce to 1 epoch, and consider using LoRA or increased dropout to limit the model's capacity for memorization",
        "Switch to a smaller model, since the 13B model has too few parameters to properly fit the 5,000 training examples"
      ],
      correct: 2,
      explanation: "This is textbook SFT overfitting: decreasing train loss + increasing eval loss + repetitive outputs. The model is memorizing training examples rather than learning generalizable patterns. Remedies: (1) stop at 1 epoch, (2) reduce effective capacity with LoRA (low-rank updates can't memorize as much) or dropout, (3) verify data diversity (repetitive data accelerates overfitting). The learning rate should be *decreased*, not increased — a large learning rate would additionally risk catastrophic forgetting."
    },
    {
      type: "info",
      title: "Learning Rate: The Stability-Plasticity Dilemma",
      content: "The pretrained model sits in a well-conditioned region of the loss landscape — a \"basin\" whose geometry encodes linguistic knowledge, factual recall, and reasoning patterns. SFT must modify this model *just enough* to adopt a new behavioral format without destroying what it already knows.\n\nThis is a concrete instance of the **stability-plasticity dilemma** (Goodfellow et al., 2016, §8.7.4 on catastrophic forgetting):\n- **Too much plasticity** (high learning rate): The optimizer takes large steps that leave the pretrained basin, destroying encoded knowledge. The model \"forgets\" how to reason, write coherent text, or recall facts.\n- **Too much stability** (low learning rate): The optimizer barely moves, and the model retains base behavior — it doesn't learn to follow instructions or adopt the assistant format.\n\nTypical SFT learning rates are **10–100x smaller** than pretraining:\n- Pretraining: $\\sim 3 \\times 10^{-4}$\n- SFT: $\\sim 1 \\times 10^{-5}$ to $5 \\times 10^{-5}$\n\nSFT also commonly uses a **cosine schedule** that decays the learning rate to near zero, with a brief linear warmup (1–5% of steps) to avoid early instability when the gradients from the new data distribution first interact with the pretrained parameters."
    },
    {
      type: "mc",
      question: "You fine-tune a pretrained model with a learning rate of $3 \\times 10^{-4}$ (the same as pretraining). After 100 steps, the model produces grammatically broken, incoherent text and has forgotten basic factual knowledge. What happened?",
      options: [
        "The learning rate is too low — the model couldn't adapt to the new distribution and is stuck generating pretraining-style text",
        "The model's tokenizer is incompatible with the fine-tuning data, causing token misalignment that corrupts outputs",
        "The learning rate is too high — the large gradient steps pushed the model out of its pretrained basin, causing catastrophic forgetting of language capabilities",
        "This is normal early-training behavior — the model needs at least 1,000 steps before coherent outputs emerge during SFT"
      ],
      correct: 2,
      explanation: "A learning rate of $3 \\times 10^{-4}$ is appropriate for pretraining from scratch but far too aggressive for SFT. The pretrained weights encode representations in a specific region of parameter space. Large updates \"kick\" the model out of this basin, destroying the structure that enables coherent language generation. This is catastrophic forgetting — the model loses previously learned capabilities. SFT learning rates should be 10–100x smaller to make incremental adjustments while preserving the pretrained representation."
    },
    {
      type: "info",
      title: "Data Quality Over Quantity",
      content: "The LIMA paper (Zhou et al., 2023) demonstrated that **1,000 carefully curated examples** could produce a model competitive with those trained on 50K+ examples. This finding has been replicated across model sizes and domains.\n\nWhy does quality dominate? Because of the Superficial Alignment Hypothesis: SFT only needs to teach format and behavioral norms, not factual knowledge. A small number of examples is sufficient to cover the core patterns:\n\n**What makes a high-quality SFT example?**\n1. **Correctness**: The response is factually accurate and well-reasoned\n2. **Diversity**: The dataset covers a broad range of task types (QA, coding, math, creative writing, multi-turn)\n3. **Appropriate complexity**: Responses match the difficulty of the prompt — neither oversimplified nor unnecessarily complex\n4. **Consistent style**: A coherent \"voice\" that the model can learn reliably\n\n**What makes low-quality data actively harmful?**\n- Contradictory styles (some examples verbose, others terse, with no clear pattern)\n- Incorrect or hallucinated content (teaches the model to confidently state falsehoods)\n- Overly agreeable responses (\"sycophantic\" data that teaches the model to validate rather than inform)\n\nThe sycophancy problem is particularly insidious: human annotators writing \"ideal\" responses tend to be accommodating, so the training data is biased toward agreement. The model learns that agreeing with the user is the statistically dominant pattern — a problem that SFT alone cannot fix, requiring RLHF or DPO to provide contrastive preference signal."
    },
    {
      type: "mc",
      question: "Two SFT datasets are prepared for the same base model. Dataset A has 50,000 examples scraped from the web with automated filtering. Dataset B has 3,000 examples written by domain experts with careful quality review. Based on the LIMA findings, which outcome is most likely?",
      options: [
        "Dataset A wins decisively — 50K examples provide 16x more gradient signal, and automated filtering is sufficient for quality",
        "Dataset B produces a more capable model — curated quality outweighs quantity because SFT is teaching format, not knowledge, and noise in Dataset A teaches bad patterns",
        "Both produce identical models because the pretrained model's capabilities dominate, making SFT dataset differences negligible",
        "Dataset A wins on common tasks but Dataset B wins on niche tasks, creating a clean partition based on dataset coverage"
      ],
      correct: 1,
      explanation: "Multiple studies converge on this finding: data quality dominates quantity for SFT. Dataset B's 3,000 curated examples teach a clean, consistent behavioral pattern that the model can learn in 1–2 epochs. Dataset A's 50,000 noisy examples contain contradictory styles, factual errors, and formatting inconsistencies that the model also learns. Since SFT is teaching *format and behavior* rather than *knowledge*, the quality of the behavioral signal matters more than the volume of training data."
    },
    {
      type: "info",
      title: "LoRA and the Low-Rank Nature of Alignment",
      content: "**Low-Rank Adaptation (LoRA)** freezes the pretrained weights $W$ and trains a low-rank update $\\Delta W = BA$, where $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times d}$, and $r \\ll d$.\n\nThe effective weight at inference is $W + BA$, but during training, only $B$ and $A$ are updated — reducing trainable parameters by 100–1000x.\n\nLoRA's success for SFT provides **empirical evidence** for the Superficial Alignment Hypothesis from a parameter-space perspective:\n\n- If alignment required overhauling the model's representations, a low-rank update ($r = 8$ or $r = 16$) could not capture it — you'd need full-rank changes across many layers.\n- But LoRA with small $r$ often **matches full fine-tuning** on SFT benchmarks. This means the SFT update lives in a **small subspace** of the full parameter space.\n- Aghajanyan et al. (2021) formalized this as \"intrinsic dimensionality\": the effective number of parameters needed for a downstream task is orders of magnitude smaller than the model's total parameter count.\n\nThis connects to a practical advantage: by constraining the update to a small subspace, LoRA acts as an implicit regularizer, reducing the risk of catastrophic forgetting and overfitting — two of SFT's primary failure modes."
    },
    {
      type: "mc",
      question: "LoRA with rank $r = 16$ on a 7B model often matches full fine-tuning on SFT benchmarks. What does this tell us about the nature of the SFT weight update?",
      options: [
        "The weight update lives in a low-dimensional subspace — alignment is a small directional adjustment, not a full overhaul of the model's representations",
        "Full fine-tuning is wasteful because 99.99% of parameters are frozen by the optimizer anyway, even without explicit LoRA constraints",
        "LoRA succeeds because it trains entirely different parameters from those used during pretraining, avoiding any interference with learned knowledge",
        "The rank-16 constraint forces the model to memorize fewer training examples, which coincidentally matches the optimal SFT data efficiency"
      ],
      correct: 0,
      explanation: "LoRA's success demonstrates that the effective dimensionality of the SFT update is far smaller than the full parameter count. A rank-16 update modifies each weight matrix along only 16 directions — and this is sufficient to transform a base model into an instruction-following assistant. This is strong evidence that SFT is a surface-level behavioral adjustment (changing *how* the model presents knowledge) rather than a deep representational change (changing *what* the model knows). The low-rank constraint also provides implicit regularization, reducing overfitting."
    }
  ]
};
