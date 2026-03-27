// Focused learning module: Supervised Fine-Tuning (SFT) Mechanics
// Section A.1: Supervised Fine-Tuning
// Covers: why SFT works (superficial alignment hypothesis), the SFT training
// objective as maximum likelihood, loss masking, chat templates, catastrophic
// forgetting, data quality vs quantity, and practical training considerations.
// Single-concept module building from first principles.
// Grounded in Goodfellow et al. (2016) Ch. 5 (MLE) and Ch. 8 (optimization).

export const sftMechanicsLearning = {
  id: "A.1-sft-learning-easy",
  sectionId: "A.1",
  title: "Supervised Fine-Tuning Mechanics",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 25,
  steps: [
    // Step 1: Why SFT?
    {
      type: "info",
      title: "From Pretrained Model to Assistant",
      content: "A pretrained language model is trained to predict the next token on internet text. It can complete sentences and generate coherent text, but it doesn't know how to follow instructions or have a conversation. If you ask it a question, it's as likely to generate a follow-up question as an answer — it's mimicking the distribution of internet text, not acting as a helpful assistant.\n\n**Supervised Fine-Tuning (SFT)** bridges this gap. The idea is simple: take the pretrained model and continue training it on a curated dataset of (instruction, response) pairs — demonstrations of the behavior you want.\n\nThe key insight, known as the **Superficial Alignment Hypothesis** (Zhou et al., 2023 — the LIMA paper), is that SFT doesn't teach the model new knowledge. The model already \"knows\" how to answer questions, write code, and reason — these capabilities were acquired during pretraining on trillions of tokens. What SFT does is teach the model a new **format**: given an instruction, produce a helpful response rather than continuing the text in a random direction.\n\nThis explains a striking empirical finding: LIMA achieved strong performance with only **1,000 carefully curated examples**. If SFT were teaching knowledge, thousands of examples would be far too few. But if SFT is just teaching the model to access its existing knowledge in a particular format, a small number of high-quality demonstrations suffices."
    },
    // Step 2: MC — superficial alignment
    {
      type: "mc",
      question: "The LIMA paper demonstrated competitive instruction-following with only 1,000 training examples. What does this support about SFT's role?",
      options: [
        "SFT is the primary source of a model's factual knowledge, and 1,000 examples is sufficient because each example teaches hundreds of facts through generalization",
        "SFT primarily teaches the model a response format — how to access capabilities already learned during pretraining — so a small number of high-quality demonstrations suffices",
        "SFT works by memorizing the training examples verbatim and interpolating between them at inference time, which requires only a small lookup table",
        "SFT is unnecessary for instruction-following — the LIMA result shows that prompting alone achieves the same performance without any fine-tuning"
      ],
      correct: 1,
      explanation: "The Superficial Alignment Hypothesis proposes that SFT teaches format, not knowledge. The pretrained model already has the capability to answer questions, reason, and generate code — it learned these from pretraining on trillions of tokens. SFT's job is to teach the model to channel these capabilities into a particular output format (instruction → response). This explains why 1,000 high-quality examples can be sufficient: you're not teaching the model what to know, you're teaching it how to present what it already knows."
    },
    // Step 3: The SFT training objective
    {
      type: "info",
      title: "The SFT Objective: Maximum Likelihood",
      content: "SFT uses the same training objective as pretraining: **next-token cross-entropy loss**. Given a sequence of tokens $y_1, y_2, \\ldots, y_T$, the loss is:\n\n$$\\mathcal{L}(\\theta) = -\\frac{1}{T}\\sum_{t=1}^{T} \\log p_\\theta(y_t | y_{<t})$$\n\nThis is **maximum likelihood estimation (MLE)** — we find parameters $\\theta$ that maximize the probability of the training data under the model. As Goodfellow et al. (2016, Ch. 5.5) show, minimizing cross-entropy is equivalent to minimizing the KL divergence $D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$ between the empirical data distribution and the model.\n\nThis is the **forward KL divergence** — the same one from information theory. Recall that forward KL is **mode-covering**: the model tries to place probability mass everywhere the data distribution has mass. This has an important consequence for SFT: if the training data contains diverse response styles (formal, casual, verbose, concise), the model will try to cover all of them, potentially producing outputs that are an averaged mixture of styles rather than consistently good in any single style.\n\nThis mode-covering property is a fundamental limitation of SFT that motivates the need for preference-based training (RLHF, DPO) which uses reverse KL instead."
    },
    // Step 4: MC — forward KL and SFT
    {
      type: "mc",
      question: "SFT minimizes the forward KL divergence $D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$. If the SFT training data contains two distinct response styles — concise technical answers and verbose conversational answers — what behavior does the forward KL objective encourage?",
      options: [
        "The model selects whichever style appears more frequently in the training data and completely ignores the minority style",
        "The model collapses to the shorter responses because cross-entropy loss penalizes longer sequences more heavily",
        "The model assigns equal probability to exactly two modes and samples cleanly from one or the other at each generation",
        "The model places probability mass on both styles, potentially generating responses that are an awkward blend of concise and verbose tendencies"
      ],
      correct: 3,
      explanation: "Forward KL is mode-covering: $D_{\\text{KL}}(p_{\\text{data}} \\| p_\\theta)$ penalizes the model for assigning zero probability anywhere $p_{\\text{data}}$ has mass. This forces the model to spread probability across all modes in the data. With two styles present, the model doesn't cleanly separate them — it covers both, producing outputs that can blend or hedge between styles. This is why curating consistent SFT data is important, and why preference-based methods (which use reverse KL, a mode-seeking objective) can produce more focused outputs."
    },
    // Step 5: Loss masking — completion only
    {
      type: "info",
      title: "Loss Masking: Training on Responses Only",
      content: "A critical difference between pretraining and SFT: in SFT, we typically compute loss **only on the response tokens**, not on the instruction/prompt tokens. This is called **completion-only training** or **loss masking**.\n\nConsider a training example with prompt $x = (x_1, \\ldots, x_P)$ and response $y = (y_1, \\ldots, y_R)$. The full sequence is $[x_1, \\ldots, x_P, y_1, \\ldots, y_R]$. With loss masking:\n\n$$\\mathcal{L}(\\theta) = -\\frac{1}{R}\\sum_{t=1}^{R} \\log p_\\theta(y_t | x, y_{<t})$$\n\nThe prompt tokens still flow through the forward pass (they provide context), but they contribute zero gradient. Only the response tokens generate loss.\n\nWhy? The model should learn to **generate good responses given instructions**, not to generate instructions. Including the prompt in the loss wastes gradient signal on predicting prompt tokens (which the model already handles well from pretraining) and can bias the model toward generating instruction-like text rather than response-like text.\n\nImplementation: a **label mask** is created where prompt positions are set to $-100$ (PyTorch's ignore index), and response positions contain the actual token IDs. The cross-entropy loss function ignores any position with a $-100$ label."
    },
    // Step 6: MC — loss masking
    {
      type: "mc",
      question: "An SFT training example has a 200-token prompt and a 50-token response. With completion-only loss masking, how many tokens contribute to the gradient update?",
      options: [
        "250 tokens — all tokens in the sequence contribute to the loss, since the prompt provides important context for response generation",
        "200 tokens — only the prompt tokens are trained, since the model needs to learn to understand instructions",
        "50 tokens — only the response tokens generate loss; the prompt tokens flow through the forward pass but contribute zero gradient",
        "1 token — only the final token of the response contributes, since cross-entropy is computed at the sequence level"
      ],
      correct: 2,
      explanation: "With completion-only training, the label mask sets all 200 prompt positions to the ignore index ($-100$). Only the 50 response tokens have valid labels and contribute to the cross-entropy loss and gradient. The prompt tokens are still processed in the forward pass — they're essential for providing context — but they don't generate any gradient signal. This focuses the learning signal entirely on response generation."
    },
    // Step 7: Chat templates and multi-turn
    {
      type: "info",
      title: "Chat Templates and Role Attribution",
      content: "Modern SFT uses **chat templates** — structured formats that mark the boundaries between different roles (system, user, assistant) in a conversation. For example, the ChatML format:\n\n```\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain entropy.<|im_end|>\n<|im_start|>assistant\nEntropy measures the average uncertainty...<|im_end|>\n```\n\nThe special tokens (`<|im_start|>`, `<|im_end|>`) are added to the tokenizer's vocabulary and serve as **structural delimiters** — they tell the model where each turn begins and ends, and who is speaking.\n\nFor **multi-turn conversations**, the loss mask must be applied carefully. Consider a 3-turn dialogue:\n- Turn 1: user message → assistant response\n- Turn 2: user follow-up → assistant response\n- Turn 3: user question → assistant response\n\nThe loss should be computed only on **all assistant response tokens** across all turns, not just the final response. The user messages and system prompt are masked. This teaches the model to produce appropriate responses at every point in a conversation, not just at the end.\n\nGetting the mask wrong — for instance, accidentally training on user messages — can cause the model to start generating user-like text (asking questions back instead of answering them)."
    },
    // Step 8: MC — chat templates
    {
      type: "mc",
      question: "In a multi-turn SFT training example with 3 user messages and 3 assistant responses, which tokens should have their labels masked (set to ignore)?",
      options: [
        "All user message tokens and the system prompt are masked; all three assistant responses contribute to the loss",
        "Only the system prompt is masked; all user and assistant tokens contribute to the loss since multi-turn coherence requires training on the full conversation",
        "Only the first two assistant responses are masked; the model trains only on the final response to avoid learning from potentially suboptimal earlier turns",
        "All tokens are trained with equal weight — masking in multi-turn conversations causes the model to lose conversational context"
      ],
      correct: 0,
      explanation: "The loss mask should cover all non-assistant tokens: the system prompt, all user messages, and all special/role tokens. All three assistant responses contribute to the loss — the model needs to learn to generate appropriate responses at every turn in the conversation, not just the final one. Training only on the last response would waste the learning signal from earlier turns and could lead to inconsistent multi-turn behavior."
    },
    // Step 9: Catastrophic forgetting
    {
      type: "info",
      title: "Catastrophic Forgetting and the Stability-Plasticity Tradeoff",
      content: "A persistent challenge in SFT is **catastrophic forgetting**: as the model adapts to the fine-tuning distribution, it can lose capabilities acquired during pretraining. For instance, a model fine-tuned exclusively on English instruction data may lose its multilingual abilities.\n\nThis reflects a fundamental tension in neural network training that Goodfellow et al. (2016, Ch. 8) describe as the **stability-plasticity dilemma**: the same properties that let the model quickly adapt to new data (plasticity) also let it overwrite old knowledge (instability).\n\nIn practice, SFT mitigates forgetting through several mechanisms:\n\n1. **Small learning rates**: SFT typically uses learning rates 10-100$\\times$ smaller than pretraining (e.g., $1\\text{-}5 \\times 10^{-5}$ vs. $10^{-3}$). Smaller steps mean smaller perturbations to the pretrained weights.\n\n2. **Few epochs**: SFT datasets are usually trained for only **1-3 epochs**. Empirically, performance on held-out instructions often peaks at 1-2 epochs and then degrades — the model starts memorizing the specific examples rather than learning the format.\n\n3. **Data mixing**: Including a fraction of pretraining data (or diverse tasks) in the SFT mixture helps retain general capabilities.\n\nThe learning rate sensitivity is extreme: a rate that's too high causes the model to forget rapidly, while a rate that's too low means it never learns the new format. This narrow window is one reason why SFT requires careful hyperparameter tuning."
    },
    // Step 10: MC — catastrophic forgetting
    {
      type: "mc",
      question: "An SFT run uses a learning rate of $3 \\times 10^{-4}$ (the same as pretraining) and trains for 10 epochs on 5,000 examples. The model quickly learns to follow instructions but loses its code generation ability. What is the most likely cause and fix?",
      options: [
        "The dataset is too small — scaling to 50,000 examples would prevent forgetting by providing enough diverse signal to reinforce all pretrained capabilities during fine-tuning",
        "Code generation was never a robust pretrained capability — it only appeared to work because of memorized training examples that happened to overlap with the evaluation benchmarks used",
        "Full fine-tuning inherently causes catastrophic forgetting regardless of hyperparameters — the architecture must be modified with frozen layers or adapter modules before any SFT",
        "The learning rate is too high and epoch count too large for SFT — $3 \\times 10^{-4}$ for 10 epochs causes excessive weight perturbation; reducing to $2 \\times 10^{-5}$ for 2 epochs would help"
      ],
      correct: 3,
      explanation: "SFT learning rates are typically 10-100$\\times$ smaller than pretraining rates ($1\\text{-}5 \\times 10^{-5}$ vs. $10^{-3}\\text{-}10^{-4}$), and training usually lasts 1-3 epochs. Using $3 \\times 10^{-4}$ for 10 epochs is far too aggressive — the large weight updates rapidly overwrite the pretrained representations, including code generation capabilities. Reducing the learning rate and epochs shrinks the perturbation to the pretrained weights, preserving more of the original capabilities while still teaching the instruction-following format."
    },
    // Step 11: Data quality dominates
    {
      type: "info",
      title: "Data Quality Over Quantity",
      content: "One of the most consistent findings in SFT research is that **data quality dominates data quantity**. A small set of high-quality, diverse examples outperforms a much larger set of mediocre ones.\n\nWhat makes SFT data \"high quality\"?\n\n1. **Correctness**: Responses must be factually accurate and well-reasoned. Errors in the training data teach the model to produce errors with confidence.\n\n2. **Consistency of style**: The responses should follow a consistent voice and format. Mixing wildly different styles (terse vs. verbose, formal vs. casual) leads to the mode-covering problem discussed earlier — the model averages across styles rather than excelling at any one.\n\n3. **Appropriate difficulty**: The instructions should cover the full range of complexity the model will encounter, from simple factual questions to complex multi-step reasoning. Overrepresenting simple queries teaches the model to give shallow answers even to deep questions.\n\n4. **Diversity of tasks**: Covering many task types (QA, summarization, coding, math, creative writing) teaches the model that the instruction format applies broadly, not just to one domain.\n\nA concrete example: **sycophancy** — the tendency of fine-tuned models to agree with the user even when the user is wrong — is largely a data quality problem. If training examples reward agreeable responses over accurate ones, the model learns to prioritize agreement. This is a systematic bias in the training data, not a model architecture problem."
    },
    // Step 12: MC — data quality
    {
      type: "mc",
      question: "A team fine-tunes a model on 100,000 SFT examples sourced from a crowdsourcing platform with minimal quality control. A second team fine-tunes the same base model on 1,000 examples that were expert-curated and carefully verified. Based on SFT research, what outcome is most likely?",
      options: [
        "The 100K model performs significantly better because the larger dataset provides more diverse coverage that prevents overfitting and teaches the model a broader range of response patterns",
        "Both models perform similarly because the base model's pretrained knowledge dominates the fine-tuning signal regardless of SFT data quality or the number of training examples used",
        "The 1K expert-curated model likely matches or exceeds the 100K model, because SFT teaches format not knowledge, and clean demonstrations are more effective than noisy ones",
        "The 100K model outperforms on easy tasks while the 1K model outperforms on hard tasks, because larger datasets provide breadth of coverage while smaller datasets provide depth",
      ],
      correct: 2,
      explanation: "Research consistently shows that data quality dominates quantity for SFT. The LIMA paper demonstrated that 1,000 high-quality examples can match models trained on much larger datasets. Since SFT primarily teaches the model a response format (the Superficial Alignment Hypothesis), clean demonstrations of that format are more valuable than noisy ones. A large but poorly curated dataset introduces inconsistent response styles, errors, and biases (like sycophancy) that can actually degrade performance compared to a small, clean dataset."
    },
    // Step 13: SFT as the foundation for alignment
    {
      type: "info",
      title: "SFT's Role in the Alignment Pipeline",
      content: "SFT is the **first step** of the modern alignment pipeline, followed by preference-based methods (RLHF or DPO). Understanding why SFT alone is insufficient clarifies the motivation for these later steps.\n\nSFT's limitation is structural: it can only learn from **demonstrations** (examples of correct behavior). It cannot learn from **comparisons** (which of two responses is better). This distinction matters because:\n\n1. **Demonstrations require knowing the best response.** For complex, ambiguous queries, even experts disagree on the single best answer. But comparing two responses and saying \"this one is better\" is much easier and more reliable.\n\n2. **The mode-covering problem persists.** MLE forces the model to spread probability over all training responses. Even if you curate perfect data, any remaining diversity in response style gets averaged. Preference methods use reverse KL, which is mode-seeking — the model can focus on one high-quality mode.\n\n3. **SFT can't distinguish \"good\" from \"great.\"** Two responses might both be correct, but one is more helpful, more concise, or better structured. SFT treats them identically if both appear in the training data. RLHF/DPO can express these gradations through reward modeling.\n\nThe pipeline is:\n- **SFT**: Teach the format (instruction → response). This gets you from a text-completion model to an instruction-following model.\n- **RLHF/DPO**: Refine the quality within that format. This gets you from an instruction-following model to a model that produces consistently high-quality, preferred responses."
    },
    // Step 14: MC — SFT in the pipeline
    {
      type: "mc",
      question: "Why is SFT typically performed before RLHF/DPO rather than skipping directly to preference-based training on the pretrained model?",
      options: [
        "RLHF/DPO needs the model to already generate instruction-formatted responses — without SFT, the model produces continuation-style text that can't be meaningfully compared for quality",
        "SFT reduces the model's effective parameter count through implicit pruning, making the subsequent RLHF optimization computationally feasible on standard training hardware",
        "RLHF/DPO algorithms mathematically require the policy and reference policy to share the same tokenizer vocabulary, which is only guaranteed after SFT alignment",
        "SFT precomputes the reward model's output distribution over response candidates, which RLHF uses as the initialization for the value function network"
      ],
      correct: 0,
      explanation: "SFT establishes the response format that preference training builds upon. A raw pretrained model generates text completions, not instruction responses — its outputs can't be meaningfully compared as \"which response is more helpful.\" SFT teaches the model to produce instruction-formatted responses, creating a reference policy ($\\pi_{\\text{ref}}$) that generates coherent, on-topic responses. RLHF/DPO then refines the quality of these responses through preference data. The pipeline is sequential by necessity: format first (SFT), then quality refinement (RLHF/DPO)."
    }
  ]
};
