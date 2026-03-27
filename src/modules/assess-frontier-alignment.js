// Section A.5: Frontier Alignment Assessment

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
      options: ["Strong models trained with weak supervision consistently outperform their weak supervisors, recovering a significant fraction of the gap between weak and strong performance — suggesting that strong models can partially \"generalize beyond\" noisy labels", "Weak supervisors always fail — the strong model degrades to weak-model performance because it cannot distinguish signal from noise in the supervisor's labels", "Weak and strong models always achieve identical performance regardless of their capacity difference, since the supervision signal is the binding constraint on quality", "The strong model learns to ignore the weak supervisor entirely and instead relies on its pretrained representations, making the supervision step unnecessary"],
      correct: 0,
      explanation: "Burns et al. found that when a weak model (e.g., GPT-2-level) provides labels for training a strong model (e.g., GPT-4-level), the strong model consistently exceeds the weak supervisor's performance. This is encouraging for scalable oversight: it suggests that superhuman models might partially self-correct even when trained with imperfect human feedback. However, the recovery is incomplete — there is a \"alignment tax\" — and the gap grows on harder tasks, indicating that weak-to-strong generalization alone is insufficient for frontier alignment."
    },
    {
      type: "mc",
      question: "In AI safety via debate (Irving et al., 2018), two AI agents argue opposing sides while a human judge decides the winner. Why does this protocol theoretically scale to superhuman capabilities?",
      options: [
        "The debate always converges to the truth in finite rounds because each exchange eliminates at least one incorrect hypothesis from consideration",
        "A human only needs to verify individual arguments/evidence (a simpler task) rather than generate the full solution — if lying requires more complex arguments than truth-telling, the honest debater has a structural advantage",
        "The human judge can always tell when an AI is lying because deceptive arguments exhibit detectable statistical patterns in their linguistic structure",
        "Debate eliminates the need for human oversight entirely by ensuring that the Nash equilibrium of the game always corresponds to the truthful outcome"
      ],
      correct: 1,
      explanation: "The key insight is *asymmetric verification*: checking an argument is easier than generating one. In debate, a lie can be exposed by pointing to a specific flaw, which the human can verify. Under the conjecture that truth is \"simpler\" than consistent deception (lies require more complex supporting arguments), the honest debater has an advantage in the Nash equilibrium. This allows a human with limited capabilities to adjudicate between superhuman arguments. The protocol assumes the human can verify atomic claims — the debaters recursively decompose complex arguments until reaching human-verifiable steps."
    },
    {
      type: "mc",
      question: "Representation engineering / steering vectors involve finding directions $\\mathbf{v}$ in a model's activation space such that adding $\\alpha \\mathbf{v}$ to intermediate activations controls a specific behavior (e.g., honesty). These vectors are typically found by:",
      options: ["Random search in the activation space, sampling directions until one produces the desired behavioral shift when added during inference", "Pruning attention heads until the behavior changes, identifying which heads encode the target concept by measuring behavioral impact of ablation", "Training a separate classifier on the model's outputs and using the learned classifier weights as the steering direction for activation intervention", "Computing the difference in mean activations between contrastive prompt pairs (e.g., honest vs. dishonest responses) across a dataset — the resulting direction captures the \"concept direction\" in representation space"],
      correct: 3,
      explanation: "The standard approach: (1) construct contrastive pairs — prompts that elicit opposite behaviors (e.g., truthful vs. deceptive responses). (2) Run both through the model, extract activations at a chosen layer. (3) Compute $\\mathbf{v} = \\mathbb{E}[\\mathbf{h}_{\\text{positive}}] - \\mathbb{E}[\\mathbf{h}_{\\text{negative}}]$. This difference vector captures the linear direction in representation space corresponding to the concept. Adding $\\alpha \\mathbf{v}$ at inference time \"steers\" the model along this direction. More sophisticated methods use PCA or linear probes on the contrastive activations."
    },
    {
      type: "mc",
      question: "OpenAI's o1/o3 models use RL for reasoning, training on chain-of-thought traces. A critical design choice is training the model to produce long \"thinking\" traces before answering. What RL signal structure makes this work?",
      options: ["Reward is given at every token of the reasoning trace, providing dense per-token supervision that guides each individual generation step toward correctness", "The model is only trained with supervised learning on expert traces collected from human mathematicians, with no reinforcement learning component involved", "Outcome-based reward (correctness of the final answer) combined with process supervision — the RL algorithm must learn to credit intermediate reasoning steps that lead to correct answers despite extremely sparse reward", "Reward is given only for short responses to encourage computational efficiency, penalizing any reasoning trace that exceeds a fixed token budget threshold"],
      correct: 2,
      explanation: "The o1/o3 paradigm uses RL with sparse outcome reward (is the final answer correct?) to train extended reasoning. The challenge is credit assignment: a 10,000-token reasoning trace might have one reward signal. Process reward models (PRMs) and process supervision help bridge this by evaluating intermediate steps. The RL training (likely a variant of PPO or GRPO) must learn which reasoning patterns — backtracking, verification, decomposition — lead to correct outcomes. This is \"RL for inference-time compute scaling\" — the model learns to think longer and more carefully."
    },
    {
      type: "mc",
      question: "Process supervision (as in the \"Let's Verify Step by Step\" paper, Lightman et al. 2023) provides feedback on each reasoning step. Compared to outcome supervision, process supervision:",
      options: ["Provides denser reward signal, enables better credit assignment, and allows the model to be guided away from flawed reasoning even when it accidentally reaches correct answers — but requires significantly more expensive per-step human annotations", "Always produces worse results but is cheaper to implement, since it avoids the overhead of training a separate step-level evaluation model alongside the policy", "Does not require any human annotations because the process reward can be derived automatically from the logical structure of the reasoning chain", "Is only applicable to non-mathematical tasks where the reasoning steps are expressed in natural language rather than formal symbolic notation"],
      correct: 0,
      explanation: "Process supervision labels each step as correct/incorrect, creating a rich training signal. Benefits: (1) **Credit assignment**: identifies exactly where reasoning fails, rather than penalizing the entire trace for a wrong final answer. (2) **Avoiding reward hacking**: a correct final answer via flawed reasoning gets negative process feedback. (3) **Denser signal**: reduces variance in policy gradients. The cost is annotation: labeling each step requires expert annotators who understand the reasoning, which is far more expensive than checking final answers. Lightman et al. showed process supervision substantially outperforms outcome supervision on math reasoning."
    },
    {
      type: "mc",
      question: "Red-teaming in the context of LLM safety involves systematically trying to elicit harmful or undesired outputs. The most effective red-teaming approaches combine:",
      options: [
        "Only automated tools (e.g., gradient-based adversarial search, fuzzing) with no human involvement — humans introduce bias that narrows the space of discovered vulnerabilities",
        "Human creativity for discovering novel attack vectors with automated methods (e.g., adversarial prompt optimization like GCG, model-based red-teaming) for scaling coverage — humans find qualitatively new failures while automation explores variations",
        "Only manual testing by a small team of domain experts, since automated methods generate superficial attacks that rarely uncover meaningful safety-relevant failure modes",
        "Random input generation without any guided structure, relying on statistical coverage of the input space to eventually trigger failures through brute-force exploration"
      ],
      correct: 1,
      explanation: "Effective red-teaming requires both modalities: (1) **Human red-teamers** discover novel failure modes that require creativity, cultural knowledge, and adversarial reasoning (e.g., role-playing attacks, multi-turn manipulation). (2) **Automated methods** like GCG (gradient-based adversarial suffix optimization), model-based red-teaming (using another LLM to generate attacks), and fuzzing scale to thousands of attack variations. The most comprehensive programs (e.g., Anthropic's, Meta's) layer both: humans identify attack categories, automation fills in the coverage matrix."
    },
    {
      type: "mc",
      question: "A steering vector $\\mathbf{v}$ is applied as $\\mathbf{h}'_l = \\mathbf{h}_l + \\alpha \\mathbf{v}$ at layer $l$. The coefficient $\\alpha$ controls steering strength. A known failure mode is that large $|\\alpha|$ values cause:",
      options: ["The model to produce higher quality outputs because the stronger signal more decisively activates the target behavioral direction in representation space", "Faster inference speed because the larger activation perturbation causes the model to converge to high-confidence token predictions more quickly", "The model to ignore the steering vector entirely because a saturation effect in the layer normalization neutralizes perturbations above a threshold", "Distribution shift in the activations — the modified $\\mathbf{h}'_l$ moves outside the manifold of activations the downstream layers were trained on, causing incoherent or degenerate text even if the desired behavioral shift is achieved"],
      correct: 3,
      explanation: "The model's downstream layers (layers $> l$) are trained on activations from the natural distribution. Adding a large $\\alpha \\mathbf{v}$ pushes activations off-manifold — the downstream layers receive inputs they've never seen during training. This is an out-of-distribution problem: the layers may produce unpredictable outputs. In practice, moderate $\\alpha$ produces interpretable behavioral shifts, but large $\\alpha$ degrades coherence. This is analogous to the \"curse of representation engineering\" — effective steering requires staying within the model's operational distribution."
    },
    {
      type: "mc",
      question: "The concept of \"alignment tax\" refers to:",
      options: ["Government taxes on AI companies specifically levied to fund safety compliance audits and regulatory oversight of deployed models", "The computational cost of training larger models, which grows superlinearly with parameter count due to the increased data requirements", "The capability cost of alignment training — the observation that safety training (RLHF, refusals, guardrails) can reduce the model's performance on benign tasks, creating a tradeoff between safety and helpfulness", "The salary cost of hiring alignment researchers, which diverts engineering resources from capability development toward safety-focused work"],
      correct: 2,
      explanation: "Alignment tax measures how much capability is lost to make a model safe. An ideal alignment method has zero tax — the model is both maximally capable and perfectly safe. In practice, safety training introduces refusals, hedging, and conservatism that can degrade performance: (1) over-refusal on benign queries, (2) reduced creativity due to conservative generation, (3) loss of calibration from RLHF. Minimizing alignment tax is a key research goal — methods like Constitutional AI and careful reward modeling aim to maintain capabilities while improving safety."
    },
    {
      type: "mc",
      question: "Scalable oversight is the problem of providing reliable training signal for models that exceed human capabilities in some domains. The recursive reward modeling (RRM) approach proposes to:",
      options: ["Use an AI assistant (itself aligned by human feedback) to help humans evaluate the next-level model's outputs — creating a chain where each model helps align its successor, with humans retaining oversight at each stage", "Use the same human annotators for all model generations, relying on their accumulated expertise to keep pace with increasingly capable model outputs", "Remove human oversight entirely and rely on self-play between model copies, trusting that adversarial dynamics will converge to aligned behavior", "Train only on tasks where human performance is superior to the model's, gradually shrinking the training distribution as the model improves"],
      correct: 0,
      explanation: "RRM creates a bootstrap chain: (1) Humans directly evaluate model $M_1$. (2) $M_1$ assists humans in evaluating the more capable $M_2$. (3) $M_2$ assists in evaluating $M_3$, and so on. At each stage, humans make the final judgment but are aided by the previous model. The key assumption is that *evaluating with assistance* is easier than *evaluating alone*, even as models become superhuman. This is related to the debate approach — both leverage asymmetric verification. The risk is that errors compound across the chain."
    },
    {
      type: "mc",
      question: "Consider a model trained with RLHF where the reward model was trained on human preferences. The model now encounters a novel domain (e.g., advanced scientific reasoning) where the reward model was never evaluated. According to Goodhart's taxonomy, which form of Goodhart's law is most relevant?",
      options: [
        "Regressional Goodhart — the reward model's statistical noise is exploited by the policy, which finds inputs that trigger high-variance reward predictions rather than genuinely high quality",
        "Causal Goodhart — the reward model captures correlations that are not causal, so optimizing the proxy in a new domain breaks the correlation structure that held in the training distribution",
        "Extremal Goodhart — the policy operates in a region of the output space far from the training distribution, where the reward model's predictions become unreliable extrapolations",
        "All forms of Goodhart's law are equally relevant in any domain shift scenario, since each variant contributes proportionally to the total divergence between proxy and true reward"
      ],
      correct: 1,
      explanation: "Causal Goodhart applies when the proxy (learned reward) correlates with the true objective through confounders or non-causal pathways. In the training domain, \"well-structured reasoning\" may correlate with \"correct answers\" because annotators rewarded both. In a novel scientific domain, the RM may still reward well-structured-*looking* reasoning even when the conclusions are wrong — the causal pathway (domain expertise) is absent. This is distinct from regressional Goodhart (exploiting noise) and extremal Goodhart (out-of-distribution behavior at optimization extremes), though all three can co-occur in practice."
    }
  ]
};
