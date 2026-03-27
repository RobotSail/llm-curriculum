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
      options: [
        "Strong models trained with weak supervision consistently exceed their weak supervisors, recovering a significant fraction of the weak-to-strong performance gap",
        "Weak supervisors always fail — the strong model degrades to weak-level performance because it cannot distinguish signal from noise in the supervisor's labels",
        "Weak and strong models achieve identical performance regardless of capacity difference, since the supervision signal is the binding constraint on final quality",
        "The strong model learns to completely ignore the weak supervisor and instead relies on its pretrained representations, making the supervision step unnecessary"
      ],
      correct: 0,
      explanation: "Burns et al. found that when a weak model (e.g., GPT-2-level) provides labels for training a strong model (e.g., GPT-4-level), the strong model consistently exceeds the weak supervisor's performance. This is encouraging for scalable oversight: it suggests that superhuman models might partially self-correct even when trained with imperfect human feedback. However, the recovery is incomplete — there is a \"alignment tax\" — and the gap grows on harder tasks, indicating that weak-to-strong generalization alone is insufficient for frontier alignment."
    },
    {
      type: "mc",
      question: "In AI safety via debate (Irving et al., 2018), two AI agents argue opposing sides while a human judge decides the winner. Why does this protocol theoretically scale to superhuman capabilities?",
      options: [
        "The debate always converges to truth in finite rounds because each exchange provably eliminates at least one incorrect hypothesis from consideration",
        "A human only needs to verify arguments (a simpler task) rather than generate solutions — if lying requires more complex arguments, the honest debater has an advantage",
        "The human judge can always detect AI deception because lying produces detectable statistical artifacts in the linguistic structure of the arguments",
        "Debate removes the need for human oversight entirely by ensuring that the Nash equilibrium of the adversarial game always produces truthful outcomes"
      ],
      correct: 1,
      explanation: "The key insight is *asymmetric verification*: checking an argument is easier than generating one. In debate, a lie can be exposed by pointing to a specific flaw, which the human can verify. Under the conjecture that truth is \"simpler\" than consistent deception (lies require more complex supporting arguments), the honest debater has an advantage in the Nash equilibrium. This allows a human with limited capabilities to adjudicate between superhuman arguments. The protocol assumes the human can verify atomic claims — the debaters recursively decompose complex arguments until reaching human-verifiable steps."
    },
    {
      type: "mc",
      question: "Representation engineering / steering vectors involve finding directions $\\mathbf{v}$ in a model's activation space such that adding $\\alpha \\mathbf{v}$ to intermediate activations controls a specific behavior (e.g., honesty). These vectors are typically found by:",
      options: [
        "Random search in activation space, sampling directions until one produces the desired behavioral shift when applied as an additive intervention during inference",
        "Pruning attention heads one at a time until behavior changes, then identifying which heads encode the target concept via ablation-based behavioral analysis",
        "Training a separate binary classifier on model outputs and repurposing the learned classifier's weight vector as the steering direction for activation intervention",
        "Computing the difference in mean activations between contrastive prompt pairs (e.g., honest vs. dishonest), capturing the concept direction in representation space"
      ],
      correct: 3,
      explanation: "The standard approach: (1) construct contrastive pairs — prompts that elicit opposite behaviors (e.g., truthful vs. deceptive responses). (2) Run both through the model, extract activations at a chosen layer. (3) Compute $\\mathbf{v} = \\mathbb{E}[\\mathbf{h}_{\\text{positive}}] - \\mathbb{E}[\\mathbf{h}_{\\text{negative}}]$. This difference vector captures the linear direction in representation space corresponding to the concept. Adding $\\alpha \\mathbf{v}$ at inference time \"steers\" the model along this direction. More sophisticated methods use PCA or linear probes on the contrastive activations."
    },
    {
      type: "mc",
      question: "OpenAI's o1/o3 models use RL for reasoning, training on chain-of-thought traces. A critical design choice is training the model to produce long \"thinking\" traces before answering. What RL signal structure makes this work?",
      options: [
        "Dense per-token reward is given at every step of the reasoning trace, providing fine-grained supervision that guides each individual generation step toward correctness",
        "The model is trained purely with supervised learning on expert traces from human mathematicians, with no reinforcement learning component used in the pipeline",
        "Outcome-based reward (final answer correctness) combined with process supervision — the RL algorithm must credit intermediate reasoning steps despite sparse reward",
        "Reward is given only for short responses to encourage efficiency, penalizing any reasoning trace that exceeds a fixed token budget threshold during training"
      ],
      correct: 2,
      explanation: "The o1/o3 paradigm uses RL with sparse outcome reward (is the final answer correct?) to train extended reasoning. The challenge is credit assignment: a 10,000-token reasoning trace might have one reward signal. Process reward models (PRMs) and process supervision help bridge this by evaluating intermediate steps. The RL training (likely a variant of PPO or GRPO) must learn which reasoning patterns — backtracking, verification, decomposition — lead to correct outcomes. This is \"RL for inference-time compute scaling\" — the model learns to think longer and more carefully."
    },
    {
      type: "mc",
      question: "Process supervision (as in the \"Let's Verify Step by Step\" paper, Lightman et al. 2023) provides feedback on each reasoning step. Compared to outcome supervision, process supervision:",
      options: [
        "Provides denser reward, enables better credit assignment, and catches flawed reasoning even when the final answer is accidentally correct — but requires costly per-step annotations",
        "Always produces worse results but is cheaper to implement, since it avoids the computational overhead of training a separate step-level evaluation model alongside the policy",
        "Does not require any human annotations because the process reward signal can be derived automatically from the logical structure of the reasoning chain alone",
        "Is only applicable to non-mathematical tasks where reasoning steps are expressed in natural language sentences rather than formal symbolic mathematical notation"
      ],
      correct: 0,
      explanation: "Process supervision labels each step as correct/incorrect, creating a rich training signal. Benefits: (1) **Credit assignment**: identifies exactly where reasoning fails, rather than penalizing the entire trace for a wrong final answer. (2) **Avoiding reward hacking**: a correct final answer via flawed reasoning gets negative process feedback. (3) **Denser signal**: reduces variance in policy gradients. The cost is annotation: labeling each step requires expert annotators who understand the reasoning, which is far more expensive than checking final answers. Lightman et al. showed process supervision substantially outperforms outcome supervision on math reasoning."
    },
    {
      type: "mc",
      question: "Red-teaming in the context of LLM safety involves systematically trying to elicit harmful or undesired outputs. The most effective red-teaming approaches combine:",
      options: [
        "Only automated tools (gradient-based adversarial search, fuzzing) with no human involvement — humans introduce systematic bias that narrows discovered vulnerabilities",
        "Human creativity for novel attack vectors with automated methods (adversarial prompt optimization, model-based red-teaming) for scaling coverage across variations",
        "Only manual testing by a small expert team, since automated methods generate exclusively superficial attacks that rarely uncover meaningful safety-relevant failure modes",
        "Random input generation without guided structure, relying on statistical coverage of the full input space to eventually trigger safety failures through brute force"
      ],
      correct: 1,
      explanation: "Effective red-teaming requires both modalities: (1) **Human red-teamers** discover novel failure modes that require creativity, cultural knowledge, and adversarial reasoning (e.g., role-playing attacks, multi-turn manipulation). (2) **Automated methods** like GCG (gradient-based adversarial suffix optimization), model-based red-teaming (using another LLM to generate attacks), and fuzzing scale to thousands of attack variations. The most comprehensive programs (e.g., Anthropic's, Meta's) layer both: humans identify attack categories, automation fills in the coverage matrix."
    },
    {
      type: "mc",
      question: "A steering vector $\\mathbf{v}$ is applied as $\\mathbf{h}'_l = \\mathbf{h}_l + \\alpha \\mathbf{v}$ at layer $l$. The coefficient $\\alpha$ controls steering strength. A known failure mode is that large $|\\alpha|$ values cause:",
      options: [
        "Higher quality outputs because the stronger signal more decisively activates the target behavioral direction in the model's learned representation space",
        "Faster inference because the larger activation perturbation causes the model to converge on high-confidence token predictions more quickly during decoding",
        "The model to ignore the vector entirely because layer normalization saturates and neutralizes any perturbation that exceeds its learned normalization threshold",
        "Distribution shift — the modified $\\mathbf{h}'_l$ moves off the activation manifold downstream layers expect, causing incoherent or degenerate text generation"
      ],
      correct: 3,
      explanation: "The model's downstream layers (layers $> l$) are trained on activations from the natural distribution. Adding a large $\\alpha \\mathbf{v}$ pushes activations off-manifold — the downstream layers receive inputs they've never seen during training. This is an out-of-distribution problem: the layers may produce unpredictable outputs. In practice, moderate $\\alpha$ produces interpretable behavioral shifts, but large $\\alpha$ degrades coherence. This is analogous to the \"curse of representation engineering\" — effective steering requires staying within the model's operational distribution."
    },
    {
      type: "mc",
      question: "The concept of \"alignment tax\" refers to:",
      options: [
        "Government taxes on AI companies specifically levied to fund mandatory safety compliance audits and regulatory oversight of all deployed frontier models",
        "The computational cost of training larger models, which grows superlinearly with parameter count due to the correspondingly increased data and compute requirements",
        "The capability cost of safety training — RLHF, refusals, and guardrails can reduce performance on benign tasks, creating a safety-helpfulness tradeoff",
        "The salary cost of hiring alignment researchers, which diverts engineering resources from capability development toward safety-focused research and evaluation work"
      ],
      correct: 2,
      explanation: "Alignment tax measures how much capability is lost to make a model safe. An ideal alignment method has zero tax — the model is both maximally capable and perfectly safe. In practice, safety training introduces refusals, hedging, and conservatism that can degrade performance: (1) over-refusal on benign queries, (2) reduced creativity due to conservative generation, (3) loss of calibration from RLHF. Minimizing alignment tax is a key research goal — methods like Constitutional AI and careful reward modeling aim to maintain capabilities while improving safety."
    },
    {
      type: "mc",
      question: "Scalable oversight is the problem of providing reliable training signal for models that exceed human capabilities in some domains. The recursive reward modeling (RRM) approach proposes to:",
      options: [
        "Use an AI assistant (aligned by human feedback) to help humans evaluate the next model — creating a chain where each model helps align its successor",
        "Use the same human annotators for all model generations, relying on their accumulated domain expertise to keep pace with increasingly capable model outputs",
        "Remove human oversight entirely and rely on self-play between model copies, trusting that adversarial dynamics will converge to well-aligned behavioral equilibria",
        "Train only on tasks where human performance exceeds the model's, gradually shrinking the training distribution as the model's capabilities continue to improve"
      ],
      correct: 0,
      explanation: "RRM creates a bootstrap chain: (1) Humans directly evaluate model $M_1$. (2) $M_1$ assists humans in evaluating the more capable $M_2$. (3) $M_2$ assists in evaluating $M_3$, and so on. At each stage, humans make the final judgment but are aided by the previous model. The key assumption is that *evaluating with assistance* is easier than *evaluating alone*, even as models become superhuman. This is related to the debate approach — both leverage asymmetric verification. The risk is that errors compound across the chain."
    },
    {
      type: "mc",
      question: "Consider a model trained with RLHF where the reward model was trained on human preferences. The model now encounters a novel domain (e.g., advanced scientific reasoning) where the reward model was never evaluated. According to Goodhart's taxonomy, which form of Goodhart's law is most relevant?",
      options: [
        "Regressional Goodhart — the policy exploits statistical noise in the RM's predictions, finding inputs that trigger high-variance reward estimates rather than genuine quality",
        "Causal Goodhart — the RM captures correlations that aren't causal, so optimizing the proxy in a new domain breaks the correlation structure from training",
        "Extremal Goodhart — the policy operates far from the RM's training distribution, where reward predictions become unreliable out-of-distribution extrapolations",
        "All forms are equally relevant in any domain shift scenario, since each variant contributes proportionally to the total proxy-vs-true reward divergence"
      ],
      correct: 1,
      explanation: "Causal Goodhart applies when the proxy (learned reward) correlates with the true objective through confounders or non-causal pathways. In the training domain, \"well-structured reasoning\" may correlate with \"correct answers\" because annotators rewarded both. In a novel scientific domain, the RM may still reward well-structured-*looking* reasoning even when the conclusions are wrong — the causal pathway (domain expertise) is absent. This is distinct from regressional Goodhart (exploiting noise) and extremal Goodhart (out-of-distribution behavior at optimization extremes), though all three can co-occur in practice."
    }
  ]
};
