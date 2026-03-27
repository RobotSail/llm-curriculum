// Focused module: Group Relative Policy Optimization (GRPO)
// Covers GRPO's value-free advantage estimation, group sampling, and its use in DeepSeek-R1.

export const grpoLearning = {
  id: "A.3-grpo-learning-medium",
  sectionId: "A.3",
  title: "Group Relative Policy Optimization (GRPO)",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Value Function Problem in LLM RL",
      content: "PPO for language models requires two large neural networks: the **policy** $\\pi_\\theta$ (the LLM being fine-tuned) and a **value function** $V_\\phi$ (a critic that predicts expected return at each token position).\n\nThe value function causes three practical problems:\n\n1. **Memory**: For a 7B parameter LLM, the value network is another 7B model — doubling GPU memory from ~28 GB to ~56 GB (in FP16). With optimizer states, the total can exceed what fits on available hardware.\n\n2. **Approximation error**: $V_\\phi$ must predict future returns at every token position, but the reward is typically sparse (only at the end). Errors in $V_\\phi$ propagate through advantage estimation, producing noisy gradients that can destabilize training.\n\n3. **Training complexity**: The value function must be trained alongside the policy, with its own learning rate, loss function, and training schedule. Getting this co-training right requires careful hyperparameter tuning.\n\n**GRPO** (Group Relative Policy Optimization), introduced in DeepSeek-Math and used prominently in DeepSeek-R1, eliminates the value function entirely by computing advantages from groups of sampled responses."
    },
    {
      type: "mc",
      question: "A team fine-tuning a 70B LLM with PPO finds that GPU memory is the binding constraint. The value function alone consumes approximately:",
      options: [
        "A negligible amount of memory because the value head is just a single linear layer on top of the frozen policy",
        "Exactly half the memory of the policy because the value function only processes the prompt, not the generated response tokens",
        "More memory than the policy model because the value function requires storing return targets for every token position in the rollout buffer",
        "Roughly the same memory as the policy model — another 70B parameters plus its optimizer states, since the value network is initialized from the same pretrained LM"
      ],
      correct: 3,
      explanation: "In standard RLHF implementations (e.g., TRL, OpenRLHF), the value function is initialized from the same pretrained LM as the policy, with a scalar head replacing the LM head. This means it has nearly the same parameter count as the policy. For a 70B model in FP16, that's ~140 GB just for weights, plus optimizer states (2x for Adam = ~280 GB for moments). The value network's memory cost is essentially equal to the policy's."
    },
    {
      type: "info",
      title: "GRPO's Core Idea: Group Sampling",
      content: "Instead of learning a value function to estimate expected return, GRPO takes a simpler approach: **sample multiple responses per prompt and use their relative rewards as advantages**.\n\nFor each prompt $x$ in a batch:\n1. Sample $G$ complete responses $\\{y_1, y_2, \\dots, y_G\\}$ from the current policy $\\pi_\\theta$\n2. Score each response with the reward model: $\\{r_1, r_2, \\dots, r_G\\}$\n3. Normalize rewards within the group:\n$$\\hat{A}_i = \\frac{r_i - \\text{mean}(\\mathbf{r})}{\\text{std}(\\mathbf{r})}$$\n\nThis normalized reward $\\hat{A}_i$ serves as the advantage for response $y_i$. Responses better than the group average get positive advantage; responses worse than average get negative advantage.\n\nThe group mean $\\text{mean}(\\mathbf{r})$ acts as a **prompt-specific baseline** — it adapts naturally to each prompt's difficulty. Hard prompts where all responses score low will have a low baseline; easy prompts will have a high baseline. No learned value function needed."
    },
    {
      type: "mc",
      question: "GRPO samples $G=8$ responses for a prompt. All responses receive rewards between 0.85 and 0.92. Under PPO with a poorly calibrated value function predicting $V(s) = 0.5$, versus GRPO, what happens?",
      options: [
        "PPO correctly handles this by clipping the importance ratio, while GRPO fails because all advantages are near zero after normalization",
        "PPO computes large positive advantages for all responses (since all rewards exceed $V=0.5$), reinforcing everything uniformly, while GRPO still discriminates between 0.85 and 0.92 through group normalization",
        "Both methods produce identical gradient updates because the absolute reward scale cancels out in the policy gradient formula",
        "GRPO produces unstable gradients because the standard deviation of the group rewards is very small, causing division by near-zero"
      ],
      correct: 1,
      explanation: "With $V(s) = 0.5$ and all rewards near $0.9$, PPO's advantages are all approximately $+0.4$ — it reinforces every response almost equally, learning nothing about which responses are better. GRPO normalizes within the group: the 0.92 response gets a positive advantage, the 0.85 response gets a negative one. The model learns to distinguish good from mediocre. Small std is handled by adding a small $\\epsilon$ to the denominator in practice."
    },
    {
      type: "info",
      title: "The GRPO Objective",
      content: "GRPO combines its group-normalized advantages with PPO-style clipping and a KL penalty. For each response $y_i$ in the group, the per-token objective is:\n\n$$L_t^{\\text{GRPO}} = \\min\\left(\\rho_t \\hat{A}_i, \\; \\text{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon) \\hat{A}_i\\right) - \\beta \\, D_{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})$$\n\nwhere $\\rho_t = \\frac{\\pi_\\theta(y_t | s_t)}{\\pi_{\\text{old}}(y_t | s_t)}$ is the importance sampling ratio.\n\nKey differences from PPO:\n- **Advantage $\\hat{A}_i$** is the same for all tokens in response $i$ — it's a sequence-level signal, not token-level\n- **No GAE**: Since there's no value function, there's no TD error and no Generalized Advantage Estimation. The advantage is simply the normalized reward.\n- **KL penalty** is applied explicitly in the objective rather than as a per-token reward shaping term\n\nThe clipping mechanism works identically to PPO: it prevents the policy from changing too much in a single update by capping the benefit of large probability ratio changes."
    },
    {
      type: "mc",
      question: "In GRPO, the advantage $\\hat{A}_i$ is constant across all tokens in response $y_i$. Compared to PPO's per-token advantages from GAE, what is the tradeoff?",
      options: [
        "GRPO sacrifices fine-grained credit assignment (which tokens were good/bad) but avoids the bias and instability introduced by an imperfect value function",
        "Per-token advantages are always strictly better because they assign more credit to tokens that actually caused the high or low reward",
        "GRPO's constant advantage provides better credit assignment because it ensures every token receives the same learning signal uniformly",
        "There is no meaningful difference because GAE with $\\lambda=1$ also produces constant advantages across all tokens in a sequence"
      ],
      correct: 0,
      explanation: "This is GRPO's fundamental tradeoff. PPO with GAE can theoretically assign different advantages to different tokens (early tokens that set up a good response vs. filler tokens). But this requires an accurate value function, which is hard to learn. GRPO gives up this per-token discrimination in exchange for removing the value function entirely. In practice, the instability from value function errors often negates the theoretical benefit of per-token advantages. GAE with $\\lambda=1$ gives Monte Carlo returns minus baseline, which varies per token due to different $V(s_t)$ at each position."
    },
    {
      type: "info",
      title: "Group Size and Variance Reduction",
      content: "The group size $G$ controls the quality of GRPO's advantage estimates:\n\n**Small $G$ (e.g., 2-4)**:\n- Cheap to sample (fewer forward passes)\n- Noisy baseline: the group mean of 2 samples is a poor estimate of the true expected reward\n- High variance in advantage estimates\n\n**Large $G$ (e.g., 16-64)**:\n- Better baseline: the group mean converges toward $\\mathbb{E}_{\\pi}[r(y) | x]$\n- More diverse responses for ranking — the model sees a wider range of quality levels\n- But: $G$ times more forward passes per prompt, proportionally more compute\n\nDeepSeek-R1 uses $G = 16$ as a sweet spot. The compute cost is $16\\times$ more generation per prompt, but this is partially offset by eliminating the value network's forward and backward passes.\n\n**Variance analysis**: The variance of the group mean as a baseline estimator is $\\text{Var}(r) / G$. Doubling $G$ halves the baseline variance. Compare to a learned value function, which has fixed approximation bias regardless of batch size — GRPO's statistical baseline can always be improved by sampling more."
    },
    {
      type: "mc",
      question: "A team switches from PPO to GRPO with $G=16$ for a 7B model. How does total compute per training step change?",
      options: [
        "Compute decreases because eliminating the value network removes half of all forward and backward passes during the optimization phase",
        "Compute stays roughly the same: the 16x increase in generation is exactly offset by removing the value network's training cost",
        "Compute is unpredictable because it depends entirely on the average response length, which varies per prompt",
        "Compute increases overall — generating 16 responses per prompt costs more than PPO's single response plus value function, though memory usage drops significantly"
      ],
      correct: 3,
      explanation: "PPO generates 1 response per prompt and runs both policy and value network forward/backward passes. GRPO generates 16 responses per prompt (16x generation cost) but eliminates the value network entirely. Generation (autoregressive decoding) is expensive — it's 16x the inference cost. The value network savings (roughly 1x forward + 1x backward) don't compensate. However, GRPO's major win is memory: no value network means the full GPU memory goes to the policy, enabling larger models or larger batch sizes."
    },
    {
      type: "info",
      title: "GRPO in DeepSeek-R1: Scaling RL for Reasoning",
      content: "DeepSeek-R1 demonstrated GRPO at scale for training reasoning capabilities. Key design choices:\n\n**Rule-based rewards**: For math and coding tasks, DeepSeek-R1 uses **verifiable rewards** — checking whether the final answer matches the ground truth or whether code passes test cases. No reward model needed, eliminating reward hacking entirely.\n\n**Emergent chain-of-thought**: With GRPO and verifiable rewards, models spontaneously develop structured reasoning (\"Let me think step by step...\") without being explicitly trained to do so. The RL signal from correctness is sufficient.\n\n**Multi-stage training**:\n1. Cold start with SFT on curated reasoning examples\n2. GRPO with rule-based rewards on reasoning tasks\n3. Additional SFT for formatting and helpfulness\n4. Final GRPO round with mixed reward signals (rule-based + RM)\n\n**Why GRPO works well for reasoning**: Reasoning tasks have **high outcome variance** — for a given math problem, some solution attempts succeed and others fail. GRPO's group sampling naturally captures this variance, creating strong training signal. The group might contain 3 correct and 13 incorrect solutions, providing clear contrast for learning."
    },
    {
      type: "mc",
      question: "DeepSeek-R1 uses rule-based rewards (correctness checking) instead of a learned reward model for math reasoning. What advantage does this specifically provide when combined with GRPO?",
      options: [
        "Rule-based rewards are faster to compute, which offsets GRPO's 16x generation overhead and makes the total training pipeline cheaper than PPO with a reward model",
        "Rule-based rewards produce continuous scores between 0 and 1, which GRPO requires for meaningful group normalization and advantage calculation",
        "Rule-based rewards eliminate reward hacking entirely — the signal is ground-truth correctness, so the model cannot exploit reward model weaknesses regardless of how much it is optimized",
        "Rule-based rewards enable GRPO to use larger group sizes because they scale linearly with batch size, unlike neural reward models that have quadratic memory scaling"
      ],
      correct: 2,
      explanation: "The core advantage of rule-based (verifiable) rewards is robustness: the reward is binary (correct/incorrect) and cannot be gamed. A learned RM can be exploited — the model might find outputs that score high on the RM but are actually wrong or incoherent. With ground-truth checking, optimization pressure always points toward genuine correctness. This is especially powerful with GRPO's intensive sampling: the model tries many approaches and gets rewarded only for ones that actually work."
    },
    {
      type: "info",
      title: "GRPO vs PPO vs REINFORCE",
      content: "Positioning GRPO relative to other policy gradient methods:\n\n| Aspect | REINFORCE | PPO | GRPO |\n|--------|-----------|-----|------|\n| Baseline | Optional (learned or constant) | Learned $V_\\phi(s)$ | Group mean $\\mu_G$ |\n| Advantage granularity | Per-sequence | Per-token (via GAE) | Per-sequence |\n| Value network needed | No | Yes | No |\n| Memory overhead | Low | High (~2x model) | Low |\n| Clipping | No | Yes | Yes |\n| Samples per prompt | 1+ | 1 | $G$ (typically 8-64) |\n| Credit assignment | Coarse | Fine-grained | Coarse |\n\n**When to use GRPO over PPO**:\n- Memory-constrained settings (large models, limited GPUs)\n- Tasks with verifiable rewards where the value function adds little benefit\n- When value function instability is causing training problems\n\n**When to use PPO over GRPO**:\n- When per-token credit assignment matters (e.g., specific formatting requirements)\n- When compute budget is tight and generating $G$ responses per prompt is too expensive\n- When a reliable value function can be trained (e.g., using process reward models)"
    },
    {
      type: "mc",
      question: "A researcher is fine-tuning a 70B model for summarization quality using a learned reward model. They have 8 A100-80GB GPUs. Which method is most practical?",
      options: [
        "PPO, because the value function's per-token advantages are essential for learning good summarization structure and the hardware can support both networks with model parallelism",
        "GRPO with $G=64$, because more samples always produce better advantages and the compute cost is irrelevant with 8 GPUs",
        "GRPO with moderate $G$ (e.g., 8-16), because a 70B policy + 70B value network + optimizer states would exceed 640 GB, while GRPO fits the policy alone with room for generation batches",
        "REINFORCE without any baseline, because it uses the least memory and baselines don't help when rewards come from a neural reward model"
      ],
      correct: 2,
      explanation: "A 70B model in FP16 is ~140 GB for weights alone. With Adam optimizer states (2x), that's ~420 GB per model. PPO would need this for both policy and value network: ~840 GB total — exceeding 8 × 80 GB = 640 GB even before activations. GRPO with the policy alone fits comfortably (~420 GB for weights + optimizer), leaving room for generation batches. $G=64$ would be very expensive in generation compute for a 70B model. REINFORCE without baseline would have extreme variance."
    },
    {
      type: "mc",
      question: "GRPO normalizes rewards within each prompt's group: $\\hat{A}_i = (r_i - \\mu_G) / \\sigma_G$. If a prompt is so easy that all $G$ responses receive nearly identical high rewards, what happens to the advantages?",
      options: [
        "All advantages become very large and positive, causing the policy to strongly reinforce all responses for this prompt equally",
        "The advantages are numerically unstable due to division by near-zero $\\sigma_G$, but in practice a small $\\epsilon$ is added to the denominator, yielding near-zero advantages that produce minimal gradient",
        "The group normalization detects this case and falls back to using absolute rewards instead of relative advantages for this prompt",
        "The advantages alternate between positive and negative because normalization forces them to have zero mean, even when all rewards are identical"
      ],
      correct: 1,
      explanation: "When all rewards are nearly identical, $\\sigma_G \\approx 0$. Without protection, division would explode. In practice, implementations add $\\epsilon$ (e.g., $10^{-8}$) to the denominator: $\\hat{A}_i = (r_i - \\mu_G) / (\\sigma_G + \\epsilon)$. With tiny $\\sigma_G$, the numerator $(r_i - \\mu_G)$ is also tiny, so advantages are near zero. This is correct behavior: if all responses are equally good, there's nothing to learn from this prompt. The gradient contribution is minimal."
    }
  ]
};
