// TBD placeholder modules for the Muon Optimizer in RL curriculum.
// These contain no MC questions, so they won't appear in warmup.
// They exist so the curriculum can reference them and show "Coming Soon".

export const muonOptimizerLearning = {
  id: "0.3-muon-learning-easy",
  sectionId: "0.3",
  title: "Muon Optimizer Fundamentals",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module will cover the Muon optimizer: Newton-Schulz iteration for approximate matrix inversion, spectral steepest descent, the relationship to Shampoo and SOAP optimizers, and why Muon's per-layer orthogonalization of gradients leads to better conditioning.\n\nTopics planned:\n- Steepest descent under the spectral norm\n- Newton-Schulz iterations as a matrix-free approximation\n- Comparison with Shampoo's Kronecker-factored preconditioning\n- Per-layer update normalization and its effect on training dynamics"
    }
  ]
};

export const onlineRlLlmLearning = {
  id: "A.3-online-rl-learning-medium",
  sectionId: "A.3",
  title: "Online RL for LLMs: Plasticity & Forgetting",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 25,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module will cover online vs offline reinforcement learning in the context of LLM fine-tuning, with a focus on why online RL forgets less (RL's Razor).\n\nTopics planned:\n- On-policy (PPO, REINFORCE) vs off-policy (DPO, offline RLHF) training\n- Loss of plasticity in neural networks during RL fine-tuning\n- Why online RL naturally regularizes: the razor principle\n- KL penalty mechanics and its interaction with optimizer choice\n- Practical implications for reward hacking and alignment stability"
    }
  ]
};

export const rlSubnetworksLearning = {
  id: "A.3-rl-subnets-learning-medium",
  sectionId: "A.3",
  title: "RL Fine-Tuning & Subnetwork Structure",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module will cover how reinforcement learning fine-tuning affects small subnetworks within large language models.\n\nTopics planned:\n- Weight change analysis during RL fine-tuning: which parameters move?\n- Sparse update patterns and the lottery ticket hypothesis connection\n- Layer-wise analysis of RL-induced changes\n- How optimizer choice (Adam vs Muon) affects which subnetworks are activated\n- Implications for parameter-efficient RL fine-tuning"
    }
  ]
};

export const muonVsAdamLearning = {
  id: "B.4-muon-vs-adam-learning-hard",
  sectionId: "B.4",
  title: "Muon vs AdamW: Spectral Analysis & Tail Learning",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 30,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module will provide a deep comparison between the Muon and AdamW optimizers, focusing on their spectral properties and behavior in tail-end learning.\n\nTopics planned:\n- Spectral analysis of weight updates: how Muon and Adam differ in singular value space\n- Associative memory formation and why Muon excels at tail-end patterns\n- Muon's scalability properties for large LLM training\n- When to use Muon vs AdamW: practical decision framework\n- Distributed training considerations for Muon (all-reduce of full gradient matrices)\n- Interaction between optimizer choice and RL fine-tuning dynamics"
    }
  ]
};
