// TBD placeholder modules for topics in the Muon/RL curriculum that need full content.
// Each placeholder has a detailed outline of what the module will cover.
// They show "Coming Soon" in the curriculum UI.

export const matrixNormsLearning = {
  id: "0.1-norms-learning-easy",
  sectionId: "0.1",
  title: "Matrix Norms for Machine Learning",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module will build intuition for matrix norms from first principles — the different ways to measure the \"size\" of a matrix and why the choice matters for optimization.\n\nTopics planned:\n- Frobenius norm as element-wise L2: $\\|A\\|_F = \\sqrt{\\sum_{ij} A_{ij}^2}$\n- Spectral norm as the largest singular value: $\\|A\\|_\\sigma = \\sigma_1(A)$\n- Operator norm interpretation: maximum stretching of a unit vector\n- Nuclear norm as the sum of singular values\n- How norm choice defines what \"small update\" means for an optimizer\n- Worked examples with small matrices to build geometric intuition"
    }
  ]
};

export const steepestDescentNormsLearning = {
  id: "0.3-steepest-descent-learning-medium",
  sectionId: "0.3",
  title: "Steepest Descent Under Different Norms",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module derives the steepest descent direction for different norm constraints and shows how norm choice fundamentally changes optimizer behavior.\n\nTopics planned:\n- Steepest descent framework: $\\min_\\Delta \\langle G, \\Delta \\rangle$ subject to $\\|\\Delta\\| \\leq \\epsilon$\n- Frobenius norm → gradient descent (update proportional to $G$)\n- Spectral norm → orthogonal projection $UV^T$ (Muon's update rule)\n- Element-wise $\\ell_\\infty$ norm → sign of gradient (SignSGD / Adam approximation)\n- Dual norms and why the gradient's SVD appears in the spectral case\n- Visual comparison of update directions for the same gradient under different norms"
    }
  ]
};

export const newtonSchulzLearning = {
  id: "0.3-newton-schulz-learning-medium",
  sectionId: "0.3",
  title: "Newton-Schulz Iteration for Matrix Orthogonalization",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 18,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module covers the Newton-Schulz iteration — the core computational primitive that makes Muon practical.\n\nTopics planned:\n- The matrix sign function and its connection to polar decomposition\n- Newton-Schulz iteration: $X_{k+1} = a_k X_k + b_k (X_kX_k^T)X_k + c_k (X_kX_k^T)^2 X_k$\n- Why 5 iterations of a 5th-order polynomial suffice for gradient orthogonalization\n- Convergence rate analysis: how initial scaling affects convergence\n- Computational cost: matrix multiplications on GPU vs SVD\n- Comparison with other orthogonalization methods (QR, Gram-Schmidt, SVD)\n- Numerical stability considerations in mixed-precision training"
    }
  ]
};

export const plasticityForgettingLearning = {
  id: "A.3-plasticity-learning-medium",
  sectionId: "A.3",
  title: "Plasticity & Catastrophic Forgetting in Fine-Tuning",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module covers the mechanisms of plasticity loss and catastrophic forgetting during LLM fine-tuning.\n\nTopics planned:\n- What is plasticity? A network's capacity to continue learning new tasks\n- Feature rank collapse: representations becoming low-dimensional\n- Dead units: neurons that output zero for all training inputs\n- Weight norm growth and effective learning rate decay\n- Gradient stiffening: why confident models produce less diverse gradients\n- How different training paradigms (SFT vs RL vs DPO) trigger these mechanisms differently\n- Measurement techniques: rank of activations, gradient diversity metrics, benchmark degradation curves"
    }
  ]
};

export const muonAtScaleLearning = {
  id: "B.4-muon-scale-learning-hard",
  sectionId: "B.4",
  title: "Muon at Scale: Distributed Training Considerations",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module covers the practical challenges and solutions for using Muon at large scale, based on 'Muon is Scalable for LLM Training'.\n\nTopics planned:\n- Why Muon's Newton-Schulz iteration requires the full (not sharded) gradient matrix\n- All-reduce patterns for Muon vs AdamW in data-parallel training\n- Tensor parallelism: how Muon interacts with column/row parallelism\n- Memory footprint comparison at 7B, 13B, and 70B scales\n- Mixed-precision considerations for Newton-Schulz numerical stability\n- Scaling curves: Muon vs AdamW loss at matched compute budgets\n- Practical recommendations for when Muon's overhead is worth the convergence benefit"
    }
  ]
};

export const muonVsAdamLearning = {
  id: "B.4-muon-vs-adam-learning-hard",
  sectionId: "B.4",
  title: "Muon vs AdamW: Singular Value Dynamics",
  moduleType: "learning",
  difficulty: "hard",
  estimatedMinutes: 25,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module provides a deep comparison of Muon and AdamW focusing on their spectral behavior, based on 'Muon Outperforms Adam in Tail-End Associative Memory Learning'.\n\nTopics planned:\n- How Adam's per-element scaling affects the singular value structure of weight updates\n- How Muon's orthogonalization equalizes singular value directions\n- Associative memory formation in transformer weight matrices\n- Tail-end learning: why rare input-output associations require small singular value directions\n- Why Muon captures tail patterns that Adam misses\n- Spectral analysis of weight evolution during training: Adam vs Muon trajectories\n- When Adam wins: scenarios where per-element adaptivity matters more than spectral structure"
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
      content: "This module covers how RL fine-tuning modifies sparse subnetworks within LLMs, based on 'Reinforcement Learning Finetunes Small Subnetworks in Large Language Models'.\n\nTopics planned:\n- Weight change analysis: measuring $\\|W_{\\text{final}} - W_{\\text{init}}\\|$ per layer\n- The finding: 3-10% of parameters account for 90%+ of weight change during RL\n- Layer-wise patterns: early layers barely change, late layers change most\n- MLP vs attention: why MLP layers show more concentrated changes\n- Comparison with SFT: broader vs sparser update patterns\n- Connection to the lottery ticket hypothesis\n- Implications for parameter-efficient RL (LoRA rank allocation)\n- How optimizer choice affects which subnetworks are activated"
    }
  ]
};

export const onlineRlLlmLearning = {
  id: "A.3-online-rl-learning-medium",
  sectionId: "A.3",
  title: "RL's Razor: Why Online RL Forgets Less",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  tbd: true,
  steps: [
    {
      type: "info",
      title: "Coming Soon",
      content: "This module covers the RL's Razor principle — why on-policy RL methods cause less catastrophic forgetting than off-policy methods, based on 'RL's Razor: Why Online Reinforcement Learning Forgets Less'.\n\nTopics planned:\n- Empirical evidence: PPO vs DPO forgetting curves on pretraining benchmarks\n- The razor principle: on-policy training as implicit regularization\n- Why on-policy KL constraints are self-tightening\n- The off-policy feedback loop: mismatch → large updates → more mismatch\n- Interaction between on-policy training and optimizer choice\n- Why Muon + on-policy RL is theoretically well-motivated\n- Practical implications for choosing between PPO and DPO"
    }
  ]
};
