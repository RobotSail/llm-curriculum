import { MODULES } from '../modules';

/**
 * Build a flat lookup: moduleId -> module object (with sectionId attached).
 * The MODULES registry is keyed by sectionId, so we invert it once.
 */
export function buildModuleLookup() {
  const lookup = {};
  for (const [sectionId, mods] of Object.entries(MODULES)) {
    for (const mod of mods) {
      lookup[mod.id] = { ...mod, sectionId: mod.sectionId || sectionId };
    }
  }
  return lookup;
}

export const RECOMMENDED_CURRICULUMS = [
  {
    id: "muon-rl",
    title: "Muon Optimizer in RL for LLMs",
    description: "A focused, first-principles study path for understanding the Muon optimizer and its application in reinforcement learning fine-tuning of large language models. Builds up from matrix norms and KL divergence through optimizer internals and RL mechanics to the cutting-edge research on Muon vs AdamW in RL settings.",
    color: "#8B5CF6",
    items: [
      // === Phase 1: Linear Algebra Foundations ===
      {
        moduleId: "0.1-norms-learning-easy",
        phase: "Linear Algebra Foundations",
        note: "Matrix norms (Frobenius, spectral, nuclear) define what 'small update' means for an optimizer. Muon uses spectral norm; Adam implicitly uses element-wise norms.",
        tbd: true
      },
      {
        moduleId: "0.1-easy",
        phase: "Linear Algebra Foundations",
        note: "SVD, eigenvalues, and spectral decomposition — the language of Muon's update rule and the basis for understanding singular value dynamics."
      },
      {
        moduleId: "0.1-medium",
        phase: "Linear Algebra Foundations",
        note: "Matrix calculus and low-rank structure. Muon's Newton-Schulz iteration operates on gradient matrices; Jacobians and Eckart-Young theorem provide context."
      },

      // === Phase 2: Information Theory for RL ===
      {
        moduleId: "0.2-forward-kl-learning-easy",
        phase: "Information Theory for RL",
        note: "Forward KL divergence — mode-covering behavior. Standard pretraining (cross-entropy) minimizes forward KL. Understanding this direction is prerequisite for reverse KL."
      },
      {
        moduleId: "0.2-reverse-kl-learning-easy",
        phase: "Information Theory for RL",
        note: "Reverse KL divergence — mode-seeking behavior. The RLHF KL penalty is reverse KL. This module covers why it prevents reward hacking and how β controls the constraint."
      },
      {
        moduleId: "0.2-assess-divergences",
        phase: "Information Theory for RL",
        note: "Test your understanding of KL divergence, JS divergence, and f-divergences. These appear throughout RL fine-tuning objectives."
      },

      // === Phase 3: Optimization Prerequisites ===
      {
        moduleId: "0.3-adam-learning-easy",
        phase: "Optimization Prerequisites",
        note: "Adam & AdamW from first principles: momentum, adaptive learning rates, weight decay. You need to deeply understand Adam before you can appreciate what Muon does differently."
      },
      {
        moduleId: "0.3-assess",
        phase: "Optimization Prerequisites",
        note: "Assess your optimization theory baseline — SGD, momentum, second-order methods, convergence. Identify gaps before diving into Muon."
      },
      {
        moduleId: "0.3-steepest-descent-learning-medium",
        phase: "Optimization Prerequisites",
        note: "Steepest descent under different norms: Frobenius → gradient descent, spectral → Muon, element-wise ℓ∞ → SignSGD/Adam. This is the conceptual bridge to Muon.",
        tbd: true
      },
      {
        moduleId: "0.3-newton-schulz-learning-medium",
        phase: "Optimization Prerequisites",
        note: "Newton-Schulz iteration — the computational trick that makes Muon practical. Approximate matrix orthogonalization using only matrix multiplications.",
        tbd: true
      },

      // === Phase 4: The Muon Optimizer ===
      {
        moduleId: "0.3-muon-learning-easy",
        phase: "The Muon Optimizer",
        note: "Muon fundamentals: spectral steepest descent, Newton-Schulz orthogonalization, relationship to Shampoo/SOAP, per-layer normalization, memory and compute costs."
      },
      {
        moduleId: "B.4-muon-vs-adam-learning-hard",
        phase: "The Muon Optimizer",
        note: "Deep comparison of Muon vs AdamW: singular value dynamics, tail-end associative memory learning, and when each optimizer wins.",
        tbd: true
      },
      {
        moduleId: "B.4-muon-scale-learning-hard",
        phase: "The Muon Optimizer",
        note: "Muon at scale: distributed training all-reduce patterns, tensor parallelism interaction, scaling curves vs AdamW at matched compute.",
        tbd: true
      },

      // === Phase 5: RL Fine-Tuning Mechanics ===
      {
        moduleId: "A.3-policy-gradients-learning-easy",
        phase: "RL Fine-Tuning Mechanics",
        note: "Policy gradients from first principles: REINFORCE, the log-probability trick, baselines, variance reduction. The foundation for understanding PPO."
      },
      {
        moduleId: "A.3-ppo-learning-medium",
        phase: "RL Fine-Tuning Mechanics",
        note: "PPO mechanics: trust regions, the clipped surrogate objective, the LLM training loop, and how clipping interacts with optimizer behavior."
      },
      {
        moduleId: "A.2-assess",
        phase: "RL Fine-Tuning Mechanics",
        note: "Reward modeling assessment — understanding how reward models are trained and their failure modes is prerequisite for RLHF."
      },
      {
        moduleId: "A.3-assess",
        phase: "RL Fine-Tuning Mechanics",
        note: "Full RLHF assessment: PPO, KL penalties, reward hacking, alignment. Gauge your readiness before the advanced RL topics."
      },

      // === Phase 6: On-Policy vs Off-Policy & Forgetting ===
      {
        moduleId: "A.3-on-off-policy-learning-medium",
        phase: "On-Policy vs Off-Policy & Forgetting",
        note: "The on-policy vs off-policy distinction — the single most important concept for understanding why online RL and DPO behave so differently."
      },
      {
        moduleId: "A.3-plasticity-learning-medium",
        phase: "On-Policy vs Off-Policy & Forgetting",
        note: "Plasticity and catastrophic forgetting: feature rank collapse, dead units, weight norm growth. Why fine-tuning degrades pretrained capabilities.",
        tbd: true
      },
      {
        moduleId: "A.3-online-rl-learning-medium",
        phase: "On-Policy vs Off-Policy & Forgetting",
        note: "RL's Razor: why online RL forgets less. Self-consistency of on-policy data, the off-policy feedback loop, and interaction with optimizer choice.",
        tbd: true
      },
      {
        moduleId: "A.3-rl-subnets-learning-medium",
        phase: "On-Policy vs Off-Policy & Forgetting",
        note: "RL fine-tuning modifies sparse subnetworks — 3-10% of parameters carry 90%+ of weight change. Implications for optimizer and PEFT method selection.",
        tbd: true
      },

      // === Phase 7: Infrastructure ===
      {
        moduleId: "1.6-assess",
        phase: "Infrastructure",
        note: "Distributed training fundamentals: Muon requires all-reduce of full gradient matrices before orthogonalization, making distributed strategy important."
      },
      {
        moduleId: "G.2-assess",
        phase: "Infrastructure",
        note: "Memory-efficient training: Muon's reduced optimizer state vs Adam's two-buffer overhead. Gradient checkpointing and mixed-precision considerations."
      },
    ]
  },
];
