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
    description: "A focused study path for understanding the Muon optimizer and its application in reinforcement learning fine-tuning of large language models. Covers the mathematical foundations (spectral methods, divergences), core ML concepts (transformers, pretraining, RLHF), and specialized topics (Muon internals, online RL plasticity, subnetwork analysis).",
    color: "#8B5CF6",
    items: [
      // === Phase 1: Mathematical Foundations ===
      {
        moduleId: "0.1-easy",
        phase: "Mathematical Foundations",
        note: "Eigenvalues, SVD, and spectral norms are the language of Muon's update rule — it performs steepest descent under the spectral norm."
      },
      {
        moduleId: "0.1-medium",
        phase: "Mathematical Foundations",
        note: "Matrix calculus and low-rank structure. Muon's Newton-Schulz iteration operates on gradient matrices; understanding Jacobians and the Eckart-Young theorem helps."
      },
      {
        moduleId: "0.1-hard",
        phase: "Mathematical Foundations",
        note: "Advanced linear algebra rounds out the foundation for understanding spectral methods in optimization."
      },
      {
        moduleId: "0.3-assess",
        phase: "Mathematical Foundations",
        note: "Assess your understanding of optimization theory — Adam, SGD, momentum, second-order methods. This is the baseline before diving into Muon."
      },
      {
        moduleId: "0.2-entropy-easy",
        phase: "Mathematical Foundations",
        note: "Entropy and cross-entropy underpin LLM training losses and the KL divergence terms in RL objectives."
      },
      {
        moduleId: "0.2-easy",
        phase: "Mathematical Foundations",
        note: "KL divergence asymmetry and mode-covering vs mode-seeking behavior directly affect RLHF training — understanding f-divergences is critical."
      },
      {
        moduleId: "0.2-assess-divergences",
        phase: "Mathematical Foundations",
        note: "Test your divergence knowledge. Forward vs reverse KL appears in every RLHF paper."
      },

      // === Phase 2: Core LLM Knowledge ===
      {
        moduleId: "1.1-assess",
        phase: "Core LLM Knowledge",
        note: "Transformer architecture knowledge is essential — RL fine-tuning and Muon both operate on transformer weight matrices."
      },
      {
        moduleId: "1.3-assess",
        phase: "Core LLM Knowledge",
        note: "Pretraining dynamics context: how models learn during pretraining sets the stage for understanding what RL fine-tuning changes."
      },
      {
        moduleId: "B.1-assess",
        phase: "Core LLM Knowledge",
        note: "Scaling laws matter because Muon's scalability claims need to be understood in the context of compute-optimal training."
      },
      {
        moduleId: "B.4-assess",
        phase: "Core LLM Knowledge",
        note: "Training stability and dynamics: loss landscapes, gradient flow, and how optimizer choice affects convergence."
      },

      // === Phase 3: Muon Optimizer Deep Dive ===
      {
        moduleId: "0.3-muon-learning-easy",
        phase: "Muon Optimizer Deep Dive",
        note: "Muon fundamentals: Newton-Schulz iteration, spectral steepest descent, relationship to Shampoo/SOAP.",
        tbd: true
      },
      {
        moduleId: "B.4-muon-vs-adam-learning-hard",
        phase: "Muon Optimizer Deep Dive",
        note: "Deep comparison of Muon vs AdamW: spectral properties, tail-end associative memory learning, and when each optimizer wins.",
        tbd: true
      },

      // === Phase 4: RL Fine-Tuning ===
      {
        moduleId: "A.2-assess",
        phase: "RL Fine-Tuning",
        note: "Reward modeling is the foundation of RLHF — understanding reward model training is prerequisite for RL policy optimization."
      },
      {
        moduleId: "A.3-assess",
        phase: "RL Fine-Tuning",
        note: "Core RLHF and policy optimization: PPO, KL penalties, reward hacking — the setting where Muon is being applied."
      },
      {
        moduleId: "A.3-online-rl-learning-medium",
        phase: "RL Fine-Tuning",
        note: "Online RL for LLMs: why on-policy methods forget less (RL's Razor), plasticity preservation, and implications for optimizer selection.",
        tbd: true
      },
      {
        moduleId: "A.3-rl-subnets-learning-medium",
        phase: "RL Fine-Tuning",
        note: "How RL fine-tuning targets small subnetworks in LLMs — sparse update patterns and what this means for Muon vs Adam.",
        tbd: true
      },

      // === Phase 5: Scaling & Infrastructure ===
      {
        moduleId: "1.6-assess",
        phase: "Scaling & Infrastructure",
        note: "Distributed training fundamentals: Muon requires all-reduce of full gradient matrices, making distributed strategy important."
      },
      {
        moduleId: "G.2-assess",
        phase: "Scaling & Infrastructure",
        note: "Memory-efficient training: Muon's memory footprint differs from Adam's — understanding gradient checkpointing and mixed precision matters."
      },
    ]
  },
];
