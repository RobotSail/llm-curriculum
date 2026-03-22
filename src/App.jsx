import { useState, useCallback, useMemo } from "react";
import ModuleView, { getModuleProgress, getGaps, removeGap, clearAllGaps } from './components/ModuleView';
import WarmupView from './components/WarmupView';
import MathText from './components/MathText';
import { MODULES } from './modules';

const LAST_VISIT_KEY = 'llm-curriculum-last-visit';

const CURRICULUM = [
  {
    id: "t0",
    label: "Tier 0",
    title: "Prerequisites",
    color: "#888780",
    colorLight: "#F1EFE8",
    sections: [
      {
        id: "0.1",
        title: "Linear algebra (beyond intro)",
        motivatingProblems: [
          "LoRA works by injecting low-rank weight updates \u2014 but why should task-specific adaptations be low-rank at all? Understanding SVD and the spectral structure of weight matrices reveals why most of the \"information\" in fine-tuning lives in a small subspace.",
          "During training, some weight matrices develop outlier singular values 100\u00D7 larger than the bulk. Random matrix theory (Marchenko-Pastur) tells you which eigenvalues are signal vs. noise \u2014 this directly informs pruning, compression, and diagnosing training pathology.",
          "Backpropagation through attention, LayerNorm, and linear layers requires chain-ruling through matrix expressions. Without matrix calculus fluency, you can\u2019t derive gradients for novel architectures or understand why certain parameterizations are more stable."
        ],
        subtopics: [
          "Matrix calculus: Jacobians, Hessians, chain rule over matrix expressions",
          "Eigendecomposition, spectral theory, positive-definiteness",
          "SVD and low-rank approximations",
          "Random matrix theory basics",
          "Tensor algebra: einsum fluency"
        ]
      },
      {
        id: "0.2",
        title: "Probability, statistics & information theory",
        motivatingProblems: [
          "RLHF uses KL divergence to keep the policy close to a reference model \u2014 but forward KL (mode-covering) and reverse KL (mode-seeking) produce radically different behaviors. Choosing wrong means your model either hedges uselessly or collapses to a single mode.",
          "The entire training objective (cross-entropy loss) is an information-theoretic quantity. Without understanding entropy, mutual information, and cross-entropy, the loss function is just a number going down rather than a meaningful measure of how well your model approximates the data distribution.",
          "Variational inference (ELBO, mean-field) underpins VAEs, latent variable models, and the theoretical framework behind DPO\u2019s derivation. It\u2019s the bridge between Bayesian reasoning and tractable training."
        ],
        subtopics: [
          "Probability foundations: Bayes' theorem, conditional probability, common distributions",
          "Entropy, cross-entropy, mutual information, and perplexity",
          "Exponential family distributions, sufficient statistics, and MLE",
          "KL divergence \u2014 forward vs. reverse, f-divergences, Jensen-Shannon, R\u00E9nyi",
          "Bayesian inference, variational methods, ELBO, and amortized inference",
          "Sampling methods: importance sampling, MCMC, rejection sampling, Gumbel-max",
          "Concentration inequalities: Hoeffding, Chernoff, sub-Gaussian bounds",
          "Information theory in practice: LLM loss, temperature, RLHF, bits-per-byte"
        ]
      },
      {
        id: "0.3",
        title: "Optimization theory",
        motivatingProblems: [
          "RLHF\u2019s objective is a KL-constrained reward maximization \u2014 literally a constrained optimization problem. Understanding Lagrangian duality and KKT conditions tells you why the \u03B2 parameter in the KL penalty behaves the way it does, and how to set it.",
          "Why does Adam work so well for transformers but plain SGD struggles? The answer involves adaptive preconditioning of the loss landscape, the role of gradient noise as implicit regularization, and why second-order curvature information (Fisher matrix, natural gradient) explains what Adam approximates.",
          "Loss spikes during training can destroy a run that cost millions of dollars. Understanding sharp vs. flat minima, saddle point geometry, and how learning rate interacts with batch size is what separates debugging from guessing."
        ],
        subtopics: [
          "Convex optimization: duality, KKT conditions, Lagrangian methods",
          "Non-convex optimization: saddle points, loss landscape geometry",
          "SGD theory: convergence, learning rate schedules, noise",
          "Adaptive methods: Adam internals, AdaFactor, LAMB/LARS",
          "Constrained optimization and penalty methods",
          "Second-order methods and natural gradient"
        ]
      },
      {
        id: "0.4",
        title: "Systems & infrastructure literacy",
        motivatingProblems: [
          "A single H100 has 80GB HBM3 but only 50MB of SRAM per SM. FlashAttention\u2019s entire speedup comes from tiling attention to fit in SRAM \u2014 if you don\u2019t understand the memory hierarchy, you can\u2019t reason about why some operations are 10\u00D7 slower than expected despite having plenty of FLOPs.",
          "Training a 70B model across 256 GPUs means network bandwidth between nodes becomes the bottleneck, not compute. Understanding ring-allreduce, NVLink topology, and gradient compression is the difference between 40% and 80% GPU utilization.",
          "torch.compile and Triton can fuse operations to eliminate memory round-trips, but only if you understand what operator fusion means at the hardware level. The gap between \u2018it runs\u2019 and \u2018it runs efficiently\u2019 is often 3-5\u00D7."
        ],
        subtopics: [
          "PyTorch internals: autograd, memory, torch.compile",
          "GPU architecture: SMs, warps, memory hierarchy, occupancy",
          "Distributed computing: parameter server vs. all-reduce",
          "Networking: InfiniBand, NVLink/NVSwitch topology",
          "Profiling: torch profiler, Nsight",
          "Compiler optimization: operator fusion, Triton"
        ]
      }
    ]
  },
  {
    id: "t1",
    label: "Tier 1",
    title: "Foundational core",
    color: "#378ADD",
    colorLight: "#E6F1FB",
    sections: [
      {
        id: "1.1",
        title: "Transformer architecture \u2014 deep understanding",
        motivatingProblems: [
          "Attention scales as O(n\u00B2) in sequence length, which means a 128K context model spends most of its compute on attention. Understanding why 1/\u221Ad\u2096 scaling prevents softmax saturation and entropy collapse explains both the mechanism and the motivation behind every efficient attention variant.",
          "RoPE encodes position through rotation in the complex plane, enabling length generalization \u2014 but it breaks at positions far beyond training. Why positional encoding is still an open problem (not a solved one) frames ongoing work on ALiBi, NTK-aware scaling, and YaRN.",
          "The \u2018residual stream as communication bus\u2019 mental model (from Anthropic\u2019s circuits work) reframes the transformer: layers read from and write to a shared stream rather than sequentially transforming representations. This is the foundation for understanding superposition, feature extraction, and steering."
        ],
        subtopics: [
          "Attention mechanism and information retrieval interpretation",
          "Scaling, softmax saturation, entropy collapse",
          "Positional encoding: RoPE, ALiBi, and open problems",
          "FFN blocks: SwiGLU, key-value memory interpretation",
          "LayerNorm variants and training stability",
          "Residual stream view and superposition",
          "Architectural decisions at scale: GQA, MQA, parallel blocks"
        ]
      },
      {
        id: "1.2",
        title: "Tokenization",
        motivatingProblems: [
          "GPT-4 can\u2019t reliably count the letters in a word because tokenization fragments words unpredictably. BPE, WordPiece, and Unigram each handle this differently \u2014 the choice of tokenizer cascades into arithmetic ability, code generation, and even hallucination patterns.",
          "A model trained with a 32K English-optimized vocabulary needs 3\u00D7 more tokens to represent the same Korean text, making it 3\u00D7 slower and 3\u00D7 more expensive for Korean users. Multilingual tokenization is an unsolved fairness and efficiency problem.",
          "Byte-level models (like MegaByte) avoid tokenization entirely \u2014 but then the model has to learn character-to-word composition from scratch. Understanding the compression ratio vs. representation quality trade-off is the core tension."
        ],
        subtopics: [
          "BPE, WordPiece, Unigram LM algorithms",
          "Vocabulary size trade-offs",
          "Multilingual tokenization challenges",
          "Downstream effects on arithmetic, code, reasoning",
          "Byte-level architectures"
        ]
      },
      {
        id: "1.3",
        title: "Pretraining objectives & dynamics",
        motivatingProblems: [
          "Chinchilla showed that most LLMs were massively over-parameterized relative to their data budget \u2014 GPT-3 should have been trained on 5\u00D7 more data at the same compute. This single insight redirected billions of dollars of compute allocation across the industry.",
          "Models exhibit sudden \u2018phase transitions\u2019 where a capability appears to emerge discontinuously \u2014 is this real emergence or just a measurement artifact? The answer determines whether we can predict what a 10\u00D7 larger model will be able to do.",
          "A training run costs $10M+ and a single loss spike from gradient explosion can waste weeks. Understanding why warmup matters, how learning rate interacts with batch size (the linear scaling rule), and what causes instabilities is essential risk management."
        ],
        subtopics: [
          "Autoregressive LM objective and teacher forcing",
          "Training dynamics: grokking, phase transitions, emergence",
          "Learning rate warmup, cosine/WSD schedules",
          "Batch size scaling and the linear scaling rule",
          "Weight initialization",
          "Training instabilities and loss spikes",
          "Compute-optimal training and scaling laws"
        ]
      },
      {
        id: "1.4",
        title: "Data",
        motivatingProblems: [
          "Llama\u2019s success came largely from data quality \u2014 aggressive deduplication and perplexity filtering on Common Crawl. But if you filter too aggressively, you lose diversity; too little, and you train on SEO spam. The data mixing problem has no clean solution.",
          "Synthetic data from a model\u2019s own outputs can cause \u2018model collapse\u2019 where the distribution narrows over generations. But carefully constructed synthetic data (like Phi\u2019s \u2018textbook-quality\u2019 approach) can dramatically improve sample efficiency. Knowing when synthetic data helps vs. hurts is an open research question.",
          "Benchmark contamination is rampant \u2014 models that score 90% on MMLU may have seen the test questions during pretraining. Detection is hard because near-duplicates (rephrased questions) are as problematic as exact matches."
        ],
        subtopics: [
          "Web-scale data pipelines and deduplication",
          "Data quality signals: perplexity filtering, classifiers",
          "Data mixing and curriculum strategies",
          "Synthetic data: benefits, model collapse risks",
          "Contamination and benchmark leakage",
          "Licensing, copyright, PII"
        ]
      },
      {
        id: "1.5",
        title: "Evaluation",
        motivatingProblems: [
          "A model can score 85% on MMLU but be useless as a coding assistant. Perplexity correlates with nothing users care about. The \u2018elicitation gap\u2019 \u2014 the difference between what a model knows and what you can get it to show \u2014 means evaluation is inseparable from prompting and post-training.",
          "Every benchmark gets Goodharted within months of release. Models are optimized for MMLU, HumanEval, GSM8K until the numbers stop meaning anything. Chatbot Arena (live human ranking) resists this but is expensive and slow. Evaluation may be the hardest open problem in the field.",
          "LLM-as-judge (using GPT-4 to evaluate other models) has known biases: it prefers verbose answers, is sensitive to response position, and exhibits self-preference. But human evaluation doesn\u2019t scale. Building reliable automated evaluation is unsolved."
        ],
        subtopics: [
          "Perplexity and its limitations",
          "N-shot prompting evaluation design",
          "Benchmark suites: what each measures and misses",
          "Contamination-robust evaluation",
          "LLM-as-judge biases",
          "Elicitation gap",
          "Goodhart\u2019s law in evaluation"
        ]
      },
      {
        id: "1.6",
        title: "Distributed training infrastructure",
        motivatingProblems: [
          "A 70B parameter model with Adam optimizer states requires ~1.2TB of memory \u2014 no single GPU can hold it. ZeRO/FSDP shards optimizer states, gradients, and parameters across GPUs, but each sharding stage trades memory for communication. Choosing the right ZeRO stage is a constraint satisfaction problem over memory, bandwidth, and compute.",
          "Pipeline parallelism splits layers across GPUs, but naive splitting leaves most GPUs idle (the \u2018pipeline bubble\u2019). Interleaved schedules (1F1B, etc.) shrink the bubble but complicate gradient accumulation. The scheduling problem is NP-hard in general.",
          "BF16 training eliminates the need for loss scaling that FP16 requires, because BF16\u2019s wider exponent range prevents underflow \u2014 but BF16 has less mantissa precision. Understanding when this precision loss matters (it usually doesn\u2019t, but sometimes it catastrophically does) requires knowing the numerical properties."
        ],
        subtopics: [
          "Data parallelism (DDP)",
          "Model parallelism: tensor and pipeline parallelism",
          "FSDP / ZeRO stages 1-3",
          "Sequence and context parallelism",
          "Mixed precision: FP16, BF16, FP8",
          "Activation checkpointing",
          "Fault tolerance and checkpoint management",
          "Frameworks: Megatron-LM, DeepSpeed, FSDP"
        ]
      }
    ]
  },
  {
    id: "brA",
    label: "Branch A",
    title: "Post-training & alignment",
    color: "#7F77DD",
    colorLight: "#EEEDFE",
    sections: [
      {
        id: "A.1",
        title: "Supervised fine-tuning (SFT)",
        motivatingProblems: [
          "Base models \u2018know\u2019 how to follow instructions but won\u2019t do so reliably. SFT\u2019s job is \u2018unlocking\u2019 \u2014 not teaching new knowledge, but formatting the model\u2019s existing capabilities into a usable interface. LIMA showed you can do this with only 1,000 examples, but subsequent work showed this breaks on complex multi-turn tasks.",
          "SFT overfits catastrophically fast \u2014 often within 1-2 epochs on small datasets. The learning rate, data ordering, and when to stop are all critical, and the optimal settings are very different from pretraining. Getting this wrong produces a model that parrots training examples instead of generalizing."
        ],
        subtopics: [
          "Dataset construction and conversation structure",
          "Quality vs. quantity: LIMA and its limits",
          "SFT as unlocking vs. teaching",
          "Hyperparameter sensitivity and overfitting",
          "Open datasets: FLAN, Tulu-3, Capybara",
          "Regularization during SFT"
        ]
      },
      {
        id: "A.2",
        title: "Reward modeling",
        motivatingProblems: [
          "Humans can say which of two responses is better, but not assign a numerical score. The Bradley-Terry model converts pairwise comparisons into a scalar reward \u2014 but assumes transitivity (if A > B and B > C, then A > C), which human preferences routinely violate.",
          "Reward models are susceptible to \u2018reward hacking\u2019 \u2014 the policy finds outputs that score high on the RM but are actually terrible (e.g., overly verbose, sycophantic, or adversarially formatted). The RM can\u2019t perfectly represent human preferences, and the gap gets exploited.",
          "Process reward models (PRMs) give step-level feedback on reasoning chains, while outcome reward models (ORMs) only judge the final answer. PRMs are much more useful for math/code, but require step-level human labels which are 10\u00D7 more expensive to collect."
        ],
        subtopics: [
          "Bradley-Terry preference model",
          "Reward model training and overoptimization",
          "Reward model scaling behavior",
          "Process vs. outcome reward models",
          "Implicit vs. explicit rewards",
          "Multi-objective reward modeling",
          "Generative reward models and LLM-as-judge"
        ]
      },
      {
        id: "A.3",
        title: "RLHF / policy optimization",
        motivatingProblems: [
          "PPO for LLMs is notoriously unstable \u2014 it requires carefully balancing the clipped surrogate objective, KL penalty, value function baseline, and reward normalization. A single bad hyperparameter makes the model either not learn at all or collapse to degenerate outputs. This is why many teams abandoned PPO for DPO.",
          "The KL penalty prevents the policy from drifting too far from the reference model, but forward KL and reverse KL penalize different failure modes. Adaptive KL controllers (adjusting \u03B2 during training) are necessary in practice but theoretically underprincipled.",
          "GRPO (group relative policy optimization, used in DeepSeek-R1) eliminates the value function entirely by using group-relative baselines \u2014 comparing outputs within a batch rather than estimating absolute value. This is simpler than PPO but changes what the model optimizes."
        ],
        subtopics: [
          "PPO for LLMs: clipped objective, GAE, value baseline",
          "KL penalty: \u03B2 trade-off, forward vs. reverse KL",
          "Importance sampling and off-policy corrections",
          "REINFORCE variants and GRPO",
          "Practical RLHF: rollout strategies, memory choreography",
          "PPO vs. alternatives debate"
        ]
      },
      {
        id: "A.4",
        title: "Direct alignment algorithms",
        motivatingProblems: [
          "DPO\u2019s key insight: you can rearrange the RLHF objective into a closed-form solution that uses the policy itself as an implicit reward model, eliminating reward model training entirely. But this elegance comes at a cost \u2014 DPO suffers from distribution shift because it trains on a fixed preference dataset while the policy changes.",
          "The post-DPO zoo (IPO, KTO, ORPO, SimPO) each fix one DPO failure mode but introduce new ones. IPO fixes overconfident preferences, KTO works with only thumbs-up/down data, SimPO removes the reference model \u2014 but no single method dominates across all settings.",
          "Online/iterative DPO (generating fresh preference pairs from the current policy) consistently outperforms offline DPO, suggesting that the distribution shift problem is the dominant failure mode. But online DPO reintroduces much of the infrastructure complexity that DPO was supposed to eliminate."
        ],
        subtopics: [
          "DPO derivation and closed-form interpretation",
          "DPO failure modes: distribution shift, overfitting",
          "IPO, KTO, ORPO, SimPO landscape",
          "Online vs. offline preference optimization",
          "Rejection sampling and Best-of-N",
          "Constitutional AI / RLAIF"
        ]
      },
      {
        id: "A.5",
        title: "Frontier alignment topics",
        motivatingProblems: [
          "As models become superhuman at specific tasks, humans can no longer reliably evaluate their outputs. Scalable oversight asks: how do you align a model that\u2019s smarter than the humans providing feedback? Approaches include weak-to-strong generalization, debate, and recursive reward modeling \u2014 all unsolved.",
          "RL for reasoning (training models to think step-by-step, as in o1/o3) fuses post-training with test-time compute. The model learns to allocate internal reasoning tokens, but the reward signal for \u2018good thinking\u2019 is extremely sparse and hard to define.",
          "Representation engineering and steering vectors allow direct manipulation of model behavior through activation-space interventions \u2014 potentially more robust than prompt-based alignment. But the theoretical foundation for why this works is thin."
        ],
        subtopics: [
          "Scalable oversight: weak-to-strong, debate",
          "Safety training and red-teaming",
          "Representation engineering and steering vectors",
          "RLHF at scale: multi-turn RL, long-horizon objectives",
          "RL for reasoning and process supervision",
          "Synthetic preference generation at scale"
        ]
      }
    ]
  },
  {
    id: "brB",
    label: "Branch B",
    title: "Pretraining & architecture research",
    color: "#1D9E75",
    colorLight: "#E1F5EE",
    sections: [
      {
        id: "B.1",
        title: "Scaling laws & compute-optimal training",
        motivatingProblems: [
          "Kaplan et al. predicted that model performance follows a power law with parameters and data \u2014 but Chinchilla showed their exponents were wrong. The revised law said \u2018train on much more data with a smaller model,\u2019 redirecting billions in compute allocation. Which scaling law is right depends on whether you\u2019re optimizing for training cost or inference cost.",
          "\u03BCP (maximal update parameterization) promises that hyperparameters transfer across scales \u2014 tune on a small model, use those settings for the large one. If this works reliably, it eliminates the most expensive part of large-scale training (hyperparameter search). In practice, it partially works."
        ],
        subtopics: [
          "Kaplan vs. Chinchilla scaling laws",
          "Predicting downstream performance from loss",
          "\u03BCP and hyperparameter transfer",
          "Inference-aware scaling"
        ]
      },
      {
        id: "B.2",
        title: "Architecture innovations",
        motivatingProblems: [
          "Mixture of Experts (MoE) can scale parameter count 4-8\u00D7 with only 2\u00D7 the compute, because only a subset of experts activate per token. But expert load balancing is unstable \u2014 some experts hoard all the traffic (routing collapse) while others go unused. Solving this is active research.",
          "Attention is O(n\u00B2) in sequence length, which makes million-token contexts prohibitively expensive. Linear attention, state-space models (Mamba/S4), and RetNet try to get O(n) scaling \u2014 but consistently underperform attention on tasks requiring long-range retrieval. The question is whether this is fundamental or fixable.",
          "Mixture-of-depths and early exit allow the model to spend more compute on hard tokens and less on easy ones \u2014 but knowing which tokens are \u2018hard\u2019 requires processing them first, creating a chicken-and-egg problem."
        ],
        subtopics: [
          "MoE: sparse gating, load balancing, routing",
          "Linear attention and efficient variants",
          "State-space models: Mamba, S4, RetNet, RWKV",
          "Long-context architectures",
          "Depth-wise innovations: mixture-of-depths, early exit",
          "Hybrid architectures"
        ]
      },
      {
        id: "B.3",
        title: "Data-centric pretraining research",
        motivatingProblems: [
          "If you knew which training examples were responsible for a specific model behavior, you could surgically fix problems by removing or reweighting data. Influence functions attempt this, but scale terribly \u2014 applying them to a 70B model on trillions of tokens is computationally infeasible with current methods.",
          "Continual pretraining (updating a model on new data) causes catastrophic forgetting of old knowledge. Learning rate rewarming and replay buffers help, but there\u2019s no principled way to know how much old data to replay or when forgetting becomes unacceptable."
        ],
        subtopics: [
          "Influence functions and data attribution at scale",
          "Data selection: DSIR, DoReMi, skill-based mixing",
          "Continual pretraining and catastrophic forgetting",
          "Domain-specific pretraining trade-offs",
          "Multilingual training and cross-lingual transfer"
        ]
      },
      {
        id: "B.4",
        title: "Training stability & dynamics research",
        motivatingProblems: [
          "The \u2018edge of stability\u2019 phenomenon: training loss doesn\u2019t decrease monotonically; instead, the loss oscillates at the edge of divergence, and the optimizer implicitly self-regularizes. Why does this happen, and does it explain why overparameterized networks generalize?",
          "Induction heads (circuits that implement in-context learning) form in a sudden phase transition during training. Understanding when and why specific capabilities crystallize could let us predict \u2014 rather than discover post-hoc \u2014 what a model will be able to do.",
          "Loss landscape mode connectivity: two independently trained models can be connected by a path of low loss, suggesting the landscape has a simple global structure despite being non-convex. This has implications for model merging, ensembling, and our understanding of why training works at all."
        ],
        subtopics: [
          "Edge of stability phenomenon",
          "Feature learning vs. kernel regime",
          "Mechanistic understanding: induction heads, ICL emergence",
          "Loss landscape geometry and mode connectivity",
          "\u03BCP and hyperparameter transfer across scales"
        ]
      },
      {
        id: "B.5",
        title: "Novel pretraining objectives",
        motivatingProblems: [
          "Autoregressive models generate left-to-right, which is a poor fit for tasks like infilling, editing, and bidirectional understanding. Masked language modeling handles these but can\u2019t generate fluently. UL2 tried unifying both \u2014 but no approach fully resolves the tension.",
          "Diffusion-based language models could enable iterative refinement (editing all tokens simultaneously) rather than sequential generation. But discrete diffusion for text is much harder than continuous diffusion for images, and current results trail autoregressive models significantly."
        ],
        subtopics: [
          "Masked language modeling and modern descendants",
          "Diffusion-based language models",
          "Energy-based approaches",
          "Non-autoregressive generation"
        ]
      }
    ]
  },
  {
    id: "brC",
    label: "Branch C",
    title: "Inference & deployment",
    color: "#D85A30",
    colorLight: "#FAECE7",
    sections: [
      {
        id: "C.1",
        title: "Quantization",
        motivatingProblems: [
          "A 70B model in FP16 needs 140GB VRAM \u2014 two H100s. In 4-bit, it fits in 35GB \u2014 one GPU. Quantization can cut inference cost by 4\u00D7 with minimal quality loss, but activation quantization is much harder than weight quantization because activations have heavy-tailed outliers that 4-bit integers can\u2019t represent.",
          "BitNet (1.58-bit) suggests that models could be trained natively in extreme low-precision. If this works at scale, it would fundamentally change the hardware requirements for inference \u2014 but current results are limited to small models."
        ],
        subtopics: [
          "Weight quantization: GPTQ, AWQ, SqueezeLLM",
          "Activation quantization and outlier features",
          "Low-bit frontiers: 4/3/2/1.58-bit",
          "QAT vs. PTQ trade-offs",
          "Mixed-precision per-layer sensitivity"
        ]
      },
      {
        id: "C.2",
        title: "Efficient decoding",
        motivatingProblems: [
          "Autoregressive decoding is inherently sequential \u2014 each token depends on all previous ones. Speculative decoding breaks this by drafting N tokens with a small model and verifying them in parallel with the large model, getting 2-3\u00D7 speedup with mathematically identical outputs. The open question is how to choose optimal draft models.",
          "The KV-cache for a 70B model with 128K context can exceed 50GB, meaning memory \u2014 not compute \u2014 is the bottleneck for long-context inference. Paged attention (vLLM) manages this like virtual memory, but KV-cache compression (quantizing or evicting old entries) is an active research area."
        ],
        subtopics: [
          "KV-cache: memory scaling, paged attention, compression",
          "Speculative decoding: Medusa, EAGLE",
          "Continuous batching and scheduling",
          "Prefix caching and prompt optimization",
          "Parallel decoding methods"
        ]
      },
      {
        id: "C.3",
        title: "Serving infrastructure",
        motivatingProblems: [
          "Prefill (processing the prompt) is compute-bound while decode (generating tokens) is memory-bandwidth-bound. They have fundamentally different hardware requirements, which is why disaggregated inference (separate prefill and decode pools) can improve throughput by 2\u00D7 \u2014 but adds system complexity.",
          "Serving 100 different LoRA adapters from a single base model (multi-tenant inference) requires loading and swapping adapters efficiently without reloading the full model. This is an unsolved systems problem that determines whether fine-tuned models are economically viable to serve."
        ],
        subtopics: [
          "Serving frameworks: vLLM, TensorRT-LLM, SGLang",
          "Disaggregated inference: prefill vs. decode",
          "Multi-LoRA serving",
          "Hardware selection and memory bandwidth bottlenecks"
        ]
      },
      {
        id: "C.4",
        title: "Compression & distillation",
        motivatingProblems: [
          "Knowledge distillation can compress a 70B model into a 7B model that retains 85-90% of the capability. But the best distillation method depends on whether you match logits (cheap but lossy), features (better but expensive), or generate on-policy data (best but requires the teacher at inference time).",
          "LoRA merging allows combining multiple specialized adapters into a single model \u2014 but merging in weight space doesn\u2019t preserve the individual adapters\u2019 behaviors. Multi-LoRA routing (choosing which adapter to apply per input) is more principled but harder to deploy."
        ],
        subtopics: [
          "Knowledge distillation approaches",
          "Structured vs. unstructured pruning",
          "LoRA merging and multi-LoRA routing",
          "Neural architecture search for efficient LMs"
        ]
      }
    ]
  },
  {
    id: "brD",
    label: "Branch D",
    title: "Reasoning, agents & test-time compute",
    color: "#D4537E",
    colorLight: "#FBEAF0",
    sections: [
      {
        id: "D.1",
        title: "Chain-of-thought & reasoning",
        motivatingProblems: [
          "Asking a model to \u2018think step by step\u2019 improves accuracy on math problems from ~20% to ~80%. The \u2018scratchpad hypothesis\u2019 says CoT works by giving the model working memory, but we don\u2019t actually know if the model follows its stated reasoning or just uses the extra tokens for pattern matching.",
          "Self-consistency (sampling multiple reasoning chains and majority-voting) works remarkably well, suggesting that errors are somewhat random rather than systematic. But this means spending 10\u00D7 more compute at inference \u2014 when is this cost-effective vs. just using a bigger model?"
        ],
        subtopics: [
          "Chain-of-thought and the scratchpad hypothesis",
          "Self-consistency and majority voting",
          "Tree-of-thought and structured search",
          "Process supervision for reasoning",
          "Reasoning vs. pattern matching debate"
        ]
      },
      {
        id: "D.2",
        title: "Test-time compute scaling",
        motivatingProblems: [
          "o1/o3 demonstrated that training models to \u2018think longer\u2019 (using more tokens for internal reasoning) can dramatically improve performance on hard problems. This suggests an entirely new scaling axis: instead of making models bigger, make them think longer. The key question is whether there are diminishing returns.",
          "MCTS (Monte Carlo Tree Search) applied to LLM generation treats each token-generation step as a tree search problem, using a value model to evaluate partial completions. This is how AlphaGo worked, but applying it to language is harder because the branching factor is 50,000+ (vocabulary size) vs. 361 (Go board positions)."
        ],
        subtopics: [
          "Best-of-N sampling with reward models",
          "Iterative refinement and self-correction",
          "MCTS for LLM generation",
          "Compute-optimal inference allocation",
          "Extended thinking and reasoning tokens",
          "Inference scaling laws"
        ]
      },
      {
        id: "D.3",
        title: "Tool use & function calling",
        motivatingProblems: [
          "Language models can\u2019t reliably do arithmetic, access current information, or execute code. Function calling turns the model into an orchestrator that delegates to specialized tools \u2014 but training models to use tools reliably (correct syntax, appropriate tool selection, error recovery) is its own alignment challenge.",
          "RAG (retrieval-augmented generation) grounds the model in external knowledge, reducing hallucination \u2014 but chunking strategies, embedding model choice, and reranking all dramatically affect quality. Multi-step retrieval (the model decides what to search for iteratively) is more powerful but much harder to make reliable."
        ],
        subtopics: [
          "Function calling training and format design",
          "Code generation as tool use",
          "RAG architecture and embedding models",
          "Multi-step and agentic RAG"
        ]
      },
      {
        id: "D.4",
        title: "Agentic systems",
        motivatingProblems: [
          "An LLM agent that can browse the web, write code, and call APIs could automate complex knowledge work \u2014 but compounding errors over multi-step trajectories means a 95% per-step accuracy yields ~60% accuracy over 10 steps. Making agents reliable enough for real-world deployment requires fundamentally better error recovery.",
          "Evaluating agents is much harder than evaluating single-turn responses \u2014 you need trajectory-level metrics, and the same goal can be achieved via very different action sequences. There\u2019s no equivalent of \u2018MMLU for agents\u2019 that the field agrees on."
        ],
        subtopics: [
          "Agent frameworks: ReAct, tool-augmented loops",
          "Memory architectures: short-term, long-term, episodic",
          "Multi-agent systems: debate, coordination",
          "Agent evaluation and trajectory metrics",
          "Grounding and hallucination reduction"
        ]
      }
    ]
  },
  {
    id: "brE",
    label: "Branch E",
    title: "Multimodality",
    color: "#BA7517",
    colorLight: "#FAEEDA",
    sections: [
      {
        id: "E.1",
        title: "Vision-language models",
        motivatingProblems: [
          "How do you align a vision encoder (trained with contrastive learning on image-text pairs) with a language model (trained autoregressively on text)? The architectural choice \u2014 cross-attention, early fusion, or a simple projection layer (LLaVA-style) \u2014 determines whether the model can do fine-grained visual reasoning or just caption images.",
          "Resolution is a fundamental trade-off: higher resolution means more visual tokens (quadratically more!), which explodes compute. Dynamic resolution and tiling strategies try to allocate visual tokens where they matter, but this is still crude compared to how humans allocate visual attention."
        ],
        subtopics: [
          "Vision encoders: ViT, SigLIP, CLIP",
          "VLM architectures: cross-attention vs. fusion vs. projection",
          "Visual instruction tuning",
          "Resolution handling and tiling strategies"
        ]
      },
      {
        id: "E.2",
        title: "Image generation",
        motivatingProblems: [
          "Diffusion models work by learning to reverse a noise process \u2014 but the forward process (adding noise) is trivially simple while the reverse (removing noise) requires learning the full data distribution. Classifier-free guidance lets you trade off diversity for quality by interpolating between conditioned and unconditioned denoising, but the interpolation weight is a magic number.",
          "Autoregressive image generation (VQ-VAE + transformer) enables unified \u2018understand and generate\u2019 architectures, but quantizing continuous images into discrete tokens lossy \u2014 fine details get smeared. The tension between discrete (compositional, language-compatible) and continuous (high-fidelity) representations is unresolved."
        ],
        subtopics: [
          "Diffusion models and classifier-free guidance",
          "Latent diffusion and Stable Diffusion lineage",
          "Autoregressive image generation",
          "Unified generation and understanding models"
        ]
      },
      {
        id: "E.3",
        title: "Audio & speech",
        motivatingProblems: [
          "Speech is a continuous signal but language models work with discrete tokens. Encodec and SpeechTokenizer convert audio into discrete tokens, but the tokenization loses prosody, emotion, and speaker identity to varying degrees. The choice of discrete vs. continuous representations determines what the model can do.",
          "Real-time speech interaction requires streaming architectures that generate responses before the user finishes speaking. This is fundamentally different from the \u2018wait for complete input, then process\u2019 paradigm that text models use \u2014 it requires speculative generation and backtracking."
        ],
        subtopics: [
          "Speech tokenization: Encodec, continuous vs. discrete",
          "Speech-language models: AudioLM, VALL-E, Whisper",
          "Real-time speech and streaming architectures"
        ]
      },
      {
        id: "E.4",
        title: "Video & beyond",
        motivatingProblems: [
          "A 1-minute video at 30fps is 1,800 frames \u2014 each with more information than a typical text prompt. Even with aggressive frame sampling, the token count is enormous. Temporal consistency (making sure a generated video doesn\u2019t flicker or have objects teleporting between frames) is the core challenge for video generation.",
          "The trajectory toward \u2018omni-modal\u2019 models (one architecture that consumes and produces text, images, audio, video, code) requires solving modality alignment \u2014 making sure the shared representation space doesn\u2019t privilege one modality at the expense of others."
        ],
        subtopics: [
          "Video understanding and temporal modeling",
          "Video generation and temporal consistency",
          "Omni-modal models and modality alignment"
        ]
      }
    ]
  },
  {
    id: "brF",
    label: "Branch F",
    title: "Interpretability & mechanistic understanding",
    color: "#534AB7",
    colorLight: "#EEEDFE",
    sections: [
      {
        id: "F.1",
        title: "Probing & behavioral analysis",
        motivatingProblems: [
          "Linear probes can detect that a model \u2018knows\u2019 facts (gender, location, sentiment) in its intermediate representations \u2014 but a positive probe result might just mean the information is linearly accessible, not that the model actually uses it. The methodological debate about what probing proves is ongoing.",
          "Causal interventions (activation patching) go beyond correlation: by surgically replacing activations from one input with another, you can identify which components causally contribute to a behavior. This is the basis of ROME/MEMIT for knowledge editing, but scaling causal tracing to complex behaviors is hard."
        ],
        subtopics: [
          "Linear probing methodology and limitations",
          "Causal interventions and activation patching",
          "Behavioral testing: CheckList-style evaluation"
        ]
      },
      {
        id: "F.2",
        title: "Mechanistic interpretability",
        motivatingProblems: [
          "Models store far more \u2018features\u2019 than they have dimensions \u2014 a phenomenon called superposition. A 768-dimensional residual stream might encode tens of thousands of interpretable features by using nearly-orthogonal directions. Sparse autoencoders (SAEs) try to decompose this, but we don\u2019t know if the decomposition is unique or meaningful.",
          "Induction heads (attention patterns that implement \u2018if A follows B in context, predict A after B next time\u2019) are among the only circuits we fully understand. Scaling this level of understanding to complex behaviors (deception, sycophancy, reasoning) would transform alignment \u2014 but it\u2019s unclear if circuits compose cleanly enough to analyze."
        ],
        subtopics: [
          "Induction heads and circuit-level analysis",
          "Superposition and the toy model framework",
          "Sparse autoencoders for feature extraction",
          "Feature visualization and steering"
        ]
      },
      {
        id: "F.3",
        title: "Training dynamics interpretability",
        motivatingProblems: [
          "Grokking: a model memorizes training data, achieving perfect train accuracy but zero test accuracy \u2014 then, thousands of steps later, suddenly generalizes. The mechanistic explanation involves the model transitioning from a memorization circuit to an algorithmic one. Understanding why this transition happens (and how to trigger it faster) could inform how to train better models.",
          "Singular learning theory (the Watanabe framework) provides a mathematical explanation for why neural networks generalize despite being overparameterized. The key quantity is the \u2018learning coefficient\u2019 (effective dimensionality), which may predict generalization better than parameter count."
        ],
        subtopics: [
          "Phase transitions and sudden capability emergence",
          "Grokking: delayed generalization",
          "Lottery ticket hypothesis",
          "Singular learning theory and generalization"
        ]
      },
      {
        id: "F.4",
        title: "Formal & theoretical approaches",
        motivatingProblems: [
          "What can transformers compute? They can simulate bounded-depth circuits, implement certain algorithms via in-context learning, and approximate Turing machines with sufficient depth \u2014 but the precise characterization of their computational class is unknown.",
          "The \u2018mesa-optimization\u2019 hypothesis: transformers might implement internal optimization algorithms (like gradient descent in their forward pass) during in-context learning. If true, this means transformers aren\u2019t just pattern matchers \u2014 they\u2019re running search/optimization at inference time, with implications for alignment and capability."
        ],
        subtopics: [
          "Expressiveness results for transformers",
          "Complexity-theoretic perspectives on ICL",
          "Transformers as mesa-optimizers"
        ]
      }
    ]
  },
  {
    id: "brG",
    label: "Branch G",
    title: "Efficient training & parameter-efficient methods",
    color: "#639922",
    colorLight: "#EAF3DE",
    sections: [
      {
        id: "G.1",
        title: "Parameter-efficient fine-tuning (PEFT)",
        motivatingProblems: [
          "Full fine-tuning of a 70B model requires storing a separate copy of all 70B parameters per task \u2014 at 140GB each, serving 10 tasks means 1.4TB just for weights. LoRA\u2019s insight: the weight update for fine-tuning is low-rank, so you can represent it with two small matrices (e.g., rank 16 = 0.01% of parameters). But choosing the right rank is task-dependent and poorly understood.",
          "QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning of a 65B model on a single 48GB GPU \u2014 democratizing access to LLM training. But quantization introduces noise into the frozen weights, and whether this noise helps or hurts depends on the task in non-obvious ways.",
          "The LoRA variant explosion (DoRA, LoRA+, rsLoRA, PiSSA) reflects the fact that vanilla LoRA has systematic failure modes: it under-trains the B matrix, uses suboptimal learning rates for A vs. B, and initializes suboptimally. Each variant fixes one issue, but no variant fixes all of them."
        ],
        subtopics: [
          "LoRA: low-rank hypothesis, rank selection, \u03B1 scaling",
          "LoRA variants: QLoRA, DoRA, LoRA+, rsLoRA, PiSSA",
          "Adapter methods: prefix tuning, prompt tuning",
          "When PEFT matches full fine-tuning"
        ]
      },
      {
        id: "G.2",
        title: "Memory-efficient training",
        motivatingProblems: [
          "Adam stores two additional fp32 tensors (first and second moments) per parameter \u2014 for a 70B model, that\u2019s 560GB just for optimizer states. 8-bit Adam, Adafactor, and GaLore each reduce this from different angles, but each makes different trade-offs with training stability.",
          "GaLore (Gradient Low-Rank Projection) projects gradients into a low-rank subspace, dramatically reducing optimizer memory \u2014 but the projection must be re-computed periodically, and choosing the re-projection frequency is another hyperparameter that affects convergence."
        ],
        subtopics: [
          "Gradient checkpointing strategies",
          "Optimizer state reduction: 8-bit Adam, Adafactor, GaLore",
          "Communication-efficient distributed training"
        ]
      },
      {
        id: "G.3",
        title: "Hardware-aware optimization",
        motivatingProblems: [
          "FlashAttention is 2-4\u00D7 faster than standard attention not because it reduces FLOPs \u2014 it actually does the same number \u2014 but because it tiles the computation to fit in SRAM, avoiding HBM round-trips. The \u2018IO-awareness principle\u2019 (optimizing for memory access patterns, not just FLOPs) is the key insight that applies far beyond attention.",
          "The roofline model tells you whether an operation is compute-bound or memory-bound. For transformers, attention is typically memory-bound at inference (KV-cache reads dominate) but compute-bound at training (matmuls dominate). This determines which optimizations help: parallelism for compute-bound, memory management for bandwidth-bound."
        ],
        subtopics: [
          "FlashAttention and IO-awareness",
          "Kernel writing with Triton",
          "Roofline model for transformers",
          "FP8 training: scaling factors, per-tensor vs. per-channel"
        ]
      }
    ]
  }
];

const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{flexShrink:0}}>
    <path d="M3 7.5L5.5 10L11 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ChevronIcon = ({ open }) => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{transform:open?'rotate(90deg)':'rotate(0)',transition:'transform 0.15s ease',flexShrink:0}}>
    <path d="M6 4L10 8L6 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const STORAGE_KEY = "llm-curriculum-v2";

export default function App() {
  const [completed, setCompleted] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  });
  const [expandedSections, setExpandedSections] = useState({});
  const [expandedTiers, setExpandedTiers] = useState(() => {
    const o = {};
    CURRICULUM.forEach(t => o[t.id] = true);
    return o;
  });
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [activeModule, setActiveModule] = useState(null);
  const [showWarmup, setShowWarmup] = useState(false);
  const [showWarmupPrompt, setShowWarmupPrompt] = useState(() => {
    try {
      const last = localStorage.getItem(LAST_VISIT_KEY);
      const today = new Date().toDateString();
      if (last !== today) {
        localStorage.setItem(LAST_VISIT_KEY, today);
        return true;
      }
      return false;
    } catch { return false; }
  });
  const [showGaps, setShowGaps] = useState(false);
  const [gapsVersion, setGapsVersion] = useState(0);
  const [copied, setCopied] = useState(false);

  const gaps = useMemo(() => getGaps(), [showGaps, gapsVersion]);

  const getSectionTitle = useCallback((sectionId) => {
    for (const tier of CURRICULUM) {
      for (const sec of tier.sections) {
        if (sec.id === sectionId) return sec.title;
      }
    }
    return sectionId;
  }, []);

  const copyGapsToClipboard = useCallback(() => {
    if (gaps.length === 0) return;
    const grouped = {};
    gaps.forEach(g => {
      if (!grouped[g.sectionId]) grouped[g.sectionId] = [];
      grouped[g.sectionId].push(g);
    });

    // Collect existing module titles per section for context
    const existingBySection = {};
    for (const sectionId of Object.keys(grouped)) {
      const mods = MODULES[sectionId];
      if (mods) {
        existingBySection[sectionId] = mods.map(m =>
          `"${m.title}" (${m.difficulty}, ${m.estimatedMinutes}m)`
        );
      }
    }

    let text = "I need deeper learning content on these topics from my LLM curriculum.\n";
    text += "Please create interactive modules (easy/medium/hard with Brilliant-style questions) covering each gap.\n\n";
    for (const [sectionId, items] of Object.entries(grouped)) {
      text += `Section ${sectionId} \u2014 ${getSectionTitle(sectionId)}:\n`;
      text += "  Gaps to cover:\n";
      items.forEach(g => {
        text += `    - "${g.label}" (from ${g.moduleTitle}, ${g.difficulty})\n`;
      });
      if (existingBySection[sectionId]) {
        text += "  Existing modules (avoid duplicating these):\n";
        existingBySection[sectionId].forEach(m => {
          text += `    - ${m}\n`;
        });
      }
      text += "\n";
    }
    navigator.clipboard.writeText(text.trim());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [gaps, getSectionTitle]);

  const handleRemoveGap = useCallback((id) => {
    removeGap(id);
    setGapsVersion(v => v + 1);
  }, []);

  const handleClearGaps = useCallback(() => {
    if (confirm("Clear all learning gaps?")) {
      clearAllGaps();
      setGapsVersion(v => v + 1);
    }
  }, []);

  const save = useCallback((next) => {
    setCompleted(next);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch {}
  }, []);

  const toggle = useCallback((id) => {
    setCompleted(prev => {
      const next = { ...prev, [id]: !prev[id] };
      save(next);
      return next;
    });
  }, [save]);

  const toggleSection = useCallback((id) => {
    setExpandedSections(p => ({ ...p, [id]: !p[id] }));
  }, []);

  const toggleTier = useCallback((id) => {
    setExpandedTiers(p => ({ ...p, [id]: !p[id] }));
  }, []);

  const stats = useMemo(() => {
    const out = {};
    let totalAll = 0, doneAll = 0;
    CURRICULUM.forEach(tier => {
      let total = 0, done = 0;
      tier.sections.forEach(s => {
        total++;
        totalAll++;
        if (completed[s.id]) { done++; doneAll++; }
      });
      out[tier.id] = { total, done };
    });
    out._all = { total: totalAll, done: doneAll };
    return out;
  }, [completed]);

  const filteredCurriculum = useMemo(() => {
    let items = CURRICULUM;
    if (filter !== "all") {
      if (filter === "completed") {
        items = items.map(t => ({
          ...t,
          sections: t.sections.filter(s => completed[s.id])
        })).filter(t => t.sections.length > 0);
      } else {
        items = items.map(t => ({
          ...t,
          sections: t.sections.filter(s => !completed[s.id])
        })).filter(t => t.sections.length > 0);
      }
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      items = items.map(t => ({
        ...t,
        sections: t.sections.filter(s =>
          s.title.toLowerCase().includes(q) ||
          s.subtopics.some(st => st.toLowerCase().includes(q)) ||
          s.motivatingProblems.some(mp => mp.toLowerCase().includes(q))
        )
      })).filter(t => t.sections.length > 0);
    }
    return items;
  }, [filter, search, completed]);

  const resetAll = () => {
    if (confirm("Reset all progress? This cannot be undone.")) {
      setCompleted({});
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify({})); } catch {}
    }
  };

  const pctAll = stats._all.total > 0 ? Math.round(stats._all.done / stats._all.total * 100) : 0;

  return (
    <>
    {showWarmup && (
      <WarmupView onClose={() => setShowWarmup(false)} />
    )}
    {activeModule && (
      <ModuleView
        module={activeModule.module}
        tierColor={activeModule.tierColor}
        onClose={() => { setActiveModule(null); setGapsVersion(v => v + 1); }}
      />
    )}
    {showGaps && (
      <div style={{position:'fixed',inset:0,zIndex:2000,display:'flex',justifyContent:'center',alignItems:'flex-start'}}>
        <div onClick={() => setShowGaps(false)} style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.4)'}}/>
        <div style={{position:'relative',background:'var(--color-background-primary)',borderRadius:'var(--border-radius-lg)',maxWidth:560,width:'100%',margin:'48px 16px',maxHeight:'calc(100vh - 96px)',overflow:'auto',border:'0.5px solid var(--color-border-tertiary)',padding:'24px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px'}}>
            <h3 style={{fontSize:16,fontWeight:600,margin:0}}>Learning Gaps</h3>
            <button onClick={() => setShowGaps(false)} style={{background:'transparent',border:'none',fontSize:20,cursor:'pointer',color:'var(--color-text-tertiary)',fontFamily:'inherit',padding:'0 4px'}}>&times;</button>
          </div>
          <div style={{display:'flex',gap:8,marginBottom:'20px'}}>
            <button onClick={copyGapsToClipboard} style={{
              fontSize:13,padding:'6px 16px',borderRadius:'var(--border-radius-md)',
              border:'none',background: copied ? '#1D9E75' : '#BA7517',color:'white',
              cursor:'pointer',fontFamily:'inherit',transition:'background 0.15s',
            }}>
              {copied ? '\u2713 Copied!' : 'Copy as prompt'}
            </button>
            <button onClick={handleClearGaps} style={{
              fontSize:13,padding:'6px 16px',borderRadius:'var(--border-radius-md)',
              border:'0.5px solid var(--color-border-tertiary)',background:'transparent',
              color:'var(--color-text-danger)',cursor:'pointer',fontFamily:'inherit',
            }}>
              Clear all
            </button>
          </div>
          <p style={{fontSize:12,color:'var(--color-text-tertiary)',marginBottom:'16px',lineHeight:1.5}}>
            Flag topics you need to study more. Copy them as a prompt to generate new learning modules.
          </p>
          {gaps.length === 0 ? (
            <p style={{textAlign:'center',color:'var(--color-text-tertiary)',padding:'2rem 0',fontSize:14}}>No gaps flagged yet. Use the &quot;Need to learn this&quot; button during modules.</p>
          ) : (
            Object.entries(
              gaps.reduce((acc, g) => { (acc[g.sectionId] = acc[g.sectionId] || []).push(g); return acc; }, {})
            ).map(([sectionId, items]) => (
              <div key={sectionId} style={{marginBottom:'16px'}}>
                <div style={{fontSize:11,fontWeight:500,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--color-text-tertiary)',marginBottom:'6px'}}>
                  {sectionId} &mdash; {getSectionTitle(sectionId)}
                </div>
                {items.map(g => (
                  <div key={g.id} style={{display:'flex',alignItems:'flex-start',gap:8,padding:'8px 10px',borderRadius:'var(--border-radius-md)',background:'var(--color-background-secondary)',marginBottom:4,border:'0.5px solid var(--color-border-tertiary)'}}>
                    <div style={{flex:1,minWidth:0}}>
                      <MathText as="div" style={{fontSize:13,lineHeight:1.5,color:'var(--color-text-primary)',overflow:'hidden',textOverflow:'ellipsis'}}>{g.label}</MathText>
                      <div style={{fontSize:11,color:'var(--color-text-tertiary)',marginTop:2}}>
                        <span style={{fontSize:10,fontWeight:600,padding:'1px 5px',borderRadius:3,
                          background:({easy:'#1D9E7518',medium:'#BA751718',hard:'#D85A3018'})[g.difficulty],
                          color:({easy:'#1D9E75',medium:'#BA7517',hard:'#D85A30'})[g.difficulty],
                          textTransform:'uppercase',letterSpacing:'0.04em',marginRight:6
                        }}>{g.difficulty}</span>
                        {g.moduleTitle}
                      </div>
                    </div>
                    <button onClick={() => handleRemoveGap(g.id)} style={{background:'transparent',border:'none',color:'var(--color-text-tertiary)',cursor:'pointer',fontSize:16,padding:'0 2px',fontFamily:'inherit',flexShrink:0}}>&times;</button>
                  </div>
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    )}
    <div style={{maxWidth:'800px',margin:'0 auto',padding:'1.5rem 0'}}>
      <div style={{marginBottom:'2rem'}}>
        <div style={{display:'flex',alignItems:'baseline',justifyContent:'space-between',marginBottom:'8px',flexWrap:'wrap',gap:'8px'}}>
          <span style={{fontSize:'14px',color:'var(--color-text-secondary)'}}>
            {stats._all.done} of {stats._all.total} topics completed
          </span>
          <span style={{fontSize:'22px',fontWeight:500}}>{pctAll}%</span>
        </div>
        <div style={{height:'6px',background:'var(--color-background-secondary)',borderRadius:'3px',overflow:'hidden'}}>
          <div style={{height:'100%',width:`${pctAll}%`,background:'var(--color-text-info)',borderRadius:'3px',transition:'width 0.3s ease'}}/>
        </div>
      </div>

      {showWarmupPrompt && (
        <div style={{display:'flex',alignItems:'center',gap:'12px',padding:'12px 16px',marginBottom:'16px',borderRadius:'var(--border-radius-lg)',background:'#378ADD10',border:'1px solid #378ADD33'}}>
          <span style={{fontSize:20,flexShrink:0}}>&#9728;&#65039;</span>
          <div style={{flex:1}}>
            <div style={{fontSize:14,fontWeight:500,color:'var(--color-text-primary)',marginBottom:2}}>Daily Warmup</div>
            <div style={{fontSize:12,color:'var(--color-text-secondary)'}}>10 questions across your curriculum to start the day.</div>
          </div>
          <button onClick={() => { setShowWarmupPrompt(false); setShowWarmup(true); }} style={{
            padding:'6px 16px',borderRadius:'var(--border-radius-md)',border:'none',
            background:'#378ADD',color:'white',fontSize:13,fontWeight:500,
            cursor:'pointer',fontFamily:'inherit',flexShrink:0,
          }}>Start</button>
          <button onClick={() => setShowWarmupPrompt(false)} style={{
            background:'transparent',border:'none',color:'var(--color-text-tertiary)',
            cursor:'pointer',fontSize:16,fontFamily:'inherit',padding:'0 4px',flexShrink:0,
          }}>&times;</button>
        </div>
      )}

      <div style={{display:'flex',gap:'8px',marginBottom:'1.5rem',flexWrap:'wrap',alignItems:'center'}}>
        <input
          type="text"
          placeholder="Search topics..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{flex:'1 1 200px',minWidth:'160px'}}
        />
        <div style={{display:'flex',gap:'4px'}}>
          {[["all","All"],["remaining","Remaining"],["completed","Done"]].map(([v,l]) => (
            <button key={v} onClick={() => setFilter(v)} style={{
              background: filter===v ? 'var(--color-text-primary)' : 'transparent',
              color: filter===v ? 'var(--color-background-primary)' : 'var(--color-text-secondary)',
              fontSize:'13px',padding:'4px 12px',borderRadius:'var(--border-radius-md)',
              border: filter===v ? 'none' : '0.5px solid var(--color-border-tertiary)',
              cursor:'pointer',fontFamily:'inherit'
            }}>{l}</button>
          ))}
        </div>
        <button onClick={() => setShowWarmup(true)} style={{
          fontSize:'12px',padding:'4px 12px',borderRadius:'var(--border-radius-md)',
          border:'1px solid #378ADD44',background:'#378ADD08',
          color:'#378ADD',cursor:'pointer',fontFamily:'inherit',
        }}>
          Warmup
        </button>
        {gaps.length > 0 && (
          <button onClick={() => setShowGaps(true)} style={{
            fontSize:'12px',padding:'4px 12px',borderRadius:'var(--border-radius-md)',
            border:'1px solid #BA751744',background:'#BA751708',
            color:'#BA7517',cursor:'pointer',fontFamily:'inherit',
          }}>
            Learning Gaps ({gaps.length})
          </button>
        )}
        <button onClick={resetAll} style={{fontSize:'12px',color:'var(--color-text-danger)',background:'transparent',border:'none',cursor:'pointer',padding:'4px 8px',fontFamily:'inherit'}}>
          Reset
        </button>
      </div>

      {filteredCurriculum.map(tier => {
        const s = stats[tier.id] || {total:0,done:0};
        const pct = s.total > 0 ? Math.round(s.done/s.total*100) : 0;
        const isOpen = expandedTiers[tier.id];
        return (
          <div key={tier.id} style={{marginBottom:'12px'}}>
            <div
              onClick={() => toggleTier(tier.id)}
              style={{
                display:'flex',alignItems:'center',gap:'10px',padding:'12px 16px',
                background:'var(--color-background-secondary)',borderRadius:'var(--border-radius-lg)',
                cursor:'pointer',userSelect:'none',
                border:'0.5px solid var(--color-border-tertiary)'
              }}
            >
              <ChevronIcon open={isOpen}/>
              <div style={{width:'10px',height:'10px',borderRadius:'50%',background:tier.color,flexShrink:0}}/>
              <div style={{flex:1}}>
                <span style={{fontSize:'13px',color:'var(--color-text-secondary)',marginRight:'8px'}}>{tier.label}</span>
                <span style={{fontSize:'15px',fontWeight:500}}>{tier.title}</span>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:'8px',flexShrink:0}}>
                <span style={{fontSize:'12px',color:'var(--color-text-secondary)'}}>{s.done}/{s.total}</span>
                <div style={{width:'60px',height:'4px',background:'var(--color-border-tertiary)',borderRadius:'2px',overflow:'hidden'}}>
                  <div style={{height:'100%',width:`${pct}%`,background:tier.color,borderRadius:'2px',transition:'width 0.3s'}}/>
                </div>
              </div>
            </div>

            {isOpen && (
              <div style={{marginLeft:'20px',borderLeft:`2px solid ${tier.color}22`,paddingLeft:'16px',marginTop:'4px'}}>
                {tier.sections.map(sec => {
                  const isDone = !!completed[sec.id];
                  const isExp = !!expandedSections[sec.id];
                  return (
                    <div key={sec.id} style={{marginBottom:'2px'}}>
                      <div style={{display:'flex',alignItems:'flex-start',gap:'8px',padding:'10px 12px',borderRadius:'var(--border-radius-md)',cursor:'pointer',
                        background: isExp ? 'var(--color-background-secondary)' : 'transparent',
                      }}>
                        <div
                          onClick={(e) => {e.stopPropagation(); toggle(sec.id);}}
                          style={{
                            width:'22px',height:'22px',borderRadius:'var(--border-radius-md)',
                            border: isDone ? 'none' : '1.5px solid var(--color-border-secondary)',
                            background: isDone ? tier.color : 'transparent',
                            color:'white',
                            display:'flex',alignItems:'center',justifyContent:'center',
                            cursor:'pointer',flexShrink:0,marginTop:'1px',
                            transition:'all 0.15s ease'
                          }}
                        >
                          {isDone && <CheckIcon/>}
                        </div>
                        <div style={{flex:1}} onClick={() => toggleSection(sec.id)}>
                          <div style={{display:'flex',alignItems:'center',gap:'6px'}}>
                            <span style={{fontSize:'12px',color:tier.color,fontWeight:500,fontFamily:'var(--font-mono)'}}>{sec.id}</span>
                            <span style={{fontSize:'14px',fontWeight:500,textDecoration:isDone?'line-through':'none',
                              color:isDone?'var(--color-text-tertiary)':'var(--color-text-primary)',
                              transition:'color 0.15s'
                            }}>{sec.title}</span>
                          </div>
                          {!isExp && (
                            <div style={{fontSize:'12px',color:'var(--color-text-tertiary)',marginTop:'2px'}}>
                              {sec.motivatingProblems.length} motivating problem{sec.motivatingProblems.length!==1?'s':''} · {sec.subtopics.length} subtopic{sec.subtopics.length!==1?'s':''}
                            </div>
                          )}
                        </div>
                        <div onClick={() => toggleSection(sec.id)} style={{cursor:'pointer',padding:'2px',color:'var(--color-text-tertiary)'}}>
                          <ChevronIcon open={isExp}/>
                        </div>
                      </div>

                      {isExp && (
                        <div style={{marginLeft:'30px',paddingBottom:'12px'}}>
                          <div style={{marginBottom:'16px'}}>
                            <div style={{fontSize:'11px',fontWeight:500,textTransform:'uppercase',letterSpacing:'0.05em',color:tier.color,marginBottom:'8px'}}>
                              Why study this — the hard problems
                            </div>
                            {sec.motivatingProblems.map((mp, i) => (
                              <div key={i} style={{
                                fontSize:'13px',lineHeight:'1.65',color:'var(--color-text-secondary)',
                                marginBottom:'10px',paddingLeft:'12px',
                                borderLeft:`2px solid ${tier.color}44`
                              }}>
                                {mp}
                              </div>
                            ))}
                          </div>
                          <div>
                            <div style={{fontSize:'11px',fontWeight:500,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--color-text-tertiary)',marginBottom:'6px'}}>
                              Subtopics to cover
                            </div>
                            <div style={{display:'flex',flexWrap:'wrap',gap:'4px'}}>
                              {sec.subtopics.map((st, i) => (
                                <span key={i} style={{
                                  fontSize:'12px',padding:'3px 10px',
                                  background:'var(--color-background-secondary)',
                                  border:'0.5px solid var(--color-border-tertiary)',
                                  borderRadius:'var(--border-radius-md)',
                                  color:'var(--color-text-secondary)',lineHeight:'1.5'
                                }}>{st}</span>
                              ))}
                            </div>
                          </div>
                          {MODULES[sec.id] && (
                            <div style={{marginTop:'16px'}}>
                              <div style={{fontSize:'11px',fontWeight:500,textTransform:'uppercase',letterSpacing:'0.05em',color:tier.color,marginBottom:'8px'}}>
                                Interactive Modules
                              </div>
                              <div style={{display:'flex',gap:'8px',flexWrap:'wrap'}}>
                                {MODULES[sec.id].map(mod => {
                                  const prog = getModuleProgress()[mod.id];
                                  const dc = {easy:'#1D9E75',medium:'#BA7517',hard:'#D85A30'}[mod.difficulty];
                                  return (
                                    <button key={mod.id} onClick={(e) => {e.stopPropagation(); setActiveModule({module:mod,tierColor:tier.color});}}
                                      style={{
                                        display:'flex',alignItems:'center',gap:'8px',
                                        padding:'8px 14px',borderRadius:'var(--border-radius-md)',
                                        background:'var(--color-background-primary)',
                                        border:'0.5px solid var(--color-border-tertiary)',
                                        cursor:'pointer',fontFamily:'inherit',textAlign:'left',
                                        color:'var(--color-text-primary)',
                                      }}>
                                      <span style={{fontSize:'10px',fontWeight:600,padding:'2px 6px',borderRadius:'4px',
                                        background:dc+'18',color:dc,textTransform:'uppercase',letterSpacing:'0.04em',flexShrink:0}}>
                                        {mod.difficulty}
                                      </span>
                                      <span style={{fontSize:'13px',flex:1}}>{mod.title}</span>
                                      <span style={{fontSize:'11px',color: prog?.completed ? '#1D9E75' : 'var(--color-text-tertiary)',flexShrink:0}}>
                                        {prog?.completed ? '\u2713 Done' : `~${mod.estimatedMinutes}m`}
                                      </span>
                                    </button>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {filteredCurriculum.length === 0 && (
        <div style={{textAlign:'center',padding:'3rem 1rem',color:'var(--color-text-tertiary)'}}>
          {search ? "No topics match your search." : "No topics in this filter."}
        </div>
      )}
    </div>
    </>
  );
}
