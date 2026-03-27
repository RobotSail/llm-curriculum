// Focused module: Pretraining Data Mixture Optimization
// Covers: domain weighting, DoReMi, scaling-aware mixing, proxy-based optimization.
// Section B.3: Data-Centric Pretraining

export const dataMixingLearning = {
  id: "B.3-data-mixing-learning-medium",
  sectionId: "B.3",
  title: "Pretraining Data Mixture Optimization",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "The Mixture Problem",
      content: "LLM pretraining data comes from many **domains**: web crawls, books, code, Wikipedia, academic papers, social media, and more. Each domain has different characteristics — vocabulary, style, factual density, and relevance to downstream tasks.\n\nThe **data mixture** specifies the proportion of tokens drawn from each domain during training. For example, a mixture might be: 60% web, 15% code, 10% books, 8% Wikipedia, 7% academic.\n\nThis mixture profoundly affects the model's capabilities. Too much web text and the model writes fluently but reasons poorly. Too much code and it becomes an excellent programmer but a mediocre conversationalist. Too little academic text and it lacks factual grounding.\n\nThe core question: **given a fixed compute budget and a pool of data sources, what mixture of domains produces the best model?** This is not a simple engineering decision — it is a research problem with surprising structure."
    },
    {
      type: "mc",
      question: "A team trains two 7B models with identical architectures and compute budgets but different data mixtures: Model A uses 80% web / 20% code, while Model B uses 50% web / 50% code. On a downstream benchmark suite covering both language understanding and programming tasks, which outcome is most likely?",
      options: [
        "Model B dominates on all benchmarks because code data teaches structured reasoning that transfers to language understanding tasks as well",
        "Model A dominates on all benchmarks because web text is more diverse and subsumes the patterns found in code data",
        "Neither model dominates — Model A outperforms on language benchmarks while Model B outperforms on coding benchmarks, with overall performance depending on the benchmark weights",
        "Both models achieve identical overall performance because total compute, not mixture composition, determines final capability"
      ],
      correct: 2,
      explanation: "Data mixture creates Pareto tradeoffs between domain-specific capabilities. More code tokens improve coding at the expense of language understanding (and vice versa). There is no free lunch — the optimal mixture depends on what you want the model to do. This is why mixture optimization requires specifying a target evaluation distribution or weighting over downstream tasks."
    },
    {
      type: "info",
      title: "The Naive Approach: Natural Proportions",
      content: "The simplest mixture strategy is to use the **natural proportions** of available data — if your crawl produces 70% web, 15% code, and 15% other, train on those ratios.\n\nThis approach has two fundamental problems:\n\n**1. Data availability ≠ data value.** Web text is abundant but noisy and repetitive. Academic papers are scarce but dense with factual knowledge. Natural proportions massively over-weight low-information-density sources.\n\n**2. Domain-specific token efficiency varies.** The model may need 100B web tokens to extract most of the learnable signal (since web text is redundant), but only 10B tokens of well-curated code to learn programming. Training on natural proportions wastes compute on diminishing returns from overrepresented domains.\n\nEmpirical evidence is stark: the Llama 2 paper showed that **upsampling Wikipedia and books by 2x** relative to natural proportions improved knowledge-intensive benchmarks by 5-10% with no degradation on other tasks. The natural distribution is not optimal."
    },
    {
      type: "mc",
      question: "A pretraining corpus contains 1T tokens of web text and 50B tokens of scientific papers. Training for 300B tokens at natural proportions would use approximately 285B web tokens and 15B scientific tokens. A researcher instead upsamples science to 30% of the mixture (90B science tokens). What is the main risk of this aggressive upsampling?",
      options: [
        "The model will become unable to process web-style text because the training distribution has shifted too far from natural language usage patterns",
        "The scientific tokens will be repeated ~1.8 epochs while web tokens see ~0.3 epochs, risking memorization of scientific text and under-learning from the larger web corpus",
        "Upsampling science data beyond 10% violates the scaling law relationship between data quantity and model performance established by Chinchilla",
        "The tokenizer will produce suboptimal encodings of scientific text because it was designed for web-crawl vocabulary distributions"
      ],
      correct: 1,
      explanation: "With only 50B science tokens available, a 30% mixture at 300B total means 90B science tokens — repeating the science corpus ~1.8 times. Repeated data risks memorization: the model may learn to recite scientific passages rather than internalize concepts. Meanwhile, only 210B of 1T web tokens are used (0.21 epochs), meaning the model has barely scratched the surface of web data diversity. The optimal mixture must balance domain value against repetition."
    },
    {
      type: "info",
      title: "DoReMi: Proxy-Based Mixture Optimization",
      content: "**DoReMi** (Xie et al., 2023) introduced an elegant two-stage approach to finding optimal mixtures:\n\n**Stage 1 — Train a small proxy model with DRO.** Train a small model (e.g., 280M parameters) using **distributionally robust optimization (DRO)**. At each step, DRO dynamically upweights domains where the model has higher **excess loss** (current loss minus a reference model's loss). Domains the model struggles with get more training weight.\n\n**Stage 2 — Transfer the learned weights.** Take the domain weights learned by the small proxy model and use them to train the full-scale model (e.g., 8B parameters). The key finding: **domain weights learned at small scale transfer to large scale**.\n\nThe reference model prevents a degenerate solution: without it, DRO would permanently upweight the hardest domains (like encrypted text or random bytes) that are unlearnable. The excess loss isolates domains where the model can genuinely improve.\n\nDoReMi improved average downstream performance by 6.5% over natural proportions, and it did this with only a cheap proxy experiment — no expensive large-scale ablations over mixture ratios."
    },
    {
      type: "mc",
      question: "In DoReMi, the reference model's loss is subtracted from the current model's loss to compute excess loss per domain. Why is this subtraction critical?",
      options: [
        "It isolates learnable signal by removing baseline difficulty, preventing DRO from wasting compute on inherently hard but unlearnable domains like encrypted text",
        "It normalizes the gradient magnitudes across domains, preventing any single domain from dominating the optimizer's update step during training",
        "It ensures that the domain weights sum to exactly 1.0 at every training step, which is a required condition for DRO's convergence guarantee",
        "It implements a form of curriculum learning that schedules easy domains early in training and hard domains later in the training process"
      ],
      correct: 0,
      explanation: "Some domains are inherently harder than others regardless of model quality — random byte sequences, encrypted text, or highly technical content may always have high loss. Without the reference subtraction, DRO would spend all its budget on these unlearnable domains. The excess loss (current minus reference) isolates the learnable component: domains where the current model is underperforming relative to what is achievable. This focuses the mixture on domains where additional training actually helps."
    },
    {
      type: "info",
      title: "Scaling-Aware Mixture Optimization",
      content: "A critical discovery in data mixing research: **optimal mixtures change with model scale**.\n\nSmaller models benefit more from clean, curated data (Wikipedia, books) because they have limited capacity and need high-density information. Larger models can extract value from noisier sources (web crawls) because they have the capacity to filter signal from noise internally.\n\nThis creates a practical problem: if you tune your mixture on small proxy experiments (as DoReMi does), the optimal mixture may shift by the time you scale up. Recent work addresses this:\n\n**Scaling law extrapolation**: Fit per-domain loss curves as a function of domain token count and model size. Use the fitted curves to predict optimal mixtures at target scale. This requires running many small experiments across different mixtures and sizes, then fitting a parametric model.\n\n**Online mixture adjustment**: Start with proxy-optimized weights and adjust them during training based on the actual model's per-domain loss trajectory. If a domain's loss plateaus (diminishing returns), reduce its weight and redistribute to domains still improving.\n\nThe practical takeaway: mixture optimization at scale is an **ongoing process**, not a one-time decision. The best training runs monitor per-domain loss curves and adjust accordingly."
    },
    {
      type: "mc",
      question: "A team finds that their 1B proxy model's optimal mixture assigns 25% weight to code data. When they train a 70B model with this same mixture, code benchmark performance is lower than expected. What is the most likely explanation?",
      options: [
        "The 70B model requires a fundamentally different tokenizer optimized for code, and the shared tokenizer degrades code representation quality at larger scale",
        "Larger models are inherently worse at code tasks because their increased parameter count causes them to memorize rather than generalize from code patterns",
        "The 70B model needs a lower code fraction because at larger scale, general language understanding transfers more effectively to code tasks, reducing the marginal value of dedicated code training",
        "The 70B model needs a higher code fraction because its greater capacity can extract more value from code data before hitting diminishing returns, making the 1B-optimized mixture under-allocate code"
      ],
      correct: 3,
      explanation: "Larger models have more capacity to absorb and generalize from structured data like code. A 1B model may saturate on code patterns at 25% allocation, but a 70B model can continue extracting useful signal well beyond that point. The optimal code fraction typically increases with scale. This is why proxy-based mixture optimization (like DoReMi) must account for scale — naively transferring small-model weights underestimates the data appetite of large models for high-information-density domains."
    },
    {
      type: "info",
      title: "Data Selection vs. Data Mixing",
      content: "Mixture optimization is related to but distinct from **data selection** (choosing which individual documents to include).\n\n**Data mixing** decides the proportions: \"Train on 60% web, 25% code, 15% curated.\" It operates at the domain level.\n\n**Data selection** decides which documents within each domain to keep: \"Include this web page but exclude that one.\" Methods like DSIR (Data Selection with Importance Resampling) use density ratios to select documents that match a target distribution.\n\nThe two interact: a web domain with aggressive quality filtering (data selection) is higher-value per token, which changes its optimal weight in the mixture (data mixing). The best training pipelines optimize both:\n\n1. **Within-domain selection**: Filter each domain for quality using classifiers, perplexity scores, or heuristic rules\n2. **Cross-domain mixing**: Set domain proportions using DoReMi-style optimization or scaling law extrapolation\n3. **Deduplication**: Remove near-duplicates within and across domains to avoid memorization\n\nThese three decisions — what to keep, how much of each, and what to remove — define the effective training distribution. Getting them right is often more impactful than architectural innovations at the same compute budget."
    },
    {
      type: "mc",
      question: "A team applies aggressive quality filtering to their web crawl, removing 80% of documents. They then optimize their data mixture using DoReMi. Compared to running DoReMi on the unfiltered web data, what should they expect?",
      options: [
        "DoReMi will likely assign higher weight to the filtered web domain because each remaining web token is higher quality, shifting the optimal mixture toward using more of it",
        "The results will be identical because DoReMi's DRO mechanism internally performs the same filtering that was applied externally",
        "DoReMi will assign lower weight to the filtered web domain because it now contains fewer tokens, and domain weight is proportional to domain size",
        "DoReMi cannot function after quality filtering because the reference model was trained on the unfiltered distribution and the loss comparison becomes invalid"
      ],
      correct: 0,
      explanation: "Quality filtering increases the per-token value of the web domain by removing noise, boilerplate, and low-quality content. DoReMi assigns weight based on where the model can improve most (excess loss), and a cleaner domain offers more learnable signal per token. The filtered web data now competes more effectively with high-quality curated sources like books and Wikipedia. This interaction illustrates why within-domain selection and cross-domain mixing must be co-optimized."
    },
    {
      type: "info",
      title: "Practical Mixture Design",
      content: "In practice, mixture optimization is constrained by several factors beyond pure loss minimization:\n\n**Data availability**: You cannot upweight a domain beyond what repetition allows. If you have 20B tokens of math data, allocating 30% of a 300B training run (90B tokens) means 4.5 epochs of repetition — well past the point of diminishing returns.\n\n**Legal and safety constraints**: Some data sources have licensing restrictions. Others contain toxic or harmful content that must be limited regardless of its training value.\n\n**Capability priorities**: The mixture should reflect the intended use case. A coding assistant needs more code; a medical AI needs more biomedical text. There is no universally optimal mixture.\n\n**Epoch budgets**: Muennighoff et al. (2023) showed that repeating data up to ~4 epochs causes minimal degradation, but beyond that, memorization effects dominate. This sets a ceiling on how much you can upweight scarce domains.\n\nA practical starting point used by many teams:\n- Run DoReMi-style proxy experiments at 1-5% of target compute\n- Apply the learned weights, capping any domain at 3-4 epochs\n- Monitor per-domain validation loss during training and adjust if curves plateau"
    },
    {
      type: "mc",
      question: "During a 500B-token training run, per-domain validation loss tracking shows that Wikipedia loss plateaued 100B tokens ago (no improvement despite continued training on Wikipedia tokens), while code loss is still decreasing steadily. What should the team do?",
      options: [
        "Stop training entirely since the plateau indicates the model has converged and further training will only cause overfitting across all domains",
        "Remove Wikipedia from the mixture entirely and redistribute its allocation to code, since Wikipedia has contributed all possible learning signal",
        "Reduce Wikipedia's weight and increase code's weight for the remaining training budget, since Wikipedia shows diminishing returns while code still offers learnable signal",
        "Double the learning rate to break through the Wikipedia plateau, since the stagnation indicates the optimizer is stuck in a local minimum for that domain"
      ],
      correct: 2,
      explanation: "A domain-specific loss plateau means the model has extracted most learnable signal from that domain at the current mixture rate. Continuing at the same rate wastes compute on near-zero improvement. The correct response is to redistribute: reduce Wikipedia weight (not eliminate — maintaining some prevents forgetting) and increase code weight where learning is still active. This online adjustment is the practical version of scaling-aware mixture optimization — adjusting the mixture mid-run based on observed learning dynamics."
    },
    {
      type: "mc",
      question: "A researcher claims they can skip mixture optimization entirely by simply training on all available data in its natural proportions, arguing that 'more diverse data is always better.' Which empirical finding most directly contradicts this claim?",
      options: [
        "Models trained on natural proportions show the same downstream performance as mixture-optimized models but require 10x more training compute to reach that level",
        "Natural-proportion training causes catastrophic forgetting of low-resource domains because the model's parameters are overwhelmed by the high-resource domains",
        "Natural-proportion training consistently outperforms mixture-optimized training when the total dataset exceeds 10T tokens, invalidating the need for optimization at frontier scale",
        "DoReMi's optimized mixture improved downstream performance by 6.5% over natural proportions at the same compute budget, demonstrating that proportion matters as much as diversity"
      ],
      correct: 3,
      explanation: "The DoReMi result directly demonstrates that mixture composition matters: at identical compute and data diversity, simply changing the proportions improved downstream metrics by 6.5%. Natural proportions over-weight cheap but redundant web data and under-weight high-value domains like code and academic text. The improvement is free — it requires no additional data, compute, or architectural changes, only a smarter allocation of existing resources."
    }
  ]
};
