// Section B.3: Data-Centric Pretraining Assessment

export const dataCentricAssessment = {
  id: "B.3-assess",
  sectionId: "B.3",
  title: "Assessment: Data-Centric Pretraining",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Influence functions estimate how a model's prediction would change if a specific training example were removed (or upweighted). The classic formula involves $\\mathcal{I}(z, z_{\\text{test}}) = -\\nabla_\\theta \\ell(z_{\\text{test}})^\\top H_\\theta^{-1} \\nabla_\\theta \\ell(z)$. Why do influence functions not scale to modern LLMs?",
      options: ["The loss function of LLMs is not twice-differentiable due to the discrete argmax in token selection, making the Hessian undefined", "The gradient $\\nabla_\\theta \\ell(z)$ is always zero at the optimum, so the influence function evaluates to zero for any training example", "Influence functions require the model to be trained to full convergence on the training set, which is deliberately avoided in LLM pretraining", "Computing or approximating the inverse Hessian $H_\\theta^{-1}$ is intractable for billions of parameters, and the quadratic approximation breaks down in the non-convex, overparameterized regime where LLMs operate"],
      correct: 3,
      explanation: "The Hessian $H_\\theta$ is an $N \\times N$ matrix where $N$ is the parameter count — storing it is impossible for LLMs (e.g., $70\\text{B}^2$ entries). Even Hessian-vector product approximations (like LiSSA) are noisy and expensive. Furthermore, influence functions assume a convex loss landscape near the optimum, which does not hold for deep networks. Recent work (TRAK, datamodels) uses random projection-based approximations that trade fidelity for scalability."
    },
    {
      type: "mc",
      question: "Data attribution methods like TRAK (Tracing with Randomly-projected After Kernel) address the scalability limitations of influence functions by:",
      options: ["Using only the first-order gradient without any Hessian information, computing a simple dot product between training and test gradients as the influence proxy", "Training a separate neural network to predict influence scores from input features, bypassing the need to differentiate through the original model entirely", "Projecting per-example gradients into a low-dimensional random subspace, then computing attribution scores via a linear model in that projected space — trading exact inverse-Hessian computation for tractable random projections", "Computing influence only for the last layer of the model, where gradients are largest and most informative about the mapping from representations to predictions"],
      correct: 2,
      explanation: "TRAK projects the high-dimensional gradient vectors $\\nabla_\\theta \\ell(z) \\in \\mathbb{R}^N$ down to $\\mathbb{R}^k$ (with $k \\ll N$) using random matrices. In this compressed space, it fits a linear model that predicts test loss from training example features. This is motivated by the neural tangent kernel (NTK) perspective: near convergence, the model behaves approximately linearly in the projected gradient space. TRAK is orders of magnitude cheaper than exact influence functions."
    },
    {
      type: "mc",
      question: "DSIR (Data Selection with Importance Resampling) selects pretraining data that resembles a target distribution. The core mechanism is:",
      options: ["Computing importance weights $w(x) = p_{\\text{target}}(x) / p_{\\text{source}}(x)$ using n-gram language model ratios, then resampling the source corpus according to these weights", "Training a binary classifier to label each document as \"good\" or \"bad\" and keeping only those predicted as positive for pretraining inclusion", "Clustering the data into semantic groups and selecting the clusters whose centroids are closest to the target distribution's centroid in embedding space", "Using perplexity under a target-domain language model as the sole selection criterion, discarding documents above a fixed perplexity threshold"],
      correct: 0,
      explanation: "DSIR fits lightweight n-gram models to both the target domain and the source corpus, then computes importance weights as the density ratio. Data points that are more likely under the target distribution (relative to the source) get upweighted. Resampling according to these weights yields a subset whose distribution approximates the target. This is much cheaper than training a neural classifier, and importance resampling has well-understood statistical properties."
    },
    {
      type: "mc",
      question: "DoReMi (Xie et al., 2023) optimizes the domain mixing proportions for pretraining data (e.g., how much web text vs. code vs. Wikipedia). How does it determine the optimal mixture?",
      options: [
        "It uses the proportion of each domain in the raw crawl as the optimal mixture, assuming the natural distribution reflects the ideal training balance",
        "It trains a small proxy model using distributionally robust optimization (DRO) to upweight domains where the model struggles most, then uses those optimized proportions to train the large model",
        "It computes the KL divergence between each domain and the target evaluation distribution, then selects the domains with the lowest divergence scores",
        "It alternates between domains in round-robin fashion during training, giving each domain equal exposure regardless of its size or difficulty"
      ],
      correct: 1,
      explanation: "DoReMi uses a two-stage process: (1) train a small reference model on the default mixture, (2) train another small model with group DRO, which dynamically upweights domains with higher excess loss (current loss minus reference loss). The domain weights learned by the small DRO model transfer to the large-scale training run. This avoids expensive large-scale ablations over mixture proportions."
    },
    {
      type: "mc",
      question: "Catastrophic forgetting in continual pretraining occurs when a model fine-tuned on domain-specific data loses its general capabilities. Which of the following is NOT a standard mitigation strategy?",
      options: ["Mixing domain-specific data with a fraction of the original pretraining distribution during continued training to maintain general capabilities", "Using elastic weight consolidation (EWC) or similar regularization that penalizes changes to parameters important for previous tasks", "Replaying a small buffer of original pretraining data alongside the new domain data to maintain the model's prior knowledge", "Training on the new domain for exactly one epoch to prevent overfitting, relying on the single-pass constraint to limit forgetting"],
      correct: 3,
      explanation: "Training for exactly one epoch is not a principled forgetting mitigation — forgetting depends on the degree of distribution shift, not epochs. The other three are well-established approaches: data mixing (most common in practice), EWC-style regularization (penalizes parameter drift weighted by Fisher information), and replay buffers (store and interleave old examples). In practice, simple data mixing (e.g., 90% domain + 10% general) is the most widely used because it is effective and easy to implement."
    },
    {
      type: "mc",
      question: "Learning rate rewarming is a technique used when continuing pretraining on a new data distribution. The practice involves:",
      options: ["Resetting the learning rate to its initial maximum and repeating the full warmup + decay schedule from scratch as if starting pretraining over", "Using a constant learning rate throughout continual pretraining to maintain a steady adaptation rate across the entire new data distribution", "Briefly increasing the learning rate back to a moderate value before decaying again, which helps the model escape the loss basin of the original training distribution and adapt to the new data", "Reducing the learning rate to near zero to prevent catastrophic forgetting by ensuring the model's weights change as little as possible during adaptation"],
      correct: 2,
      explanation: "After pretraining, the LR has decayed to a very small value. If you continue training at this low LR on new data, the model adapts very slowly. Rewarming briefly raises the LR (typically not to the original maximum, but to a meaningful fraction) and then decays again. This lets the model move away from its current minimum to better accommodate the new distribution. The Gupta et al. (2023) work on continual pretraining found rewarming essential for efficient adaptation."
    },
    {
      type: "mc",
      question: "When building a domain-specific LLM (e.g., for biomedicine), you can either (A) pretrain from scratch on domain data, or (B) continue pretraining a general-purpose LLM on domain data. Which statement is most accurate?",
      options: ["Continued pretraining is almost always more compute-efficient: general LLMs have already learned syntax, reasoning, and world knowledge that transfers to the domain, so domain adaptation requires far fewer tokens than learning everything from scratch", "From-scratch pretraining always produces superior domain models because the tokenizer can be optimized for domain-specific vocabulary and subword patterns", "The two approaches yield identical results given the same total compute, since the final loss depends only on aggregate FLOPs regardless of training trajectory", "Continued pretraining cannot work because the general tokenizer lacks domain-specific tokens, causing excessive fragmentation of specialized terminology"],
      correct: 0,
      explanation: "Continued pretraining leverages transfer learning: a 7B model pretrained on 2T tokens has learned language structure, reasoning patterns, and broad knowledge. Adapting it to biomedicine with 50-100B domain tokens is far cheaper than training a biomedical model from scratch on hundreds of billions of tokens. The tokenizer concern is real but secondary — subword tokenizers handle unseen terms by decomposition, and domain terms can be added. Models like BioMedLM, PMC-LLaMA, and SciLLM all use continued pretraining."
    },
    {
      type: "mc",
      question: "Data deduplication before pretraining is considered essential. What is the primary failure mode if near-duplicate documents are not removed?",
      options: [
        "The model's effective vocabulary size grows because duplicated documents introduce redundant surface-level n-gram patterns, fragmenting the learned token representations across synonymous forms",
        "Training loss decreases artificially without improving generalization — the model memorizes duplicated sequences, inflating training metrics while wasting compute on redundant updates and increasing verbatim memorization risks",
        "The optimizer diverges due to repeated gradient directions from the same data points, creating a degenerate update trajectory that spirals away from any stable minimum",
        "Attention heads become specialized for duplicated content, allocating a disproportionate fraction of the model's representational capacity to memorizing those patterns rather than learning general features"
      ],
      correct: 1,
      explanation: "Lee et al. (2022) showed that deduplication improves both training efficiency and downstream performance. Duplicated data means the model sees certain patterns disproportionately often, leading to memorization rather than generalization. It also wastes compute — tokens spent on duplicates could have been spent on diverse examples. MinHash-based near-deduplication is standard practice. Carlini et al. showed that memorization rates correlate strongly with duplication frequency."
    },
    {
      type: "mc",
      question: "In the context of data quality filtering for pretraining, a perplexity-based filter uses a reference language model to score each document. What is a known failure mode of naive perplexity filtering?",
      options: ["It cannot process documents longer than the reference model's context window, causing systematic exclusion of long-form content such as technical papers and books", "It is too slow to apply to web-scale corpora because computing perplexity requires a full forward pass of the reference model over every candidate document", "It removes all non-English text regardless of quality, since the reference language model assigns high perplexity to any text in an unfamiliar language", "It systematically biases the pretraining data toward the style and domain of the reference model's training data — e.g., a Wikipedia-trained reference model will favor Wikipedia-like text and discard informal but informative content"],
      correct: 3,
      explanation: "A reference LM assigns low perplexity to text similar to its own training distribution. A Wikipedia-trained filter will favor encyclopedic prose and penalize code, dialogue, informal writing, and domain-specific jargon — all of which may be high-quality and valuable for a general-purpose LLM. The C4 dataset used a Wikipedia perplexity filter, which is now recognized as having been too aggressive. Modern pipelines use classifier-based quality scoring with more diverse positive examples."
    },
    {
      type: "mc",
      question: "When selecting data for continued pretraining of an LLM on a specialized domain, the optimal strategy with respect to data mixing is:",
      options: [
        "Use only domain-specific data to maximize specialization, since the general capabilities learned during pretraining are robust enough to persist without reinforcement",
        "Use only general data but increase the learning rate to capture domain knowledge from the few relevant examples that happen to appear in the general corpus",
        "Mix domain-specific data with general-purpose data, tuning the ratio empirically — too much domain data causes forgetting of general capabilities, too little yields insufficient specialization",
        "Alternate between pure domain and pure general data in separate training phases, letting the model fully adapt to each distribution before switching to the other"
      ],
      correct: 2,
      explanation: "Data mixing is a Pareto optimization between domain performance and general capability retention. The optimal ratio depends on (1) how different the domain is from general text, (2) how much domain data is available, and (3) which general capabilities matter for the application. Typical ratios range from 50-90% domain data. Pure domain training causes rapid forgetting; phase alternation creates oscillation in capabilities. Continuous mixing provides the smoothest learning dynamics."
    }
  ]
};
