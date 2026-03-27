// Assessment: Probing & Behavioral Analysis (F.1)
// Pure assessment — no info steps

export const probingAssessment = {
  id: "F.1-assess",
  sectionId: "F.1",
  title: "Assessment: Probing & Behavioral Analysis",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "A linear probe trained on a frozen LLM's hidden representations achieves 92% accuracy at classifying part-of-speech tags. What can we validly conclude from this result?",
      options: ["The model actively uses POS information during its forward pass to make next-token predictions, since a linear probe would not achieve high accuracy unless the information were causally relevant", "The model has explicitly learned an internal POS tagging algorithm during pretraining, with dedicated neurons that classify each token's grammatical role before processing it further", "POS information is stored in a specific identifiable attention head that we have now localized, since linear probes can only achieve high accuracy when the information is concentrated in a single component", "The hidden representations contain linearly accessible information about POS, but this does not prove the model uses that information for downstream tasks -- it may be a byproduct of encoding other features"],
      correct: 3,
      explanation: "A linear probe demonstrates that the information is linearly decodable from the representation, not that the model causally uses it. This is a fundamental limitation: representations are high-dimensional, and a probe might extract information that is a byproduct of encoding other features. The model may encode POS-correlated structure without ever \"reading\" it. Causal methods like activation patching are needed to establish whether information is actually used."
    },
    {
      type: "mc",
      question: "Activation patching (also called causal tracing) works by running a model on a clean input and a corrupted input, then selectively restoring activations from the clean run into the corrupted run at specific positions and layers. If restoring the residual stream at layer $l$, position $t$ recovers the model's original output, what does this demonstrate?",
      options: ["Layer $l$ is the only layer that matters for this prediction, since restoring activations there fully recovers the output while all other layers contribute only noise to the residual stream", "Position $t$ always contains the most important information for any prediction regardless of the specific input, reflecting a fixed architectural bottleneck at that sequence position", "The attention heads at layer $l$ have directly memorized the correct answer as a stored key-value pair, and restoring their activations simply retrieves this memorized association from the model's parameters", "The residual stream at $(l, t)$ carries information that is causally necessary for the model to produce its output on this input, because restoring it suffices to recover the clean behavior in the otherwise-corrupted computation"],
      correct: 3,
      explanation: "Activation patching establishes a causal claim: the information flowing through the residual stream at that specific (layer, position) is sufficient to restore clean behavior when the rest of the computation uses corrupted activations. This is stronger than probing because it demonstrates causal relevance, not just information presence. The original causal tracing work by Meng et al. used this to localize factual associations to mid-layer MLPs at the last subject token position."
    },
    {
      type: "mc",
      question: "ROME (Rank-One Model Editing) edits factual knowledge in a transformer by modifying a single MLP weight matrix. Specifically, it targets the second MLP matrix $W_{\\text{out}}$ at a critical layer identified via causal tracing. What mathematical operation does ROME perform?",
      options: ["It identifies and deletes the specific attention head responsible for retrieving the old fact, then retrains a replacement head from scratch using only the corrected input-output pair as supervision", "It fine-tunes the entire MLP layer using gradient descent on a corrective loss that maximizes the probability of the new target while minimizing KL divergence from the original model on unrelated inputs", "It applies a rank-one update $W_{\\text{out}} \\leftarrow W_{\\text{out}} + \\Delta$ where $\\Delta = \\frac{(v_* - W_{\\text{out}} k_*) k_*^T}{k_*^T k_*}$ is chosen so the MLP maps the subject's key vector $k_*$ to a new value vector $v_*$ encoding the desired fact", "It inserts a new dedicated neuron into the MLP that activates selectively for the edited subject entity, with its output weights set to produce the desired value vector when the subject is detected"],
      correct: 2,
      explanation: "ROME treats the MLP as a key-value memory where the first projection computes keys and the second projection stores values. The rank-one update is a constrained least-squares solution: it modifies $W_{\\text{out}}$ minimally (in Frobenius norm) such that $W_{\\text{out}}^{\\text{new}} k_* = v_*$, where $v_*$ is optimized so the model's output distribution assigns high probability to the new target. This is equivalent to writing a single new key-value association into the MLP's implicit memory."
    },
    {
      type: "mc",
      question: "MEMIT extends ROME to edit multiple facts simultaneously. What is the key difference in MEMIT's approach compared to applying ROME sequentially for each fact?",
      options: [
        "MEMIT uses a completely different mechanism than ROME, targeting attention head QK circuits instead of MLP layers to store the edited facts as modified attention patterns",
        "MEMIT simply applies independent ROME rank-one edits in parallel on different GPUs for speed, with each edit targeting the same single critical layer identified by causal tracing",
        "MEMIT spreads the parameter update across a range of critical layers rather than a single layer, and solves a batched constrained least-squares problem to simultaneously satisfy all edit constraints while minimizing disruption",
        "MEMIT retrains the entire model from scratch with the new facts included as additional examples in the pretraining data, using a modified curriculum that up-weights the corrected associations"
      ],
      correct: 2,
      explanation: "Sequential ROME edits interact destructively: each rank-one update changes the function at one layer, potentially undoing previous edits. MEMIT distributes the update across layers $L = \\{l_1, \\ldots, l_n\\}$ by solving a multi-layer constrained optimization: minimize $\\sum_{l \\in L} \\|\\Delta_l\\|_F^2$ subject to the model producing the correct output for all edited facts. This spreads the perturbation, reducing interference. Experiments show MEMIT scales to thousands of simultaneous edits while maintaining model quality."
    },
    {
      type: "mc",
      question: "Amnesic probing addresses a key limitation of standard linear probing. Instead of training a probe to extract a feature, amnesic probing removes the feature's linear subspace from the representation and measures the effect on downstream task performance. What does this method reveal that standard probing cannot?",
      options: ["Whether the model's downstream behavior actually depends on the linearly-encoded feature, by showing performance degrades when that information is projected out, thereby establishing a causal rather than merely correlational link", "The exact individual neurons responsible for encoding the feature, since projecting out the feature's subspace isolates the specific neurons whose activations change, revealing the encoding mechanism", "Whether the feature is present anywhere in the representation, since standard probing cannot determine if the feature exists at all without the contrastive removal step that amnesic probing provides", "The specific training data examples that caused the feature to be learned during pretraining, since the projection identifies which training distribution characteristics shaped the feature's linear subspace"],
      correct: 0,
      explanation: "Standard probing shows information presence; amnesic probing shows information usage. By computing the linear subspace encoding a feature (e.g., via INLP — Iterative Null-space Projection) and projecting it out, we can measure whether the model's behavior changes. If performance drops, the model's computation causally depends on that linearly-encoded information. If performance is unchanged despite high probe accuracy, the information was present but unused — a \"ghost feature.\" This bridges the gap between probing and causal analysis."
    },
    {
      type: "mc",
      question: "CheckList is a behavioral testing framework for NLP models that organizes tests into a matrix of linguistic capabilities versus test types. Which of the following best describes CheckList's test type taxonomy?",
      options: ["Minimum Functionality Tests (simple sanity checks), Invariance tests (perturbations that should not change the output), and Directional Expectation tests (perturbations that should change the output in a predictable direction)", "Accuracy tests (measuring overall correctness), robustness tests (evaluating stability under noise), and fairness tests (checking demographic parity) -- these three categories cover all relevant model behaviors", "Unit tests on individual neurons (verifying single-neuron activation patterns), integration tests on layers (checking inter-layer information flow), and system tests on full predictions (end-to-end output correctness)", "Training-time tests (evaluating learning dynamics), validation-time tests (measuring held-out generalization), and deployment-time tests (monitoring production performance drift over time)"],
      correct: 0,
      explanation: "CheckList defines three test types: MFT (Minimum Functionality Tests) target simple behaviors the model should handle trivially, akin to unit tests; INV (Invariance) tests apply label-preserving perturbations (e.g., adding irrelevant context, paraphrasing) and check that predictions remain stable; DIR (Directional Expectation) tests apply perturbations with known expected effects (e.g., adding \"This is great\" to a review should increase sentiment score). These are crossed with capabilities like negation, temporal reasoning, and coreference to create a comprehensive behavioral matrix."
    },
    {
      type: "mc",
      question: "A researcher trains a nonlinear (2-layer MLP) probe on frozen GPT-2 representations and achieves 95% accuracy on a syntactic task, while a linear probe achieves only 60%. A colleague argues this proves GPT-2 has rich syntactic representations. What is the strongest methodological concern?",
      options: ["GPT-2 is too small to have learned meaningful syntactic representations during pretraining, so the high probe accuracy must be an artifact of the probe overfitting to superficial statistical patterns in the evaluation dataset", "Nonlinear probes are always methodologically invalid for interpretability research because their excess capacity makes any positive result uninterpretable, regardless of the representation being probed or the control experiments conducted", "A sufficiently expressive nonlinear probe can learn the task itself rather than merely extracting information from the representation — high accuracy may reflect the probe's computational power rather than information present in the frozen features, making it essential to compare against a control baseline (e.g., probing random representations of the same dimensionality)", "MLP probes cannot be reliably trained on frozen representations because the gradient signal from the frozen encoder is too noisy, causing the probe to learn spurious input-output mappings rather than genuine feature extraction"],
      correct: 2,
      explanation: "This is the selectivity concern raised by Hewitt & Liang (2019). A nonlinear probe with enough capacity can achieve high accuracy even on random representations of sufficient dimension, effectively learning a lookup table. The key diagnostic is to measure selectivity: accuracy on the linguistic task minus accuracy when probing random (control) representations. If selectivity is low, the probe is doing the computational work. Linear probes are preferred precisely because their limited capacity makes high accuracy more attributable to the representation rather than the probe."
    },
    {
      type: "mc",
      question: "In causal tracing for factual recall (e.g., \"The Eiffel Tower is located in [Paris]\"), Meng et al. found that restoring activations at a specific component and position most strongly recovers the correct prediction. Which combination was identified as the critical site?",
      options: [
        "Attention head outputs at the final token position in early layers (approximately layers 1-5 in GPT-2 XL), suggesting early attention heads serve as the primary retrieval mechanism for factual associations",
        "MLP outputs at the last subject token position in middle layers (approximately layers 15-25 in GPT-2 XL), suggesting MLPs serve as the primary storage for factual key-value associations",
        "The token embedding layer at the first token position of the subject, suggesting factual information is already encoded in the initial embeddings before any transformer computation occurs",
        "Layer normalization scale parameters at all token positions uniformly across all layers, suggesting that factual knowledge is distributed across the normalization statistics rather than localized in any specific component"
      ],
      correct: 1,
      explanation: "Meng et al.'s causal tracing revealed a striking pattern: corrupting the subject tokens (\"Eiffel Tower\") destroys the prediction, and restoring MLP activations at the last subject token in mid-layers (around layer 17-24 in GPT-2 XL) is sufficient to recover it. This suggests a mechanism where attention heads gather subject information to the last subject position, and mid-layer MLPs act as key-value stores that map subject representations to associated attributes. This finding directly motivated the ROME editing technique."
    },
    {
      type: "mc",
      question: "A CheckList invariance test for a sentiment classifier perturbs inputs by changing named entities (e.g., replacing \"John\" with \"Mary\") and finds that 23% of predictions flip. What does this reveal?",
      options: ["The model has correctly learned that different people express sentiments differently, reflecting a genuine linguistic pattern where speaker identity correlates with sentiment expression style in natural text", "The model exhibits a systematic failure of invariance to identity-irrelevant perturbations, indicating it has learned spurious correlations between names and sentiment labels rather than genuine linguistic understanding of sentiment-bearing content", "The test is methodologically invalid because proper nouns can legitimately affect sentiment in certain contexts, making name-swap perturbations an inappropriate test of invariance for sentiment classification models", "The model simply needs more training data containing those specific names in diverse sentiment contexts, since the prediction flips reflect insufficient coverage rather than a fundamental modeling failure"],
      correct: 1,
      explanation: "Name changes should not affect sentiment predictions in most contexts (e.g., \"John/Mary loved the movie\"). A 23% flip rate reveals the model has learned spurious associations, likely from training data biases where certain names co-occur with certain sentiment distributions. This is precisely the type of fragile, non-generalizable behavior CheckList was designed to detect. It exposes failures that aggregate accuracy metrics miss: the model might achieve 95% accuracy on a test set while harboring systematic biases that manifest in deployment."
    },
    {
      type: "mc",
      question: "Which of the following represents a fundamental limitation shared by all probing-based interpretability methods, regardless of whether the probe is linear or nonlinear?",
      options: ["Probes cannot be trained on GPU-accelerated hardware due to the requirement of computing second-order derivatives through the frozen model's parameters, limiting scalability to small models only", "Probing requires access to the model's original pretraining data distribution, which is usually proprietary and unavailable, making it impossible to ensure the probe's training distribution matches the model's learned representations", "Probes are only applicable to English-language models because the linguistic features they target (POS tags, dependency relations, constituency structure) are defined only for well-resourced languages with established annotation standards", "Probing is purely correlational — it demonstrates what information is decodable from a representation but cannot establish whether the model's own computation accesses or uses that information during its forward pass, because the probe is an external decoder separate from the model's computational graph"],
      correct: 3,
      explanation: "This is the core epistemological limitation of probing. A representation is a high-dimensional object, and any sufficiently rich representation will contain many decodable features as mathematical byproducts, even if the model never \"reads\" them. The probe is an external function that we attach; it has no connection to the model's actual computation. To establish causal usage, one must turn to interventional methods: activation patching (does restoring this information recover the output?), amnesic probing (does removing this information degrade the output?), or ablation studies."
    }
  ]
};
