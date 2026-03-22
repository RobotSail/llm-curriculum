// Branch F Assessments: Interpretability & Theory
// F.1: Probing & Behavioral Analysis, F.2: Mechanistic Interpretability,
// F.3: Training Dynamics Interpretability, F.4: Formal Theory of Transformers
// Pure assessment — no info steps

// ============================================================================
// F.1: Probing & Behavioral Analysis
// ============================================================================
export const probingAssessment = {
  id: "F.1-assess",
  sectionId: "F.1",
  title: "Assessment: Probing & Behavioral Analysis",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "A linear probe trained on a frozen LLM's hidden representations achieves 92% accuracy at classifying part-of-speech tags. What can we validly conclude from this result?",
      options: [
        "The model actively uses POS information during its forward pass to make predictions",
        "The hidden representations contain linearly accessible information about POS, but this does not prove the model uses that information for downstream tasks",
        "The model has explicitly learned a POS tagging algorithm during pretraining",
        "POS information is stored in a specific attention head that we have now identified"
      ],
      correct: 1,
      explanation: "A linear probe demonstrates that the information is linearly decodable from the representation, not that the model causally uses it. This is a fundamental limitation: representations are high-dimensional, and a probe might extract information that is a byproduct of encoding other features. The model may encode POS-correlated structure without ever \"reading\" it. Causal methods like activation patching are needed to establish whether information is actually used."
    },
    {
      type: "mc",
      question: "Activation patching (also called causal tracing) works by running a model on a clean input and a corrupted input, then selectively restoring activations from the clean run into the corrupted run at specific positions and layers. If restoring the residual stream at layer $l$, position $t$ recovers the model's original output, what does this demonstrate?",
      options: [
        "Layer $l$ is the only layer that matters for this prediction",
        "The residual stream at $(l, t)$ carries information that is causally necessary for the model to produce its output on this input, because restoring it suffices to recover the clean behavior in the otherwise-corrupted computation",
        "The attention heads at layer $l$ have memorized the answer",
        "Position $t$ always contains the most important information regardless of the input"
      ],
      correct: 1,
      explanation: "Activation patching establishes a causal claim: the information flowing through the residual stream at that specific (layer, position) is sufficient to restore clean behavior when the rest of the computation uses corrupted activations. This is stronger than probing because it demonstrates causal relevance, not just information presence. The original causal tracing work by Meng et al. used this to localize factual associations to mid-layer MLPs at the last subject token position."
    },
    {
      type: "mc",
      question: "ROME (Rank-One Model Editing) edits factual knowledge in a transformer by modifying a single MLP weight matrix. Specifically, it targets the second MLP matrix $W_{\\text{out}}$ at a critical layer identified via causal tracing. What mathematical operation does ROME perform?",
      options: [
        "It fine-tunes the entire MLP layer with gradient descent on a corrective loss",
        "It applies a rank-one update $W_{\\text{out}} \\leftarrow W_{\\text{out}} + \\Delta$ where $\\Delta = \\frac{(v_* - W_{\\text{out}} k_*) k_*^T}{k_*^T k_*}$ is chosen so the MLP maps the subject's key vector $k_*$ to a new value vector $v_*$ encoding the desired fact",
        "It deletes the attention head responsible for the old fact and retrains a new one",
        "It adds a new neuron to the MLP that fires only for the edited fact"
      ],
      correct: 1,
      explanation: "ROME treats the MLP as a key-value memory where the first projection computes keys and the second projection stores values. The rank-one update is a constrained least-squares solution: it modifies $W_{\\text{out}}$ minimally (in Frobenius norm) such that $W_{\\text{out}}^{\\text{new}} k_* = v_*$, where $v_*$ is optimized so the model's output distribution assigns high probability to the new target. This is equivalent to writing a single new key-value association into the MLP's implicit memory."
    },
    {
      type: "mc",
      question: "MEMIT extends ROME to edit multiple facts simultaneously. What is the key difference in MEMIT's approach compared to applying ROME sequentially for each fact?",
      options: [
        "MEMIT uses a completely different architecture than ROME, relying on attention heads instead of MLPs",
        "MEMIT spreads the parameter update across a range of critical layers rather than a single layer, and solves a batched constrained least-squares problem to simultaneously satisfy all edit constraints while minimizing disruption",
        "MEMIT simply applies ROME edits in parallel on different GPUs for speed",
        "MEMIT retrains the entire model from scratch with the new facts included in the training data"
      ],
      correct: 1,
      explanation: "Sequential ROME edits interact destructively: each rank-one update changes the function at one layer, potentially undoing previous edits. MEMIT distributes the update across layers $L = \\{l_1, \\ldots, l_n\\}$ by solving a multi-layer constrained optimization: minimize $\\sum_{l \\in L} \\|\\Delta_l\\|_F^2$ subject to the model producing the correct output for all edited facts. This spreads the perturbation, reducing interference. Experiments show MEMIT scales to thousands of simultaneous edits while maintaining model quality."
    },
    {
      type: "mc",
      question: "Amnesic probing addresses a key limitation of standard linear probing. Instead of training a probe to extract a feature, amnesic probing removes the feature's linear subspace from the representation and measures the effect on downstream task performance. What does this method reveal that standard probing cannot?",
      options: [
        "Whether the feature is present in the representation at all",
        "Whether the model's downstream behavior actually depends on the linearly-encoded feature, by showing performance degrades when that information is projected out, thereby establishing a causal rather than merely correlational link",
        "The exact neurons responsible for encoding the feature",
        "The training data examples that caused the feature to be learned"
      ],
      correct: 1,
      explanation: "Standard probing shows information presence; amnesic probing shows information usage. By computing the linear subspace encoding a feature (e.g., via INLP — Iterative Null-space Projection) and projecting it out, we can measure whether the model's behavior changes. If performance drops, the model's computation causally depends on that linearly-encoded information. If performance is unchanged despite high probe accuracy, the information was present but unused — a \"ghost feature.\" This bridges the gap between probing and causal analysis."
    },
    {
      type: "mc",
      question: "CheckList is a behavioral testing framework for NLP models that organizes tests into a matrix of linguistic capabilities versus test types. Which of the following best describes CheckList's test type taxonomy?",
      options: [
        "Training tests, validation tests, and deployment tests",
        "Minimum Functionality Tests (simple sanity checks), Invariance tests (perturbations that should not change the output), and Directional Expectation tests (perturbations that should change the output in a predictable direction)",
        "Unit tests on individual neurons, integration tests on layers, and system tests on full predictions",
        "Accuracy tests, robustness tests, and fairness tests only"
      ],
      correct: 1,
      explanation: "CheckList defines three test types: MFT (Minimum Functionality Tests) target simple behaviors the model should handle trivially, akin to unit tests; INV (Invariance) tests apply label-preserving perturbations (e.g., adding irrelevant context, paraphrasing) and check that predictions remain stable; DIR (Directional Expectation) tests apply perturbations with known expected effects (e.g., adding \"This is great\" to a review should increase sentiment score). These are crossed with capabilities like negation, temporal reasoning, and coreference to create a comprehensive behavioral matrix."
    },
    {
      type: "mc",
      question: "A researcher trains a nonlinear (2-layer MLP) probe on frozen GPT-2 representations and achieves 95% accuracy on a syntactic task, while a linear probe achieves only 60%. A colleague argues this proves GPT-2 has rich syntactic representations. What is the strongest methodological concern?",
      options: [
        "Nonlinear probes are always invalid because they are too powerful",
        "A sufficiently expressive nonlinear probe can learn the task itself rather than merely extracting information from the representation — high accuracy may reflect the probe's computational power rather than information present in the frozen features, making it essential to compare against a control baseline (e.g., probing random representations of the same dimensionality)",
        "GPT-2 is too small to learn syntax so the result must be wrong",
        "MLP probes cannot be trained on frozen representations"
      ],
      correct: 1,
      explanation: "This is the selectivity concern raised by Hewitt & Liang (2019). A nonlinear probe with enough capacity can achieve high accuracy even on random representations of sufficient dimension, effectively learning a lookup table. The key diagnostic is to measure selectivity: accuracy on the linguistic task minus accuracy when probing random (control) representations. If selectivity is low, the probe is doing the computational work. Linear probes are preferred precisely because their limited capacity makes high accuracy more attributable to the representation rather than the probe."
    },
    {
      type: "mc",
      question: "In causal tracing for factual recall (e.g., \"The Eiffel Tower is located in [Paris]\"), Meng et al. found that restoring activations at a specific component and position most strongly recovers the correct prediction. Which combination was identified as the critical site?",
      options: [
        "Attention heads at the final token position in early layers",
        "MLP outputs at the last subject token position in middle layers (approximately layers 15-25 in GPT-2 XL), suggesting MLPs serve as the primary storage for factual key-value associations",
        "The embedding layer at the first token position",
        "Layer normalization parameters at all positions uniformly"
      ],
      correct: 1,
      explanation: "Meng et al.'s causal tracing revealed a striking pattern: corrupting the subject tokens (\"Eiffel Tower\") destroys the prediction, and restoring MLP activations at the last subject token in mid-layers (around layer 17-24 in GPT-2 XL) is sufficient to recover it. This suggests a mechanism where attention heads gather subject information to the last subject position, and mid-layer MLPs act as key-value stores that map subject representations to associated attributes. This finding directly motivated the ROME editing technique."
    },
    {
      type: "mc",
      question: "A CheckList invariance test for a sentiment classifier perturbs inputs by changing named entities (e.g., replacing \"John\" with \"Mary\") and finds that 23% of predictions flip. What does this reveal?",
      options: [
        "The model has correctly learned that different people have different sentiments",
        "The model exhibits a systematic failure of invariance to identity-irrelevant perturbations, indicating it has learned spurious correlations between names and sentiment labels rather than genuine linguistic understanding of sentiment-bearing content",
        "The test is invalid because names can affect sentiment",
        "The model needs more training data with those specific names"
      ],
      correct: 1,
      explanation: "Name changes should not affect sentiment predictions in most contexts (e.g., \"John/Mary loved the movie\"). A 23% flip rate reveals the model has learned spurious associations, likely from training data biases where certain names co-occur with certain sentiment distributions. This is precisely the type of fragile, non-generalizable behavior CheckList was designed to detect. It exposes failures that aggregate accuracy metrics miss: the model might achieve 95% accuracy on a test set while harboring systematic biases that manifest in deployment."
    },
    {
      type: "mc",
      question: "Which of the following represents a fundamental limitation shared by all probing-based interpretability methods, regardless of whether the probe is linear or nonlinear?",
      options: [
        "Probes cannot be trained on GPU-accelerated hardware",
        "Probing is purely correlational — it demonstrates what information is decodable from a representation but cannot establish whether the model's own computation accesses or uses that information during its forward pass, because the probe is an external decoder separate from the model's computational graph",
        "Probes only work on English-language models",
        "Probing requires access to the model's training data, which is usually unavailable"
      ],
      correct: 1,
      explanation: "This is the core epistemological limitation of probing. A representation is a high-dimensional object, and any sufficiently rich representation will contain many decodable features as mathematical byproducts, even if the model never \"reads\" them. The probe is an external function that we attach; it has no connection to the model's actual computation. To establish causal usage, one must turn to interventional methods: activation patching (does restoring this information recover the output?), amnesic probing (does removing this information degrade the output?), or ablation studies."
    }
  ]
};

// ============================================================================
// F.2: Mechanistic Interpretability
// ============================================================================
export const mechInterpAssessment = {
  id: "F.2-assess",
  sectionId: "F.2",
  title: "Assessment: Mechanistic Interpretability",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "Superposition in neural networks refers to the phenomenon where a model represents more features than it has dimensions. If a model has $d$ dimensions and needs to represent $m \\gg d$ features, how does it accomplish this?",
      options: [
        "It discards all but the $d$ most important features and ignores the rest",
        "It assigns nearly-orthogonal directions to different features, tolerating small interference between them — features are packed as approximately orthogonal vectors in $\\mathbb{R}^d$, exploiting the fact that $\\mathbb{R}^d$ can contain far more nearly-orthogonal vectors than exactly orthogonal ones",
        "It increases $d$ dynamically during inference to accommodate all features",
        "It stores extra features in the model's parameter memory rather than in activations"
      ],
      correct: 1,
      explanation: "The key mathematical insight is that while $\\mathbb{R}^d$ has at most $d$ mutually orthogonal vectors, it can contain exponentially many nearly-orthogonal vectors. Specifically, the Johnson-Lindenstrauss lemma implies that $\\exp(\\Omega(d\\epsilon^2))$ vectors can be packed with pairwise dot products bounded by $\\epsilon$. Superposition exploits this: each feature gets a direction, and the model tolerates the small cross-talk (interference) between features. Toy models show this emerges naturally when features are sparse — the expected interference is proportional to feature co-occurrence probability."
    },
    {
      type: "mc",
      question: "Sparse autoencoders (SAEs) are used to decompose superposition in neural network activations. Given an activation vector $x \\in \\mathbb{R}^d$, an SAE with hidden dimension $m \\gg d$ computes $f(x) = \\text{ReLU}(W_{\\text{enc}}(x - b_{\\text{dec}}) + b_{\\text{enc}})$ and reconstructs $\\hat{x} = W_{\\text{dec}} f(x) + b_{\\text{dec}}$. What role does the sparsity penalty play?",
      options: [
        "It ensures the decoder weights are orthonormal",
        "It encourages each activation $x$ to be reconstructed using only a few dictionary elements, so that each active element corresponds to a single interpretable feature rather than a dense entangled mixture — without sparsity, the autoencoder would learn a trivial dense code that does not disentangle features",
        "It reduces the computational cost of the forward pass by pruning neurons",
        "It forces the encoder and decoder to be transposes of each other"
      ],
      correct: 1,
      explanation: "The sparsity penalty (typically an $L_1$ penalty on $f(x)$, i.e., $\\lambda \\|f(x)\\|_1$) is what makes the autoencoder learn a meaningful decomposition. Without it, the autoencoder would find a dense code where every dictionary element is slightly active — just a rotation of the original space. With sparsity, each input activates only $k \\ll m$ dictionary elements, and each element tends to correspond to a single interpretable feature. The total loss is $\\|x - \\hat{x}\\|_2^2 + \\lambda \\|f(x)\\|_1$, balancing reconstruction fidelity against code sparsity."
    },
    {
      type: "mc",
      question: "Induction heads are a specific circuit motif discovered in transformers. They implement the pattern: having seen \"A B ... A\" in context, predict \"B\" as the next token. What is the two-head mechanism that implements this?",
      options: [
        "A single attention head that memorizes all bigrams from training data",
        "A \"previous token head\" in an earlier layer copies information about B to the A token's residual stream, and then an \"induction head\" in a later layer attends from the current A position back to positions where B's information was written, retrieving B as the prediction — the composition happens through the residual stream connecting the two layers",
        "Two MLP layers that jointly perform a lookup table operation",
        "A head that attends to the first occurrence of A and another that attends to the last token"
      ],
      correct: 1,
      explanation: "The induction head circuit involves composition between two attention heads across layers. In the earlier layer, a \"previous token head\" uses a shifted attention pattern to write the identity of each token's predecessor into the residual stream (at position $i$, it writes information about token $i-1$). In the later layer, the induction head uses key-query matching: when it sees an A token at the current position, it matches against positions where A appeared before, but reads the value written by the previous token head — which is B. This is K-composition: the later head's key/query computation uses information written by the earlier head."
    },
    {
      type: "mc",
      question: "Circuit-level analysis in mechanistic interpretability involves identifying the minimal subgraph of a transformer that implements a specific behavior. When analyzing a circuit for \"indirect object identification\" (e.g., in \"Mary gave the book to John, so John...\"), researchers found a circuit involving approximately 26 attention heads across multiple layers. What technique is used to verify that this circuit is both necessary and sufficient?",
      options: [
        "Training a new model from scratch with only those heads and checking it works",
        "Knockout/ablation experiments: ablating heads outside the circuit should not affect performance on the task (sufficiency), and ablating heads inside the circuit should degrade performance (necessity) — together establishing that the identified subgraph is the minimal computational unit responsible for the behavior",
        "Computing the gradient magnitude of each head's output with respect to the loss",
        "Visualizing attention patterns and selecting heads that attend to the correct tokens"
      ],
      correct: 1,
      explanation: "Circuit verification requires demonstrating both necessity and sufficiency through interventions. For sufficiency, all heads outside the proposed circuit are ablated (e.g., by mean-ablating their outputs), and the model should still perform the task. For necessity, ablating any head inside the circuit should degrade performance. The IOI circuit analysis (Wang et al., 2022) used this methodology, identifying specific head roles: S-inhibition heads, name mover heads, backup name mover heads, and duplicate token heads, each serving a specific function in the information flow that resolves indirect object references."
    },
    {
      type: "mc",
      question: "Steering vectors are computed by taking the difference in mean activations between two sets of inputs (e.g., positive vs. negative sentiment) at a specific layer: $v = \\mathbb{E}[h_l | \\text{positive}] - \\mathbb{E}[h_l | \\text{negative}]$. Adding $\\alpha \\cdot v$ to the residual stream during generation steers the model's behavior. What is a key theoretical concern with this approach?",
      options: [
        "Steering vectors only work with models that have exactly 12 layers",
        "The difference-in-means direction may conflate multiple independent features that happen to correlate with the contrast set, and steering along this direction may activate unintended features — superposition means concepts are not axis-aligned, so a single vector may be a mixture of several feature directions",
        "Adding vectors to the residual stream always causes the model to output gibberish",
        "Steering vectors require retraining the model, making them impractical"
      ],
      correct: 1,
      explanation: "Because of superposition, the direction $v$ obtained from difference-in-means is not guaranteed to correspond to a single clean feature. It may be a linear combination of several feature directions that happen to correlate with the contrast. When you steer along $v$, you may activate all of these features simultaneously, producing unintended side effects. For example, a \"truthfulness\" steering vector might also encode formality, topic, or other correlated attributes. SAE-based feature finding can help isolate cleaner individual feature directions for more precise steering."
    },
    {
      type: "mc",
      question: "In the toy models of superposition studied by Elhage et al. (2022), a key finding was that features organize into specific geometric structures. When features have similar importance and sparsity, what geometric arrangement emerges?",
      options: [
        "Features are always aligned with the standard basis vectors $e_1, e_2, \\ldots$",
        "Features arrange into regular geometric structures such as antipodal pairs, triangles, pentagons, and tetrahedra — maximizing the number of nearly-orthogonal directions in the available dimensions while maintaining uniform angular separation",
        "Features are randomly distributed with no discernible pattern",
        "Features collapse into a single shared direction"
      ],
      correct: 1,
      explanation: "Toy model experiments reveal that superposition is not random but highly structured. With 2 dimensions and uniform feature importance, features form regular polygons (pentagons, hexagons) that tile the plane. In 3D, they form platonic solid vertices. These are solutions to the Thomson problem (placing points on a sphere to minimize energy), which is equivalent to maximizing minimum pairwise angle. When feature importances vary, the geometry breaks symmetry: more important features get directions closer to orthogonal, while less important features are packed more tightly at the cost of higher interference."
    },
    {
      type: "mc",
      question: "A sparse autoencoder trained on an LLM's residual stream activations produces a dictionary of 32,768 features from a 4,096-dimensional space. Researchers find that feature #7,291 activates strongly on text discussing \"the Golden Gate Bridge\" and related San Francisco landmarks. When this feature's activation is artificially amplified during generation, the model steers toward discussing San Francisco. What does this demonstrate?",
      options: [
        "That the model has memorized all facts about the Golden Gate Bridge in a single neuron",
        "That SAEs can identify monosemantic (single-concept) feature directions from the polysemantic superposition in the residual stream, and that these learned features correspond to causally active directions — amplifying them changes model behavior in a predictable, concept-specific way",
        "That the model's knowledge of San Francisco is stored only in this single feature",
        "That SAEs always find geographically-organized features"
      ],
      correct: 1,
      explanation: "This example (from Anthropic's work on SAEs applied to Claude) demonstrates two key properties: (1) SAEs succeed at decomposing polysemantic activations into monosemantic features — individual dictionary elements that respond to coherent, interpretable concepts; (2) these features are not just decodable but causally active — intervening on them changes behavior. This is stronger than probing because the features are identified unsupervised (not trained for a specific labeling task) and verified through causal intervention. The 8x expansion ratio (32,768/4,096) reflects the degree of superposition."
    },
    {
      type: "mc",
      question: "K-composition, V-composition, and Q-composition describe how attention heads in different layers interact through the residual stream. In K-composition, a head $H_2$ in layer $l_2$ uses information written to the residual stream by head $H_1$ in layer $l_1 < l_2$ when computing its keys. Which of the following correctly describes a consequence?",
      options: [
        "K-composition allows $H_2$ to attend based on features computed by $H_1$, meaning $H_2$'s attention pattern depends on what $H_1$ wrote — this enables complex matching rules like \"attend to positions where the previous token was X\" (as in induction heads), which would be impossible with a single layer",
        "K-composition means $H_2$ always copies $H_1$'s attention pattern exactly",
        "K-composition only occurs between adjacent layers",
        "K-composition requires $H_1$ and $H_2$ to have the same number of heads"
      ],
      correct: 0,
      explanation: "In K-composition, $H_2$'s key vectors $K_2 = W_K^{(2)} \\cdot (\\text{residual at } l_2)$ incorporate the output of $H_1$ via the residual stream. This means $H_2$ can match queries against keys that encode information from $H_1$'s computation. The induction head is the canonical example: $H_1$ (previous token head) writes token $i-1$'s identity at position $i$, and $H_2$'s keys reflect this, allowing $H_2$ to attend to positions based on what preceded them. Similarly, V-composition means $H_2$ reads values enriched by $H_1$, and Q-composition means queries are enriched."
    },
    {
      type: "mc",
      question: "Feature visualization in vision models involves optimizing an input image $x$ to maximize a specific neuron's activation: $x^* = \\arg\\max_x \\, a_k(x) - \\lambda R(x)$, where $a_k$ is the neuron's activation and $R(x)$ is a regularizer. Why is the regularizer essential?",
      options: [
        "Without regularization, the optimization always converges to a uniform gray image",
        "Without regularization, the optimized input exploits high-frequency adversarial patterns that maximally activate the neuron but are visually meaningless — the regularizer (e.g., total variation, Gaussian blur, transformation robustness) constrains the solution to the natural image manifold so the visualization reflects what the neuron responds to on real inputs",
        "The regularizer prevents the optimization from taking too many gradient steps",
        "Regularization is only needed for convolutional networks, not transformers"
      ],
      correct: 1,
      explanation: "Unregularized feature visualization produces inputs with high-frequency noise patterns — adversarial-style artifacts that exploit the neuron's full receptive field in non-natural ways. These technically maximize activation but reveal the neuron's sensitivity to out-of-distribution inputs rather than its role in processing natural images. Common regularizers include: total variation penalty (encourages spatial smoothness), jitter/rotation augmentation (ensures the feature is robust to transforms), frequency penalization (suppresses high-frequency components), and learned priors from generative models."
    },
    {
      type: "mc",
      question: "A mechanistic interpretability researcher identifies a circuit responsible for a specific behavior and publishes the finding. A critic argues that the circuit explanation may be a \"just-so story\" — post-hoc rationalization that does not actually capture the model's computation. Which experimental approach best addresses this criticism?",
      options: [
        "Showing that the circuit's attention patterns look interpretable in visualization",
        "Demonstrating that the circuit makes novel, testable predictions about model behavior on held-out inputs that are confirmed empirically — for example, predicting that the model will fail on specific adversarial constructions that the circuit analysis identifies as edge cases, or predicting transfer behavior to modified architectures",
        "Training a larger model and checking if the same circuit appears",
        "Publishing the code so others can replicate the attention pattern visualizations"
      ],
      correct: 1,
      explanation: "The strongest validation of a mechanistic explanation is novel prediction. If the circuit analysis is a genuine understanding of the mechanism (not post-hoc rationalization), it should predict behaviors that were not used to construct the explanation. For example, the IOI circuit analysis predicted specific failure modes (e.g., performance degradation with more than two names) that were confirmed. The induction head theory predicted a phase transition in loss during training, which was observed. Predictive power distinguishes genuine mechanistic understanding from pattern-matching on cherry-picked examples."
    }
  ]
};

// ============================================================================
// F.3: Training Dynamics Interpretability
// ============================================================================
export const trainingInterpAssessment = {
  id: "F.3-assess",
  sectionId: "F.3",
  title: "Assessment: Training Dynamics Interpretability",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "Grokking refers to a phenomenon where a neural network first memorizes the training data (achieving near-zero training loss while validation loss remains high), and then, thousands of steps later, suddenly generalizes (validation loss drops sharply). On which type of task was grokking first demonstrated?",
      options: [
        "Large-scale language modeling on internet text",
        "Modular arithmetic operations (e.g., $a \\circ b \\mod p$ for a binary operation $\\circ$ on elements of $\\mathbb{Z}/p\\mathbb{Z}$), where the model transitions from memorizing a lookup table to learning the underlying algebraic algorithm long after achieving perfect training accuracy",
        "Image classification on ImageNet",
        "Machine translation between English and French"
      ],
      correct: 1,
      explanation: "Power et al. (2022) discovered grokking on algorithmic tasks over finite groups, particularly modular arithmetic (addition, multiplication, etc. mod a prime $p$). With a small training fraction (~30-50% of all $(a, b)$ pairs), the model achieves 100% training accuracy within ~$10^3$ steps by memorizing, but validation accuracy remains at chance until ~$10^5$ steps, when it suddenly jumps to ~100%. Mechanistic analysis by Neel Nanda et al. revealed the model learns interpretable Fourier-based algorithms: it represents numbers as points on a circle and computes the operation using trigonometric identities."
    },
    {
      type: "mc",
      question: "In the context of grokking, weight decay plays a critical role. Without weight decay, grokking either does not occur or takes much longer. What is the mechanistic explanation for why weight decay promotes grokking?",
      options: [
        "Weight decay prevents the model from learning at all, forcing it to randomly guess until it finds the right algorithm",
        "Weight decay provides continuous pressure toward lower-norm solutions, and the generalizing circuit (which exploits algebraic structure) has lower parameter norm than the memorizing circuit (which requires a large lookup table) — over time, weight decay erodes the memorization solution and the structured algorithm emerges as the lower-norm attractor",
        "Weight decay makes the learning rate effectively larger, speeding up all learning uniformly",
        "Weight decay only affects the bias terms, which are irrelevant to memorization"
      ],
      correct: 1,
      explanation: "The memorizing solution requires large weights to implement a lookup table over all training examples, while the generalizing solution (e.g., the Fourier algorithm for modular addition) uses structured, lower-norm weights. Weight decay penalizes $\\|\\theta\\|_2^2$, creating a persistent force toward the origin in parameter space. Initially, the memorization solution dominates because it's found first via gradient descent. But weight decay continuously shrinks it, eventually allowing the generalizing solution — which is a lower-norm fixed point — to take over. This explains the delayed generalization: it's the timescale of weight decay eroding a local minimum."
    },
    {
      type: "mc",
      question: "Phase transitions during neural network training refer to abrupt, qualitative changes in model behavior or internal representations. Which of the following best describes a phase transition observed in transformer language model training?",
      options: [
        "The loss decreases linearly throughout training with no sudden changes",
        "Induction heads form at a specific point during training, coinciding with a discrete drop in loss on sequences requiring in-context pattern completion — before this transition, the model cannot perform in-context learning of bigram statistics; after it, the model suddenly acquires this capability",
        "The model's vocabulary size increases during training",
        "All attention heads simultaneously learn their final attention patterns in the first 100 steps"
      ],
      correct: 1,
      explanation: "Olsson et al. (2022) identified a sharp phase transition in transformer training where induction heads form. Before the transition, attention heads show only local patterns. At the transition (which occurs at a specific training step), induction heads emerge and the model acquires the ability to complete in-context \"A B ... A -> B\" patterns. This manifests as a visible bump/drop in the loss curve. The transition is abrupt rather than gradual, suggesting a bifurcation in the learning dynamics — the circuit \"snaps\" into place as compositions between layers become reinforcing."
    },
    {
      type: "mc",
      question: "The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) states that dense, randomly-initialized neural networks contain sparse subnetworks that, when trained in isolation from the same initialization, can match the full network's performance. What is the formal procedure for finding these \"winning tickets\"?",
      options: [
        "Randomly remove 90% of weights and train the remaining 10% from a new random initialization",
        "Train the full network to completion, prune the smallest-magnitude weights to obtain a mask $m$, then reset the surviving weights to their original initialization $\\theta_0$ and retrain the sparse network $m \\odot \\theta_0$ — the winning ticket is the pair $(m, \\theta_0)$ where the mask is found by training and the initialization is the original one",
        "Use a neural architecture search to find the optimal sparse architecture from scratch",
        "Distill the dense network into a smaller dense network with fewer layers"
      ],
      correct: 1,
      explanation: "The original LTH procedure is: (1) Initialize the network with parameters $\\theta_0$; (2) Train to convergence, obtaining $\\theta_T$; (3) Prune the $p\\%$ smallest-magnitude weights, creating binary mask $m$; (4) Reset remaining weights to $\\theta_0$ (not to $\\theta_T$ or a new random init); (5) Retrain $m \\odot \\theta_0$. The crucial finding is that the original initialization $\\theta_0$ is essential — the same mask with a different random init often fails. This implies the initialization contains structure that the sparse subnetwork exploits. Winning tickets at 90%+ sparsity can match or exceed dense network performance."
    },
    {
      type: "mc",
      question: "Later work on the Lottery Ticket Hypothesis found that for larger networks, the original initialization $\\theta_0$ must be replaced with weights from early in training $\\theta_k$ (\"rewinding\" to step $k$) for the sparse network to train successfully. What does this suggest about the training dynamics?",
      options: [
        "The original hypothesis was completely wrong and should be abandoned",
        "The early phase of training (steps $0$ to $k$) performs a critical restructuring of the initialization — it moves the parameters into a basin of attraction from which sparse training can succeed, suggesting that the \"lottery\" is not purely in the initialization but in the early training dynamics that establish the right loss landscape geometry for sparse optimization",
        "Weight rewinding is just a regularization trick with no deeper meaning",
        "Larger networks cannot be pruned at all"
      ],
      correct: 1,
      explanation: "Frankle et al. (2020) showed that for larger models (e.g., ResNet-50), rewinding to $\\theta_0$ fails but rewinding to $\\theta_k$ (typically $k$ = 1-5% of total training) succeeds. This reveals that early training performs a kind of \"alignment\" — moving weights into a region of parameter space where the loss landscape supports sparse optimization. This connects to other findings about distinct training phases: an early \"chaotic\" phase where the network explores, followed by a more structured phase. The rewinding point $k$ often coincides with the stabilization of the loss Hessian's top eigenvectors."
    },
    {
      type: "mc",
      question: "Singular Learning Theory (SLT), developed by Watanabe, provides a Bayesian framework for understanding generalization in singular (non-regular) statistical models like neural networks. A key quantity is the learning coefficient $\\lambda$ (also called the real log canonical threshold, RLCT). How does $\\lambda$ relate to generalization?",
      options: [
        "$\\lambda$ is identical to the number of parameters $d$ and provides no additional information",
        "The learning coefficient $\\lambda$ replaces $d/2$ in the BIC/MDL formula: the free energy scales as $\\lambda \\ln n$ (where $n$ is sample size) rather than $(d/2) \\ln n$ — for singular models, $\\lambda \\leq d/2$ because symmetries and degeneracies in the parameter space reduce the effective complexity, making $\\lambda$ a better predictor of generalization than raw parameter count",
        "$\\lambda$ measures the model's training speed and has no connection to generalization",
        "$\\lambda$ is only defined for linear models and cannot be applied to neural networks"
      ],
      correct: 1,
      explanation: "In regular statistical models, the Bayesian free energy (negative log marginal likelihood) is approximately $nL_n(\\hat{\\theta}) + (d/2)\\ln n$ (BIC). But neural networks are singular: the Fisher information matrix is degenerate, and the map from parameters to distributions is not one-to-one. Watanabe proved that for singular models, $d/2$ is replaced by the RLCT $\\lambda$, which captures the effective dimensionality of the model near the true distribution. Since $\\lambda \\leq d/2$, singular models are less complex than their parameter count suggests. This explains why overparameterized networks generalize: their effective complexity (measured by $\\lambda$) is much smaller than $d$."
    },
    {
      type: "mc",
      question: "In the context of grokking, the training process can be decomposed into distinct phases visible in the weight norm dynamics. Which sequence of phases is observed?",
      options: [
        "Random initialization, immediate generalization, followed by gradual forgetting",
        "An initial phase where both training loss drops and weight norm grows (memorization via embedding growth), a plateau where training accuracy is perfect but generalization is absent, then a phase where weight norm decreases while validation accuracy suddenly improves (the generalizing circuit overtakes the memorizing one as weight decay compresses the representation)",
        "Weight norm decreases monotonically throughout training while both losses decrease together",
        "Weight norm remains constant while the model alternates between memorizing and generalizing"
      ],
      correct: 1,
      explanation: "The weight norm dynamics reveal the mechanistic story of grokking: (1) Weight norm grows as the model memorizes — it needs large weights to implement the lookup table; (2) A long plateau where training is perfect but generalization is absent — the model is stuck in the memorization basin; (3) Weight norm begins decreasing as weight decay overwhelms the memorization signal (since training loss is already zero, there is no gradient pressure to maintain large weights); (4) As the memorization circuit shrinks, the generalizing circuit — which has been slowly growing in the background — becomes dominant, causing sudden generalization."
    },
    {
      type: "mc",
      question: "The phenomenon of \"double descent\" in the bias-variance tradeoff challenges classical statistical learning theory. In the modern interpolation regime, what is observed as model complexity increases past the interpolation threshold (the point where the model can exactly fit the training data)?",
      options: [
        "Test error increases monotonically, exactly as classical theory predicts",
        "Test error spikes at the interpolation threshold (where model capacity equals the effective number of training constraints) and then decreases again as the model becomes further overparameterized — the model transitions from barely fitting the data with large-norm solutions to smoothly interpolating with low-norm solutions that generalize well",
        "Test error drops to zero and remains there for all larger models",
        "The model becomes unable to fit the training data beyond the threshold"
      ],
      correct: 1,
      explanation: "Double descent shows that the classical U-shaped bias-variance curve is incomplete. At the interpolation threshold, the model has just enough capacity to fit the training data, and it must use extreme (large-norm) parameter values to do so — these solutions are brittle and generalize poorly, causing the error spike. Beyond this threshold, the model has many solutions that fit the training data, and implicit regularization (e.g., gradient descent's bias toward minimum-norm solutions) selects smooth interpolations that generalize well. This resolves the paradox of why overparameterized models work: more parameters enable simpler solutions."
    },
    {
      type: "mc",
      question: "In Singular Learning Theory, the multiplicity $m$ (the order of the largest pole of the zeta function) appears alongside the learning coefficient $\\lambda$ in the asymptotic expansion of the free energy. What role does $m$ play?",
      options: [
        "$m$ is the number of layers in the neural network",
        "$m$ controls the coefficient of the $\\ln \\ln n$ term in the free energy expansion: $F_n = nL_n(\\hat{\\theta}) + \\lambda \\ln n - (m-1) \\ln \\ln n + O(1)$, reflecting the degree of degeneracy of the singularity — higher $m$ means the model has more symmetries near the optimal parameters, leading to slightly better generalization at finite sample sizes",
        "$m$ is the batch size used during training",
        "$m$ counts the number of local minima in the loss landscape"
      ],
      correct: 1,
      explanation: "The free energy asymptotic expansion in SLT is $F_n = nL_n(w_0) + \\lambda \\ln n - (m-1) \\ln \\ln n + O_p(1)$, where $\\lambda$ is the RLCT and $m$ is the multiplicity. While $\\lambda$ dominates at large $n$ (controlling the leading complexity term), $m$ provides a sub-leading correction. A larger $m$ indicates more degenerate singularities — more ways the parameters can be equivalent — which provides a modest additional compression benefit. In practice, $\\lambda$ is far more important for model selection, but $m$ breaks ties between models with equal $\\lambda$."
    },
    {
      type: "mc",
      question: "A researcher observes that a transformer trained on a synthetic task exhibits a sharp phase transition at step 5,000: a specific capability (e.g., multi-step reasoning) suddenly appears. They hypothesize this is due to a circuit \"snapping\" into place. Which empirical measurement would most directly support this hypothesis?",
      options: [
        "Showing that the loss curve has a kink at step 5,000",
        "Demonstrating that specific attention head composition scores (measuring how much one head's output is read by another head's keys/queries) show a discontinuous jump at step 5,000, that ablating the composed heads after the transition destroys the capability, and that the circuit did not exist in any functional form before the transition — establishing that a specific computational structure crystallized at the transition point",
        "Showing that gradient norms are largest at step 5,000",
        "Demonstrating that the model's parameter count effectively increases at step 5,000"
      ],
      correct: 1,
      explanation: "A genuine circuit phase transition requires showing: (1) the circuit components (specific heads and their compositions) are functionally absent before the transition; (2) they appear abruptly at the transition point, measurable via composition scores $\\|W_{QK}^{(H_2)} W_{OV}^{(H_1)}\\|_F$ between heads; (3) the emerged circuit is causally necessary for the new capability (ablation destroys it). A loss curve kink alone is suggestive but insufficient — it could reflect a gradual improvement becoming visible. The composition score measurement directly tracks the formation of the computational structure."
    }
  ]
};

// ============================================================================
// F.4: Formal Theory of Transformers
// ============================================================================
export const formalTheoryAssessment = {
  id: "F.4-assess",
  sectionId: "F.4",
  title: "Assessment: Formal Theory of Transformers",
  difficulty: "easy",
  estimatedMinutes: 12,
  assessmentOnly: true,
  steps: [
    {
      type: "mc",
      question: "Transformers with hard attention (each head attends to exactly one position) and fixed precision have been analyzed in terms of circuit complexity classes. A constant-depth, polynomial-size, hard-attention transformer with $O(\\log n)$-precision arithmetic can be simulated by which circuit complexity class?",
      options: [
        "$\\text{P}$ (polynomial time Turing machines)",
        "$\\text{TC}^0$ (constant-depth, polynomial-size threshold circuits) — transformers with logarithmic precision can compute any function in $\\text{TC}^0$, and $\\text{TC}^0$ can simulate such transformers, establishing an equivalence between constant-depth transformers and this class, which includes multiplication, sorting, and iterated addition",
        "$\\text{NC}^1$ (logarithmic-depth, polynomial-size fan-in-2 circuits)",
        "$\\text{EXPTIME}$ (exponential time Turing machines)"
      ],
      correct: 1,
      explanation: "Merrill & Sabharwal (2023) and related work showed that constant-depth transformers with $O(\\log n)$-precision (necessary to address $n$ positions) can be captured by $\\text{TC}^0$. The key insight is that attention is essentially a weighted average (a sum of products), which is a threshold operation. $\\text{TC}^0$ is powerful — it contains multiplication, division, sorting, and counting — but cannot solve problems requiring inherently sequential computation like evaluating arbitrary Boolean formulas ($\\text{NC}^1$-complete) or graph connectivity. This sets a formal ceiling on what fixed-depth transformers can compute."
    },
    {
      type: "mc",
      question: "Von Oswald et al. (2023) showed that a single layer of a linear self-attention transformer, when processing in-context learning examples, implements one step of an algorithm familiar from optimization. Specifically, given in-context examples $(x_1, y_1), \\ldots, (x_k, y_k)$ and a query $x_{k+1}$, the transformer's output at the query position approximates what?",
      options: [
        "A nearest-neighbor lookup that returns the $y_i$ whose $x_i$ is closest to $x_{k+1}$",
        "One step of gradient descent on a linear regression objective: the construction sets $W_K$ and $W_Q$ so that $\\text{Attn}(X)$ computes $x_{k+1}^T (\\sum_i x_i x_i^T)^{-1} (\\sum_i x_i y_i^T)$ — equivalent to the prediction of a linear model trained by one step of gradient descent (or the closed-form OLS solution with appropriate token construction) on the in-context examples",
        "A random prediction unrelated to the in-context examples",
        "The mean of all $y_i$ values regardless of the query"
      ],
      correct: 1,
      explanation: "Von Oswald et al. demonstrated a remarkable equivalence: if tokens encode $(x_i, y_i)$ pairs as concatenated vectors, then self-attention with appropriately constructed weight matrices computes: $\\hat{y}_{k+1} = (\\sum_i v_i k_i^T) q_{k+1}$, where this outer product sum is equivalent to the gradient update of a linear model. With one layer, this is one step of GD; multi-layer transformers apply iterative GD. This provides a mechanistic explanation for in-context learning: the transformer does not retrieve examples but implicitly fits a model to them, with each layer performing one optimization step."
    },
    {
      type: "mc",
      question: "Mesa-optimization refers to the phenomenon where a trained model (the base optimizer's output) internally implements its own optimization process. In the context of transformers, which of the following would constitute evidence of mesa-optimization?",
      options: [
        "The model achieves low training loss, indicating the base optimizer succeeded",
        "The model's forward pass implements an iterative search or optimization algorithm within its layers — for example, each layer refines a solution by approximately minimizing an internal objective, and the model's in-context performance improves with depth in a manner consistent with running more steps of an internal optimizer rather than performing a fixed feedforward computation",
        "The model's weights change during inference due to online learning",
        "The model uses more FLOPs than a linear model"
      ],
      correct: 1,
      explanation: "Mesa-optimization (Hubinger et al., 2019) is when the learned model itself contains an optimization algorithm. Evidence requires showing that the forward pass performs search/optimization, not just a fixed function. For transformers, this could mean: (1) representations at successive layers approximate iterates of an optimization algorithm; (2) in-context performance scales with depth in a way consistent with more optimization steps; (3) the model exhibits goal-directed behavior that generalizes beyond training distribution in ways consistent with optimizing an internal objective. The in-context learning as implicit GD result is arguably a concrete example of mesa-optimization."
    },
    {
      type: "mc",
      question: "A transformer with unbounded depth (i.e., the number of layers can grow with input length) and bounded precision has been shown to be Turing complete. What additional mechanism is required beyond standard self-attention and feedforward layers to achieve Turing completeness?",
      options: [
        "No additional mechanism is needed — standard transformers are Turing complete with any finite depth",
        "The ability to adaptively decide when to stop (a halting mechanism) and to use the depth parameter as a function of input length — with depth $T(n)$ scaling with input size, the transformer can simulate $T(n)$ steps of a Turing machine, where each layer simulates one step by using attention to read from and write to a tape encoded in the sequence positions",
        "An external memory module like a Neural Turing Machine tape",
        "Replacing softmax attention with a lookup table"
      ],
      correct: 1,
      explanation: "Pérez et al. (2021) showed that transformers with unbounded depth are Turing complete. The construction encodes the Turing machine's tape in the sequence positions and state in the residual stream. Each layer simulates one transition: attention reads the current tape cell (matching the head position), the FFN computes the transition function (new state, write symbol, head movement), and the result is written back to the residual stream. The depth must scale with the number of simulation steps. A fixed-depth transformer cannot be Turing complete because it corresponds to $\\text{TC}^0$, which is strictly weaker than $\\text{P}$."
    },
    {
      type: "mc",
      question: "The expressiveness gap between soft attention (standard softmax attention) and hard attention (attending to exactly one position) has formal implications. Which statement correctly characterizes this gap?",
      options: [
        "Soft and hard attention are computationally equivalent for all tasks",
        "Soft attention with $O(\\log n)$ precision is strictly more powerful than hard attention with $O(\\log n)$ precision — soft attention can compute weighted averages that aggregate information from all positions, enabling computations like counting and majority that require accumulating evidence across the entire input, while hard attention can only route information from a single position per head",
        "Hard attention is more powerful because it can make discrete decisions",
        "The distinction only matters for tasks with more than 1 million tokens"
      ],
      correct: 1,
      explanation: "Hard attention selects a single position per head: $\\text{HardAttn}(Q, K, V)_i = V_{\\arg\\max_j Q_i^T K_j}$. This limits each head to routing information from one position. Soft attention computes weighted averages: $\\text{SoftAttn}(Q, K, V)_i = \\sum_j \\alpha_{ij} V_j$, allowing it to aggregate information (counts, means, sums) across all positions. With $O(\\log n)$ precision, soft-attention transformers can solve problems like MAJORITY (are there more 1s than 0s?) that hard-attention transformers with constant heads cannot, because aggregation over all positions is essential and cannot be decomposed into single-position lookups."
    },
    {
      type: "mc",
      question: "The chain-of-thought (CoT) prompting technique allows transformers to solve problems that appear to exceed their constant-depth computational power. From a computational complexity perspective, how does CoT extend a transformer's expressiveness?",
      options: [
        "CoT has no effect on expressiveness and is purely a prompt engineering trick",
        "CoT effectively increases the computational depth by allowing the model to write intermediate results to the context and read them back in subsequent generation steps — a transformer generating $T$ CoT tokens performs $T \\times L$ layers of computation (where $L$ is model depth), enabling it to simulate $T$ steps of a sequential algorithm and thus solve problems in $\\text{P}$ that fixed-depth $\\text{TC}^0$ cannot",
        "CoT makes the model use more parameters, increasing its capacity",
        "CoT works by accessing an external database of solutions"
      ],
      correct: 1,
      explanation: "A fixed-depth transformer is limited to $\\text{TC}^0$ computations. But with CoT, each generated token constitutes an external memory write that is fed back as input. This converts the constant-depth transformer into a recurrent computation: $T$ autoregressive steps give $T \\times L$ effective layers. Feng et al. (2023) formalized this: a polynomial-size transformer with polynomial CoT length can solve any problem in $\\text{P}$. This explains empirical findings where CoT enables arithmetic, multi-step reasoning, and other inherently sequential tasks that standard (non-CoT) prompting fails at — these tasks require depth beyond $\\text{TC}^0$."
    },
    {
      type: "mc",
      question: "Garg et al. (2022) trained transformers to perform in-context learning of function classes (linear functions, sparse linear functions, decision trees, etc.) by presenting $(x_1, f(x_1)), \\ldots, (x_k, f(x_k)), x_{k+1}$ and training the model to predict $f(x_{k+1})$. A key finding compared the transformer's in-context learning performance to the optimal algorithm for each function class. What was observed?",
      options: [
        "The transformer always performed worse than the optimal algorithm",
        "The transformer approximately matched the performance of the Bayes-optimal predictor for each function class — for linear functions it matched OLS, for sparse linear functions it matched Lasso, and for decision trees it matched the optimal tree learner — suggesting that transformers implicitly learn to implement near-optimal learning algorithms for the function classes in their training distribution",
        "The transformer memorized the training functions and could not generalize to new functions",
        "The transformer performed equally well on all function classes regardless of difficulty"
      ],
      correct: 1,
      explanation: "This result is remarkable because the transformer is trained with a single architecture and objective, yet its in-context predictions match specialized algorithms: OLS for linear regression, Lasso for sparse regression, etc. The transformer learns a general-purpose in-context learning algorithm that adapts to the function class. This supports the mesa-optimization view: the transformer has learned an internal algorithm flexible enough to approximate optimal estimation for each class. However, performance degrades when tested on function classes outside the training distribution, showing the learned algorithm is not fully general."
    },
    {
      type: "mc",
      question: "The problem of length generalization in transformers (the failure to generalize from training on short sequences to testing on long sequences) has been studied formally. What is the primary formal barrier to length generalization in standard transformers?",
      options: [
        "The vocabulary size is too small for longer sequences",
        "Standard positional encodings (both sinusoidal and learned) create a distribution shift: the model sees position embeddings during testing that were never encountered during training — and more fundamentally, for tasks requiring sequential computation, the fixed depth means the model cannot increase its computation to match the increased sequence length, since the computational pattern must extrapolate rather than merely extend",
        "Longer sequences require more parameters, but the model size is fixed",
        "The softmax function becomes numerically unstable with longer sequences"
      ],
      correct: 1,
      explanation: "Length generalization failure has two components: (1) Positional encoding OOD: learned positions beyond training length are untrained; sinusoidal positions have seen the frequencies but not the specific phase combinations. ALiBi, RoPE with NTK-aware scaling, and relative position encodings partially address this. (2) Computational insufficiency: for tasks like multi-digit addition or parity, the required computation grows with input length, but a fixed-depth transformer has constant compute per position. Even with perfect position handling, the model cannot allocate more \"thinking\" to longer inputs. This is why CoT helps — it provides additional computation proportional to the problem size."
    },
    {
      type: "mc",
      question: "Bai et al. (2023) proved that transformers can implement a variety of machine learning algorithms in-context, including gradient descent, ridge regression, and even more complex iterative algorithms. Their construction shows that a transformer with $L$ layers performing in-context linear regression can be understood as implementing what?",
      options: [
        "A random forest with $L$ trees",
        "$L$ steps of preconditioned gradient descent, where each layer applies one step of the form $W_{t+1} = W_t - \\eta_t P_t \\nabla \\mathcal{L}(W_t)$, with the preconditioner $P_t$ and learning rate $\\eta_t$ implicitly parameterized by the attention and MLP weights — deeper transformers implement more optimization steps, converging closer to the optimal solution",
        "A k-nearest-neighbors algorithm with $k = L$",
        "An ensemble of $L$ independent linear models whose predictions are averaged"
      ],
      correct: 1,
      explanation: "Bai et al. provided a constructive proof that each transformer layer can implement one step of preconditioned gradient descent. The attention mechanism computes the gradient of the in-context least-squares loss by attending to the context examples, and the MLP applies the update with an implicit preconditioner. This extends the von Oswald et al. result by showing: (1) multi-layer transformers implement multi-step GD, not just one step; (2) the preconditioner can implement algorithms like ridge regression ($P = (X^TX + \\lambda I)^{-1}$); (3) the construction works for more general loss functions. Deeper models provably get closer to the optimal solution."
    },
    {
      type: "mc",
      question: "Consider a transformer operating on Boolean strings of length $n$ with $O(\\log n)$-precision activations and constant depth $L$. Which of the following computational problems provably CANNOT be solved by this transformer, assuming $\\text{TC}^0 \\neq \\text{NC}^1$?",
      options: [
        "Sorting $n$ numbers (each $O(\\log n)$ bits)",
        "Evaluating an arbitrary Boolean formula (a tree of AND, OR, NOT gates) of size $n$ — this problem is $\\text{NC}^1$-complete under $\\text{AC}^0$ reductions, and since $\\text{TC}^0 \\subseteq \\text{NC}^1$ (with equality being a major open question believed false), a constant-depth transformer cannot solve it if the containment is strict",
        "Computing the majority function (are more than $n/2$ input bits set to 1?)",
        "Integer multiplication of two $n$-bit numbers"
      ],
      correct: 1,
      explanation: "Constant-depth transformers with log-precision correspond to $\\text{TC}^0$. Sorting and multiplication are in $\\text{TC}^0$ (iterated addition reduces to threshold circuits). Majority is the canonical $\\text{TC}^0$-complete problem. But Boolean formula evaluation (BFV) is $\\text{NC}^1$-complete: it requires $\\Omega(\\log n)$ depth with bounded fan-in. If $\\text{TC}^0 \\subsetneq \\text{NC}^1$ (widely believed), then no constant-depth transformer can evaluate arbitrary Boolean formulas. This provides a concrete theoretical limit on transformer capabilities without chain-of-thought or increased depth — the model provably cannot perform arbitrary depth-$\\log n$ sequential reasoning in a single forward pass."
    }
  ]
};
