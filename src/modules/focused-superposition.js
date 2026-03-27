// Focused module: Superposition in Neural Networks
// Section F.2: Mechanistic Interpretability
// ONE concept: How neural networks represent more features than dimensions
// through nearly-orthogonal directions, and how sparse autoencoders
// decompose this superposition into interpretable features.

export const superpositionLearning = {
  id: "F.2-superposition-learning",
  sectionId: "F.2",
  title: "Superposition in Neural Networks",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Polysemanticity Problem",
      content: "When researchers examine individual neurons in trained neural networks, they find something frustrating: most neurons respond to **multiple unrelated concepts**. A single neuron in a language model might activate for academic citations, DNA sequences, and Korean text — concepts with no obvious semantic connection.\n\nThis **polysemanticity** means we cannot interpret the network by reading off what individual neurons represent. If neuron 347 fires, we don't know which of its several \"meanings\" is active.\n\nBut here is the puzzle: the model clearly *does* distinguish these concepts — it processes Korean text differently from DNA sequences. The information must be encoded somewhere. The resolution is that the model does not store one concept per neuron. Instead, concepts are represented as **directions in activation space** — linear combinations of neurons — and many more concepts exist than there are neurons.\n\nThis is **superposition**: the network represents $m$ features using only $d$ dimensions, where $m \\gg d$. Understanding superposition is the foundation of modern mechanistic interpretability."
    },
    {
      type: "mc",
      question: "A 512-dimensional MLP layer in a language model consistently distinguishes thousands of distinct semantic concepts (sentiment, syntax, entities, topics). Since 512 neurons exist but thousands of concepts are tracked, what must be true?",
      options: [
        "The model stores most concepts in its attention layers and only routes a small subset through the MLP, keeping MLP representations monosemantic",
        "Only 512 concepts are actually represented — the apparent thousands of concepts are artifacts of overlapping activation patterns that don't encode real distinctions",
        "The model silently expands its hidden dimension during inference through dynamic memory allocation to accommodate additional feature representations",
        "Concepts are encoded as directions in the 512-dimensional space rather than individual neurons, with different concepts corresponding to different linear combinations"
      ],
      correct: 3,
      explanation: "This is the core of superposition. A 512-dimensional space has only 512 orthogonal axes (one per neuron), but it has infinitely many *directions*. Concepts are stored as directions — linear combinations of neurons. Neuron 347 firing doesn't mean \"concept 347 is active\"; it means some combination of concepts whose direction has a positive component along the 347th axis is active. This is why individual neurons appear polysemantic: they participate in the directions of multiple unrelated concepts."
    },
    {
      type: "info",
      title: "Nearly-Orthogonal Vectors: The Geometry of Superposition",
      content: "How can $m \\gg d$ features coexist in $d$ dimensions without catastrophic interference? The answer lies in high-dimensional geometry.\n\nIn $\\mathbb{R}^d$, you can have at most $d$ **mutually orthogonal** vectors (the standard basis). If features must be perfectly orthogonal, you can store only $d$ of them. But features don't need to be *perfectly* orthogonal — they just need to be **nearly** orthogonal.\n\nThe Johnson-Lindenstrauss lemma guarantees that $\\mathbb{R}^d$ can contain $\\exp(\\Omega(d \\epsilon^2))$ vectors with all pairwise dot products bounded by $\\epsilon$. For $d = 512$ and tolerance $\\epsilon = 0.1$, this is an astronomically large number — far more than the thousands of features a model needs.\n\nWhen feature $i$ is represented by direction $v_i$ and activated with magnitude $x$, the interference it causes in feature $j$ is:\n\n$$\\text{interference} = x \\cdot (v_i \\cdot v_j)$$\n\nIf $v_i \\cdot v_j \\approx 0.01$ (nearly orthogonal) and $x = 10$, the interference is only $0.1$ — small enough to be filtered out by nonlinearities like ReLU. The model pays a small cost in cross-talk between features but gains an enormous increase in representational capacity."
    },
    {
      type: "mc",
      question: "A model represents 2,000 features in a 256-dimensional space using nearly-orthogonal directions. What determines whether two features $v_i$ and $v_j$ interfere destructively in practice?",
      options: [
        "Whether their dot product $v_i \\cdot v_j$ is nonzero AND both features are simultaneously active on the same input — interference only matters during co-activation",
        "Whether both features were learned during the same training epoch, since features learned at different times occupy non-overlapping subspaces",
        "Whether the features belong to the same semantic category, since semantically related features always share the same direction in activation space",
        "Whether the model's layer normalization is applied before or after the features are read out from the residual stream"
      ],
      correct: 0,
      explanation: "Interference requires two conditions: (1) nonzero dot product between feature directions (geometric overlap), and (2) simultaneous activation (both features active for the same input). This is why **sparsity** is crucial for superposition — if features are rarely active, they rarely co-occur, so the geometric overlap rarely manifests as actual interference. A model can tolerate higher dot products between features that are almost never active at the same time."
    },
    {
      type: "info",
      title: "Sparsity Enables Superposition",
      content: "Sparsity is the key ingredient that makes superposition work. If a feature is active on only 1% of inputs, it rarely co-occurs with any particular other feature, so the interference from near-orthogonality rarely matters in practice.\n\nConsider the trade-off the model faces for each feature:\n- **Dedicate a dimension** (monosemantic): zero interference, but uses up one of the $d$ precious dimensions\n- **Superpose** (pack in with other features): small interference when co-active, but saves dimensions for other features\n\nFor **dense** features (active on most inputs), the interference cost is high — these features co-occur frequently, so their cross-talk accumulates. The model dedicates dimensions to them.\n\nFor **sparse** features (active on <1% of inputs), the interference cost is low — they rarely co-occur, so cross-talk is negligible. The model superposes them.\n\nELhage et al. (2022) demonstrated this in toy models: as feature sparsity increases, the model undergoes a **phase transition** from monosemantic to superposed representation. The transition is sharp — at a critical sparsity level, the model suddenly switches from dedicating a dimension to a feature to packing it into superposition. This mirrors the trade-off: when sparsity crosses a threshold, the cost of interference drops below the benefit of freeing a dimension."
    },
    {
      type: "mc",
      question: "In a toy model with 5 dimensions and 20 features, feature A is active on 50% of inputs and feature B is active on 0.1% of inputs. How does the model likely represent each?",
      options: [
        "Both are superposed — the model always maximizes representational capacity regardless of individual feature activation frequency",
        "Feature A gets a dedicated dimension (monosemantic) because its frequent activation makes interference costly, while B is superposed with other sparse features",
        "Feature B gets a dedicated dimension because rare features are more important to represent precisely, while A is superposed as a common baseline",
        "Both get dedicated dimensions — the model has 5 dimensions and only needs 2 for these features, leaving 3 dimensions unused"
      ],
      correct: 1,
      explanation: "Feature A activates on half of all inputs, so it co-occurs with almost every other feature. If superposed, its interference with other features would cause errors on ~50% of inputs — unacceptable. The model dedicates one of its 5 dimensions to A. Feature B activates on 0.1% of inputs, so its interference with any given feature occurs on at most 0.1% of inputs — negligible. The model superimposes B along a nearly-orthogonal direction shared with other sparse features, saving the dimension for something else."
    },
    {
      type: "info",
      title: "Geometric Structures in Superposition",
      content: "A surprising finding from Elhage et al.'s toy models: features in superposition don't arrange randomly. They organize into **regular geometric structures** — polytopes — that minimize interference.\n\nWith 2 dimensions and several features of equal importance:\n- 2 features → orthogonal (standard axes)\n- 3 features → equilateral triangle inscribed in the unit circle (120° apart)\n- 5 features → regular pentagon (72° apart)\n- 6 features → hexagon\n\nWith 3 dimensions: features arrange as vertices of Platonic solids (tetrahedron, octahedron, icosahedron).\n\nThese structures solve the **Thomson problem** — placing points on a sphere to maximize minimum pairwise angle. This is exactly the optimization pressure superposition creates: pack as many nearly-orthogonal directions as possible.\n\nWhen feature importances are unequal, the symmetry breaks. More important features get directions closer to orthogonal (lower interference), while less important features are packed more tightly. The geometry reflects a priority ordering: the model allocates the most \"angular space\" to the features that matter most."
    },
    {
      type: "mc",
      question: "A toy model with 2 dimensions represents 5 equally important, equally sparse features. The model arranges them as a regular pentagon inscribed in the unit circle (72° between adjacent features). Why this specific geometry?",
      options: [
        "The pentagon maximizes the minimum angle between any pair of feature directions, minimizing worst-case interference while packing 5 features into 2 dimensions",
        "The pentagon is the only configuration where all 5 feature vectors have exactly unit norm, which is required for the ReLU activation to function correctly",
        "The optimization landscape has a single global minimum for 5 features in 2 dimensions, and gradient descent always converges to the pentagonal solution",
        "The pentagon allows each feature to be perfectly reconstructed from its two nearest neighbors, providing redundancy that protects against gradient noise"
      ],
      correct: 0,
      explanation: "The regular pentagon places 5 points on the unit circle with maximum minimum pairwise angle (72°). Any other arrangement of 5 points would have some pair closer together, creating higher worst-case interference. This is the Thomson problem: given the constraint of 2 dimensions and 5 equally important features, the pentagon is the optimal packing. For unequally important features, the geometry deforms — important features spread out (larger angles) while less important features cluster (smaller angles, higher interference tolerated)."
    },
    {
      type: "info",
      title: "Sparse Autoencoders: Decomposing Superposition",
      content: "If features are stored in superposition, how do we extract them? This is where **sparse autoencoders** (SAEs) come in.\n\nAn SAE takes a $d$-dimensional activation vector $x$ and maps it to a much larger $m$-dimensional sparse representation, then reconstructs:\n\n$$f(x) = \\text{ReLU}(W_{\\text{enc}}(x - b_{\\text{dec}}) + b_{\\text{enc}}) \\quad \\text{(encode to } m \\text{ dimensions)}$$\n$$\\hat{x} = W_{\\text{dec}} \\, f(x) + b_{\\text{dec}} \\quad \\text{(reconstruct to } d \\text{ dimensions)}$$\n\nThe training loss has two terms:\n\n$$\\mathcal{L} = \\underbrace{\\|x - \\hat{x}\\|_2^2}_{\\text{reconstruction}} + \\underbrace{\\lambda \\|f(x)\\|_1}_{\\text{sparsity}}$$\n\nThe reconstruction term ensures the SAE preserves the information in $x$. The $L_1$ sparsity penalty ensures each input activates only a few of the $m$ features — typically 5-50 out of tens of thousands.\n\nThe decoder columns $W_{\\text{dec}}[:, j]$ learn the **feature directions** in the original $d$-dimensional space. Each column is a direction that corresponds to one interpretable concept. The sparse code $f(x)$ tells us which features are active for this input and how strongly."
    },
    {
      type: "mc",
      question: "An SAE with $m = 32{,}768$ features is trained on activations from a $d = 4{,}096$ dimensional residual stream. The expansion ratio is $8\\times$. Why must $m \\gg d$?",
      options: [
        "Larger $m$ reduces the computational cost of the encoder matrix multiplication by enabling more efficient GPU parallelism and memory access patterns",
        "With $m = d$, the autoencoder would learn an identity function, which is computationally trivial and provides no useful decomposition of the activation space",
        "The model stores approximately $8\\times$ more features than dimensions through superposition, so the SAE needs at least that many dictionary elements to capture them all",
        "The expansion ratio must exactly equal the model's number of attention heads, since each SAE feature corresponds to one head's contribution to the residual stream"
      ],
      correct: 2,
      explanation: "The entire point of SAEs is to reverse superposition — extracting the $m \\gg d$ features packed into $d$ dimensions. If $m = d$, the SAE would at best learn a rotation of the original space, not a decomposition. By making $m$ much larger than $d$ (with sparsity enforcing that only a few features activate per input), the SAE provides an overcomplete dictionary where each element can correspond to a single interpretable concept. The 8x ratio is empirical — it reflects the estimated degree of superposition in the layer."
    },
    {
      type: "info",
      title: "Validating SAE Features",
      content: "How do we know SAE features are meaningful and not just mathematical artifacts? Validation uses multiple independent lines of evidence:\n\n**1. Activation pattern analysis**: For each feature, collect the inputs where it activates most strongly. If feature #7,291 activates on text about the Golden Gate Bridge, San Francisco landmarks, and Bay Area geography — and almost nothing else — that's evidence of monosemanticity.\n\n**2. Logit attribution**: Compute how each feature influences the model's next-token predictions. A monosemantic feature should have coherent logit effects — e.g., a \"French language\" feature should boost the probability of French words.\n\n**3. Causal intervention (the gold standard)**: Artificially amplify or suppress a feature during generation and observe the effect. If amplifying the Golden Gate Bridge feature makes the model discuss San Francisco, and ablating it removes that tendency, the feature is **causally active** — it genuinely participates in the model's computation, not just correlates with it.\n\nThe distinction between correlation and causation matters. A feature might activate on Golden Gate Bridge text simply because it's a statistical artifact of the SAE training — but if intervening on it changes behavior, we know it's real.\n\nBricken et al. (2023) validated SAE features on a small transformer and found the majority were interpretable. Anthropic's \"Scaling Monosemanticity\" (2024) extended this to Claude 3 Sonnet, extracting millions of interpretable features."
    },
    {
      type: "mc",
      question: "An SAE feature activates strongly on examples containing legal contract language. To validate that this feature is causally meaningful (not just a statistical artifact), which experiment is most informative?",
      options: [
        "Computing the cosine similarity between this feature's decoder direction and the embedding vectors of legal terms in the model's vocabulary",
        "Checking whether the feature also activates on non-English legal text, since a genuine legal-language feature should generalize across languages",
        "Amplifying the feature during generation on a neutral prompt and observing whether the model's output shifts toward legal language and contract terminology",
        "Training a second SAE with different random initialization and checking whether an identical feature emerges at the same dictionary index position"
      ],
      correct: 2,
      explanation: "Causal intervention is the gold standard for feature validation. If amplifying the feature on a neutral prompt (e.g., \"The company decided to\") causes the model to generate legal contract language (\"enter into an agreement pursuant to...\"), we know the feature causally participates in producing legal text — it's not just a passive correlate. Cosine similarity shows geometric relatedness but not causal influence. Cross-lingual generalization tests breadth, not causality. A second SAE would test stability of the training procedure, not whether the feature is real."
    },
    {
      type: "info",
      title: "From Superposition to Mechanistic Understanding",
      content: "Superposition and SAEs are not just theoretical curiosities — they provide the foundation for understanding how models actually compute.\n\n**The interpretability pipeline:**\n1. Extract features from superposition using SAEs\n2. Validate that features are monosemantic and causally active\n3. Trace how features interact across layers to form **circuits** — the computational pathways that implement specific behaviors\n\nFor example, understanding the induction head circuit (\"having seen A B ... A, predict B\") required identifying specific features that track token identity, previous-token context, and pattern completion. These features exist in superposition — individual neurons don't cleanly correspond to these roles. SAEs make the features visible.\n\n**Practical implications for AI safety:**\n- **Detecting deception**: If features exist for \"what I believe\" vs. \"what I'm saying,\" divergence between them signals potential deceptive behavior\n- **Steering behavior**: Amplifying or suppressing specific features can modify model behavior more precisely than fine-tuning or prompting\n- **Monitoring**: Track features related to dangerous capabilities or misalignment during deployment\n\nThe field is still early — current SAEs capture many but not all features, and the relationship between features and complex behaviors is not fully understood. But the superposition framework provides the first principled approach to opening the black box of large language models."
    },
    {
      type: "mc",
      question: "A researcher finds an SAE feature that activates when the model is about to produce a factually incorrect statement. They propose using this feature as a real-time hallucination detector. What is the main limitation of this approach?",
      options: [
        "SAE features only exist in the residual stream, not in attention layers, so they cannot detect hallucinations that originate from incorrect attention patterns",
        "Hallucination detection requires comparing model outputs against a ground-truth database, which no internal feature can accomplish since features only represent the model's beliefs",
        "SAE features are computed from static activations and cannot be evaluated during the sequential token-by-token generation process that produces hallucinations",
        "The feature may capture correlation with uncertainty or hedging language rather than factual incorrectness itself — superposition means a single direction can conflate multiple related concepts"
      ],
      correct: 3,
      explanation: "This is a direct consequence of superposition theory. The difference-in-means direction that separates \"about to hallucinate\" from \"about to state truth\" may conflate factual incorrectness with correlated attributes: uncertainty, hedging, unusual topic domains, or low-confidence generation. A single direction in activation space can mix multiple features that happen to correlate with hallucination. More precise detection would require decomposing this direction into its component features using SAEs and identifying which specific feature captures factual accuracy versus mere uncertainty."
    }
  ]
};
