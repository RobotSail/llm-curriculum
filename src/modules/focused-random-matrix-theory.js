// Module: Random Matrix Theory for Weight Analysis
// Section 0.1: Marchenko-Pastur law, spectral gaps, signal vs noise
// Single-concept module: RMT applied to understanding neural network weights

export const randomMatrixTheoryLearning = {
  id: "0.1-rmt-learning-medium",
  sectionId: "0.1",
  title: "Random Matrix Theory for Weight Analysis",
  difficulty: "medium",
  moduleType: "learning",
  estimatedMinutes: 15,
  optional: true,
  steps: [
    {
      type: "info",
      title: "Random Matrix Theory: Separating Signal from Noise",
      content: "**Random Matrix Theory** (RMT) provides the theoretical baseline for what a weight matrix's singular value spectrum should look like if it contained **no learned structure** — just random noise.\n\nThe key result is the **Marchenko-Pastur law**: for a random matrix $W \\in \\mathbb{R}^{m \\times n}$ with i.i.d. entries $W_{ij} \\sim \\mathcal{N}(0, 1/n)$, as $m, n \\to \\infty$ with $m/n \\to \\gamma$, the eigenvalues of $\\frac{1}{n}W^\\top W$ concentrate in the interval $[(1 - \\sqrt{\\gamma})^2, (1 + \\sqrt{\\gamma})^2]$.\n\nAny eigenvalue **above** this bulk is an outlier that represents learned structure. This gives a principled way to:\n- Determine how many singular values to keep when compressing\n- Identify which layers have learned the most structured representations\n- Distinguish signal from noise in weight matrices"
    },
    {
      type: "mc",
      question: "A weight matrix $W \\in \\mathbb{R}^{m \\times n}$ is initialized with i.i.d. entries $W_{ij} \\sim \\mathcal{N}(0, 1/n)$. As $m, n \\to \\infty$ with $m/n \\to \\gamma$, the empirical distribution of eigenvalues of $\\frac{1}{n}W^\\top W$ follows the **Marchenko-Pastur** law. What does an eigenvalue **far above** the bulk of this distribution indicate?",
      options: [
        "A numerical precision error in the weight initialization causing spurious large eigenvalues",
        "A direction in weight space that has learned structured signal beyond random noise",
        "That the matrix is rank-deficient and has lost important representational capacity",
        "That the learning rate was too large and caused eigenvalues to diverge during training"
      ],
      correct: 1,
      explanation: "Marchenko-Pastur gives the theoretical eigenvalue distribution for a **purely random** matrix. Eigenvalues within the bulk $[(1 - \\sqrt{\\gamma})^2, (1 + \\sqrt{\\gamma})^2]$ are consistent with random structure (noise). Eigenvalues **above** the bulk are outliers that cannot be explained by randomness alone — they correspond to directions where the matrix has learned structured signal. In trained LLMs, outlier singular values in attention weight matrices correspond to important representational directions."
    },
    {
      type: "mc",
      question: "You analyze the singular value spectrum of a trained attention weight matrix and observe 15 singular values above the Marchenko-Pastur edge, with the remaining ~4000 falling within the predicted bulk. A colleague suggests compressing this layer. What rank $r$ would RMT suggest?",
      options: [
        "$r \\approx 2000$ — keep half the singular values as a conservative estimate",
        "$r \\approx 15$ — retain exactly the outlier singular values above the MP bulk",
        "$r \\approx 4000$ — the bulk singular values still carry important distributional information",
        "$r = 1$ — only the largest singular value matters for the layer's function"
      ],
      correct: 1,
      explanation: "RMT provides a principled cutoff: singular values within the Marchenko-Pastur bulk are consistent with random noise and carry no structured information. The 15 outliers above the MP edge represent learned signal. Setting $r \\approx 15$ discards only the noise-consistent components. This is more principled than arbitrary rank choices — it uses the theoretical null model to determine where signal ends and noise begins."
    },
    {
      type: "mc",
      question: "Two layers of a transformer have been trained. Layer A has 5 singular values above the Marchenko-Pastur edge; Layer B has 200 singular values above the edge. What does this comparison tell you?",
      options: [
        "Layer A is broken and needs reinitialization since it has learned too little structure",
        "Layer B has a higher learning rate and is overfitting to the training data",
        "Layer B has learned a higher-dimensional structured representation than Layer A",
        "The layers are equivalent — the number of outliers depends only on the layer dimensions"
      ],
      correct: 2,
      explanation: "The number of outlier singular values (above the MP edge) reflects the **effective dimensionality** of the learned representation. Layer B with 200 outliers has learned structure along 200 independent directions, while Layer A's representation is effectively 5-dimensional. This doesn't mean Layer A is broken — some layers (e.g., early embedding projections) may inherently need fewer dimensions. This analysis helps decide per-layer compression ratios: Layer A can be aggressively compressed to rank ~5, while Layer B needs rank ~200."
    }
  ]
};
