// Module: Eigendecomposition & Spectral Theory
// Section 0.1: Eigenvalues, PSD matrices, spectral norm, matrix powers
// Single-concept module: eigendecomposition and its properties

export const eigendecompositionLearning = {
  id: "0.1-eigen-learning-easy",
  sectionId: "0.1",
  title: "Eigendecomposition and Spectral Theory",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "Eigendecomposition: The Spectral Perspective",
      content: "For a real symmetric matrix $A$, the **Spectral Theorem** guarantees a decomposition $A = Q \\Lambda Q^\\top$ where $Q$ is orthogonal and $\\Lambda$ is diagonal with real eigenvalues.\n\nThis decomposition is central to:\n- Understanding **positive definiteness** (all eigenvalues $> 0$) and **positive semi-definiteness** ($\\geq 0$)\n- Computing **matrix powers** and **matrix exponentials** efficiently\n- Analyzing the **spectral norm** — the largest singular value — which controls gradient flow\n- Understanding Hessians and curvature in optimization\n\nThe key insight: eigendecomposition lets you analyze a matrix by looking at what it does along each eigenvector independently."
    },
    {
      type: "mc",
      question: "A real symmetric matrix $A$ has eigendecomposition $A = Q \\Lambda Q^\\top$. Which statement is **always** true?",
      options: ["The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are strictly positive", "The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are real-valued", "All eigenvalues are distinct, so the eigenvectors are linearly independent by default", "$A$ must be positive definite, since only PD matrices admit a full eigendecomposition"],
      correct: 1,
      explanation: "By the **Spectral Theorem**, every real symmetric matrix has real eigenvalues and an orthonormal eigenbasis — so $Q$ is orthogonal ($Q^\\top Q = I$) and $\\Lambda$ is real diagonal. Eigenvalues need not be positive or distinct; positive-definiteness is an additional condition requiring all $\\lambda_i > 0$."
    },
    {
      type: "mc",
      question: "A matrix $A$ is **positive semi-definite (PSD)** if and only if:",
      options: ["$x^\\top A x \\geq 0$ for all $x \\in \\mathbb{R}^n$", "All entries $A_{ij}$ of $A$ are non-negative", "$A$ is symmetric and invertible ($\\det A > 0$)", "$A$ has a non-negative determinant ($\\det A \\geq 0$)"],
      correct: 0,
      explanation: "$A$ is PSD iff $x^\\top A x \\geq 0$ for all $x$ — this is the definition. Equivalently (for symmetric $A$), all eigenvalues are $\\geq 0$. Hessians of convex functions are PSD; this is why checking eigenvalues of the Hessian tells you about convexity. Non-negative entries is a much weaker (and unrelated) condition."
    },
    {
      type: "mc",
      question: "The **spectral norm** $\\|A\\|_2$ of a matrix equals:",
      options: ["The sum of all singular values (the nuclear norm)", "The largest singular value $\\sigma_1$ of the matrix", "The square root of the sum of all squared entries", "The largest absolute eigenvalue of the matrix"],
      correct: 1,
      explanation: "$\\|A\\|_2 = \\sigma_1$, the largest singular value. It equals $\\max_{\\|x\\|=1} \\|Ax\\|$, i.e., how much $A$ stretches a unit vector at most. This is the operationally important norm for stability analysis — gradient explosion occurs when products of weight matrices have spectral norms $\\gg 1$. (The sum of singular values is the nuclear norm; the Frobenius norm is the square root of sum of squares.)"
    },
    {
      type: "mc",
      question: "Computing the eigendecomposition $A = Q \\Lambda Q^\\top$ gives a fast way to compute $A^k$ for large integer $k$. What is $A^k$?",
      options: ["$Q \\Lambda Q^\\top + k(Q \\Lambda Q^\\top)$ — a linear combination of the original decomposition", "$Q^k \\Lambda Q^{\\top k}$ — each factor in the decomposition is raised to the $k$-th power", "$Q \\Lambda^k Q^\\top$ — only the diagonal eigenvalue matrix is raised to the $k$-th power", "$k \\cdot Q \\Lambda Q^\\top$ — the matrix power equals a scalar multiple of the decomposition"],
      correct: 2,
      explanation: "$A^k = (Q \\Lambda Q^\\top)^k = Q \\Lambda^k Q^\\top$, since $Q^\\top Q = I$ collapses all the middle factors. This makes matrix powers cheap: just raise the diagonal entries to the $k$-th power. The same trick gives $\\exp(A) = Q \\exp(\\Lambda) Q^\\top$ and $A^{-1} = Q \\Lambda^{-1} Q^\\top$ (when $A$ is invertible). This is the basis for efficient computation in Hessian-based optimizers."
    },
    {
      type: "mc",
      question: "The Hessian $H = \\nabla^2 L$ of a loss function is always symmetric. At a local minimum, what can you say about $H$?",
      options: ["$H$ must be the identity matrix at any local minimum of the loss", "$H$ must be positive definite, meaning all eigenvalues satisfy $\\lambda_i > 0$", "$H$ can have eigenvalues of any sign, including negative eigenvalues", "$H$ must be positive semi-definite, meaning all eigenvalues satisfy $\\lambda_i \\geq 0$"],
      correct: 3,
      explanation: "The second-order **necessary** condition at a local minimum is $H \\succeq 0$ (PSD — all eigenvalues $\\geq 0$). Positive definiteness ($H \\succ 0$) is the **sufficient** condition that *confirms* a minimum, but it is not required — local minima with zero eigenvalues (flat directions) are common and valid. In neural network training, Hessians at minima routinely have many near-zero eigenvalues. The key distinction: if any eigenvalue is **negative**, the point is a saddle, not a minimum; if all are $\\geq 0$, it can be a minimum."
    },
    {
      type: "mc",
      question: "**Spectral normalization** divides each weight matrix by its spectral norm: $\\hat{W} = W / \\sigma_1(W)$. This ensures $\\|\\hat{W}\\|_2 = 1$. Why does this stabilize training?",
      options: ["It makes the weight matrices orthogonal, ensuring gradients neither grow nor shrink", "It ensures the Jacobian of each layer is an isometry, preserving exact gradient norms", "It makes the loss landscape convex by restricting the weight space to a convex set", "It bounds the Lipschitz constant of each linear layer to 1, preventing gradient explosion"],
      correct: 3,
      explanation: "For a linear map $x \\mapsto Wx$, the Lipschitz constant is exactly $\\|W\\|_2 = \\sigma_1(W)$. After spectral normalization, $\\|\\hat{W}x - \\hat{W}y\\|_2 \\leq \\|x - y\\|_2$ for all $x, y$ (1-Lipschitz). In a deep network, the global Lipschitz constant is at most the product of per-layer Lipschitz constants — spectral normalization keeps each factor $\\leq 1$, preventing gradient explosion through deep chains of linear layers."
    }
  ]
};
