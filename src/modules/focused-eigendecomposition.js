// Module: Eigendecomposition & Spectral Theory
// Section 0.1: Eigenvalues, eigenvectors, PSD, spectral norm, matrix powers
// Single-concept module following Goodfellow et al. Ch. 2.7
// Proper learning module with alternating info/mc steps

export const eigendecompositionLearning = {
  id: "0.1-eigen-learning-easy",
  sectionId: "0.1",
  title: "Eigendecomposition and Spectral Theory",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "What is an Eigenvector?",
      content: "An **eigenvector** of a square matrix $A$ is a nonzero vector $v$ such that multiplying $A$ by $v$ only changes the **scale** of $v$, not its direction:\n\n$$Av = \\lambda v$$\n\nThe scalar $\\lambda$ is called the **eigenvalue** corresponding to that eigenvector.\n\nGeometric intuition: most vectors change direction when multiplied by $A$. Eigenvectors are the special directions that $A$ merely stretches or compresses. If $\\lambda = 2$, the vector doubles in length. If $\\lambda = -1$, it flips direction. If $\\lambda = 0$, it gets crushed to the zero vector.\n\nWhy this matters: eigenvalues reveal the **intrinsic behavior** of a matrix. Rather than thinking of $A$ as a table of $n^2$ numbers, you can understand it as \"stretch by $\\lambda_1$ along direction $v_1$, stretch by $\\lambda_2$ along direction $v_2$, ...\" — a much more interpretable description."
    },
    {
      type: "mc",
      question: "If $v$ is an eigenvector of $A$ with eigenvalue $\\lambda = 3$, what is $Av$?",
      options: ["$v + 3$, since the eigenvalue is added to each component of the vector", "$3v$, since $A$ scales the eigenvector by its eigenvalue without changing direction", "$A + 3v$, since the eigenvalue shifts the matrix applied to the vector", "$v / 3$, since the eigenvalue divides the eigenvector's magnitude"],
      correct: 1,
      explanation: "By definition, $Av = \\lambda v = 3v$. The matrix $A$ acts on its eigenvector $v$ by simply scaling it by the eigenvalue. The direction is preserved — only the magnitude changes. This is what makes eigenvectors special: they are the directions along which a matrix behaves like a scalar."
    },
    {
      type: "info",
      title: "The Eigendecomposition",
      content: "If a matrix $A$ has $n$ linearly independent eigenvectors $v_1, \\ldots, v_n$ with eigenvalues $\\lambda_1, \\ldots, \\lambda_n$, we can write the **eigendecomposition**:\n\n$$A = V \\text{diag}(\\lambda) V^{-1}$$\n\nwhere $V$ is the matrix whose columns are the eigenvectors, and $\\text{diag}(\\lambda)$ is a diagonal matrix of eigenvalues.\n\nThis factorization says: to apply $A$ to any vector $x$, you can (1) express $x$ in the eigenvector basis ($V^{-1}x$), (2) scale each component by its eigenvalue ($\\text{diag}(\\lambda)$), and (3) convert back ($V$).\n\nNot every matrix has an eigendecomposition — it requires $n$ linearly independent eigenvectors. But there is an important class of matrices that always does: **real symmetric matrices**."
    },
    {
      type: "mc",
      question: "A matrix $A$ has eigendecomposition $A = V \\text{diag}(\\lambda) V^{-1}$. One eigenvalue is $\\lambda_3 = 0$. What does this tell you about $A$?",
      options: ["$A$ is the zero matrix, since any zero eigenvalue forces all entries to be zero", "$A$ has rank $n$, since the zero eigenvalue does not affect the rank", "$A$ is symmetric, since zero eigenvalues only occur in symmetric matrices", "$A$ is singular (not invertible), because it collapses some direction to zero"],
      correct: 3,
      explanation: "An eigenvalue of 0 means $Av_3 = 0 \\cdot v_3 = 0$ — the matrix sends the eigenvector $v_3$ to the zero vector. This means $A$ has a nontrivial null space, so $A$ is singular (not invertible). In general, $A$ is invertible if and only if **all** eigenvalues are nonzero. The number of zero eigenvalues tells you the dimension of the null space."
    },
    {
      type: "info",
      title: "The Spectral Theorem for Symmetric Matrices",
      content: "**Real symmetric matrices** ($A = A^\\top$) are the most important case in machine learning. The **Spectral Theorem** guarantees three powerful properties:\n\n1. All eigenvalues are **real** (no complex numbers)\n2. The eigenvectors can be chosen to be **orthonormal** (perpendicular and unit-length)\n3. The decomposition becomes: $A = Q \\Lambda Q^\\top$\n\nwhere $Q$ is an **orthogonal matrix** ($Q^\\top Q = QQ^\\top = I$, so $Q^{-1} = Q^\\top$) whose columns are the orthonormal eigenvectors, and $\\Lambda$ is a diagonal matrix of real eigenvalues.\n\nThis is cleaner than the general case because $Q^{-1} = Q^\\top$ — no matrix inversion needed. You'll encounter symmetric matrices constantly: covariance matrices $X^\\top X$, Hessians $\\nabla^2 L$, and kernel matrices are all symmetric.\n\nFollowing Goodfellow et al. (Deep Learning, §2.7): we can think of $A$ as scaling space by $\\lambda_i$ along each eigenvector direction $q_i$. The matrix decomposes into a sum of rank-1 contributions:\n\n$$A = \\sum_{i=1}^n \\lambda_i q_i q_i^\\top$$"
    },
    {
      type: "mc",
      question: "A real symmetric matrix $A$ has eigendecomposition $A = Q \\Lambda Q^\\top$. Which statement is **always** true?",
      options: ["The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are real-valued", "The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are strictly positive", "All eigenvalues are distinct, so the eigenvectors are linearly independent by default", "$A$ must be positive definite, since only PD matrices admit a full eigendecomposition"],
      correct: 0,
      explanation: "By the Spectral Theorem, every real symmetric matrix has real eigenvalues and an orthonormal eigenbasis — so $Q$ is orthogonal ($Q^\\top Q = I$) and $\\Lambda$ is real diagonal. Eigenvalues need not be positive or distinct; positive-definiteness is an additional condition requiring all $\\lambda_i > 0$. Any real symmetric matrix — including those with zero or negative eigenvalues — has this decomposition."
    },
    {
      type: "info",
      title: "Positive Definiteness: What Eigenvalues Tell You",
      content: "The sign pattern of the eigenvalues classifies a symmetric matrix into important categories:\n\n- **Positive definite (PD)**: all $\\lambda_i > 0$. The matrix curves \"upward\" in every direction. A scalar analogy: like $f(x) = x^2$ which has positive curvature everywhere.\n- **Positive semi-definite (PSD)**: all $\\lambda_i \\geq 0$. Like PD but some directions may be flat (zero curvature).\n- **Indefinite**: some $\\lambda_i > 0$ and some $\\lambda_i < 0$. The matrix curves up in some directions and down in others — this is what a **saddle point** looks like.\n\nThe formal definition of PSD: $A$ is PSD if and only if $x^\\top A x \\geq 0$ for **all** vectors $x$.\n\nWhy this matters: the **Hessian** matrix $H = \\nabla^2 L$ of a loss function is always symmetric. At a local minimum, $H$ must be PSD (all eigenvalues $\\geq 0$). If any eigenvalue is negative, you're at a saddle point, not a minimum. In deep learning, Hessians at minima typically have many near-zero eigenvalues (flat directions) and a few large positive eigenvalues (sharp directions)."
    },
    {
      type: "mc",
      question: "A matrix $A$ is **positive semi-definite (PSD)** if and only if:",
      options: ["$x^\\top A x \\geq 0$ for all $x \\in \\mathbb{R}^n$", "All entries $A_{ij}$ of $A$ are non-negative", "$A$ is symmetric and invertible ($\\det A > 0$)", "$A$ has a non-negative determinant ($\\det A \\geq 0$)"],
      correct: 0,
      explanation: "$A$ is PSD iff $x^\\top A x \\geq 0$ for all $x$ — this is the definition. Equivalently (for symmetric $A$), all eigenvalues are $\\geq 0$. Non-negative entries is a much weaker (and unrelated) condition — the matrix $\\begin{pmatrix} 1 & -2 \\\\ -2 & 5 \\end{pmatrix}$ has negative entries but is PSD (eigenvalues 0.17 and 5.83). Conversely, a matrix with all positive entries can fail to be PSD."
    },
    {
      type: "mc",
      question: "The Hessian $H = \\nabla^2 L$ of a loss function is always symmetric. At a local minimum, what can you say about $H$?",
      options: ["$H$ must be the identity matrix at any local minimum of the loss", "$H$ must be positive definite, meaning all eigenvalues satisfy $\\lambda_i > 0$", "$H$ can have eigenvalues of any sign, including negative eigenvalues", "$H$ must be positive semi-definite, meaning all eigenvalues satisfy $\\lambda_i \\geq 0$"],
      correct: 3,
      explanation: "The second-order necessary condition at a local minimum is $H \\succeq 0$ (PSD — all eigenvalues $\\geq 0$). Positive definiteness ($H \\succ 0$, all $\\lambda_i > 0$) is a sufficient condition that confirms a minimum, but it is not required — local minima with zero eigenvalues (flat directions) are common and valid. If any eigenvalue is negative, the point is a saddle, not a minimum."
    },
    {
      type: "info",
      title: "The Spectral Norm",
      content: "The **spectral norm** of a matrix is its largest singular value:\n\n$$\\|A\\|_2 = \\sigma_1(A) = \\max_{\\|x\\| = 1} \\|Ax\\|$$\n\nIt measures the **maximum stretching factor** of $A$ — how much $A$ can amplify the length of a unit vector. For a symmetric matrix, the spectral norm equals the largest absolute eigenvalue: $\\|A\\|_2 = \\max_i |\\lambda_i|$.\n\nThe spectral norm matters for training stability. When data flows through a sequence of layers with weight matrices $W_1, W_2, \\ldots, W_L$, the output scales as the product $\\|W_L\\| \\cdots \\|W_2\\| \\|W_1\\|$. If each $\\|W_i\\|_2 > 1$, this product grows exponentially with depth — **gradient explosion**. If each $\\|W_i\\|_2 < 1$, it shrinks exponentially — **gradient vanishing**.\n\nRelated norms you'll encounter:\n- **Frobenius norm**: $\\|A\\|_F = \\sqrt{\\sum_{ij} A_{ij}^2} = \\sqrt{\\sum_i \\sigma_i^2}$ — the \"total energy\" of the matrix\n- **Nuclear norm**: $\\|A\\|_* = \\sum_i \\sigma_i$ — promotes low-rank structure when used as a regularizer"
    },
    {
      type: "mc",
      question: "The **spectral norm** $\\|A\\|_2$ of a matrix equals:",
      options: ["The sum of all singular values (the nuclear norm)", "The square root of the sum of all squared entries (the Frobenius norm)", "The largest singular value $\\sigma_1$ of the matrix", "The largest absolute eigenvalue of the matrix (for any matrix)"],
      correct: 2,
      explanation: "$\\|A\\|_2 = \\sigma_1$, the largest singular value. It equals $\\max_{\\|x\\|=1} \\|Ax\\|$, i.e., how much $A$ stretches a unit vector at most. The sum of singular values is the nuclear norm; the Frobenius norm is the root sum of squares. The largest absolute eigenvalue equals the spectral norm only for symmetric matrices — for general matrices, singular values (not eigenvalues) are the right measure of \"stretching.\""
    },
    {
      type: "info",
      title: "Matrix Powers via Eigendecomposition",
      content: "One of the most useful consequences of eigendecomposition is efficient computation of **matrix powers**.\n\nSince $A = Q \\Lambda Q^\\top$:\n\n$$A^2 = (Q \\Lambda Q^\\top)(Q \\Lambda Q^\\top) = Q \\Lambda (Q^\\top Q) \\Lambda Q^\\top = Q \\Lambda^2 Q^\\top$$\n\nThe key: $Q^\\top Q = I$ (orthogonality) collapses the middle factors. By induction:\n\n$$A^k = Q \\Lambda^k Q^\\top$$\n\nRaising a diagonal matrix to a power is trivial — just raise each diagonal entry: $\\Lambda^k = \\text{diag}(\\lambda_1^k, \\ldots, \\lambda_n^k)$.\n\nThis extends to any matrix function:\n- $A^{-1} = Q \\Lambda^{-1} Q^\\top = Q \\, \\text{diag}(1/\\lambda_1, \\ldots, 1/\\lambda_n) \\, Q^\\top$\n- $\\exp(A) = Q \\, \\text{diag}(e^{\\lambda_1}, \\ldots, e^{\\lambda_n}) \\, Q^\\top$\n- $\\sqrt{A} = Q \\, \\text{diag}(\\sqrt{\\lambda_1}, \\ldots, \\sqrt{\\lambda_n}) \\, Q^\\top$ (when all $\\lambda_i \\geq 0$)\n\nThis trick is used in second-order optimizers that need to compute things like $H^{-1/2}$ (the inverse square root of the Hessian) efficiently."
    },
    {
      type: "mc",
      question: "Computing the eigendecomposition $A = Q \\Lambda Q^\\top$ gives a fast way to compute $A^k$ for large integer $k$. What is $A^k$?",
      options: ["$Q \\Lambda Q^\\top + k(Q \\Lambda Q^\\top)$ — a linear combination of the original decomposition", "$Q^k \\Lambda Q^{\\top k}$ — each factor in the decomposition is raised to the $k$-th power", "$Q \\Lambda^k Q^\\top$ — only the diagonal eigenvalue matrix is raised to the $k$-th power", "$k \\cdot Q \\Lambda Q^\\top$ — the matrix power equals a scalar multiple of the decomposition"],
      correct: 2,
      explanation: "$A^k = (Q \\Lambda Q^\\top)^k = Q \\Lambda^k Q^\\top$, since $Q^\\top Q = I$ collapses all the middle factors. This makes matrix powers cheap: just raise the diagonal entries to the $k$-th power. The same trick gives $\\exp(A) = Q \\exp(\\Lambda) Q^\\top$ and $A^{-1} = Q \\Lambda^{-1} Q^\\top$ (when $A$ is invertible)."
    },
    {
      type: "info",
      title: "Spectral Normalization: Controlling the Spectral Norm",
      content: "A direct application of spectral theory is **spectral normalization**, which divides a weight matrix by its spectral norm:\n\n$$\\hat{W} = \\frac{W}{\\sigma_1(W)}$$\n\nThis guarantees $\\|\\hat{W}\\|_2 = 1$, meaning the layer's **Lipschitz constant** is bounded by 1. The Lipschitz constant of a function $f$ measures how much it can amplify small changes in input:\n\n$$\\|f(x) - f(y)\\| \\leq L \\|x - y\\|$$\n\nFor a linear map $f(x) = Wx$, the Lipschitz constant is exactly $\\|W\\|_2 = \\sigma_1(W)$.\n\nIn a deep network with $L$ layers, the global Lipschitz constant is at most the **product** of per-layer Lipschitz constants. If each layer has Lipschitz constant $\\leq 1$ (via spectral normalization), the whole network's Lipschitz constant is $\\leq 1$ — preventing gradient explosion regardless of depth."
    },
    {
      type: "mc",
      question: "**Spectral normalization** divides each weight matrix by its spectral norm: $\\hat{W} = W / \\sigma_1(W)$, ensuring $\\|\\hat{W}\\|_2 = 1$. Why does this stabilize training?",
      options: ["It makes the weight matrices orthogonal, ensuring gradients neither grow nor shrink", "It ensures the Jacobian of each layer is an isometry, preserving exact gradient norms", "It makes the loss landscape convex by restricting the weight space to a convex set", "It bounds the Lipschitz constant of each linear layer to 1, preventing gradient explosion"],
      correct: 3,
      explanation: "For a linear map $x \\mapsto Wx$, the Lipschitz constant is exactly $\\|W\\|_2 = \\sigma_1(W)$. After spectral normalization, $\\|\\hat{W}x - \\hat{W}y\\|_2 \\leq \\|x - y\\|_2$ for all $x, y$ (1-Lipschitz). In a deep network, the global Lipschitz constant is at most the product of per-layer Lipschitz constants — spectral normalization keeps each factor $\\leq 1$, preventing gradient explosion through deep chains of linear layers."
    }
  ]
};
