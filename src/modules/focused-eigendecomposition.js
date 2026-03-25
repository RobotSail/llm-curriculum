// Focused, first-principles module for eigendecomposition and spectral theory.
// Covers eigenvalues/eigenvectors, spectral theorem, PSD matrices, spectral norm,
// matrix powers, and deep learning applications (Hessians, spectral normalization).

export const eigendecompositionLearning = {
  id: "0.1-eigendecomposition-learning-easy",
  sectionId: "0.1",
  title: "Eigendecomposition and Spectral Theory",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "Eigenvalues and Eigenvectors: The Core Idea",
      content: "Most linear transformations rotate and stretch vectors in complicated ways. But for any square matrix $A$, there are special directions where $A$ acts as **pure scaling** — no rotation, just stretching or flipping.\n\nAn **eigenvector** $v$ of $A$ is a nonzero vector satisfying:\n\n$$Av = \\lambda v$$\n\nThe scalar $\\lambda$ is the corresponding **eigenvalue**. It tells you the scaling factor: $|\\lambda| > 1$ means stretching, $|\\lambda| < 1$ means shrinking, and $\\lambda < 0$ means the direction is flipped.\n\nWhy this matters for deep learning: weight matrices in neural networks are linear maps, and their eigenvalues control whether signals grow, shrink, or stay stable as they pass through layers."
    },
    {
      type: "mc",
      question: "A matrix $A$ has eigenvector $v$ with eigenvalue $\\lambda = -0.5$. When you apply $A$ to $v$, the result $Av$ is:",
      options: [
        "A vector in the opposite direction of $v$ with half the magnitude",
        "A vector in the same direction as $v$ with half the magnitude",
        "A vector perpendicular to $v$ with half the magnitude",
        "A vector in the opposite direction of $v$ with double the magnitude"
      ],
      correct: 0,
      explanation: "$Av = \\lambda v = -0.5v$. The negative sign flips the direction (opposite to $v$), and $|\\lambda| = 0.5$ means the magnitude is halved. The eigenvector equation guarantees the output stays on the same line as $v$ — it cannot be perpendicular."
    },
    {
      type: "info",
      title: "The Spectral Theorem for Real Symmetric Matrices",
      content: "General matrices can have complex eigenvalues and non-orthogonal eigenvectors. But **real symmetric matrices** ($A = A^\\top$) are special — and they appear everywhere in deep learning (covariance matrices, Hessians, kernel matrices).\n\nThe **Spectral Theorem** guarantees that any real symmetric matrix $A \\in \\mathbb{R}^{n \\times n}$ can be decomposed as:\n\n$$A = Q \\Lambda Q^\\top$$\n\nwhere $Q$ is an orthogonal matrix ($Q^\\top Q = I$) whose columns are the eigenvectors, and $\\Lambda = \\text{diag}(\\lambda_1, \\ldots, \\lambda_n)$ holds the eigenvalues.\n\nThe two key guarantees are: **(1)** all eigenvalues are **real** (not complex), and **(2)** the eigenvectors form an **orthonormal basis** for $\\mathbb{R}^n$. This means you can always change to a coordinate system where $A$ is just a diagonal matrix of scalings."
    },
    {
      type: "mc",
      question: "A $3 \\times 3$ real symmetric matrix has eigenvalues $\\lambda_1 = 4$, $\\lambda_2 = -1$, $\\lambda_3 = 2$, with corresponding eigenvectors $q_1, q_2, q_3$. What is the value of $q_1^\\top q_2$?",
      options: [
        "It depends on the specific entries of $A$ — we need more information",
        "It equals $4 \\times (-1) = -4$ since the dot product scales with eigenvalues",
        "It equals zero because eigenvectors of a real symmetric matrix are orthogonal",
        "It equals one because the spectral theorem normalizes all inner products"
      ],
      correct: 2,
      explanation: "The Spectral Theorem guarantees that eigenvectors of a real symmetric matrix corresponding to distinct eigenvalues are orthogonal. Since $\\lambda_1 = 4 \\neq -1 = \\lambda_2$, we have $q_1^\\top q_2 = 0$. This is not something we need to compute — it is guaranteed by the structure of symmetric matrices."
    },
    {
      type: "info",
      title: "Positive Semi-Definite (PSD) Matrices",
      content: "A symmetric matrix $A$ is **positive semi-definite (PSD)** if for every vector $x$:\n\n$$x^\\top A x \\geq 0$$\n\nThrough the lens of eigendecomposition, there is an elegant equivalent: $A$ is PSD **if and only if all its eigenvalues are non-negative** ($\\lambda_i \\geq 0$ for all $i$).\n\nTo see why, write $A = Q\\Lambda Q^\\top$ and let $y = Q^\\top x$:\n\n$$x^\\top A x = x^\\top Q \\Lambda Q^\\top x = y^\\top \\Lambda y = \\sum_i \\lambda_i y_i^2$$\n\nSince $y_i^2 \\geq 0$, this sum is non-negative exactly when every $\\lambda_i \\geq 0$.\n\nPSD matrices are fundamental in deep learning: **covariance matrices** are always PSD, and the **Hessian at a local minimum** of a loss function must be PSD (otherwise you could find a direction that decreases the loss)."
    },
    {
      type: "mc",
      question: "A symmetric matrix has eigenvalues $\\{3, 0, -0.01, 5\\}$. Which statement is correct?",
      options: [
        "The matrix is PSD because the sum of eigenvalues is positive ($3 + 0 - 0.01 + 5 > 0$)",
        "The matrix is PSD because only one eigenvalue is negative and it is very small",
        "The matrix is not PSD because PSD requires all eigenvalues to be strictly positive",
        "The matrix is not PSD because it has a negative eigenvalue ($-0.01 < 0$)"
      ],
      correct: 3,
      explanation: "PSD requires $\\lambda_i \\geq 0$ for **all** eigenvalues. Even though $-0.01$ is tiny, it violates this condition. There exists some direction $x$ (the eigenvector for $\\lambda = -0.01$) where $x^\\top A x < 0$. The sum or average of eigenvalues is irrelevant — one negative eigenvalue is enough to break PSD. (Strictly positive eigenvalues would make it positive *definite*, which is a stronger condition than PSD.)"
    },
    {
      type: "info",
      title: "Spectral Norm and the Lipschitz Constant",
      content: "The **spectral norm** of a matrix $A$ is the maximum factor by which it can stretch any unit vector:\n\n$$\\|A\\|_2 = \\max_{\\|x\\| = 1} \\|Ax\\| = \\sigma_1(A)$$\n\nwhere $\\sigma_1$ is the **largest singular value** of $A$. For symmetric matrices, singular values equal the absolute values of eigenvalues, so $\\|A\\|_2 = \\max_i |\\lambda_i|$.\n\nThe spectral norm directly gives the **Lipschitz constant** of the linear map $x \\mapsto Ax$: for any two inputs $x, x'$, we have $\\|Ax - Ax'\\| \\leq \\|A\\|_2 \\|x - x'\\|$.\n\nIn neural networks, if a weight matrix $W$ has $\\|W\\|_2 > 1$, it amplifies perturbations. Stack many such layers and small input changes explode — this is the root cause of **gradient explosion** and training instability."
    },
    {
      type: "mc",
      question: "A symmetric matrix has eigenvalues $\\{-5, 3, -1, 2\\}$. Its spectral norm $\\|A\\|_2$ is:",
      options: [
        "$3$, because the spectral norm is the largest eigenvalue",
        "$5$, because the spectral norm is the largest absolute value among eigenvalues for a symmetric matrix",
        "$9$, because the spectral norm is the sum of absolute values of eigenvalues",
        "$\\sqrt{39}$, because the spectral norm is the square root of the sum of squared eigenvalues"
      ],
      correct: 1,
      explanation: "For a symmetric matrix, the singular values equal the absolute values of the eigenvalues. The spectral norm is the largest singular value, so $\\|A\\|_2 = \\max(|-5|, |3|, |-1|, |2|) = 5$. The largest eigenvalue (not absolute value) is $3$, which is the spectral radius but not the spectral norm when negative eigenvalues are present. The sum of absolute values is the nuclear norm, and the root-sum-of-squares is the Frobenius norm."
    },
    {
      type: "info",
      title: "Matrix Powers via Eigendecomposition",
      content: "Computing $A^k$ naively requires $k-1$ matrix multiplications. Eigendecomposition gives a shortcut. Starting from $A = Q\\Lambda Q^\\top$:\n\n$$A^2 = (Q\\Lambda Q^\\top)(Q\\Lambda Q^\\top) = Q\\Lambda (Q^\\top Q) \\Lambda Q^\\top = Q\\Lambda^2 Q^\\top$$\n\nThe $Q^\\top Q = I$ in the middle collapses, and by induction:\n\n$$A^k = Q \\Lambda^k Q^\\top$$\n\nSince $\\Lambda$ is diagonal, $\\Lambda^k = \\text{diag}(\\lambda_1^k, \\ldots, \\lambda_n^k)$. This reveals exactly what happens over many iterations: eigenvalues with $|\\lambda_i| > 1$ **explode** exponentially, eigenvalues with $|\\lambda_i| < 1$ **decay** to zero, and only $|\\lambda_i| = 1$ remains stable.\n\nThis is the mathematical reason that recurrent neural networks suffer from **vanishing and exploding gradients** — the effective Jacobian is raised to a power equal to the sequence length."
    },
    {
      type: "mc",
      question: "A symmetric matrix has eigenvalues $\\{0.98, 1.0, 1.03\\}$. After computing $A^{100}$, the effective eigenvalues $\\lambda_i^{100}$ are approximately:",
      options: [
        "$\\{0.98^{100} \\approx 0.13,\\; 1.0,\\; 1.03^{100} \\approx 19.2\\}$ — the largest eigenvalue dominates exponentially",
        "$\\{98, 100, 103\\}$ — raising to the $k$-th power multiplies each eigenvalue by $k$",
        "$\\{0.98, 1.0, 1.03\\}$ — eigenvalues this close to $1$ remain stable under powering",
        "$\\{0.02, 0.0, 0.97\\}$ — eigenvalues shrink toward zero under repeated application"
      ],
      correct: 0,
      explanation: "Each eigenvalue is raised to the 100th power independently. $0.98^{100} \\approx 0.13$ (decaying), $1.0^{100} = 1$ (stable), and $1.03^{100} \\approx 19.2$ (exploding). Even a 3% deviation from 1 leads to a 19x amplification over 100 steps. This sensitivity is why careful initialization and normalization are essential for deep networks and RNNs."
    },
    {
      type: "info",
      title: "Application: Hessians and Spectral Normalization",
      content: "**Hessians at local minima.** The Hessian $H$ of a loss function $\\mathcal{L}(\\theta)$ is symmetric, so it has a real eigendecomposition. At a local minimum, the Hessian must be PSD ($\\lambda_i \\geq 0$ for all $i$), because any negative eigenvalue would indicate a direction where the loss curves downward — contradicting minimality. The eigenvalues of $H$ describe the **curvature** of the loss landscape: large eigenvalues mean sharp directions (sensitive to perturbation), small eigenvalues mean flat directions.\n\n**Spectral normalization for training stability.** To prevent a layer from amplifying signals, we can normalize the weight matrix by its spectral norm:\n\n$$\\bar{W} = \\frac{W}{\\|W\\|_2}$$\n\nThis ensures $\\|\\bar{W}\\|_2 = 1$, so the layer is a **contraction** (or at most preserves norms). Composing $L$ such layers gives a network with Lipschitz constant at most $1^L = 1$, regardless of depth. Spectral normalization is widely used in **GANs** and **diffusion models** to stabilize training and prevent mode collapse."
    },
    {
      type: "mc",
      question: "During training, the Hessian at the current parameters has eigenvalues $\\{50, 0.3, -2, 0.01\\}$. To stabilize subsequent updates, a researcher applies spectral normalization to all weight matrices. Which combination of conclusions is correct?",
      options: [
        "The point is a local minimum, and spectral normalization will widen the basin by reducing the largest Hessian eigenvalue",
        "The point is a saddle point (the negative eigenvalue proves it), and spectral normalization helps by bounding each layer's Lipschitz constant to $1$",
        "The point is a saddle point, but spectral normalization cannot help because it only affects the Hessian eigenvalues directly",
        "The point is a local minimum with one flat direction, and spectral normalization will sharpen the curvature to improve convergence"
      ],
      correct: 1,
      explanation: "A negative eigenvalue ($\\lambda = -2$) means the Hessian is not PSD, so this cannot be a local minimum — it is a saddle point with a direction where the loss curves downward. Spectral normalization constrains each weight matrix to have $\\|\\bar{W}\\|_2 = 1$, bounding the Lipschitz constant of each layer and preventing gradient explosion during continued training. It does not directly modify Hessian eigenvalues, but it stabilizes the optimization dynamics."
    }
  ]
};
