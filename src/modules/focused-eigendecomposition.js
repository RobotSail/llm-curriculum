// Module: Eigendecomposition
// Section 0.1: Eigenvectors, eigenvalues, spectral theorem, matrix functions
// Single-concept: eigendecomposition as a factorization and what it enables
// Follows Goodfellow et al. Ch. 2.7

export const eigendecompositionLearning = {
  id: "0.1-eigen-learning-easy",
  sectionId: "0.1",
  title: "Eigendecomposition",
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
      options: [
        "$v + 3$, since the eigenvalue is added to each component of the vector",
        "$3v$, since $A$ scales the eigenvector by its eigenvalue without changing direction",
        "$A + 3v$, since the eigenvalue shifts the matrix applied to the vector",
        "$v / 3$, since the eigenvalue divides the eigenvector's magnitude"
      ],
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
      options: [
        "$A$ is the zero matrix, since any zero eigenvalue forces all entries to be zero",
        "$A$ has rank $n$, since the zero eigenvalue does not affect the rank",
        "$A$ is symmetric, since zero eigenvalues only occur in symmetric matrices",
        "$A$ is singular (not invertible), because it collapses some direction to zero"
      ],
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
      options: [
        "The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are real-valued",
        "The columns of $Q$ are orthonormal and the diagonal entries of $\\Lambda$ are strictly positive",
        "All eigenvalues are distinct, so the eigenvectors are guaranteed to be linearly independent",
        "The matrix $A$ must be invertible, since symmetric matrices always have nonzero eigenvalues"
      ],
      correct: 0,
      explanation: "By the Spectral Theorem, every real symmetric matrix has real eigenvalues and an orthonormal eigenbasis — so $Q$ is orthogonal ($Q^\\top Q = I$) and $\\Lambda$ is real diagonal. Eigenvalues need not be positive or distinct; positive-definiteness is an additional condition requiring all $\\lambda_i > 0$. Any real symmetric matrix — including those with zero or negative eigenvalues — has this decomposition."
    },
    {
      type: "info",
      title: "Eigenvalues and Iterative Dynamics",
      content: "Eigenvalues control the behavior of **iterative processes** — and deep learning training is fundamentally iterative.\n\nConsider repeatedly applying a matrix: $x, Ax, A^2x, A^3x, \\ldots$ If we decompose $x$ in the eigenvector basis, $x = \\sum_i c_i v_i$, then:\n\n$$A^k x = \\sum_i c_i \\lambda_i^k v_i$$\n\nThe component along each eigenvector gets multiplied by $\\lambda_i^k$. This means:\n- If $|\\lambda_i| > 1$: the component **grows** exponentially — the system is unstable in that direction\n- If $|\\lambda_i| < 1$: the component **decays** to zero\n- If $|\\lambda_i| = 1$: the component stays constant\n\nThe **spectral radius** $\\rho(A) = \\max_i |\\lambda_i|$ determines overall stability: the iterative process converges if and only if $\\rho(A) < 1$.\n\nThis is why eigenvalues matter for understanding gradient descent. When linearizing the dynamics around a fixed point, the eigenvalues of the Jacobian determine whether nearby trajectories converge or diverge."
    },
    {
      type: "mc",
      question: "A linear recurrence $x_{t+1} = Ax_t$ has $A$ with eigenvalues $\\{0.95, 0.3, -0.8\\}$. After many iterations, the state $x_t$ will be dominated by which eigenvector component?",
      options: [
        "The component along the eigenvector with $\\lambda = 0.3$, because smaller eigenvalues converge faster",
        "The component along the eigenvector with $\\lambda = -0.8$, because the negative sign causes oscillation that amplifies over time",
        "The component along the eigenvector with $\\lambda = 0.95$, because $|0.95|^k$ decays most slowly among the three",
        "All three components contribute equally after convergence, since the eigenvalues all have magnitude less than 1"
      ],
      correct: 2,
      explanation: "After $k$ iterations, each component scales by $\\lambda_i^k$. Since $|0.95| > |{-0.8}| > |0.3|$, the $\\lambda = 0.95$ component decays most slowly. After many steps, $0.95^k \\gg 0.8^k \\gg 0.3^k$, so the state is dominated by the eigenvector with the largest $|\\lambda_i|$. The negative sign of $-0.8$ causes sign flips each step but its magnitude still decays as $0.8^k$."
    },
    {
      type: "info",
      title: "Matrix Powers via Eigendecomposition",
      content: "One of the most useful consequences of eigendecomposition is efficient computation of **matrix powers**.\n\nSince $A = Q \\Lambda Q^\\top$:\n\n$$A^2 = (Q \\Lambda Q^\\top)(Q \\Lambda Q^\\top) = Q \\Lambda (Q^\\top Q) \\Lambda Q^\\top = Q \\Lambda^2 Q^\\top$$\n\nThe key: $Q^\\top Q = I$ (orthogonality) collapses the middle factors. By induction:\n\n$$A^k = Q \\Lambda^k Q^\\top$$\n\nRaising a diagonal matrix to a power is trivial — just raise each diagonal entry: $\\Lambda^k = \\text{diag}(\\lambda_1^k, \\ldots, \\lambda_n^k)$.\n\nThis makes computing high powers of a matrix cheap once you have the eigendecomposition. Without it, computing $A^{100}$ requires 99 matrix multiplications. With it, you just compute $\\lambda_i^{100}$ for each eigenvalue — a scalar operation."
    },
    {
      type: "mc",
      question: "Computing the eigendecomposition $A = Q \\Lambda Q^\\top$ gives a fast way to compute $A^k$ for large integer $k$. What is $A^k$?",
      options: [
        "$Q^k \\Lambda Q^{\\top k}$ — each factor in the decomposition is raised to the $k$-th power",
        "$k \\cdot Q \\Lambda Q^\\top$ — the matrix power equals a scalar multiple of the decomposition",
        "$Q \\Lambda^k Q^\\top$ — only the diagonal eigenvalue matrix is raised to the $k$-th power",
        "$Q \\Lambda Q^\\top + k(Q \\Lambda Q^\\top)$ — a linear combination of the original decomposition"
      ],
      correct: 2,
      explanation: "$A^k = (Q \\Lambda Q^\\top)^k = Q \\Lambda^k Q^\\top$, since $Q^\\top Q = I$ collapses all the middle factors. This makes matrix powers cheap: just raise the diagonal entries to the $k$-th power. The orthogonal matrices $Q$ and $Q^\\top$ serve as fixed basis transformations — only the eigenvalue scaling changes with $k$."
    },
    {
      type: "info",
      title: "Matrix Functions via Eigendecomposition",
      content: "The same trick extends to **any function** applied to a matrix. If $f$ is a scalar function, we define the **matrix function**:\n\n$$f(A) = Q \\, \\text{diag}(f(\\lambda_1), \\ldots, f(\\lambda_n)) \\, Q^\\top$$\n\nJust apply $f$ to each eigenvalue individually. Important examples:\n\n- **Inverse**: $A^{-1} = Q \\, \\text{diag}(1/\\lambda_1, \\ldots, 1/\\lambda_n) \\, Q^\\top$ (requires all $\\lambda_i \\neq 0$)\n- **Square root**: $A^{1/2} = Q \\, \\text{diag}(\\sqrt{\\lambda_1}, \\ldots, \\sqrt{\\lambda_n}) \\, Q^\\top$ (requires all $\\lambda_i \\geq 0$)\n- **Matrix exponential**: $\\exp(A) = Q \\, \\text{diag}(e^{\\lambda_1}, \\ldots, e^{\\lambda_n}) \\, Q^\\top$\n\nThese arise in second-order optimization. Natural gradient descent requires $F^{-1} g$ where $F$ is the Fisher information matrix. Shampoo-style optimizers compute preconditioners like $G^{-1/4}$. Without eigendecomposition, these matrix functions would be impractical to compute — with it, they reduce to scalar operations on eigenvalues."
    },
    {
      type: "mc",
      question: "A second-order optimizer needs to compute $H^{-1/2} g$ where $H$ is a symmetric PSD matrix (Hessian approximation) with eigendecomposition $H = Q \\Lambda Q^\\top$. Which computation is correct?",
      options: [
        "$Q \\Lambda^{-1/2} Q^\\top g$ — apply $\\lambda_i^{-1/2}$ to each eigenvalue, then multiply by $g$",
        "$\\frac{1}{2} Q \\Lambda^{-1} Q^\\top g$ — compute the full inverse and divide by 2",
        "$(Q \\Lambda Q^\\top)^{-1} \\cdot (Q \\Lambda Q^\\top)^{1/2} \\cdot g$ — multiply inverse by square root",
        "$Q^{-1/2} \\Lambda^{-1/2} Q^{\\top(-1/2)} g$ — raise each factor to the $-1/2$ power separately"
      ],
      correct: 0,
      explanation: "$H^{-1/2} = Q \\, \\text{diag}(\\lambda_1^{-1/2}, \\ldots, \\lambda_n^{-1/2}) \\, Q^\\top$. Apply the scalar function $f(\\lambda) = \\lambda^{-1/2}$ to each eigenvalue. Then $H^{-1/2} g = Q \\Lambda^{-1/2} Q^\\top g$. Option B confuses $H^{-1/2}$ with $\\frac{1}{2}H^{-1}$. Option D incorrectly applies the power to $Q$ — orthogonal matrices don't get raised to fractional powers in this factorization."
    },
    {
      type: "mc",
      question: "In deep learning, the loss Hessian $H$ typically has a few large eigenvalues and many near-zero eigenvalues. When computing $H^{-1}g$ for a Newton-like update, what practical problem does this eigenvalue spectrum cause?",
      options: [
        "The large eigenvalues make $H^{-1}$ numerically unstable because $1/\\lambda_i$ is very small for large $\\lambda_i$",
        "The orthogonal matrix $Q$ from the eigendecomposition becomes ill-conditioned when eigenvalues vary widely",
        "The eigendecomposition itself cannot be computed when eigenvalues span many orders of magnitude",
        "The near-zero eigenvalues produce enormous entries $1/\\lambda_i$ in $H^{-1}$, amplifying noise in those gradient directions and causing instability"
      ],
      correct: 3,
      explanation: "Near-zero eigenvalues $\\lambda_i \\approx 0$ yield $1/\\lambda_i \\to \\infty$, so $H^{-1}$ massively amplifies the gradient components along those near-flat directions. A tiny gradient signal in a flat direction gets scaled up enormously, producing a huge and unreliable update. This is why practical second-order methods add damping: $(H + \\mu I)^{-1}g$ clips $1/(\\lambda_i + \\mu)$ to at most $1/\\mu$, preventing explosion along near-zero eigenvalue directions."
    }
  ]
};
