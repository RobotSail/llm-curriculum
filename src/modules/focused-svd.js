// Module: Singular Value Decomposition
// Section 0.1: SVD definition, shapes, low-rank approximation, Eckart-Young
// Single-concept module following Goodfellow et al. Ch. 2.8
// Proper learning module with alternating info/mc steps

export const svdLearning = {
  id: "0.1-svd-learning-easy",
  sectionId: "0.1",
  title: "Singular Value Decomposition",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "Beyond Eigendecomposition: Why SVD?",
      content: "Eigendecomposition $A = Q \\Lambda Q^\\top$ is powerful but limited: it only works for **square** matrices, and in the general (non-symmetric) case the eigenvectors may not be orthogonal.\n\nThe **Singular Value Decomposition** (SVD) generalizes eigendecomposition to **any** matrix — square or rectangular, symmetric or not:\n\n$$A = U \\Sigma V^\\top$$\n\nwhere:\n- $U$ is an orthogonal matrix (its columns are the **left singular vectors**)\n- $V$ is an orthogonal matrix (its columns are the **right singular vectors**)\n- $\\Sigma$ is a diagonal matrix of non-negative values called **singular values** $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq 0$\n\nAs Goodfellow et al. note (Deep Learning, §2.8): \"Every real matrix has a singular value decomposition, but the same is not true of the eigenvalue decomposition.\" This universality makes SVD the most important matrix factorization in applied mathematics."
    },
    {
      type: "mc",
      question: "What is the key advantage of SVD over eigendecomposition?",
      options: ["SVD is computationally cheaper to compute than eigendecomposition for large matrices", "SVD applies to any matrix (including non-square), while eigendecomposition requires square matrices", "SVD always produces real-valued results, while eigendecomposition may produce complex eigenvalues", "SVD guarantees that all decomposition factors are invertible, unlike eigendecomposition"],
      correct: 1,
      explanation: "Eigendecomposition is only defined for square matrices, and not all square matrices have one (they need $n$ linearly independent eigenvectors). SVD exists for every real matrix of any shape — $m \\times n$ with $m \\neq n$ is fine. Both SVD and eigendecomposition of real symmetric matrices produce real results; the difference is generality, not computational cost."
    },
    {
      type: "info",
      title: "SVD Shapes and Structure",
      content: "For a matrix $A \\in \\mathbb{R}^{m \\times n}$ (say $m \\geq n$), the **full SVD** has shapes:\n\n$$\\underbrace{A}_{m \\times n} = \\underbrace{U}_{m \\times m} \\underbrace{\\Sigma}_{m \\times n} \\underbrace{V^\\top}_{n \\times n}$$\n\n- $U \\in \\mathbb{R}^{m \\times m}$: orthogonal, columns are eigenvectors of $AA^\\top$\n- $\\Sigma \\in \\mathbb{R}^{m \\times n}$: diagonal (only the top $n \\times n$ block has entries), with $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_n \\geq 0$\n- $V \\in \\mathbb{R}^{n \\times n}$: orthogonal, columns are eigenvectors of $A^\\top A$\n\nThe **thin (economy) SVD** — more common in practice — drops the extra columns of $U$:\n\n$$A = U_n \\Sigma_n V^\\top \\quad \\text{where } U_n \\in \\mathbb{R}^{m \\times n}, \\Sigma_n \\in \\mathbb{R}^{n \\times n}$$\n\nThe connection to eigendecomposition: the singular values of $A$ are the **square roots** of the eigenvalues of $A^\\top A$. So $\\sigma_i = \\sqrt{\\lambda_i(A^\\top A)}$."
    },
    {
      type: "mc",
      question: "Matrix $A \\in \\mathbb{R}^{m \\times n}$ (with $m \\geq n$) has full SVD $A = U \\Sigma V^\\top$. What are the shapes of $U$, $\\Sigma$, and $V$?",
      options: ["$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times m}$", "$U \\in \\mathbb{R}^{m \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{m \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{n \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times m},\\ V \\in \\mathbb{R}^{m \\times m}$"],
      correct: 2,
      explanation: "Full SVD: $U \\in \\mathbb{R}^{m \\times m}$ (left singular vectors), $\\Sigma \\in \\mathbb{R}^{m \\times n}$ (diagonal, with $n$ singular values on the diagonal), $V \\in \\mathbb{R}^{n \\times n}$ (right singular vectors). The product checks out: $(m \\times m)(m \\times n)(n \\times n) = m \\times n$. Both $U$ and $V$ are orthogonal. The thin SVD uses $U \\in \\mathbb{R}^{m \\times n}$ and $\\Sigma \\in \\mathbb{R}^{n \\times n}$ instead."
    },
    {
      type: "info",
      title: "Geometric Interpretation of SVD",
      content: "SVD reveals what a matrix **does** geometrically in three steps:\n\n1. **$V^\\top$**: Rotate the input into a new coordinate system (the right singular vectors)\n2. **$\\Sigma$**: Scale each axis independently by the singular values $\\sigma_i$\n3. **$U$**: Rotate the result into the output coordinate system (the left singular vectors)\n\nSo every linear transformation is a rotation, followed by axis-aligned scaling, followed by another rotation.\n\nThe singular values $\\sigma_i$ tell you how much the matrix stretches along each axis:\n- $\\sigma_1$ = maximum stretching factor (this is the spectral norm $\\|A\\|_2$)\n- $\\sigma_n$ = minimum stretching factor\n- The ratio $\\sigma_1 / \\sigma_n$ is the **condition number** — if it's large, the matrix amplifies some directions much more than others, which causes numerical instability\n\nThe **rank** of $A$ equals the number of nonzero singular values. If $A$ has rank $r < \\min(m, n)$, then $\\sigma_{r+1} = \\cdots = \\sigma_n = 0$ — the matrix maps some directions to zero."
    },
    {
      type: "mc",
      question: "A matrix $A$ has singular values $\\sigma_1 = 100, \\sigma_2 = 50, \\sigma_3 = 0.01$. What does the large ratio $\\sigma_1 / \\sigma_3 = 10000$ indicate?",
      options: ["The matrix is rank-deficient and cannot be inverted under any circumstances", "The matrix is nearly rank-2, with the third direction carrying negligible information", "The matrix is orthogonal, since the singular values span a wide range of magnitudes", "The matrix has a large condition number, meaning it is numerically ill-conditioned"],
      correct: 3,
      explanation: "The condition number $\\kappa = \\sigma_1 / \\sigma_n = 10000$ means the matrix amplifies one direction 10,000× more than another. This makes computations involving $A$ (especially $A^{-1}$) numerically unstable — small errors in the input get amplified by up to 10,000×. The matrix is not rank-deficient (all $\\sigma_i > 0$), but $\\sigma_3 = 0.01$ is so small relative to $\\sigma_1$ that it's nearly rank-2 in practice."
    },
    {
      type: "info",
      title: "Low-Rank Approximation: The Eckart-Young Theorem",
      content: "The most practically important property of SVD is the **Eckart-Young theorem**: the best rank-$k$ approximation of any matrix is its **truncated SVD**.\n\nGiven $A = \\sum_{i=1}^r \\sigma_i u_i v_i^\\top$ (the SVD written as a sum of rank-1 outer products), the truncation to the top $k$ terms:\n\n$$\\hat{A}_k = \\sum_{i=1}^k \\sigma_i u_i v_i^\\top$$\n\nis the closest rank-$k$ matrix to $A$, in both the Frobenius norm and the spectral norm:\n\n$$\\|A - \\hat{A}_k\\|_2 = \\sigma_{k+1} \\qquad \\|A - \\hat{A}_k\\|_F = \\sqrt{\\sigma_{k+1}^2 + \\cdots + \\sigma_r^2}$$\n\nNo other rank-$k$ matrix can do better. This is why SVD is the foundation of dimensionality reduction and compression: if the singular values decay quickly (e.g., $\\sigma_1 \\gg \\sigma_2 \\gg \\cdots$), then a small $k$ captures most of the matrix's information."
    },
    {
      type: "mc",
      question: "The **Eckart-Young theorem** says: among all rank-$k$ matrices, $\\hat{A}_k = \\sum_{i=1}^k \\sigma_i u_i v_i^\\top$ minimizes which quantity?",
      options: ["Both the spectral norm $\\|A - \\hat{A}_k\\|_2$ and the Frobenius norm $\\|A - \\hat{A}_k\\|_F$", "Only the Frobenius norm $\\|A - \\hat{A}_k\\|_F$, not the spectral or nuclear norms", "Only the spectral norm $\\|A - \\hat{A}_k\\|_2$, not the Frobenius or nuclear norms", "Only the nuclear norm $\\|A - \\hat{A}_k\\|_*$, not the spectral or Frobenius norms"],
      correct: 0,
      explanation: "Eckart-Young holds for both the spectral norm and the Frobenius norm simultaneously — the rank-$k$ truncated SVD is optimal under both. The residual errors are $\\|A - \\hat{A}_k\\|_2 = \\sigma_{k+1}$ and $\\|A - \\hat{A}_k\\|_F = \\sqrt{\\sum_{i>k} \\sigma_i^2}$. This dual optimality is remarkable and makes SVD the gold standard for low-rank approximation."
    },
    {
      type: "info",
      title: "SVD and Rank in Practice",
      content: "A key application of SVD is understanding the **effective rank** of a matrix — how much of its information lives in a low-dimensional subspace.\n\nConsider a $4096 \\times 4096$ weight matrix with singular values:\n- $\\sigma_1 = 850, \\sigma_2 = 832, \\ldots, \\sigma_{50} = 810$ (top 50 are large)\n- $\\sigma_{51} = 12, \\sigma_{52} = 11.5, \\ldots$ (remaining are tiny)\n\nThe sharp drop from $\\sigma_{50} \\approx 810$ to $\\sigma_{51} \\approx 12$ is a **spectral gap**. It tells you the matrix is \"effectively rank 50\" — the top 50 singular vectors capture the structured information, and the rest is noise.\n\nTruncating to rank 50 reduces storage from $4096^2 \\approx 16.8$M parameters to storing $U_{50}$ ($4096 \\times 50$) and $V_{50}^\\top$ ($50 \\times 4096$) — about $2 \\times 50 \\times 4096 = 409$K parameters, a ~41× compression.\n\nThis same principle underlies **LoRA** (Low-Rank Adaptation): instead of updating the full weight matrix during fine-tuning, you learn a low-rank update $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times d}$ with small $r$."
    },
    {
      type: "mc",
      question: "LoRA parameterizes a weight update as $\\Delta W = BA$ where $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times d}$, and $r \\ll d$. What is $\\text{rank}(\\Delta W)$?",
      options: [
        "Exactly $r$ — the product of two rank-$r$ matrices always has rank $r$",
        "At most $r$ — the rank of a product cannot exceed the rank of either factor",
        "At most $\\min(d, r)$ — the rank is bounded by the smaller of the two dimensions",
        "Exactly $d$ — the full-rank factors ensure the product spans all $d$ dimensions"
      ],
      correct: 1,
      explanation: "$\\text{rank}(BA) \\leq \\min(\\text{rank}(B), \\text{rank}(A)) \\leq r$. Equality holds when both $B$ and $A$ are full rank (rank $r$), which is the typical case after initialization. The constraint $r \\ll d$ makes $\\Delta W$ low-rank, meaning only $r$ directions in weight space are adapted — this is the key compression that makes LoRA parameter-efficient."
    },
    {
      type: "info",
      title: "SVD and the Frobenius Norm Connection",
      content: "An elegant connection: the **Frobenius norm** of a matrix equals the root sum of squares of its singular values:\n\n$$\\|A\\|_F = \\sqrt{\\sum_{i,j} A_{ij}^2} = \\sqrt{\\sum_i \\sigma_i^2}$$\n\nThis means weight decay (L2 regularization), which penalizes $\\|W\\|_F^2 = \\sum_i \\sigma_i^2$, implicitly regularizes the **singular value spectrum**.\n\nThe gradient of $\\lambda \\|W\\|_F^2$ is $2\\lambda W$, which scales each singular value proportionally: $\\sigma_i \\leftarrow (1 - 2\\lambda \\eta) \\sigma_i$ per step (where $\\eta$ is the learning rate). The absolute shrinkage of each singular value is proportional to its size: a singular value of 500 shrinks 500× faster than a singular value of 1.\n\nThis creates implicit spectral regularization: weight decay disproportionately suppresses large singular values, pushing the spectrum toward uniformity."
    },
    {
      type: "mc",
      question: "A weight matrix $W$ develops singular values $\\sigma_1 \\approx 500$ and $\\sigma_i \\approx 1$ for most other $i$. You apply weight decay (penalizing $\\|W\\|_F^2$). Which singular values does weight decay shrink **most aggressively**?",
      options: [
        "All singular values equally — each is reduced by the same absolute amount per step",
        "The smallest singular values (near 1), since there are many more of them in total",
        "The largest singular values, since they contribute most to $\\|W\\|_F^2$ and its gradient",
        "Weight decay has no effect on the singular value spectrum of the weight matrix"
      ],
      correct: 2,
      explanation: "The gradient $2\\lambda W$ contracts all singular values proportionally: $\\sigma_i \\leftarrow (1 - 2\\lambda \\eta)\\sigma_i$. The proportional factor is the same, but the absolute shrinkage is proportional to $\\sigma_i$ itself: $\\sigma_1 = 500$ shrinks by $2\\lambda \\eta \\times 500$ per step vs. $2\\lambda \\eta \\times 1$ for the small ones — 500× larger. Weight decay acts as implicit spectral regularization, suppressing outlier singular values."
    },
    {
      type: "mc",
      question: "You want to compress a trained weight matrix $W \\in \\mathbb{R}^{4096 \\times 4096}$ by keeping only its top-$r$ singular values. The spectrum shows $\\sigma_1 = 850, \\ldots, \\sigma_{50} = 810, \\sigma_{51} = 12, \\sigma_{52} = 11.5, \\ldots$ What does this suggest about an appropriate $r$?",
      options: ["$r = 4096$ — the spectrum is too dense to truncate safely without losing information", "$r = 1$ — the first singular value dominates so one component captures the matrix", "$r$ is indeterminate — the task loss must be measured for each rank to decide", "$r \\approx 50$ — a sharp spectral gap at rank 50 separates signal from noise"],
      correct: 3,
      explanation: "The spectral gap between $\\sigma_{50} \\approx 810$ and $\\sigma_{51} \\approx 12$ (a factor of ~67×) is a clear signal/noise boundary. The top 50 singular vectors capture the structured information; the rest is noise-level. Setting $r = 50$ keeps almost all the signal while compressing storage from $4096^2 \\approx 16.8$M to $2 \\times 50 \\times 4096 \\approx 409$K parameters — a ~41× reduction."
    }
  ]
};
