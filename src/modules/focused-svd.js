// Module: Singular Value Decomposition
// Section 0.1: SVD shapes, low-rank approximation, Eckart-Young theorem
// Single-concept module: SVD and its role in model compression

export const svdLearning = {
  id: "0.1-svd-learning-easy",
  sectionId: "0.1",
  title: "Singular Value Decomposition",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "SVD: The Workhorse of Model Compression",
      content: "The **Singular Value Decomposition** (SVD) factorizes any matrix $A \\in \\mathbb{R}^{m \\times n}$ into three factors: $A = U \\Sigma V^\\top$, where $U$ and $V$ are orthogonal matrices and $\\Sigma$ is diagonal with non-negative entries called **singular values**.\n\nSVD is the foundation of:\n- **LoRA** and other low-rank adaptation methods\n- **Model compression** via truncated SVD\n- Understanding **what a weight matrix has learned** by inspecting its singular value spectrum\n\nThis module covers SVD shapes, the Eckart-Young optimality theorem, and practical applications to LLM weight matrices."
    },
    {
      type: "mc",
      question: "Matrix $A \\in \\mathbb{R}^{m \\times n}$ (with $m \\geq n$) has SVD $A = U \\Sigma V^\\top$. What are the shapes of $U$, $\\Sigma$, and $V$?",
      options: ["$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times m}$", "$U \\in \\mathbb{R}^{m \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{m \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{n \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times m},\\ V \\in \\mathbb{R}^{m \\times m}$"],
      correct: 2,
      explanation: "Full SVD: $U \\in \\mathbb{R}^{m \\times m}$ (left singular vectors), $\\Sigma \\in \\mathbb{R}^{m \\times n}$ (diagonal, with $n$ singular values on the diagonal), $V \\in \\mathbb{R}^{n \\times n}$ (right singular vectors). Both $U$ and $V$ are orthogonal matrices. The 'thin' or 'economy' SVD keeps only $n$ columns of $U$, making $\\Sigma$ square $n \\times n$ — common in practice."
    },
    {
      type: "info",
      title: "Low-Rank Approximation and Eckart-Young",
      content: "The **Eckart-Young theorem** states that among all rank-$k$ matrices, the truncated SVD $\\hat{A}_k = \\sum_{i=1}^k \\sigma_i u_i v_i^\\top$ is the **best** rank-$k$ approximation of $A$.\n\nThis holds simultaneously under both the **Frobenius norm** and the **spectral norm**:\n- $\\|A - \\hat{A}_k\\|_2 = \\sigma_{k+1}$\n- $\\|A - \\hat{A}_k\\|_F = \\sqrt{\\sum_{i>k} \\sigma_i^2}$\n\nThis is why SVD-based compression works: if the singular values decay rapidly, a low-rank approximation captures most of the matrix's information."
    },
    {
      type: "mc",
      question: "The **Eckart-Young theorem** says: among all rank-$k$ matrices, $\\hat{A}_k = \\sum_{i=1}^k \\sigma_i u_i v_i^\\top$ minimizes which quantity?",
      options: ["Both the spectral norm $\\|A - \\hat{A}_k\\|_2$ and the Frobenius norm $\\|A - \\hat{A}_k\\|_F$", "Only the Frobenius norm $\\|A - \\hat{A}_k\\|_F$, not the spectral or nuclear norms", "Only the spectral norm $\\|A - \\hat{A}_k\\|_2$, not the Frobenius or nuclear norms", "Only the nuclear norm $\\|A - \\hat{A}_k\\|_*$, not the spectral or Frobenius norms"],
      correct: 0,
      explanation: "Eckart-Young holds for **both** the spectral norm and the Frobenius norm — the rank-$k$ truncated SVD is simultaneously optimal under both. The residual errors are $\\|A - \\hat{A}_k\\|_2 = \\sigma_{k+1}$ and $\\|A - \\hat{A}_k\\|_F = \\sqrt{\\sum_{i>k} \\sigma_i^2}$. This robustness to choice of norm makes SVD the gold standard for low-rank approximation."
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
      explanation: "$\\text{rank}(BA) \\leq \\min(\\text{rank}(B), \\text{rank}(A)) \\leq r$. Equality holds when both $B$ and $A$ are full rank (rank $r$), which is the typical case at initialization. The constraint $r \\ll d$ makes $\\Delta W$ low-rank, meaning only $r$ directions in weight space are adapted — the key compression in LoRA."
    },
    {
      type: "mc",
      question: "You want to compress a trained weight matrix $W \\in \\mathbb{R}^{4096 \\times 4096}$ by keeping only its top-$r$ singular values. After inspecting the singular value spectrum, you find $\\sigma_1 = 850$, $\\sigma_2 = 832$, ..., $\\sigma_{50} = 810$, $\\sigma_{51} = 12$, $\\sigma_{52} = 11.5$, .... What does this suggest about an appropriate $r$?",
      options: ["$r = 4096$ — the spectrum is too dense to truncate safely without losing information", "$r = 1$ — the first singular value dominates so one component captures the matrix", "$r \\approx 50$ — a sharp spectral gap at rank 50 separates signal from noise", "$r$ is indeterminate — the task loss must be measured for each rank to decide"],
      correct: 2,
      explanation: "The **spectral gap** between $\\sigma_{50} \\approx 810$ and $\\sigma_{51} \\approx 12$ (a factor of ~67×) is a strong signal. The top 50 singular vectors capture structured information; the rest are noise-level. Setting $r = 50$ keeps almost all the signal while compressing storage from $4096^2 \\approx 16.8$M parameters to $2 \\times 50 \\times 4096 = 409.6$K parameters — a ~41× reduction. This spectral-gap criterion is the practical heuristic behind SVD-based model compression."
    },
    {
      type: "mc",
      question: "During training of a large transformer, you observe that a specific layer's weight matrix $W$ develops singular values with $\\sigma_1 \\approx 500$ while most others are $\\sigma_i \\approx 1$. You apply **weight decay** (L2 regularization, which penalizes $\\|W\\|_F^2$). Which singular values does weight decay shrink **most aggressively**?",
      options: [
        "All singular values equally — each is reduced by the same absolute amount per step",
        "The smallest singular values (near 1), since there are many more of them in total",
        "The largest singular values, since they contribute most to $\\|W\\|_F^2$ and its gradient",
        "Weight decay has no effect on the singular value spectrum of the weight matrix"
      ],
      correct: 2,
      explanation: "$\\|W\\|_F^2 = \\sum_i \\sigma_i^2$ (Frobenius norm equals sum of squared singular values). The L2 penalty gradient $\\frac{\\partial}{\\partial W}\\lambda\\|W\\|_F^2 = 2\\lambda W$ contracts all singular values proportionally: $\\sigma_i \\leftarrow (1 - 2\\lambda)\\sigma_i$. But $\\sigma_1 = 500$ shrinks by $2\\lambda \\times 500$ per step vs. $2\\lambda \\times 1$ for the small ones — the **absolute** shrinkage is 500× larger for the dominant singular value. This is the connection between weight decay and implicit regularization of large spectral components."
    }
  ]
};
