// Module: Linear Algebra (Beyond Intro)
// Section 0.1: Linear algebra prerequisites for LLMs
// Three difficulty tracks covering matrix calculus, eigendecomposition, SVD,
// random matrix theory, and einsum/tensor algebra.

export const easyModule = {
  id: "0.1-easy",
  sectionId: "0.1",
  title: "Core Concepts: Shapes, Eigenvalues & SVD",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 15,
  steps: [
    {
      type: "info",
      title: "What This Test Covers",
      content: "This test validates foundational linear algebra for LLM work. The five areas you need to be comfortable with:\n\n1. **Matrix calculus** — Jacobians, Hessians, chain rule through matrix expressions\n2. **Eigendecomposition & spectral theory** — eigenvalues, PSD matrices, spectral norm\n3. **SVD & low-rank approximations** — the workhorse of model compression and LoRA\n4. **Random matrix theory basics** — distinguishing signal from noise in weight spectra\n5. **Tensor algebra & einsum** — the notation that runs every modern deep learning framework\n\nStart with definitions and shapes before moving to the harder application questions."
    },
    {
      type: "mc",
      question: "A function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ has a Jacobian matrix $J$. What is the shape of $J$?",
      options: ["$m \\times n$ — rows index outputs, columns index inputs", "$n \\times n$ — the Jacobian is always a square matrix", "$n \\times m$ — rows index inputs, columns index outputs", "$m \\times m$ — the shape depends only on the output dimension"],
      correct: 0,
      explanation: "The Jacobian $J \\in \\mathbb{R}^{m \\times n}$ has entry $J_{ij} = \\partial f_i / \\partial x_j$. Each row corresponds to one output dimension, each column to one input dimension. When $m = 1$ (scalar output), this reduces to the familiar gradient $\\nabla f \\in \\mathbb{R}^{1 \\times n}$ (a row vector)."
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
      question: "Matrix $A \\in \\mathbb{R}^{m \\times n}$ (with $m \\geq n$) has SVD $A = U \\Sigma V^\\top$. What are the shapes of $U$, $\\Sigma$, and $V$?",
      options: ["$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times m}$", "$U \\in \\mathbb{R}^{m \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{m \\times m},\\ \\Sigma \\in \\mathbb{R}^{m \\times n},\\ V \\in \\mathbb{R}^{n \\times n}$", "$U \\in \\mathbb{R}^{n \\times n},\\ \\Sigma \\in \\mathbb{R}^{n \\times m},\\ V \\in \\mathbb{R}^{m \\times m}$"],
      correct: 2,
      explanation: "Full SVD: $U \\in \\mathbb{R}^{m \\times m}$ (left singular vectors), $\\Sigma \\in \\mathbb{R}^{m \\times n}$ (diagonal, with $n$ singular values on the diagonal), $V \\in \\mathbb{R}^{n \\times n}$ (right singular vectors). Both $U$ and $V$ are orthogonal matrices. The 'thin' or 'economy' SVD keeps only $n$ columns of $U$, making $\\Sigma$ square $n \\times n$ — common in practice."
    },
    {
      type: "mc",
      question: "What does `torch.einsum('ij,jk->ik', A, B)` compute?",
      options: ["Element-wise (Hadamard) product of $A$ and $B$", "Trace of the matrix product $AB$, a scalar", "Outer product of $A$ and $B$, a 4D tensor", "Standard matrix product $AB$, contracting over $j$"],
      correct: 3,
      explanation: "The index $j$ appears on both inputs but not in the output, so it is summed over: $\\sum_j A_{ij} B_{jk} = (AB)_{ik}$. This is exactly matrix multiplication. Einsum notation makes the contraction axis explicit — `ij,jk->ik` is the canonical matrix multiply pattern."
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
      question: "Which einsum string computes a **batched matrix multiply** $C_{b} = A_{b} B_{b}$ for a batch of matrices?",
      options: ["`'ij,jk->ik'`", "`'bi,bj->bij'`", "`'bij,bjk->bik'`", "`'bij,bkj->bik'`"],
      correct: 2,
      explanation: "`'bij,bjk->bik'`: index $b$ appears in all three tensors so it is kept (not summed); $j$ is summed over as the shared inner dimension. The result is $C_{bik} = \\sum_j A_{bij} B_{bjk}$ — independently multiplying the $b$-th matrix pair. This is what PyTorch's `torch.bmm` does under the hood."
    }
  ]
};

export const mediumModule = {
  id: "0.1-medium",
  sectionId: "0.1",
  title: "Matrix Calculus & Low-Rank Structure",
  difficulty: "medium",
  moduleType: "learning",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Going Deeper: Derivatives Through Matrices",
      content: "The easy module covered definitions and shapes. Here you need to **reason through** matrix calculus and SVD structure — the skills required to derive gradients for novel architectures and understand why LoRA works.\n\nKey identities to have internalized:\n\n- Chain rule: $\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial W}$ where $z = Wx$\n- For $z = Wx$, $x \\in \\mathbb{R}^n$, $W \\in \\mathbb{R}^{m \\times n}$, $z \\in \\mathbb{R}^m$: $\\frac{\\partial L}{\\partial W} = \\delta z^\\top \\otimes I$ — or more concisely, if $L$ is scalar then $\\nabla_W L = (\\nabla_z L) x^\\top$\n- Eckart-Young theorem: rank-$k$ truncated SVD is the best rank-$k$ approximation in both Frobenius and spectral norm"
    },
    {
      type: "mc",
      question: "For a linear layer $z = Wx$ with $W \\in \\mathbb{R}^{m \\times n}$, $x \\in \\mathbb{R}^n$, scalar loss $L$. The upstream gradient is $\\delta = \\nabla_z L \\in \\mathbb{R}^m$. What is $\\nabla_W L$?",
      options: ["$\\delta^\\top x$ — the inner product of the upstream gradient and input", "$W^\\top \\delta$ — the transposed weight matrix times the upstream gradient", "$x \\delta^\\top$ — the outer product of the input and upstream gradient", "$\\delta x^\\top$ — the outer product of the upstream gradient and input"],
      correct: 3,
      explanation: "$\\nabla_W L = \\delta x^\\top \\in \\mathbb{R}^{m \\times n}$. This is an outer product: the gradient with respect to each weight $W_{ij}$ is $\\delta_i x_j$, since $\\partial z_i / \\partial W_{ij} = x_j$. This outer-product structure explains why gradient updates are low-rank (rank 1 per sample, rank $\\leq$ batch size for a mini-batch) — the foundation of why LoRA works."
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
      question: "What does `torch.einsum('bhij,bhjd->bhid', attn_weights, V)` compute in a multi-head attention layer? (Indices: $b$=batch, $h$=head, $i$=query position, $j$=key position, $d$=head dim)",
      options: ["The weighted sum of values: for each query position $i$, sum $V$ weighted by attention scores over $j$", "The dot-product attention logits $QK^\\top / \\sqrt{d}$ before the softmax normalization step", "The outer product of queries and keys, producing a 4D tensor of pairwise interactions", "Layer normalization applied across the head dimension to stabilize attention outputs"],
      correct: 0,
      explanation: "Index $j$ is summed over (it appears in both inputs but not the output), giving $\\text{output}_{bhid} = \\sum_j \\text{attn}_{bhij} \\cdot V_{bhjd}$. This is exactly the attention output: for each batch $b$, head $h$, and query position $i$, compute a weighted sum of the value vectors $V_{bhjd}$ across all key/value positions $j$, with weights given by the (softmaxed) attention scores."
    }
  ]
};

export const hardModule = {
  id: "0.1-hard",
  sectionId: "0.1",
  title: "Spectral Theory, Random Matrices & Advanced Einsum",
  difficulty: "hard",
  moduleType: "learning",
  estimatedMinutes: 25,
  steps: [
    {
      type: "info",
      title: "The Hard Problems: Where Linear Algebra Meets LLM Internals",
      content: "This module tests the linear algebra that most practitioners haven't internalized — but that separates people who can *read* papers from those who can *derive* results.\n\nFocal areas:\n- **Random matrix theory**: understanding the Marchenko-Pastur distribution and what outlier singular values mean for training\n- **Spectral norm and Lipschitz control**: why spectral normalization stabilizes training\n- **Advanced einsum patterns**: the attention mechanism and beyond, from first principles\n- **Matrix calculus for novel architectures**: deriving gradients for LayerNorm, attention, and non-standard parameterizations"
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
      question: "You want to compress a trained weight matrix $W \\in \\mathbb{R}^{4096 \\times 4096}$ by keeping only its top-$r$ singular values. After inspecting the singular value spectrum, you find $\\sigma_1 = 850$, $\\sigma_2 = 832$, ..., $\\sigma_{50} = 810$, $\\sigma_{51} = 12$, $\\sigma_{52} = 11.5$, .... What does this suggest about an appropriate $r$?",
      options: ["$r = 4096$ — the spectrum is too dense to truncate safely without losing information", "$r = 1$ — the first singular value dominates so one component captures the matrix", "$r \\approx 50$ — a sharp spectral gap at rank 50 separates signal from noise", "$r$ is indeterminate — the task loss must be measured for each rank to decide"],
      correct: 2,
      explanation: "The **spectral gap** between $\\sigma_{50} \\approx 810$ and $\\sigma_{51} \\approx 12$ (a factor of ~67×) is a strong signal. The top 50 singular vectors capture structured information; the rest are noise-level. Setting $r = 50$ keeps almost all the signal while compressing storage from $4096^2 \\approx 16.8$M parameters to $2 \\times 50 \\times 4096 = 409.6$K parameters — a ~41× reduction. This spectral-gap criterion is the practical heuristic behind SVD-based model compression."
    },
    {
      type: "mc",
      question: "**Spectral normalization** divides each weight matrix by its spectral norm: $\\hat{W} = W / \\sigma_1(W)$. This ensures $\\|\\hat{W}\\|_2 = 1$. Why does this stabilize training?",
      options: ["It makes the weight matrices orthogonal, ensuring gradients neither grow nor shrink", "It ensures the Jacobian of each layer is an isometry, preserving exact gradient norms", "It makes the loss landscape convex by restricting the weight space to a convex set", "It bounds the Lipschitz constant of each linear layer to 1, preventing gradient explosion"],
      correct: 3,
      explanation: "For a linear map $x \\mapsto Wx$, the Lipschitz constant is exactly $\\|W\\|_2 = \\sigma_1(W)$. After spectral normalization, $\\|\\hat{W}x - \\hat{W}y\\|_2 \\leq \\|x - y\\|_2$ for all $x, y$ (1-Lipschitz). In a deep network, the global Lipschitz constant is at most the product of per-layer Lipschitz constants — spectral normalization keeps each factor $\\leq 1$, preventing gradient explosion through deep chains of linear layers."
    },
    {
      type: "mc",
      question: "The gradient of the loss $L$ with respect to the input $x$ of a linear layer $z = Wx$ is:",
      options: ["$\\nabla_x L = W^\\top \\nabla_z L$ — multiply the transposed weight matrix by the upstream gradient", "$\\nabla_x L = \\nabla_z L \\cdot W$ — multiply the upstream gradient row vector by the weight matrix", "$\\nabla_x L = W \\nabla_z L$ — multiply the weight matrix directly by the upstream gradient vector", "$\\nabla_x L = \\nabla_z L \\cdot W^\\top$ — multiply the upstream gradient by the transposed weight matrix"],
      correct: 0,
      explanation: "By the chain rule, $(\\nabla_x L)_j = \\sum_i (\\nabla_z L)_i \\frac{\\partial z_i}{\\partial x_j} = \\sum_i (\\nabla_z L)_i W_{ij}$. In matrix form this is $W^\\top (\\nabla_z L)$. The transpose of $W$ appears naturally because backprop reverses the direction of information flow — the Jacobian of $z = Wx$ w.r.t. $x$ is $W$, and its transpose appears in the vector-Jacobian product. This is why every backward pass through a linear layer is another matrix multiply with the transposed weight."
    },
    {
      type: "mc",
      question: "Scaled dot-product attention computes $\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$. Using einsum notation with indices $b$ (batch), $h$ (head), $i$ (query pos), $j$ (key pos), $d$ (head dim): which sequence of einsum calls correctly computes the attention output?",
      options: ["`scores = einsum('bhid,bhid->bhi', Q, K)` then `out = einsum('bhi,bhid->bhid', softmax(scores/√d), V)`", "`scores = einsum('bhid,bhjd->bhij', Q, K)` then `out = einsum('bhij,bhjd->bhid', softmax(scores/√d), V)`", "`scores = einsum('bhid,bhjd->bhid', Q, K)` then `out = einsum('bhij,bhjd->bij', softmax(scores/√d), V)`", "`scores = einsum('bhij,bhjd->bhid', Q, K)` then `out = einsum('bhid,bhjd->bhij', softmax(scores/√d), V)`"],
      correct: 1,
      explanation: "Step 1: `'bhid,bhjd->bhij'` — $d$ is summed (dot product over head dim between each query $i$ and key $j$), giving attention logits of shape $(b, h, i, j)$. Step 2: `'bhij,bhjd->bhid'` — $j$ is summed (weighted sum of values over key positions), giving output of shape $(b, h, i, d)$. The full attention pattern is: contract over $d$ to get scores, softmax, then contract over $j$ to aggregate values."
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
