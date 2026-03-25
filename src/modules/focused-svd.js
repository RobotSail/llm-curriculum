// Focused module: Singular Value Decomposition from first principles.
// Covers the factorization, shapes, geometry, eigendecomposition connection,
// low-rank approximation, Eckart-Young theorem, and deep learning applications.

export const svdLearning = {
  id: "0.1-svd-learning-easy",
  sectionId: "0.1",
  title: "Singular Value Decomposition",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "The SVD Factorization",
      content: "Every matrix $A \\in \\mathbb{R}^{m \\times n}$ — regardless of shape or rank — can be decomposed as:\n\n$$A = U \\Sigma V^T$$\n\nwhere $U \\in \\mathbb{R}^{m \\times m}$ and $V \\in \\mathbb{R}^{n \\times n}$ are **orthogonal matrices** (their columns are unit vectors that are mutually perpendicular), and $\\Sigma \\in \\mathbb{R}^{m \\times n}$ is a diagonal matrix whose entries $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_{\\min(m,n)} \\geq 0$ are the **singular values**.\n\nThe columns of $U$ are called **left singular vectors**, the columns of $V$ are **right singular vectors**, and the singular values on the diagonal of $\\Sigma$ tell you how much each corresponding pair of singular vectors matters. This decomposition always exists and is essentially unique (up to sign flips of matched columns)."
    },
    {
      type: "mc",
      question: "A weight matrix $W \\in \\mathbb{R}^{4096 \\times 768}$ is decomposed as $W = U \\Sigma V^T$. What are the shapes of $U$, $\\Sigma$, and $V^T$ in the full SVD?",
      options: [
        "$U$ is $4096 \\times 4096$, $\\Sigma$ is $4096 \\times 768$, $V^T$ is $768 \\times 768$",
        "$U$ is $4096 \\times 768$, $\\Sigma$ is $768 \\times 768$, $V^T$ is $768 \\times 768$",
        "$U$ is $768 \\times 768$, $\\Sigma$ is $768 \\times 4096$, $V^T$ is $4096 \\times 4096$",
        "$U$ is $4096 \\times 4096$, $\\Sigma$ is $4096 \\times 4096$, $V^T$ is $4096 \\times 768$"
      ],
      correct: 0,
      explanation: "In the full SVD of an $m \\times n$ matrix, $U$ is $m \\times m$, $\\Sigma$ is $m \\times n$, and $V^T$ is $n \\times n$. Here $m = 4096$ and $n = 768$, so $U$ is $4096 \\times 4096$, $\\Sigma$ is $4096 \\times 768$ (with at most 768 nonzero diagonal entries), and $V^T$ is $768 \\times 768$. The product dimensions check out: $(4096 \\times 4096)(4096 \\times 768)(768 \\times 768) = 4096 \\times 768$."
    },
    {
      type: "info",
      title: "Economy (Thin) SVD: What You Actually Compute",
      content: "The full SVD of an $m \\times n$ matrix with $m > n$ produces a $U$ that is $m \\times m$ — that is huge and mostly wasted. If $m = 4096$ and $n = 768$, the full $U$ has $4096^2 \\approx 16.8M$ entries, but only the first 768 columns ever multiply a nonzero singular value.\n\nThe **economy SVD** (also called thin SVD) drops the useless parts:\n\n$$A = U_r \\Sigma_r V_r^T$$\n\nwhere $r = \\min(m, n)$, $U_r \\in \\mathbb{R}^{m \\times r}$, $\\Sigma_r \\in \\mathbb{R}^{r \\times r}$, and $V_r \\in \\mathbb{R}^{n \\times r}$. This is mathematically identical — the discarded columns of $U$ lie in the **null space** of $A^T$ and contribute nothing to the product $U \\Sigma V^T$.\n\nIn practice, `numpy.linalg.svd(A, full_matrices=False)` and `torch.linalg.svd(A, full_matrices=False)` return the economy form. This is almost always what you want."
    },
    {
      type: "mc",
      question: "You call `torch.linalg.svd(W, full_matrices=False)` on a matrix $W \\in \\mathbb{R}^{4096 \\times 768}$. What is the shape of the returned $U$ matrix?",
      options: [
        "$4096 \\times 4096$, because the left singular vectors always span the full row space",
        "$768 \\times 768$, because the economy SVD reduces $U$ to match the smaller dimension squared",
        "$4096 \\times 768$, because only 768 left singular vectors correspond to nonzero singular values",
        "$768 \\times 4096$, because the economy SVD transposes $U$ to save memory"
      ],
      correct: 2,
      explanation: "The economy SVD returns $U_r \\in \\mathbb{R}^{m \\times r}$ where $r = \\min(m, n) = 768$. So $U$ is $4096 \\times 768$: it has 4096 rows (matching $W$'s rows) but only 768 columns (one per singular value). The remaining $4096 - 768 = 3328$ columns of the full $U$ would lie in the null space of $W^T$ and are discarded."
    },
    {
      type: "info",
      title: "Geometric Interpretation: Rotate, Scale, Rotate",
      content: "The SVD reveals that **every linear transformation is a rotation, followed by axis-aligned scaling, followed by another rotation**.\n\nWhen you apply $A$ to a vector $x$:\n\n$$Ax = U \\Sigma V^T x$$\n\nThis happens in three stages:\n1. $V^T x$ — **rotate** (or reflect) $x$ into a new coordinate system aligned with the right singular vectors\n2. $\\Sigma (V^T x)$ — **scale** each coordinate independently by $\\sigma_i$\n3. $U(\\Sigma V^T x)$ — **rotate** (or reflect) from the scaled coordinates into the output space via the left singular vectors\n\nA unit sphere in the input space gets mapped to an **ellipsoid** in the output space. The semi-axes of this ellipsoid point along the columns of $U$, and their lengths are the singular values $\\sigma_1, \\sigma_2, \\ldots$ This is why singular values measure how much $A$ stretches space in each direction."
    },
    {
      type: "mc",
      question: "A matrix $A$ has singular values $\\sigma_1 = 10$ and $\\sigma_2 = 0.01$. When $A$ maps the unit circle to an ellipse, what is the ratio of the longest axis to the shortest axis of that ellipse?",
      options: [
        "100, because the ratio is $\\sigma_1 / \\sigma_2^2$ which measures how anisotropic the mapping is",
        "10.01, because the ellipse axes have lengths $\\sigma_1 + \\sigma_2$ and $\\sigma_1 - \\sigma_2$",
        "1000, because the ratio equals $\\sigma_1 \\times \\sigma_2 / \\sigma_2^2$ from the condition number formula",
        "1000, because the axis lengths are the singular values, giving $\\sigma_1 / \\sigma_2 = 10 / 0.01$"
      ],
      correct: 3,
      explanation: "The semi-axes of the image ellipse have lengths equal to the singular values. The longest axis has length $\\sigma_1 = 10$ and the shortest has length $\\sigma_2 = 0.01$, so their ratio is $10 / 0.01 = 1000$. This ratio $\\sigma_1 / \\sigma_r$ is also called the **condition number** of $A$, and it measures how much the transformation distorts space — a large condition number means the matrix nearly collapses some directions."
    },
    {
      type: "info",
      title: "SVD and Eigendecomposition: The Connection",
      content: "The singular vectors and values of $A$ are directly related to the **eigendecompositions** of $A^T A$ and $A A^T$.\n\nConsider the matrix $A^T A \\in \\mathbb{R}^{n \\times n}$. Substituting $A = U \\Sigma V^T$:\n\n$$A^T A = (U \\Sigma V^T)^T (U \\Sigma V^T) = V \\Sigma^T U^T U \\Sigma V^T = V \\Sigma^T \\Sigma V^T$$\n\nSince $U$ is orthogonal, $U^T U = I$. The product $\\Sigma^T \\Sigma$ is an $n \\times n$ diagonal matrix with entries $\\sigma_i^2$. So $A^T A = V \\, \\text{diag}(\\sigma_i^2) \\, V^T$ — this is the eigendecomposition of $A^T A$.\n\nLikewise, $A A^T = U \\, \\text{diag}(\\sigma_i^2) \\, U^T$ is the eigendecomposition of $A A^T$.\n\nThe upshot: the **right singular vectors** $V$ are the eigenvectors of $A^T A$, the **left singular vectors** $U$ are the eigenvectors of $A A^T$, and the **singular values** are the square roots of the eigenvalues of either product: $\\sigma_i = \\sqrt{\\lambda_i}$."
    },
    {
      type: "mc",
      question: "A matrix $A$ has singular values $\\sigma_1 = 5$ and $\\sigma_2 = 3$. What are the eigenvalues of $A^T A$?",
      options: [
        "$5$ and $3$, because eigenvalues of $A^T A$ equal the singular values of $A$",
        "$\\sqrt{5}$ and $\\sqrt{3}$, because forming $A^T A$ takes a square root of the singular values",
        "$10$ and $6$, because $A^T A$ doubles the singular values due to the product of two copies of $A$",
        "$25$ and $9$, because eigenvalues of $A^T A$ are the squared singular values of $A$"
      ],
      correct: 3,
      explanation: "From $A^T A = V \\, \\text{diag}(\\sigma_i^2) \\, V^T$, the eigenvalues of $A^T A$ are $\\sigma_i^2$. With $\\sigma_1 = 5$ and $\\sigma_2 = 3$, the eigenvalues are $5^2 = 25$ and $3^2 = 9$. Conversely, singular values are always the square roots of the eigenvalues of the Gram matrix $A^T A$."
    },
    {
      type: "info",
      title: "Low-Rank Approximation and Eckart-Young",
      content: "Given $A = U \\Sigma V^T$ with singular values $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots$, you can build a **rank-$k$ approximation** by keeping only the top $k$ singular values:\n\n$$A_k = U_k \\Sigma_k V_k^T = \\sum_{i=1}^{k} \\sigma_i \\, u_i v_i^T$$\n\nwhere $U_k$ is the first $k$ columns of $U$, $\\Sigma_k$ is the top-left $k \\times k$ block of $\\Sigma$, and $V_k$ is the first $k$ columns of $V$.\n\nThe **Eckart-Young theorem** says this is the best you can do: among all rank-$k$ matrices, $A_k$ minimizes the Frobenius-norm error $\\|A - A_k\\|_F$. The error is exactly:\n\n$$\\|A - A_k\\|_F = \\sqrt{\\sigma_{k+1}^2 + \\sigma_{k+2}^2 + \\cdots + \\sigma_r^2}$$\n\nSo the approximation quality depends entirely on how fast the singular values decay. If they drop off steeply, a small $k$ captures most of the matrix's \"energy\" — this is the mathematical foundation of model compression via low-rank factorization."
    },
    {
      type: "mc",
      question: "A $1000 \\times 1000$ matrix has singular values that satisfy $\\sigma_i = 100 / i$. You truncate to rank $k = 10$. What fraction of the total Frobenius-norm energy $\\|A\\|_F^2$ is captured by $A_{10}$?",
      options: [
        "About 1% — keeping only 10 out of 1000 components retains very little energy regardless of the spectrum",
        "About 67% — the top 10 singular values contribute $\\sum_{i=1}^{10} (100/i)^2$ out of $\\sum_{i=1}^{1000} (100/i)^2$, and the harmonic series converges slowly",
        "About 95% — the $1/i$ decay means the top singular values dominate since $\\sum 1/i^2$ converges quickly",
        "Exactly 99% — truncating at rank 10 always preserves at least 99% of energy for any decaying spectrum"
      ],
      correct: 2,
      explanation: "The total energy is $\\|A\\|_F^2 = \\sum_{i=1}^{1000} (100/i)^2 = 10000 \\sum_{i=1}^{1000} 1/i^2$. The series $\\sum 1/i^2$ converges to $\\pi^2/6 \\approx 1.645$, and $\\sum_{i=1}^{10} 1/i^2 \\approx 1.549$. So the top-10 fraction is $1.549 / 1.645 \\approx 94.2\\%$. The fast convergence of $\\sum 1/i^2$ means the tail contributes very little — steep singular value decay makes low-rank approximation highly effective."
    },
    {
      type: "info",
      title: "The Spectral Gap: When Compression Works",
      content: "Not every matrix compresses well. The key indicator is the **spectral gap** — the rate at which singular values decay.\n\nConsider two $1000 \\times 1000$ matrices:\n- **Matrix A**: $\\sigma_1 = 100$, $\\sigma_2 = 80$, $\\sigma_3 = 60$, ... (slow, linear decay)\n- **Matrix B**: $\\sigma_1 = 100$, $\\sigma_2 = 10$, $\\sigma_3 = 1$, ... (rapid exponential decay)\n\nMatrix B compresses beautifully to rank 1 or 2. Matrix A needs most of its components — there is no natural truncation point.\n\nIn deep learning, **weight matrices often have rapidly decaying singular value spectra**, especially in overparameterized networks. This is why techniques like LoRA work: they bet that the update to a pretrained weight matrix is approximately low-rank. If the weight update $\\Delta W$ has a steep spectral gap — meaning a few singular values dominate — then $\\Delta W \\approx BA$ with small inner dimension $r$ captures most of the adaptation signal with far fewer parameters."
    },
    {
      type: "mc",
      question: "You compute the SVD of a fine-tuning weight update $\\Delta W$ and find singular values $[50, 48, 45, 42, 0.3, 0.1, \\ldots]$. A colleague proposes using LoRA with rank $r = 2$. What do you conclude?",
      options: [
        "Rank 2 is sufficient because the two largest singular values capture over 50% of the total energy",
        "Rank 2 will miss critical information because there is no clear spectral gap until after the 4th singular value",
        "Rank 2 is optimal because LoRA always works best with the smallest possible rank to prevent overfitting",
        "The rank does not matter because LoRA learns its own factorization that is independent of $\\Delta W$'s SVD structure"
      ],
      correct: 1,
      explanation: "The singular values $[50, 48, 45, 42, 0.3, 0.1, \\ldots]$ show that the first 4 components all carry substantial energy and the gap appears between $\\sigma_4 = 42$ and $\\sigma_5 = 0.3$. Rank 2 would capture only $50^2 + 48^2 = 4804$ out of approximately $50^2 + 48^2 + 45^2 + 42^2 = 8393$ in the top group — discarding nearly half the signal. You should use at least rank 4 to sit beyond the spectral gap."
    }
  ]
};
