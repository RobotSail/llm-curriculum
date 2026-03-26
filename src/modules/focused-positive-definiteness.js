// Module: Positive Definiteness
// Section 0.1: PSD/PD classification, quadratic forms, Hessians, curvature
// Single-concept: what positive definiteness means and why it matters for optimization
// Follows Goodfellow et al. Ch. 4.3 (second-order optimization)

export const positiveDefinitenessLearning = {
  id: "0.1-psd-learning-easy",
  sectionId: "0.1",
  title: "Positive Definiteness",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Classifying Matrices by Eigenvalue Sign",
      content: "For a real symmetric matrix $A$, the signs of the eigenvalues tell you something fundamental about the matrix's geometric behavior:\n\n- **Positive definite (PD)**: all $\\lambda_i > 0$. The matrix stretches every direction outward — it never collapses or reverses anything.\n- **Positive semi-definite (PSD)**: all $\\lambda_i \\geq 0$. Like PD, but some directions may be left unchanged (zero eigenvalue = flat direction).\n- **Negative definite**: all $\\lambda_i < 0$. Every direction gets reversed.\n- **Indefinite**: some $\\lambda_i > 0$ and some $\\lambda_i < 0$. The matrix stretches some directions and reverses others.\n\nA scalar analogy: think of $\\lambda > 0$ like a positive spring constant (restoring force), $\\lambda < 0$ like a negative spring constant (destabilizing), and $\\lambda = 0$ like a free axis with no restoring force."
    },
    {
      type: "mc",
      question: "A $4 \\times 4$ symmetric matrix has eigenvalues $\\{5.2, 0.8, 0, -0.3\\}$. How is this matrix classified?",
      options: [
        "Positive semi-definite, because most eigenvalues are non-negative",
        "Positive definite, because the largest eigenvalue is positive",
        "Indefinite, because it has both positive and negative eigenvalues",
        "Negative semi-definite, because it has a zero and a negative eigenvalue"
      ],
      correct: 2,
      explanation: "The classification depends on ALL eigenvalues, not just the majority. Having even one negative eigenvalue ($-0.3$) alongside positive ones makes the matrix indefinite. PSD requires ALL eigenvalues $\\geq 0$, and PD requires ALL eigenvalues $> 0$. The presence of both positive and negative eigenvalues means the matrix stretches some directions and reverses others."
    },
    {
      type: "info",
      title: "The Quadratic Form: What PD Means Geometrically",
      content: "The formal definition of positive definiteness uses the **quadratic form**:\n\n$$A \\text{ is PD} \\iff x^\\top A x > 0 \\text{ for all nonzero } x \\in \\mathbb{R}^n$$\n$$A \\text{ is PSD} \\iff x^\\top A x \\geq 0 \\text{ for all } x \\in \\mathbb{R}^n$$\n\nThe quantity $x^\\top Ax$ is a scalar that depends on both the matrix and the direction $x$. To see why the eigenvalue condition is equivalent, expand $x$ in the eigenvector basis $x = \\sum_i c_i q_i$:\n\n$$x^\\top A x = \\sum_i \\lambda_i c_i^2$$\n\nSince $c_i^2 \\geq 0$, this sum is non-negative for all $x$ if and only if all $\\lambda_i \\geq 0$ (PSD), and strictly positive for all nonzero $x$ if and only if all $\\lambda_i > 0$ (PD).\n\nGeometric picture: $f(x) = x^\\top Ax$ defines a surface over $\\mathbb{R}^n$. If $A$ is PD, this surface is a **bowl** — it curves upward in every direction from the origin. If indefinite, it's a **saddle** — curving up in some directions and down in others."
    },
    {
      type: "mc",
      question: "The quadratic form $f(x) = x^\\top A x$ where $A$ is a $3 \\times 3$ PSD matrix with eigenvalues $\\{4, 1, 0\\}$. What is $\\min_{x: \\|x\\|=1} f(x)$?",
      options: [
        "$4$, the maximum eigenvalue, because the quadratic form on the unit sphere is maximized along the dominant eigenvector",
        "$5/3$, the average of the eigenvalues, because the unit sphere samples all directions uniformly",
        "$1$, the smallest nonzero eigenvalue, because the zero eigenvalue is excluded by the unit norm constraint",
        "$0$, achieved by setting $x$ equal to the eigenvector with eigenvalue $0$"
      ],
      correct: 3,
      explanation: "On the unit sphere, $f(x) = \\sum_i \\lambda_i c_i^2$ where $\\sum_i c_i^2 = 1$. The minimum is achieved by putting all weight on the smallest eigenvalue: $c_3 = 1$, all others zero, giving $f = \\lambda_3 = 0$. The unit norm constraint doesn't exclude the zero-eigenvalue direction — it just requires the vector has length 1. The eigenvector $q_3$ has unit length, and $f(q_3) = 0^\\top A q_3 = 0 \\cdot 1 = 0$."
    },
    {
      type: "info",
      title: "Testing for Positive Definiteness",
      content: "How do you check if a matrix is PD or PSD in practice? Several equivalent criteria exist:\n\n1. **Eigenvalue test**: Compute all eigenvalues and check their signs. Definitive but costs $O(n^3)$.\n\n2. **Cholesky decomposition**: $A$ is PD if and only if it has a **Cholesky factorization** $A = LL^\\top$ where $L$ is lower triangular with positive diagonal entries. If the algorithm succeeds, $A$ is PD. If it encounters a non-positive diagonal, $A$ is not PD. This is the standard numerical test.\n\n3. **Leading principal minors** (Sylvester's criterion): $A$ is PD if and only if every leading principal minor is positive: $\\det(A_{1:k, 1:k}) > 0$ for $k = 1, \\ldots, n$. Useful for theoretical proofs but numerically inferior to Cholesky.\n\n4. **Construction**: Matrices of the form $B^\\top B$ are always PSD (and PD when $B$ has full column rank). This is because $x^\\top B^\\top B x = \\|Bx\\|^2 \\geq 0$.\n\nIn deep learning, many matrices are PSD **by construction**: covariance matrices $X^\\top X$, the Fisher information matrix $\\mathbb{E}[g g^\\top]$, and Gram matrices $K_{ij} = k(x_i, x_j)$ with valid kernels."
    },
    {
      type: "mc",
      question: "A researcher computes $G = X^\\top X$ where $X$ is a $1000 \\times 50$ data matrix (1000 samples, 50 features) with full column rank. Without computing eigenvalues, what can you conclude about $G$?",
      options: [
        "$G$ is indefinite because real data matrices typically produce both positive and negative eigenvalues",
        "$G$ is positive definite, because $B^\\top B$ is always PSD and full column rank of $X$ ensures all eigenvalues are strictly positive",
        "$G$ is PSD but not PD, because the data matrix has more rows than columns",
        "$G$ is orthogonal, because $X^\\top X$ produces an orthogonal matrix when $X$ has full column rank"
      ],
      correct: 1,
      explanation: "$G = X^\\top X$ is always PSD since $x^\\top G x = \\|Xx\\|^2 \\geq 0$. With $X$ having full column rank (rank 50), $Xx = 0$ only when $x = 0$, so $x^\\top G x > 0$ for all nonzero $x$ — making $G$ positive definite. The $50 \\times 50$ matrix $G$ has 50 strictly positive eigenvalues. If $X$ had rank less than 50, some eigenvalues would be zero, giving PSD but not PD."
    },
    {
      type: "info",
      title: "Hessians and Critical Points in Optimization",
      content: "The **Hessian** matrix $H = \\nabla^2 L$ of a loss function $L$ contains all second-order partial derivatives:\n\n$$H_{ij} = \\frac{\\partial^2 L}{\\partial \\theta_i \\partial \\theta_j}$$\n\nSince mixed partials commute ($\\partial^2 L / \\partial \\theta_i \\partial \\theta_j = \\partial^2 L / \\partial \\theta_j \\partial \\theta_i$), the Hessian is always **symmetric**. This means eigendecomposition applies.\n\nAt a critical point ($\\nabla L = 0$), the Hessian's eigenvalues classify the geometry:\n\n- **All $\\lambda_i > 0$ (PD)**: local minimum — the loss curves upward in every direction\n- **All $\\lambda_i < 0$ (ND)**: local maximum — the loss curves downward in every direction\n- **Mixed signs (indefinite)**: saddle point — upward in some directions, downward in others\n- **Some $\\lambda_i = 0$ (PSD)**: degenerate case — some directions are flat, need higher-order analysis\n\nFollowing Goodfellow et al. (§4.3): the second-order Taylor expansion around a critical point $\\theta^*$ is:\n\n$$L(\\theta^* + \\epsilon) \\approx L(\\theta^*) + \\frac{1}{2} \\epsilon^\\top H \\epsilon$$\n\nThe term $\\epsilon^\\top H \\epsilon$ tells you whether moving in direction $\\epsilon$ increases or decreases the loss."
    },
    {
      type: "mc",
      question: "A neural network's loss Hessian at a critical point has eigenvalues $\\{12.5, 3.1, 0.01, -0.2\\}$. What type of critical point is this?",
      options: [
        "A saddle point, because there exists at least one negative eigenvalue, meaning the loss decreases in that direction",
        "A local minimum, because the sum of eigenvalues is positive ($12.5 + 3.1 + 0.01 - 0.2 > 0$)",
        "A local minimum, because only one eigenvalue is negative and the dominant curvature is positive",
        "Indeterminate — the Hessian alone cannot classify critical points without knowing the gradient"
      ],
      correct: 0,
      explanation: "The eigenvalue $-0.2$ means there exists a direction along which $\\epsilon^\\top H \\epsilon < 0$, so the loss DECREASES when moving in that direction from the critical point. This is a saddle point, not a minimum. It doesn't matter that most eigenvalues are positive or that their sum is positive — a single negative eigenvalue is enough to rule out a local minimum. At a local minimum, ALL eigenvalues must be $\\geq 0$."
    },
    {
      type: "info",
      title: "The Hessian Spectrum in Deep Learning",
      content: "For neural networks with millions or billions of parameters, computing the full Hessian is infeasible. But research using Hessian-vector products and Lanczos iteration has revealed a characteristic pattern in the **Hessian eigenvalue spectrum** at trained minima:\n\n- A **bulk** of eigenvalues clustered near zero — these correspond to the many flat directions in the loss landscape\n- A small number of **outlier** eigenvalues that are much larger — these correspond to sharp, well-defined curvature directions\n- Very few (if any) negative eigenvalues near convergence — training approximately finds PSD critical points\n\nThe **ratio** of the largest to smallest positive eigenvalue is the **condition number** $\\kappa = \\lambda_{\\max} / \\lambda_{\\min}$. When $\\kappa$ is large, the loss surface is shaped like a narrow valley: very steep along some directions (large $\\lambda$) and nearly flat along others (small $\\lambda$).\n\nThis spectrum explains why first-order optimizers struggle: gradient descent uses the same learning rate for all directions, but the optimal step size differs by a factor of $\\kappa$ between the steepest and flattest directions. Second-order methods like natural gradient or Shampoo attempt to correct for this by rescaling gradients using curvature information."
    },
    {
      type: "mc",
      question: "A loss landscape has Hessian eigenvalues ranging from $0.001$ to $100$, giving condition number $\\kappa = 100{,}000$. Gradient descent with learning rate $\\alpha$ is applied. What constraint does the condition number impose?",
      options: [
        "The learning rate must satisfy $\\alpha > 2/\\lambda_{\\min} = 2000$ to make progress along flat directions, even if this causes oscillation along steep directions",
        "The condition number only affects convergence speed logarithmically, so $\\kappa = 100{,}000$ causes at most a $\\log(100{,}000) \\approx 12\\times$ slowdown",
        "The condition number determines the minimum batch size needed for stable training: batch size must be at least $\\kappa / n$ where $n$ is the parameter count",
        "The learning rate must satisfy $\\alpha < 2/\\lambda_{\\max} = 0.02$ to avoid divergence, but this makes progress along the $\\lambda = 0.001$ direction roughly $100{,}000\\times$ slower than the steep direction"
      ],
      correct: 3,
      explanation: "For gradient descent on a quadratic, the maximum stable learning rate is $\\alpha < 2/\\lambda_{\\max}$. At this rate, progress along the eigenvector with $\\lambda = 0.001$ is proportional to $\\alpha \\lambda_{\\min} \\approx 0.02 \\times 0.001 = 0.00002$ per step, while the steep direction converges quickly. The number of iterations to converge scales as $O(\\kappa)$. This is exactly why adaptive optimizers like Adam — which maintain per-parameter learning rates — dramatically outperform plain gradient descent on ill-conditioned loss surfaces."
    },
    {
      type: "info",
      title: "PSD Structure in Key ML Matrices",
      content: "Several important matrices in machine learning are PSD by construction, and recognizing this saves analysis effort:\n\n**Covariance matrices**: $\\Sigma = \\mathbb{E}[(x - \\mu)(x - \\mu)^\\top]$. Since $\\Sigma = \\mathbb{E}[zz^\\top]$ for centered $z$, we have $v^\\top \\Sigma v = \\mathbb{E}[(v^\\top z)^2] \\geq 0$.\n\n**Fisher information matrix**: $F = \\mathbb{E}_{x \\sim p_{\\theta}}[\\nabla \\log p_\\theta(x) \\, \\nabla \\log p_\\theta(x)^\\top]$. Again $F = \\mathbb{E}[gg^\\top]$ for score vectors $g$, so PSD by the same argument.\n\n**Gram matrices**: $K_{ij} = k(x_i, x_j)$ for a valid kernel function. PSD-ness of $K$ is actually the DEFINITION of a valid kernel (Mercer's condition).\n\n**Gauss-Newton approximation**: In neural networks, the Gauss-Newton matrix $G = J^\\top J$ (where $J$ is the Jacobian of the network output w.r.t. parameters) is always PSD. It approximates the Hessian while guaranteed positive semi-definite — useful because the true Hessian can be indefinite during training.\n\nKnowing a matrix is PSD tells you: eigenvalues are non-negative, Cholesky factorization exists, and the associated quadratic form is a bowl (or flat) — never a saddle."
    },
    {
      type: "mc",
      question: "During RL fine-tuning, the loss Hessian can be indefinite (negative eigenvalues exist). A practitioner switches to the Gauss-Newton approximation $G = J^\\top J$ for their second-order optimizer. Why is PSD-ness of $G$ practically important here?",
      options: [
        "PSD matrices have smaller spectral norm than indefinite matrices, reducing the magnitude of parameter updates",
        "The Gauss-Newton matrix has exactly the same eigenvalues as the Hessian, but with signs flipped to positive",
        "PSD guarantees the resulting update direction $G^{-1}g$ is a descent direction, whereas an indefinite Hessian could produce ascent steps that increase the loss",
        "PSD matrices can be stored in half the memory of indefinite matrices because only the upper triangle is needed"
      ],
      correct: 2,
      explanation: "For a PSD matrix $G$, the update $-G^{-1}g$ satisfies $g^\\top(G^{-1}g) = g^\\top G^{-1} g \\geq 0$ (since $G^{-1}$ is also PSD), guaranteeing a descent direction. With an indefinite Hessian, $H^{-1}g$ can point uphill along negative-curvature directions, actually increasing the loss. This is the key practical benefit: the Gauss-Newton approximation trades accuracy for guaranteed descent, making optimization stable even when the true Hessian has problematic curvature."
    }
  ]
};
