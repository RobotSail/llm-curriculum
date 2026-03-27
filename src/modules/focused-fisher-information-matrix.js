// Focused learning module: Fisher Information Matrix
// Section 0.2: Probability & Information Theory
// ONE concept: The multi-parameter Fisher Information Matrix, its properties,
// and its deep connection to KL divergence as a local metric.

export const fisherInformationMatrixLearning = {
  id: "0.2-fisher-information-matrix-learning-medium",
  sectionId: "0.2",
  title: "The Fisher Information Matrix",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 25,
  steps: [
    // --- Step 1: From scalar to matrix ---
    {
      type: "info",
      title: "From Scalar to Matrix",
      content: "When a model has a single parameter $\\theta$, Fisher Information is a scalar $I(\\theta)$. But real models have parameter vectors $\\boldsymbol{\\theta} = (\\theta_1, \\dots, \\theta_d)$. The natural extension is the **Fisher Information Matrix** (FIM):\n\n$$F(\\boldsymbol{\\theta})_{ij} = \\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\frac{\\partial \\log p_\\theta(x)}{\\partial \\theta_i} \\cdot \\frac{\\partial \\log p_\\theta(x)}{\\partial \\theta_j}\\right]$$\n\nIn matrix notation, using the score vector $\\mathbf{s}(x) = \\nabla_\\theta \\log p_\\theta(x)$:\n\n$$F(\\boldsymbol{\\theta}) = \\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\mathbf{s}(x) \\, \\mathbf{s}(x)^\\top\\right]$$\n\nSince $\\mathbb{E}[\\mathbf{s}] = \\mathbf{0}$ (the zero-mean property from the scalar case extends directly), the FIM is the **covariance matrix of the score**:\n\n$$F(\\boldsymbol{\\theta}) = \\text{Cov}_{x \\sim p_\\theta}[\\mathbf{s}(x)]$$\n\nThe diagonal entries $F_{ii}$ are the scalar Fisher Information for each parameter. The off-diagonal entries $F_{ij}$ capture how the score components for parameters $i$ and $j$ co-vary — they encode **statistical dependencies** between parameter estimates."
    },
    // --- Step 2: PSD and outer product ---
    {
      type: "mc",
      question: "The FIM is defined as $F = \\mathbb{E}[\\mathbf{s} \\, \\mathbf{s}^\\top]$, an expected outer product of score vectors. What does this structure guarantee about $F$?",
      options: [
        "It guarantees $F$ is orthogonal, because score vectors from different observations are always perpendicular in expectation",
        "It guarantees $F$ is sparse, because most pairs of parameters have independent score components that average to zero",
        "It guarantees $F$ is symmetric and has determinant 1, because the score function preserves the normalization of the probability distribution",
        "It guarantees $F$ is positive semi-definite, because for any vector $\\mathbf{v}$, $\\mathbf{v}^\\top F \\mathbf{v} = \\mathbb{E}[(\\mathbf{v}^\\top \\mathbf{s})^2] \\geq 0$"
      ],
      correct: 3,
      explanation: "For any vector $\\mathbf{v}$, $\\mathbf{v}^\\top F \\mathbf{v} = \\mathbf{v}^\\top \\mathbb{E}[\\mathbf{s}\\mathbf{s}^\\top]\\mathbf{v} = \\mathbb{E}[(\\mathbf{v}^\\top \\mathbf{s})^2] \\geq 0$. An expected square is always non-negative, so $F$ is positive semi-definite (PSD). It is also symmetric by construction ($F_{ij} = F_{ji}$). PSD is crucial: it means $F$ defines a valid inner product on parameter space, which is the foundation for using it as a Riemannian metric in natural gradient methods."
    },
    // --- Step 3: Hessian equivalence ---
    {
      type: "info",
      title: "Equivalence with the Negative Expected Hessian",
      content: "Just as in the scalar case, the FIM equals the **negative expected Hessian** of the log-likelihood:\n\n$$F(\\boldsymbol{\\theta})_{ij} = -\\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\frac{\\partial^2 \\log p_\\theta(x)}{\\partial \\theta_i \\, \\partial \\theta_j}\\right]$$\n\nOr in matrix notation:\n\n$$F(\\boldsymbol{\\theta}) = -\\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\nabla^2_\\theta \\log p_\\theta(x)\\right]$$\n\nThe proof uses the same integration trick: differentiate the identity $\\int p_\\theta(x) \\, dx = 1$ twice with respect to $\\theta$, swap integration and differentiation, and simplify.\n\nThis equivalence has a powerful geometric interpretation: the FIM measures the **average curvature of the log-likelihood surface** in every direction simultaneously. The entry $F_{ij}$ tells you how the log-likelihood curves in the $(\\theta_i, \\theta_j)$ plane.\n\nFor optimization, this means the FIM captures the same information as the Hessian of the loss — but with an important advantage: the FIM is always PSD, while the Hessian can have negative eigenvalues near saddle points. This is why the natural gradient ($F^{-1}g$) is more stable than the Newton step ($H^{-1}g$)."
    },
    // --- Step 4: Multivariate Gaussian example ---
    {
      type: "info",
      title: "Example: Multivariate Gaussian",
      content: "Consider estimating the mean $\\boldsymbol{\\mu}$ of a multivariate Gaussian $\\mathcal{N}(\\boldsymbol{\\mu}, \\Sigma)$ with known covariance $\\Sigma$.\n\nThe log-likelihood for one observation $\\mathbf{x}$ is:\n$$\\log p(\\mathbf{x}; \\boldsymbol{\\mu}) = -\\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu})^\\top \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}) + \\text{const}$$\n\nThe score vector is:\n$$\\mathbf{s}(\\mathbf{x}) = \\nabla_\\mu \\log p = \\Sigma^{-1}(\\mathbf{x} - \\boldsymbol{\\mu})$$\n\nThe FIM is:\n$$F = \\mathbb{E}[\\mathbf{s} \\, \\mathbf{s}^\\top] = \\Sigma^{-1} \\mathbb{E}[(\\mathbf{x} - \\boldsymbol{\\mu})(\\mathbf{x} - \\boldsymbol{\\mu})^\\top] \\Sigma^{-1} = \\Sigma^{-1} \\Sigma \\, \\Sigma^{-1} = \\Sigma^{-1}$$\n\nSo $F = \\Sigma^{-1}$: the **precision matrix**. This makes perfect intuitive sense:\n- Directions with low variance (high precision) have high Fisher Information — the data tightly constrains $\\boldsymbol{\\mu}$ in those directions\n- Directions with high variance (low precision) have low Fisher Information — $\\boldsymbol{\\mu}$ is poorly determined\n- Correlated dimensions create off-diagonal entries that couple the estimation of different components of $\\boldsymbol{\\mu}$"
    },
    // --- Step 5: Gaussian FIM quiz ---
    {
      type: "mc",
      question: "For a 2D Gaussian with covariance $\\Sigma = \\begin{pmatrix} 1 & 0.9 \\\\ 0.9 & 1 \\end{pmatrix}$, the FIM for $\\boldsymbol{\\mu}$ is $\\Sigma^{-1} = \\frac{1}{0.19}\\begin{pmatrix} 1 & -0.9 \\\\ -0.9 & 1 \\end{pmatrix}$. What does the large negative off-diagonal entry $F_{12} \\approx -4.7$ signify?",
      options: [
        "The two mean components are impossible to estimate jointly — the data can determine $\\mu_1 + \\mu_2$ but not the individual values",
        "Estimating $\\mu_1$ too high systematically biases $\\mu_2$ too low — the score components are negatively correlated due to the positive data correlation",
        "The Fisher Information for both parameters individually is negative, indicating the model is misspecified for this data distribution",
        "The two components of the score are independent but both have negative means, producing a negative expected product"
      ],
      correct: 1,
      explanation: "When data $x_1$ and $x_2$ are positively correlated ($\\Sigma_{12} = 0.9$), observing $x_1$ above $\\mu_1$ typically coincides with $x_2$ above $\\mu_2$. The precision matrix $\\Sigma^{-1}$ has negative off-diagonal entries, meaning the score for $\\mu_1$ and the score for $\\mu_2$ are negatively correlated: evidence that pushes $\\mu_1$ up tends to also push $\\mu_2$ up, but the FIM captures that these pushes are not independent — overestimating one systematically affects the other. This coupling is exactly what the off-diagonal FIM entries encode."
    },
    // --- Step 6: KL divergence connection ---
    {
      type: "info",
      title: "Fisher Information as the Hessian of KL Divergence",
      content: "The deepest connection in information geometry: the FIM is the **local curvature of KL divergence**.\n\nConsider two nearby parameter settings $\\boldsymbol{\\theta}$ and $\\boldsymbol{\\theta} + \\boldsymbol{\\delta}$. Taylor-expand the KL divergence between their distributions:\n\n$$\\text{KL}(p_\\theta \\| p_{\\theta + \\delta}) \\approx \\frac{1}{2} \\boldsymbol{\\delta}^\\top F(\\boldsymbol{\\theta}) \\, \\boldsymbol{\\delta}$$\n\nThe first-order term vanishes (KL is minimized at $\\boldsymbol{\\delta} = 0$), and the second-order term is precisely the FIM. This means:\n\n1. **The FIM defines a local distance** between distributions. Two parameter settings that differ by $\\boldsymbol{\\delta}$ produce distributions separated by approximately $\\frac{1}{2} \\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta}$ in KL divergence.\n\n2. **Natural gradient steepest descent in KL space**. The natural gradient $F^{-1}g$ finds the parameter change that maximally reduces the loss per unit KL change in the output distribution, rather than per unit Euclidean change in parameter space.\n\n3. **Information geometry**. The FIM is a **Riemannian metric tensor** on the manifold of probability distributions. It defines lengths, angles, and geodesics in the space of distributions — a framework pioneered by Amari that underlies much of modern optimization theory for probabilistic models."
    },
    // --- Step 7: KL connection quiz ---
    {
      type: "mc",
      question: "A language model has parameters $\\boldsymbol{\\theta}$. You compute the FIM $F$ and want to find the parameter perturbation $\\boldsymbol{\\delta}$ (with $\\|\\boldsymbol{\\delta}\\|_2 = \\epsilon$) that causes the largest KL divergence $\\text{KL}(p_\\theta \\| p_{\\theta + \\delta})$. Which direction should $\\boldsymbol{\\delta}$ point?",
      options: [
        "Along the gradient $\\nabla_\\theta \\mathcal{L}$, because the loss gradient always points in the direction of maximum distribution change",
        "Along the eigenvector of $F$ with the smallest eigenvalue, because these fragile directions are most sensitive to perturbation",
        "Along the eigenvector of $F$ with the largest eigenvalue, because KL $\\approx \\frac{1}{2}\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta}$ is maximized in the highest-curvature direction",
        "Uniformly in all directions, because the FIM is an average over data and does not prefer any particular direction in parameter space"
      ],
      correct: 2,
      explanation: "The quadratic form $\\frac{1}{2}\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta}$ subject to $\\|\\boldsymbol{\\delta}\\|_2 = \\epsilon$ is maximized when $\\boldsymbol{\\delta}$ aligns with the eigenvector of $F$ corresponding to the largest eigenvalue $\\lambda_{\\max}$. This gives KL $\\approx \\frac{1}{2}\\lambda_{\\max}\\epsilon^2$. These are the directions in parameter space where the output distribution is most sensitive. Conversely, the smallest eigenvalue direction gives the minimum KL — the distribution barely changes. This eigenstructure is exactly what natural gradient methods exploit."
    },
    // --- Step 8: Empirical vs true Fisher ---
    {
      type: "info",
      title: "Empirical Fisher vs. True Fisher",
      content: "In practice, computing the true FIM is expensive because it requires expectations under the model distribution $p_\\theta$. Two common approximations are used:\n\n**True (model) Fisher**: Sample $y$ from the model's own output $p_\\theta(y|x)$ and compute:\n$$\\hat{F}_{\\text{true}} = \\frac{1}{N}\\sum_{i=1}^{N} \\nabla_\\theta \\log p_\\theta(y_i | x_i) \\, \\nabla_\\theta \\log p_\\theta(y_i | x_i)^\\top, \\quad y_i \\sim p_\\theta(\\cdot | x_i)$$\n\n**Empirical Fisher**: Use the actual labels $y^*_i$ from the training data instead of model samples:\n$$\\hat{F}_{\\text{emp}} = \\frac{1}{N}\\sum_{i=1}^{N} \\nabla_\\theta \\log p_\\theta(y^*_i | x_i) \\, \\nabla_\\theta \\log p_\\theta(y^*_i | x_i)^\\top$$\n\nThe empirical Fisher is much cheaper (no sampling needed — just use the training batch gradients you already compute). But it has a subtle flaw: the training labels come from the data distribution, not the model distribution. The two coincide only when the model perfectly fits the data ($p_\\theta = p_{\\text{data}}$).\n\nIn practice, the empirical Fisher is widely used (it is what EWC, most K-FAC implementations, and Adam's diagonal approximation actually compute). The approximation is reasonable when the model is well-trained on the data, but can be misleading early in training or for out-of-distribution data."
    },
    // --- Step 9: Empirical Fisher quiz ---
    {
      type: "mc",
      question: "Early in training, a language model assigns nearly uniform probability over the vocabulary. Why is the empirical Fisher a poor approximation to the true Fisher at this stage?",
      options: [
        "The gradients from the true labels point toward the data distribution, but the true Fisher should measure curvature around the current model distribution — which is nearly uniform and has very different geometry",
        "The empirical Fisher is undefined for uniform distributions because the log-probability gradient is zero everywhere",
        "The empirical Fisher overestimates the true Fisher because early gradients are large, while the true Fisher is always close to zero at initialization",
        "The empirical Fisher and true Fisher are identical at initialization because the model has not yet learned any structure from the data"
      ],
      correct: 0,
      explanation: "The true Fisher uses samples from $p_\\theta$ (nearly uniform early on) to compute the expected outer product. The empirical Fisher uses real labels from $p_{\\text{data}}$ (highly peaked — real text has specific tokens). These are very different distributions early in training: the model says \"any token is equally likely\" while the data says \"this specific token comes next.\" The score vectors under model samples vs. data labels have different magnitudes and directions. As training progresses and $p_\\theta$ approaches $p_{\\text{data}}$, the two Fisher matrices converge."
    },
    // --- Step 10: Cramér-Rao in matrix form ---
    {
      type: "info",
      title: "The Matrix Cramér-Rao Bound",
      content: "The Cramér-Rao bound generalizes to the multi-parameter setting:\n\n$$\\text{Cov}(\\hat{\\boldsymbol{\\theta}}) \\succeq \\frac{1}{n} F(\\boldsymbol{\\theta})^{-1}$$\n\nwhere $\\succeq$ means the difference is positive semi-definite: $\\text{Cov}(\\hat{\\boldsymbol{\\theta}}) - \\frac{1}{n}F^{-1}$ is PSD. This is a matrix inequality — it constrains the entire covariance structure, not just individual variances.\n\nTaking diagonal entries gives scalar bounds: $\\text{Var}(\\hat{\\theta}_i) \\geq [F^{-1}]_{ii}/n$. Notice that the bound for $\\theta_i$ uses $[F^{-1}]_{ii}$, **not** $1/F_{ii}$. When parameters are correlated (off-diagonal entries in $F$), $[F^{-1}]_{ii} \\geq 1/F_{ii}$ — the bound is **looser** than what you would get by ignoring correlations.\n\nThis has a practical consequence: if two parameters are highly correlated in their effect on the likelihood (large $|F_{ij}|$), they are harder to estimate individually. The FIM's off-diagonal structure quantifies this \"confounding\" between parameters — something that diagonal approximations like Adam completely ignore."
    },
    // --- Step 11: Matrix CR bound quiz ---
    {
      type: "mc",
      question: "A 2-parameter model has FIM $F = \\begin{pmatrix} 10 & 9 \\\\ 9 & 10 \\end{pmatrix}$. The inverse is $F^{-1} = \\frac{1}{19}\\begin{pmatrix} 10 & -9 \\\\ -9 & 10 \\end{pmatrix}$. What is the Cramér-Rao lower bound on $\\text{Var}(\\hat{\\theta}_1)$ with $n=1$ observation, and how does it compare to ignoring the off-diagonal?",
      options: [
        "$\\text{Var}(\\hat{\\theta}_1) \\geq 10/19 \\approx 0.53$, which is much larger than $1/F_{11} = 0.1$ — the correlation inflates the estimation difficulty by $5\\times$",
        "$\\text{Var}(\\hat{\\theta}_1) \\geq 1/10 = 0.1$, the same as ignoring the off-diagonal, because the Cramér-Rao bound only uses diagonal entries of $F$",
        "$\\text{Var}(\\hat{\\theta}_1) \\geq 19/10 = 1.9$, because the bound uses $\\det(F)/F_{22}$ from the adjugate formula",
        "$\\text{Var}(\\hat{\\theta}_1) \\geq 9/19 \\approx 0.47$, using the off-diagonal entry of $F^{-1}$ rather than the diagonal entry"
      ],
      correct: 0,
      explanation: "The matrix Cramér-Rao bound gives $\\text{Var}(\\hat{\\theta}_1) \\geq [F^{-1}]_{11} = 10/19 \\approx 0.53$. If we naively ignored the correlation and used $1/F_{11} = 1/10 = 0.1$, we would underestimate the difficulty by $5\\times$. The near-equal Fisher Information for both parameters ($F_{11} = F_{22} = 10$) combined with high correlation ($F_{12} = 9$) means the data mostly tells you about $\\theta_1 + \\theta_2$ (along the correlated direction), leaving $\\theta_1 - \\theta_2$ poorly determined. This is exactly the kind of structure that off-diagonal FIM entries capture."
    },
    // --- Step 12: Fisher and model reparameterization ---
    {
      type: "info",
      title: "Reparameterization Covariance",
      content: "A crucial property of the FIM is its transformation behavior under reparameterization. If we reparameterize $\\boldsymbol{\\theta} \\to \\boldsymbol{\\phi}(\\boldsymbol{\\theta})$ with Jacobian $J = \\partial \\boldsymbol{\\phi}/\\partial \\boldsymbol{\\theta}$, the FIM transforms as:\n\n$$F_\\phi = J^{-\\top} F_\\theta \\, J^{-1}$$\n\nThis is the transformation law for a **metric tensor** — the same rule that governs how distances transform in differential geometry. The FIM is not just any matrix: it is the unique (up to scaling) Riemannian metric on the space of probability distributions that is invariant to sufficient statistics (Čencov's theorem).\n\nWhat this means practically: the \"true\" geometric properties of the model — like how far apart two parameter settings are in distribution space — do not depend on how you parameterize the model. If you reparameterize from raw logits to softmax probabilities, the FIM changes, but the KL divergence between nearby distributions (which is $\\frac{1}{2}\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta}$) remains the same.\n\nThis is why the **natural gradient** $F^{-1}g$ is reparameterization-invariant: it measures steepest descent in the intrinsic geometry of distributions, not in the arbitrary coordinates of parameter space."
    },
    // --- Step 13: Reparameterization quiz ---
    {
      type: "mc",
      question: "You switch from parameterizing a softmax layer by logits $\\boldsymbol{z}$ to log-probabilities $\\boldsymbol{\\ell} = \\log \\text{softmax}(\\boldsymbol{z})$. The FIM in logit space is $F_z$. Under the new parameterization, $F_\\ell = J^{-\\top} F_z J^{-1}$. Which quantity remains unchanged?",
      options: [
        "The gradient $\\nabla_\\theta \\mathcal{L}$, because the loss is the same function regardless of parameterization",
        "The diagonal entries of the FIM, because per-parameter Fisher Information is a property of the parameter itself",
        "The natural gradient step $F^{-1}g$, because it cancels the coordinate-dependent factors from both $F$ and $g$",
        "The trace of the FIM, because $\\text{tr}(F)$ measures total Fisher Information which is coordinate-free"
      ],
      correct: 2,
      explanation: "Under reparameterization $\\boldsymbol{\\theta} \\to \\boldsymbol{\\phi}$, the gradient transforms as $g_\\phi = J^{-\\top} g_\\theta$ and the FIM transforms as $F_\\phi = J^{-\\top} F_\\theta J^{-1}$. The natural gradient: $F_\\phi^{-1} g_\\phi = J F_\\theta^{-1} J^\\top \\cdot J^{-\\top} g_\\theta = J F_\\theta^{-1} g_\\theta$. The extra $J$ factor accounts for the coordinate change, so the resulting distribution update is identical. The ordinary gradient $g$, the diagonal of $F$, and $\\text{tr}(F)$ all change under reparameterization."
    },
    // --- Step 14: Summary and connections ---
    {
      type: "mc",
      question: "A researcher claims: \"The FIM $F$ is the Hessian of the loss function, so $F^{-1}g$ is just Newton's method.\" What is wrong with this statement?",
      options: [
        "Nothing — the FIM is exactly the Hessian of any loss function, so the natural gradient and Newton's method are always identical",
        "The FIM is the gradient of the loss, not the Hessian, so $F^{-1}g$ divides a gradient by another gradient rather than inverting curvature",
        "The FIM is the Hessian of the KL divergence, not the log-likelihood, so it measures curvature in a completely different space that has no connection to the loss surface",
        "The FIM is the negative expected Hessian of the log-likelihood under the model distribution, not the Hessian of the empirical loss — they coincide only when the model perfectly fits the data and the loss is cross-entropy"
      ],
      correct: 3,
      explanation: "The FIM is $-\\mathbb{E}_{p_\\theta}[\\nabla^2 \\log p_\\theta(x)]$: the negative expected Hessian of the log-likelihood, where the expectation is under the model distribution. The empirical loss Hessian uses the data distribution instead, and may include terms beyond the log-likelihood (regularization, etc.). For cross-entropy loss, when the model has converged ($p_\\theta \\approx p_{\\text{data}}$), the two approximately coincide — this is the Generalized Gauss-Newton approximation. But in general, the FIM captures the geometry of the model's output distribution, while the Hessian captures the geometry of the loss surface. They are related but distinct objects."
    }
  ]
};
