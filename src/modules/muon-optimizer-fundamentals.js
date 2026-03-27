// Module: Muon Optimizer Fundamentals
// Covers Newton-Schulz iteration, spectral steepest descent, and relationship to other optimizers.
// Based on: "Muon is Scalable for LLM Training" (arXiv:2502.16982)

export const muonOptimizerFundamentals = {
  id: "0.3-muon-learning-easy",
  sectionId: "0.3",
  title: "Muon Optimizer Fundamentals",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Why a New Optimizer?",
      content: "AdamW is the default optimizer for training large language models. It tracks per-parameter first and second moment estimates and uses them to scale gradient updates. This works well, but AdamW treats each parameter independently — it has no notion of the **matrix structure** of weight tensors.\n\nA weight matrix $W \\in \\mathbb{R}^{m \\times n}$ in a transformer layer maps one representation space to another. The gradient $G = \\nabla_W L$ is also a matrix. AdamW processes each entry $G_{ij}$ independently, but the rows and columns of $G$ have geometric meaning — they correspond to input and output directions in representation space.\n\n**Muon** (Matrix Updates via Orthogonalization of Newtonized gradients) exploits this structure. Instead of scaling each gradient entry independently, Muon orthogonalizes the entire gradient matrix before applying it as an update. This makes Muon perform **steepest descent under the spectral norm** rather than the element-wise $\\ell_2$ norm that SGD uses."
    },
    {
      type: "mc",
      question: "AdamW maintains per-parameter moment estimates for gradient scaling. What structural information does this approach ignore about transformer weight matrices?",
      options: [
        "The sparsity pattern that emerges in the weight matrix as training converges toward a local minimum",
        "The total number of parameters allocated to each transformer layer, which affects relative gradient magnitudes",
        "The learning rate schedule and its interaction with momentum across the different phases of training",
        "The relationship between rows and columns of the gradient, which correspond to input and output representation directions"
      ],
      correct: 3,
      explanation: "AdamW treats each parameter independently — it scales $G_{ij}$ based only on the history of that single entry. It ignores the fact that rows and columns of a weight gradient correspond to directions in the input and output representation spaces, meaning the gradient has matrix-level structure that element-wise processing discards."
    },
    {
      type: "info",
      title: "Steepest Descent and Norm Choice",
      content: "The idea behind steepest descent is simple: at each step, find the update $\\Delta W$ that decreases the loss the most, subject to a constraint on the \"size\" of the update. Different norms give different notions of size, and therefore different optimal updates.\n\n**Frobenius norm** ($\\|\\Delta W\\|_F \\leq \\epsilon$): The optimal update is proportional to the gradient $G$ itself — this gives ordinary gradient descent.\n\n**Spectral norm** ($\\|\\Delta W\\|_{\\sigma} \\leq \\epsilon$): The optimal update is $U V^T$ where $G = U \\Sigma V^T$ is the SVD of the gradient. This is the **orthogonal projection** of the gradient — all singular values are set to 1.\n\n**Element-wise $\\ell_\\infty$ norm** ($\\max_{ij} |\\Delta W_{ij}| \\leq \\epsilon$): The optimal update is $\\text{sign}(G)$ — this is what SignSGD does, and Adam approximates this behavior.\n\nMuon performs steepest descent under the spectral norm. The update $UV^T$ treats all singular value directions equally, preventing any single direction from dominating the update. This is powerful because it means Muon distributes learning effort across **all** directions in the gradient, not just the dominant ones."
    },
    {
      type: "mc",
      question: "Steepest descent under the spectral norm replaces the gradient $G = U\\Sigma V^T$ with the update $UV^T$. What is the geometric effect of discarding $\\Sigma$?",
      options: [
        "It makes the update sparse by zeroing out small singular values that fall below a learned threshold criterion",
        "It rescales the gradient so that the Frobenius norm of the update matches the target learning rate exactly",
        "It equalizes all singular value directions, preventing dominant directions from receiving disproportionately large updates",
        "It projects the gradient onto the nearest low-rank approximation that best fits the current weight matrix"
      ],
      correct: 2,
      explanation: "The SVD decomposes $G$ into orthogonal directions weighted by singular values $\\sigma_i$. Setting all $\\sigma_i = 1$ (i.e., using $UV^T$) means every direction gets equal update magnitude. Without this, gradient descent would move mostly along the top singular directions while neglecting smaller but potentially important ones."
    },
    {
      type: "info",
      title: "The Newton-Schulz Iteration",
      content: "Computing the full SVD of every gradient matrix at every training step would be prohibitively expensive — SVD is $O(mn \\cdot \\min(m,n))$ per matrix. Muon avoids this by using the **Newton-Schulz (NS) iteration** to approximate the orthogonal projection $UV^T$ without ever computing singular values.\n\nThe NS iteration for computing the matrix sign function (which yields $UV^T$ when applied to a matrix with positive singular values) is:\n\n$$X_0 = G / \\|G\\|_F$$\n$$X_{k+1} = a_k X_k + b_k (X_k X_k^T) X_k + c_k (X_k X_k^T)^2 X_k$$\n\nwhere $a_k, b_k, c_k$ are polynomial coefficients chosen so that after a few iterations, $X_k \\approx UV^T$.\n\nIn practice, **5 iterations** of a carefully tuned 5th-order polynomial suffice. The key operations are matrix multiplications ($X X^T$ and products with $X$), which map efficiently onto GPU hardware.\n\nThe cost per iteration is a few matrix multiplies of shapes $(m \\times n)$ and $(m \\times m)$ or $(n \\times n)$, making it much cheaper than full SVD while achieving a good approximation."
    },
    {
      type: "mc",
      question: "Muon uses 5 iterations of a Newton-Schulz polynomial to approximate $UV^T$. Why is this preferred over computing the exact SVD of the gradient?",
      options: [
        "Newton-Schulz iterations only require matrix multiplications, which map efficiently onto GPU hardware, while SVD requires sequential eigenvalue solvers",
        "The SVD is numerically unstable for matrices with repeated singular values, a condition that gradient matrices commonly exhibit during training",
        "The SVD cannot handle non-square matrices without expensive padding, and transformer weight gradients are typically rectangular in shape",
        "Newton-Schulz iterations produce a better conditioned orthogonal update than exact SVD when applied in the presence of stochastic gradient noise"
      ],
      correct: 0,
      explanation: "The NS iteration consists entirely of matrix multiplications (GEMMs), which are the most optimized operations on modern GPUs. Full SVD requires iterative eigenvalue algorithms that are inherently more sequential and less GPU-friendly. SVD can handle rectangular matrices and is numerically stable — the advantage is purely computational efficiency."
    },
    {
      type: "info",
      title: "Muon's Relationship to Shampoo and SOAP",
      content: "Muon is not the first optimizer to exploit matrix structure in gradients. **Shampoo** preconditions gradients using Kronecker-factored approximations of the full Fisher/Hessian:\n\n$$\\Delta W = L^{-1/2} G R^{-1/2}$$\n\nwhere $L \\approx \\mathbb{E}[GG^T]$ and $R \\approx \\mathbb{E}[G^TG]$ are running estimates of the left and right covariance of the gradient. **SOAP** improves on Shampoo by working in the eigenbasis of these covariance matrices and applying Adam-like updates there.\n\nMuon can be seen as a **simplified extreme** of this family. Where Shampoo tracks and inverts covariance matrices, Muon directly orthogonalizes the current gradient. The relationship becomes clearer when you note that $UV^T = G(G^TG)^{-1/2}$, which is a single-sample version of Shampoo's right-preconditioning with $R = G^TG$.\n\nThe practical advantage of Muon: it requires **no running statistics** beyond a standard momentum buffer. Shampoo needs to maintain and periodically invert large covariance matrices; Muon replaces all of this with a few NS iterations applied to the current (momentum-buffered) gradient."
    },
    {
      type: "mc",
      question: "Shampoo preconditions gradients using running covariance matrices $L$ and $R$. Muon's orthogonalization $UV^T$ can be related to Shampoo by noting that $UV^T = G(G^TG)^{-1/2}$. What is the key practical difference?",
      options: [
        "Shampoo requires maintaining and periodically inverting large covariance matrices, while Muon only needs the current momentum-buffered gradient",
        "Muon converges to a sharper minimum because it computes the exact matrix inverse rather than using a running approximation",
        "Muon applies preconditioning along both the left and right dimensions simultaneously, while Shampoo only preconditions one dimension",
        "Shampoo cannot be distributed across multiple GPUs because the covariance matrices must reside together on a single device"
      ],
      correct: 0,
      explanation: "Muon's main practical advantage is simplicity: it needs no running covariance estimates, no periodic matrix inversions, and no eigenbasis tracking. It just takes the current momentum-buffered gradient and orthogonalizes it via NS iterations. Shampoo achieves similar goals but with substantially more memory and bookkeeping overhead."
    },
    {
      type: "info",
      title: "Per-Layer Update Normalization",
      content: "A subtle but important aspect of Muon is how it handles different layers. In a transformer, weight matrices vary in shape and scale: attention projection matrices ($d_{model} \\times d_{head}$), MLP up-projections ($d_{model} \\times 4d_{model}$), and embedding layers all have different dimensions.\n\nMuon's orthogonalization naturally normalizes updates per-layer. Since $\\|UV^T\\|_\\sigma = 1$ for any orthogonal matrix, the spectral norm of every layer's update is the same before the learning rate is applied. This provides an implicit form of **layer-wise learning rate normalization** — no layer can receive a disproportionately large or small update due to gradient scale differences.\n\nContrast this with Adam, where different layers can have very different effective step sizes depending on their second-moment estimates. Practitioners often use techniques like layer-wise learning rate decay (LLRD) or careful initialization schemes to manage this. Muon sidesteps the problem entirely.\n\nIn practice, Muon applies its orthogonalization to all 2D weight matrices (attention projections, MLP layers). For 1D parameters (biases, LayerNorm scales) and embedding layers, standard Adam is typically used since these don't have meaningful matrix structure."
    },
    {
      type: "mc",
      question: "A transformer has an attention projection matrix of shape $768 \\times 64$ and an MLP weight of shape $768 \\times 3072$. Under Muon, how do the spectral norms of these two layers' updates compare before the learning rate is applied?",
      options: [
        "The MLP update has a larger spectral norm because the matrix contains more parameters and accumulates a larger gradient",
        "The attention update has a larger spectral norm because the extreme aspect ratio of the matrix amplifies small singular values",
        "The relative spectral norms depend on the batch size used during training and cannot be determined from matrix shape alone",
        "Both updates have spectral norm exactly 1, since orthogonalization normalizes all singular values regardless of matrix shape"
      ],
      correct: 3,
      explanation: "After orthogonalization, the update is $UV^T$ which has all singular values equal to 1, so $\\|UV^T\\|_\\sigma = 1$ regardless of the original matrix shape or gradient magnitude. This gives Muon automatic per-layer normalization — both the $768 \\times 64$ and $768 \\times 3072$ layers receive updates with identical spectral norms."
    },
    {
      type: "info",
      title: "Memory and Compute Costs",
      content: "How does Muon compare to AdamW in resource usage?\n\n**Memory:** AdamW stores two moment buffers (first and second moment) per parameter — 2x the model size in optimizer state. Muon stores only a **momentum buffer** (1x the model size for the 2D parameters). For 1D parameters where Adam is still used, both moments are stored, but these are a tiny fraction of total parameters. Overall, Muon uses roughly **35-40% less optimizer memory** than AdamW for a typical transformer.\n\n**Compute:** The NS iterations add cost. For a weight matrix $W \\in \\mathbb{R}^{m \\times n}$, each NS iteration involves multiplications of cost $O(m^2 n)$ or $O(mn^2)$. With 5 iterations, this adds roughly **5-10% compute overhead** per step compared to AdamW. However, Muon often needs **fewer total steps** to reach the same loss, so wall-clock time can be comparable or even lower.\n\n**Communication:** In distributed training, Muon requires all-reducing the full gradient matrix before orthogonalization (since NS iterations are non-linear and cannot be applied to shards independently). This is the same communication pattern as standard all-reduce, but the non-linearity means you cannot easily overlap communication with NS computation."
    },
    {
      type: "mc",
      question: "A 7B parameter transformer is trained with AdamW, requiring roughly 14B floats of optimizer state (first + second moments). If switched to Muon (with Adam only on the ~2% of parameters that are 1D), approximately how much optimizer state memory is needed?",
      options: [
        "About 14B floats — Muon still needs two moment buffers per parameter but computes them using a different algorithm",
        "About 7.3B floats — one momentum buffer for the 98% using Muon, plus two moment buffers for the 2% using Adam",
        "About 3.5B floats — Muon stores momentum in half-precision format while Adam retains its moments in full precision",
        "About 21B floats — Muon adds a third persistent buffer to store the Newton-Schulz iteration intermediate states"
      ],
      correct: 1,
      explanation: "Muon needs 1 momentum buffer for the ~98% of parameters it handles: $0.98 \\times 7B = 6.86B$ floats. For the ~2% using Adam: $0.02 \\times 7B \\times 2 = 0.28B$ floats. Total: ~7.14B floats, roughly half of AdamW's 14B. The NS iteration intermediates are temporary and don't persist in optimizer state."
    },
    {
      type: "mc",
      question: "In distributed training with data parallelism, gradient all-reduce must happen before Muon's Newton-Schulz iterations. Why can't the NS iterations be applied to gradient shards independently and then combined?",
      options: [
        "GPU memory constraints on individual workers prevent storing the full intermediate matrices needed during the iteration",
        "The Newton-Schulz convergence rate depends on having the full gradient spectrum, which individual shards cannot provide",
        "The NS polynomial has nonlinear terms like $(XX^T)X$, so applying it to shards and summing does not equal applying it to the full sum",
        "The communication bandwidth required for the intermediate matrices exceeds the bandwidth needed for the raw gradient tensors"
      ],
      correct: 2,
      explanation: "The NS iteration applies polynomial functions involving $XX^T$ — these are nonlinear operations. For a nonlinear function $f$, $f(G_1 + G_2) \\neq f(G_1) + f(G_2)$ in general. So you must first sum (all-reduce) the gradient shards to get the full gradient $G$, then apply the NS iterations to $G$. This is a fundamental mathematical constraint, not a hardware limitation."
    }
  ]
};
