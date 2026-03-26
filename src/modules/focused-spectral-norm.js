// Module: Spectral Norm
// Section 0.1: Spectral norm, Lipschitz continuity, spectral normalization
// Single-concept module: understanding and controlling how much a matrix stretches
// Proper learning module with alternating info/mc steps

export const spectralNormLearning = {
  id: "0.1-specnorm-learning-easy",
  sectionId: "0.1",
  title: "Spectral Norm",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Measuring How Much a Matrix Stretches",
      content: "When a matrix $A$ multiplies a vector $x$, it can change the vector's length. Some vectors get stretched, others get compressed. A natural question: **what is the maximum stretching factor?**\n\nThis is exactly what the **spectral norm** measures:\n\n$$\\|A\\|_2 = \\max_{\\|x\\| = 1} \\|Ax\\|$$\n\nIn words: among all unit vectors $x$, find the one that $A$ stretches the most. The length of $Ax$ for that worst-case input is the spectral norm.\n\nIt turns out this maximum equals the **largest singular value** of $A$:\n\n$$\\|A\\|_2 = \\sigma_1(A)$$\n\nFor symmetric matrices, the spectral norm equals the largest absolute eigenvalue: $\\|A\\|_2 = \\max_i |\\lambda_i|$. For general (non-symmetric) matrices, you must use singular values — eigenvalues alone don't capture the stretching behavior."
    },
    {
      type: "mc",
      question: "A matrix $A$ has singular values $\\sigma_1 = 12, \\sigma_2 = 7, \\sigma_3 = 3$. What is $\\|A\\|_2$?",
      options: ["$12$, the largest singular value", "$\\sqrt{12^2 + 7^2 + 3^2} \\approx 14.6$, the root sum of squared singular values", "$22$, the sum of all singular values", "$7$, the median singular value"],
      correct: 0,
      explanation: "The spectral norm is the largest singular value: $\\|A\\|_2 = \\sigma_1 = 12$. This means $A$ can stretch a unit vector by at most a factor of 12. The sum of singular values ($22$) is the nuclear norm $\\|A\\|_*$. The root sum of squares ($\\approx 14.6$) is the Frobenius norm $\\|A\\|_F$. Each norm measures a different aspect of the matrix."
    },
    {
      type: "info",
      title: "Why Spectral Norm Matters: Gradient Flow in Deep Networks",
      content: "Consider a deep network with $L$ linear layers (ignoring activations for now). The output is:\n\n$$f(x) = W_L W_{L-1} \\cdots W_2 W_1 x$$\n\nHow much can a small perturbation $\\delta x$ at the input grow by the time it reaches the output? By the submultiplicativity of norms:\n\n$$\\|f(x + \\delta x) - f(x)\\| \\leq \\|W_L\\|_2 \\|W_{L-1}\\|_2 \\cdots \\|W_1\\|_2 \\cdot \\|\\delta x\\|$$\n\nThe same bound applies to **gradients flowing backward**. During backpropagation, the gradient at layer $l$ passes through $W_{l+1}^\\top, W_{l+2}^\\top, \\ldots, W_L^\\top$. If each $\\|W_i\\|_2 > 1$, the gradient grows exponentially with depth — **gradient explosion**. If each $\\|W_i\\|_2 < 1$, it shrinks exponentially — **gradient vanishing**.\n\nThis is why the spectral norm of weight matrices is the key quantity for training stability: it controls the rate of exponential growth or decay of signals through the network."
    },
    {
      type: "mc",
      question: "A 50-layer network has weight matrices where $\\|W_i\\|_2 = 1.1$ for every layer. By what factor can the gradient grow from the output to the input?",
      options: ["At most $1.1 \\times 50 = 55$, since the growth is linear in the number of layers", "At most $1.1^{50} \\approx 117$, since the spectral norms multiply across layers", "Exactly $1.1$, since only the final layer's spectral norm matters for gradient magnitude", "At most $50^{1.1} \\approx 63$, since depth is raised to the power of the spectral norm"],
      correct: 1,
      explanation: "The gradient bound is the product of spectral norms: $\\prod_{i=1}^{50} \\|W_i\\|_2 = 1.1^{50} \\approx 117$. Even a modest spectral norm of $1.1$ per layer compounds exponentially over 50 layers, amplifying gradients by over 100×. This multiplicative structure is why gradient explosion is such a persistent problem in deep networks — small deviations from $\\|W\\|_2 = 1$ compound rapidly."
    },
    {
      type: "info",
      title: "Related Matrix Norms",
      content: "The spectral norm is one of three commonly used matrix norms in deep learning. Each captures different information about the singular value spectrum $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r$:\n\n**Spectral norm** (operator norm): $\\|A\\|_2 = \\sigma_1$\n- Measures the **worst-case stretching** — maximum amplification of any direction\n- Used for: Lipschitz bounds, training stability analysis\n\n**Frobenius norm**: $\\|A\\|_F = \\sqrt{\\sum_i \\sigma_i^2} = \\sqrt{\\sum_{ij} A_{ij}^2}$\n- Measures the **total energy** across all singular values\n- Used for: weight decay ($\\lambda \\|W\\|_F^2$), distance between weight matrices\n\n**Nuclear norm** (trace norm): $\\|A\\|_* = \\sum_i \\sigma_i$\n- Measures the **sum of stretching factors** — the convex envelope of rank\n- Used for: promoting low-rank solutions in matrix completion, regularization\n\nThe three norms satisfy: $\\|A\\|_2 \\leq \\|A\\|_F \\leq \\|A\\|_*$ for any matrix $A$. The spectral norm is always the smallest (it only tracks the largest singular value), while the nuclear norm is the largest."
    },
    {
      type: "mc",
      question: "Weight decay penalizes $\\|W\\|_F^2 = \\sum_i \\sigma_i^2$. How does this differ from penalizing the spectral norm $\\|W\\|_2^2 = \\sigma_1^2$?",
      options: ["They are equivalent — the Frobenius norm squared equals the spectral norm squared for all matrices", "Frobenius penalty only affects the smallest singular values, leaving the largest unchanged", "Spectral penalty promotes low-rank structure, while Frobenius penalty promotes full-rank matrices", "Frobenius penalty shrinks all singular values, while spectral penalty only constrains the largest one"],
      correct: 3,
      explanation: "The Frobenius penalty $\\sum_i \\sigma_i^2$ has gradient $2\\sigma_i$ for each singular value — it shrinks all of them proportionally. The spectral norm penalty $\\sigma_1^2$ only has gradient with respect to $\\sigma_1$ — it only constrains the largest singular value and ignores the rest. Weight decay (Frobenius) thus provides a more uniform regularization across the entire spectrum, while spectral norm penalties are more targeted."
    },
    {
      type: "info",
      title: "Lipschitz Continuity",
      content: "A function $f$ is **$L$-Lipschitz** if it cannot amplify distances by more than a factor of $L$:\n\n$$\\|f(x) - f(y)\\| \\leq L \\|x - y\\| \\quad \\text{for all } x, y$$\n\nThe smallest such $L$ is the **Lipschitz constant** of $f$. For a linear map $f(x) = Wx$, the Lipschitz constant is exactly the spectral norm:\n\n$$\\|Wx - Wy\\| = \\|W(x - y)\\| \\leq \\|W\\|_2 \\|x - y\\|$$\n\nFor a composition of functions $f = f_L \\circ \\cdots \\circ f_1$, the Lipschitz constant is at most the product of individual Lipschitz constants: $L \\leq L_L \\cdots L_1$.\n\nThis gives a direct connection to deep networks: if each layer (including nonlinearities) has Lipschitz constant $L_i$, the network's Lipschitz constant is at most $\\prod_i L_i$. ReLU has Lipschitz constant 1 (it never stretches), so the network's Lipschitz constant is controlled entirely by the weight matrices' spectral norms."
    },
    {
      type: "mc",
      question: "A network has 3 linear layers with spectral norms $\\|W_1\\|_2 = 2$, $\\|W_2\\|_2 = 0.5$, $\\|W_3\\|_2 = 3$, with ReLU activations between them. What is the tightest upper bound on the network's Lipschitz constant?",
      options: ["$2 + 0.5 + 3 = 5.5$, the sum of spectral norms since Lipschitz constants add", "$3$, since the Lipschitz constant equals the maximum spectral norm across layers", "$2 \\times 0.5 \\times 3 = 3$, since Lipschitz constants multiply and ReLU contributes 1", "$\\sqrt{2^2 + 0.5^2 + 3^2} \\approx 3.6$, the root sum of squared spectral norms"],
      correct: 2,
      explanation: "Lipschitz constants compose multiplicatively: $L \\leq L_3 \\cdot L_{\\text{relu}} \\cdot L_2 \\cdot L_{\\text{relu}} \\cdot L_1 = 3 \\times 1 \\times 0.5 \\times 1 \\times 2 = 3$. ReLU has Lipschitz constant 1 (it maps $x \\mapsto \\max(0, x)$, which never stretches distances). The product $3$ means the network can amplify input perturbations by at most 3×."
    },
    {
      type: "info",
      title: "Spectral Normalization",
      content: "**Spectral normalization** is a technique that divides a weight matrix by its spectral norm:\n\n$$\\hat{W} = \\frac{W}{\\sigma_1(W)}$$\n\nThis guarantees $\\|\\hat{W}\\|_2 = 1$, bounding the Lipschitz constant of that linear layer to exactly 1.\n\nIn a deep network with $L$ spectrally normalized layers and ReLU activations, the global Lipschitz constant is at most $1^L = 1$ — preventing gradient explosion regardless of depth. The network cannot amplify any signal by more than a factor of 1.\n\nSpectral normalization was introduced by Miyato et al. (2018) for training GANs. The discriminator in a GAN must be Lipschitz-constrained (the original WGAN paper used weight clipping, which was crude and caused training difficulties). Spectral normalization provides a theoretically principled and practically effective alternative: normalize each layer's weight matrix by its spectral norm, and the entire discriminator becomes 1-Lipschitz.\n\nComputing $\\sigma_1(W)$ exactly via SVD every step would be expensive. In practice, the **power iteration** method estimates $\\sigma_1$ with one or two iterations per training step — cheap and accurate enough."
    },
    {
      type: "mc",
      question: "After applying spectral normalization $\\hat{W} = W / \\sigma_1(W)$ to every weight matrix in a network, what is guaranteed about the network's behavior?",
      options: ["The network's Lipschitz constant is bounded by 1, so it cannot amplify input perturbations", "The network becomes linear, since normalizing the spectral norm removes all nonlinear effects", "The network preserves all input distances exactly, acting as an isometry on the input space", "The network's weights become orthogonal matrices, ensuring perfect gradient flow"],
      correct: 0,
      explanation: "Spectral normalization ensures $\\|\\hat{W}\\|_2 = 1$ for each layer. Combined with ReLU (Lipschitz constant 1), the product of Lipschitz constants across all layers is $\\leq 1$. This means $\\|f(x) - f(y)\\| \\leq \\|x - y\\|$ — the network cannot amplify perturbations. It does NOT make the network an isometry (it can still shrink distances) or linear, and the weights are not orthogonal (only their spectral norm is 1)."
    },
    {
      type: "info",
      title: "Spectral Norm in Practice: When and Why",
      content: "Spectral normalization appears in several contexts in modern deep learning:\n\n**GAN training**: The discriminator must be Lipschitz-continuous for Wasserstein-based objectives. Spectral normalization on the discriminator's weight matrices is the standard approach — simpler and more effective than weight clipping or gradient penalty.\n\n**Diffusion models**: Some architectures apply spectral normalization to stabilize the noise prediction network, particularly for high-resolution generation where training instabilities are common.\n\n**Robustness**: A small Lipschitz constant means the network's output changes slowly as the input changes. This provides a degree of robustness to input perturbations — adversarial examples must cross a larger distance to change the output.\n\n**When NOT to use it**: Spectral normalization constrains expressiveness. If every layer has Lipschitz constant $\\leq 1$, the network cannot represent functions that genuinely need to amplify signals. For standard classification or language models, this constraint is too restrictive — you want the network to be sensitive to meaningful input differences. Spectral normalization is primarily used when the Lipschitz constraint is part of the objective (GANs) or when stability is more important than expressiveness."
    },
    {
      type: "mc",
      question: "Why is spectral normalization standard for GAN discriminators but rarely used in language models?",
      options: ["Language models use attention mechanisms, which are incompatible with spectral normalization", "Spectral normalization is too computationally expensive for the large weight matrices in language models", "Language models already achieve Lipschitz bounds through layer normalization, making spectral normalization redundant", "GAN discriminators require a Lipschitz constraint for the training objective, while language models need unrestricted sensitivity to input differences"],
      correct: 3,
      explanation: "Wasserstein GANs require the discriminator to be Lipschitz-continuous — this is a mathematical requirement of the training objective. Spectral normalization enforces this efficiently. Language models have no such requirement: they need to be highly sensitive to input differences (distinguishing \"the cat sat\" from \"the cat sat on\"), so bounding the Lipschitz constant would reduce expressiveness. Layer normalization stabilizes training but does not impose a Lipschitz bound on the overall network."
    }
  ]
};
