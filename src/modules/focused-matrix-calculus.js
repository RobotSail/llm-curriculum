export const matrixCalculusLearning = {
  id: "0.1-matrix-calculus-learning-easy",
  sectionId: "0.1",
  title: "Matrix Calculus for Deep Learning",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 18,
  steps: [
    {
      type: "info",
      title: "The Jacobian: Derivatives Go Matrix-Shaped",
      content: "In scalar calculus, the derivative of $f(x)$ is a single number. But in deep learning, our functions map **vectors to vectors** — e.g., a layer takes an input $x \\in \\mathbb{R}^n$ and produces an output $f(x) \\in \\mathbb{R}^m$. How do we represent the derivative of such a function?\n\nThe answer is the **Jacobian matrix** $J \\in \\mathbb{R}^{m \\times n}$, defined entry-wise as:\n\n$$J_{ij} = \\frac{\\partial f_i}{\\partial x_j}$$\n\nRow $i$ of the Jacobian tells you how the $i$-th output changes as you perturb each input. Column $j$ tells you how all outputs respond to a small change in the $j$-th input. The shape is always **(number of outputs) $\\times$ (number of inputs)** — this is the single most important fact to internalize."
    },
    {
      type: "mc",
      question: "A function $f: \\mathbb{R}^4 \\to \\mathbb{R}^3$ maps a 4-dimensional input to a 3-dimensional output. What is the shape of its Jacobian $J$?",
      options: [
        "$3 \\times 4$ — the Jacobian shape is (output dim) $\\times$ (input dim)",
        "$4 \\times 4$ — the Jacobian is always square, matching the input dimension",
        "$4 \\times 3$ — the Jacobian shape is (input dim) $\\times$ (output dim)",
        "$3 \\times 3$ — the Jacobian is always square, matching the output dimension"
      ],
      correct: 0,
      explanation: "The Jacobian of $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ has shape $m \\times n$, i.e., (output dim) $\\times$ (input dim). Here that's $3 \\times 4$. Each row corresponds to one output component, and each column corresponds to one input component."
    },
    {
      type: "info",
      title: "The Chain Rule in Matrix Calculus",
      content: "Deep learning models are compositions of functions: $h = f \\circ g$, meaning $h(x) = f(g(x))$. The chain rule tells us how to get the Jacobian of the composition from the Jacobians of the parts:\n\n$$J_{f \\circ g}(x) = J_f(g(x)) \\cdot J_g(x)$$\n\nThis is just **matrix multiplication** of the two Jacobians. If $g: \\mathbb{R}^n \\to \\mathbb{R}^k$ and $f: \\mathbb{R}^k \\to \\mathbb{R}^m$, then $J_g$ is $k \\times n$, $J_f$ is $m \\times k$, and their product is $m \\times n$ — exactly right for $f \\circ g: \\mathbb{R}^n \\to \\mathbb{R}^m$.\n\nThis is the mathematical backbone of backpropagation: we compute the Jacobian of the whole network by **multiplying Jacobians layer by layer**, working backward from the loss."
    },
    {
      type: "mc",
      question: "Suppose $g: \\mathbb{R}^{10} \\to \\mathbb{R}^5$ and $f: \\mathbb{R}^5 \\to \\mathbb{R}^2$. You compose them as $h = f \\circ g$. Which matrix multiplication produces $J_h$?",
      options: [
        "$J_g \\cdot J_f$, a $(5 \\times 10)(2 \\times 5)$ product giving a $5 \\times 5$ matrix",
        "$J_f \\cdot J_g$, a $(2 \\times 5)(5 \\times 10)$ product giving a $2 \\times 10$ matrix",
        "$J_g^T \\cdot J_f^T$, a $(10 \\times 5)(5 \\times 2)$ product giving a $10 \\times 2$ matrix",
        "$J_f \\cdot J_g^T$, a $(2 \\times 5)(10 \\times 5)$ product giving a $2 \\times 5$ matrix"
      ],
      correct: 1,
      explanation: "The chain rule says $J_{f \\circ g} = J_f \\cdot J_g$. Here $J_f$ is $2 \\times 5$ and $J_g$ is $5 \\times 10$, so the product is $2 \\times 10$. This matches the expected shape for $h: \\mathbb{R}^{10} \\to \\mathbb{R}^2$. The order matters — the outer function's Jacobian goes on the left."
    },
    {
      type: "info",
      title: "Gradient of a Linear Layer: The Outer Product",
      content: "Consider a single linear layer with no activation: $z = Wx$, where $W \\in \\mathbb{R}^{m \\times n}$, $x \\in \\mathbb{R}^n$, and $z \\in \\mathbb{R}^m$. Suppose we have a scalar loss $L$ and we already know the **upstream gradient** $\\delta = \\nabla_z L \\in \\mathbb{R}^m$ — the gradient of the loss with respect to this layer's output.\n\nHow does $L$ change when we perturb a single weight $W_{ij}$? Since $z_i = \\sum_k W_{ik} x_k$, only $z_i$ depends on $W_{ij}$, and $\\partial z_i / \\partial W_{ij} = x_j$. By the chain rule:\n\n$$\\frac{\\partial L}{\\partial W_{ij}} = \\delta_i \\cdot x_j$$\n\nWritten as a matrix, this is the **outer product**:\n\n$$\\nabla_W L = \\delta \\, x^T$$\n\nThis $m \\times n$ gradient matrix has a beautiful structure: it is the product of a column vector ($\\delta$) and a row vector ($x^T$)."
    },
    {
      type: "mc",
      question: "In a linear layer $z = Wx$ with $W \\in \\mathbb{R}^{64 \\times 128}$, the upstream gradient is $\\delta \\in \\mathbb{R}^{64}$ and the input is $x \\in \\mathbb{R}^{128}$. What is the rank of $\\nabla_W L = \\delta x^T$ for a single training example?",
      options: [
        "Rank 64 — determined by the output dimension since $\\delta$ has 64 entries",
        "Rank 128 — determined by the input dimension since $x$ has 128 entries",
        "Rank 1 — any outer product of two vectors is a rank-1 matrix",
        "Full rank ($\\min(64, 128) = 64$) — the gradient generally has no rank deficiency"
      ],
      correct: 2,
      explanation: "The outer product $\\delta x^T$ of two vectors is always a rank-1 matrix, regardless of the vectors' dimensions. Every row of $\\delta x^T$ is a scalar multiple of $x^T$, and every column is a scalar multiple of $\\delta$. This rank-1 structure is fundamental to understanding gradient updates."
    },
    {
      type: "info",
      title: "Why Gradient Updates Are Low-Rank — and the Link to LoRA",
      content: "We just saw that for a single example, $\\nabla_W L = \\delta x^T$ has **rank 1**. What about a mini-batch of $B$ examples? The batch gradient is an average of outer products:\n\n$$\\nabla_W L = \\frac{1}{B} \\sum_{b=1}^{B} \\delta^{(b)} (x^{(b)})^T$$\n\nA sum of $B$ rank-1 matrices has rank **at most $B$**. So a gradient computed from a batch of 32 examples gives a rank-$\\leq 32$ update to a weight matrix that might be $4096 \\times 4096$.\n\nThis is why **LoRA** (Low-Rank Adaptation) works so well in practice. LoRA constrains weight updates to $\\Delta W = AB$ where $A$ and $B$ are low-rank factors. Since SGD updates are already low-rank, LoRA's constraint is far less restrictive than it first appears — it naturally matches the structure that gradient descent would produce anyway."
    },
    {
      type: "mc",
      question: "You train a large language model layer ($W \\in \\mathbb{R}^{4096 \\times 4096}$) with batch size 64. Each gradient step updates $W$ with $\\nabla_W L$. What is the maximum rank of this gradient update?",
      options: [
        "Rank 4096 — the gradient can span the full space of the weight matrix",
        "Rank 64 — the gradient is a sum of 64 rank-1 outer products, so rank is at most 64",
        "Rank 1 — the batch gradient inherits rank 1 from each individual-example gradient",
        "Rank $\\sqrt{4096} = 64$ — the rank scales with the square root of the matrix dimension"
      ],
      correct: 1,
      explanation: "The batch gradient $\\frac{1}{B}\\sum_{b=1}^B \\delta^{(b)}(x^{(b)})^T$ is a sum of $B = 64$ rank-1 matrices. The rank of a sum of matrices is at most the sum of their ranks, giving a maximum rank of 64. In a $4096 \\times 4096$ space, this is extremely low-rank — which is exactly why LoRA with small rank can capture most of the update."
    },
    {
      type: "info",
      title: "Backprop Through a Linear Layer: Transposed Weights",
      content: "We know how to get $\\nabla_W L$ (the gradient for the weights). But during backpropagation, we also need to **pass the gradient backward** to the previous layer. That means computing $\\nabla_x L$ — how does the loss change with respect to the layer's input $x$?\n\nStarting from $z = Wx$ and using the chain rule component-wise:\n\n$$\\frac{\\partial L}{\\partial x_j} = \\sum_i \\frac{\\partial L}{\\partial z_i} \\frac{\\partial z_i}{\\partial x_j} = \\sum_i \\delta_i W_{ij}$$\n\nThat sum is exactly the $j$-th element of $W^T \\delta$. In vector form:\n\n$$\\nabla_x L = W^T \\delta$$\n\nThis is the core operation of backpropagation through a linear layer: **multiply by the transposed weight matrix**. The forward pass computes $z = Wx$; the backward pass computes $\\nabla_x L = W^T \\nabla_z L$. This elegant symmetry is why neural network training is computationally tractable."
    },
    {
      type: "mc",
      question: "During backpropagation through a linear layer $z = Wx$ (where $W \\in \\mathbb{R}^{256 \\times 512}$), you have the upstream gradient $\\delta = \\nabla_z L \\in \\mathbb{R}^{256}$. What operation produces the gradient to pass to the previous layer?",
      options: [
        "$W \\delta$, which multiplies the $256 \\times 512$ weight matrix by the $256$-dim gradient",
        "$\\delta^T W$, which left-multiplies the weight matrix by the gradient as a row vector",
        "$W^T \\delta$, which multiplies the $512 \\times 256$ transposed weight matrix by the $256$-dim gradient",
        "$\\delta W^T$, which right-multiplies the transposed weight matrix by the gradient vector"
      ],
      correct: 2,
      explanation: "The backward pass computes $\\nabla_x L = W^T \\delta$. Here $W^T$ is $512 \\times 256$ and $\\delta$ is $256$-dimensional, giving a $512$-dimensional result — matching the input dimension $x \\in \\mathbb{R}^{512}$. Note that $W\\delta$ would not even be dimensionally valid since $W$ is $256 \\times 512$ and $\\delta$ is $256 \\times 1$."
    },
    {
      type: "info",
      title: "The Hessian: Second-Order Information",
      content: "The gradient tells us the **direction** of steepest descent, but it says nothing about **curvature** — how quickly the gradient itself changes. That information lives in the **Hessian** matrix.\n\nFor a scalar loss $L(\\theta)$ with parameters $\\theta \\in \\mathbb{R}^n$, the Hessian is the $n \\times n$ matrix of second partial derivatives:\n\n$$H_{ij} = \\frac{\\partial^2 L}{\\partial \\theta_i \\, \\partial \\theta_j}$$\n\nKey properties of the Hessian in deep learning:\n\n**At a local minimum**, the Hessian is **positive semi-definite** (PSD) — all eigenvalues $\\geq 0$. Directions with large eigenvalues are \"steep valleys\" where the loss curves sharply; directions with eigenvalues near zero are **flat directions** where you can move without much change in loss.\n\nNeural network loss surfaces have many near-zero Hessian eigenvalues. This means most of the parameter space is \"flat\" — the loss is insensitive to perturbations in those directions. This abundance of flat directions is closely related to why overparameterized models can generalize: many different parameter configurations achieve similar loss."
    },
    {
      type: "mc",
      question: "A neural network has 1 million parameters. At a local minimum, you compute the Hessian and find that 999,000 eigenvalues are near zero and 1,000 eigenvalues are large and positive. What does this tell you about the loss landscape?",
      options: [
        "The minimum is unstable — near-zero eigenvalues indicate saddle-point directions where the loss can decrease further",
        "The model is underfitting — most parameter directions being flat means the model has not learned enough structure from the data",
        "The loss is sensitive to most parameters — near-zero eigenvalues mean the gradient is nearly zero, so small steps cause large loss changes",
        "The loss is effectively determined by roughly 1,000 parameter directions — the other 999,000 directions are flat and can be perturbed without significantly changing the loss"
      ],
      correct: 3,
      explanation: "Near-zero Hessian eigenvalues at a local minimum mean those directions are flat: the loss barely changes as you move along them. Only the 1,000 directions with large eigenvalues significantly affect the loss. At a local minimum (not a saddle point), near-zero eigenvalues are $\\geq 0$, so they indicate flat regions, not descent directions. This low-dimensional structure of the loss landscape is a key insight in deep learning theory."
    }
  ]
};
