// Module: Matrix Calculus for Deep Learning
// Section 0.1: Jacobians, chain rule, gradients of linear layers
// Single-concept module: matrix calculus and backprop through linear maps

export const matrixCalculusLearning = {
  id: "0.1-matcalc-learning-easy",
  sectionId: "0.1",
  title: "Matrix Calculus for Deep Learning",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 15,
  steps: [
    {
      type: "info",
      title: "Matrix Calculus: Gradients Through Neural Networks",
      content: "**Matrix calculus** is the language of backpropagation. Every gradient computation in a neural network reduces to derivatives of matrix and vector expressions.\n\nThe key objects:\n- **Jacobian** $J \\in \\mathbb{R}^{m \\times n}$: the matrix of all partial derivatives $J_{ij} = \\partial f_i / \\partial x_j$ for a function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$\n- **Chain rule**: for $z = f(g(x))$, the Jacobian is $J_f \\cdot J_g$ — backprop chains these products\n- **Key identity**: for a linear layer $z = Wx$ with scalar loss $L$, $\\nabla_W L = (\\nabla_z L) x^\\top$ (outer product)\n\nThis module tests your ability to derive and reason about gradients through linear maps — the building block of every neural network layer."
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
      question: "For a linear layer $z = Wx$ with $W \\in \\mathbb{R}^{m \\times n}$, $x \\in \\mathbb{R}^n$, scalar loss $L$. The upstream gradient is $\\delta = \\nabla_z L \\in \\mathbb{R}^m$. What is $\\nabla_W L$?",
      options: ["$\\delta^\\top x$ — the inner product of the upstream gradient and input", "$W^\\top \\delta$ — the transposed weight matrix times the upstream gradient", "$x \\delta^\\top$ — the outer product of the input and upstream gradient", "$\\delta x^\\top$ — the outer product of the upstream gradient and input"],
      correct: 3,
      explanation: "$\\nabla_W L = \\delta x^\\top \\in \\mathbb{R}^{m \\times n}$. This is an outer product: the gradient with respect to each weight $W_{ij}$ is $\\delta_i x_j$, since $\\partial z_i / \\partial W_{ij} = x_j$. This outer-product structure explains why gradient updates are low-rank (rank 1 per sample, rank $\\leq$ batch size for a mini-batch) — the foundation of why LoRA works."
    },
    {
      type: "mc",
      question: "The gradient of the loss $L$ with respect to the input $x$ of a linear layer $z = Wx$ is:",
      options: ["$\\nabla_x L = W^\\top \\nabla_z L$ — multiply the transposed weight matrix by the upstream gradient", "$\\nabla_x L = \\nabla_z L \\cdot W$ — multiply the upstream gradient row vector by the weight matrix", "$\\nabla_x L = W \\nabla_z L$ — multiply the weight matrix directly by the upstream gradient vector", "$\\nabla_x L = \\nabla_z L \\cdot W^\\top$ — multiply the upstream gradient by the transposed weight matrix"],
      correct: 0,
      explanation: "By the chain rule, $(\\nabla_x L)_j = \\sum_i (\\nabla_z L)_i \\frac{\\partial z_i}{\\partial x_j} = \\sum_i (\\nabla_z L)_i W_{ij}$. In matrix form this is $W^\\top (\\nabla_z L)$. The transpose of $W$ appears naturally because backprop reverses the direction of information flow — the Jacobian of $z = Wx$ w.r.t. $x$ is $W$, and its transpose appears in the vector-Jacobian product. This is why every backward pass through a linear layer is another matrix multiply with the transposed weight."
    },
    {
      type: "info",
      title: "Summary: The Two Key Gradient Identities",
      content: "For a linear layer $z = Wx$ with scalar loss $L$ and upstream gradient $\\delta = \\nabla_z L$:\n\n$$\\nabla_W L = \\delta x^\\top \\quad \\text{(outer product — for weight update)}$$\n$$\\nabla_x L = W^\\top \\delta \\quad \\text{(matrix-vector product — for backprop to previous layer)}$$\n\nThese two identities are the **entire backward pass** for a linear layer. Every other layer's backward pass (attention, LayerNorm, convolutions) builds on these by applying the chain rule through more complex expressions."
    }
  ]
};
