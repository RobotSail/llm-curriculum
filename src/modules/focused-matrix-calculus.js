// Module: Matrix Calculus for Deep Learning
// Section 0.1: Jacobians, Hessians, chain rule, gradients of linear layers
// Single-concept module covering the calculus needed for backpropagation
// Proper learning module with alternating info/mc steps

export const matrixCalculusLearning = {
  id: "0.1-matcalc-learning-easy",
  sectionId: "0.1",
  title: "Matrix Calculus for Deep Learning",
  difficulty: "easy",
  moduleType: "learning",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "Why Matrix Calculus?",
      content: "In single-variable calculus, a derivative $\\frac{df}{dx}$ tells you how a scalar output changes when a scalar input changes. But in deep learning, both inputs and outputs are **vectors or matrices**. Matrix calculus extends derivatives to these higher-dimensional objects.\n\nThe central question: if you have a function that takes a vector (or matrix) as input and produces a vector (or scalar) as output, how do you express \"the derivative\" of this function?\n\nThis matters because **backpropagation** — the algorithm that trains every neural network — is just the chain rule of matrix calculus applied repeatedly through the network's layers. Understanding matrix calculus means understanding exactly what backprop computes and why."
    },
    {
      type: "info",
      title: "The Jacobian Matrix",
      content: "For a function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ that maps an $n$-dimensional input to an $m$-dimensional output, the **Jacobian** $J \\in \\mathbb{R}^{m \\times n}$ collects all partial derivatives:\n\n$$J_{ij} = \\frac{\\partial f_i}{\\partial x_j}$$\n\nRow $i$ contains the partial derivatives of the $i$-th output with respect to all inputs. Column $j$ contains the partial derivatives of all outputs with respect to the $j$-th input.\n\nSpecial cases you already know:\n- When $m = 1$ (scalar output): the Jacobian is a $1 \\times n$ row vector — the familiar **gradient** $\\nabla f$\n- When $n = 1$ (scalar input): the Jacobian is an $m \\times 1$ column vector — an ordinary vector derivative\n- When $m = n = 1$: the Jacobian is a $1 \\times 1$ scalar — the ordinary derivative $f'(x)$\n\nThe Jacobian generalizes all of these into a single framework."
    },
    {
      type: "mc",
      question: "A function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ has a Jacobian matrix $J$. What is the shape of $J$?",
      options: ["$m \\times n$ — rows index outputs, columns index inputs", "$n \\times n$ — the Jacobian is always a square matrix", "$n \\times m$ — rows index inputs, columns index outputs", "$m \\times m$ — the shape depends only on the output dimension"],
      correct: 0,
      explanation: "The Jacobian $J \\in \\mathbb{R}^{m \\times n}$ has entry $J_{ij} = \\partial f_i / \\partial x_j$. Each of the $m$ rows corresponds to one output dimension; each of the $n$ columns corresponds to one input dimension. When $m = 1$ (scalar output), this reduces to the gradient $\\nabla f \\in \\mathbb{R}^{1 \\times n}$ (a row vector). The shape follows the convention: outputs determine rows, inputs determine columns."
    },
    {
      type: "info",
      title: "The Chain Rule for Vector Functions",
      content: "The chain rule generalizes beautifully to vector functions. If $z = f(g(x))$ where $g: \\mathbb{R}^n \\to \\mathbb{R}^k$ and $f: \\mathbb{R}^k \\to \\mathbb{R}^m$, the Jacobian of the composition is the **matrix product** of the individual Jacobians:\n\n$$J_{f \\circ g} = J_f \\cdot J_g$$\n\nDimensions check: $J_f$ is $m \\times k$, $J_g$ is $k \\times n$, so $J_{f \\circ g}$ is $m \\times n$ — exactly the Jacobian of the composite function from $\\mathbb{R}^n$ to $\\mathbb{R}^m$.\n\nBackpropagation uses this in reverse. For a network with layers $x \\to h_1 \\to h_2 \\to \\cdots \\to L$ (where $L$ is a scalar loss), the gradient of $L$ with respect to $x$ is:\n\n$$\\nabla_x L = \\nabla_{h_1} L \\cdot J_1 = \\nabla_{h_2} L \\cdot J_2 \\cdot J_1 = \\cdots$$\n\nEach layer contributes a Jacobian to the chain. This is why layer-by-layer Jacobians control gradient flow — if any Jacobian has large singular values, gradients explode; if small, they vanish."
    },
    {
      type: "mc",
      question: "A network computes $h = g(x)$ then $L = f(h)$, where $g: \\mathbb{R}^5 \\to \\mathbb{R}^3$ and $f: \\mathbb{R}^3 \\to \\mathbb{R}$. The Jacobian $J_g$ has shape $3 \\times 5$ and $J_f$ has shape $1 \\times 3$. What is the shape of $\\nabla_x L$?",
      options: ["$5 \\times 5$ — square since both the gradient and input live in $\\mathbb{R}^5$", "$1 \\times 5$ — the chain rule gives $J_f \\cdot J_g$ with shape $(1 \\times 3)(3 \\times 5) = 1 \\times 5$", "$3 \\times 5$ — the gradient has the same shape as $J_g$ since that is the first layer", "$1 \\times 3$ — the gradient has the same shape as $J_f$ since that is the outer function"],
      correct: 1,
      explanation: "By the chain rule, $J_{f \\circ g} = J_f \\cdot J_g$. Multiplying $(1 \\times 3)(3 \\times 5) = 1 \\times 5$. This is the Jacobian of the composite $L = f(g(x))$, which is a scalar function of 5 inputs — so it is a $1 \\times 5$ row vector, the gradient $\\nabla_x L$. The inner dimension (3, the intermediate space) cancels in the matrix product."
    },
    {
      type: "info",
      title: "The Forward Pass: Linear Layer $z = Wx$",
      content: "The simplest and most common operation in deep learning is a **linear layer**: given a weight matrix $W \\in \\mathbb{R}^{m \\times n}$ and input $x \\in \\mathbb{R}^n$, compute:\n\n$$z = Wx \\in \\mathbb{R}^m$$\n\nThe $i$-th output is $z_i = \\sum_{j=1}^n W_{ij} x_j$.\n\nThis is the building block of every neural network layer (the nonlinearity and bias are applied afterward). To train the network, we need two gradients:\n\n1. **$\\nabla_W L$**: How does the loss change when we change each weight? (Used to update the weights)\n2. **$\\nabla_x L$**: How does the loss change when the input changes? (Used to continue backprop to the previous layer)\n\nBoth follow directly from the chain rule applied to $z = Wx$."
    },
    {
      type: "info",
      title: "Gradient with Respect to the Weights",
      content: "For a linear layer $z = Wx$ with scalar loss $L$, the upstream gradient from later layers is $\\delta = \\nabla_z L \\in \\mathbb{R}^m$ (a column vector of how much $L$ changes per unit change in each $z_i$).\n\nTo find $\\nabla_W L$, note that $\\frac{\\partial z_i}{\\partial W_{ij}} = x_j$ (since $z_i = \\sum_j W_{ij} x_j$). By the chain rule:\n\n$$\\frac{\\partial L}{\\partial W_{ij}} = \\frac{\\partial L}{\\partial z_i} \\cdot \\frac{\\partial z_i}{\\partial W_{ij}} = \\delta_i \\cdot x_j$$\n\nIn matrix form, this is the **outer product**:\n\n$$\\nabla_W L = \\delta x^\\top \\in \\mathbb{R}^{m \\times n}$$\n\nThis is an important structural insight: the gradient of the loss with respect to the weights is always a rank-1 outer product (for a single input). For a mini-batch of $B$ samples, the gradient is a sum of $B$ rank-1 terms, so it has rank at most $B$. This intrinsic low-rank structure of gradients is one reason why low-rank methods like LoRA work — weight updates are naturally low-rank."
    },
    {
      type: "mc",
      question: "For a linear layer $z = Wx$ with $W \\in \\mathbb{R}^{m \\times n}$, $x \\in \\mathbb{R}^n$, scalar loss $L$. The upstream gradient is $\\delta = \\nabla_z L \\in \\mathbb{R}^m$. What is $\\nabla_W L$?",
      options: ["$\\delta^\\top x$ — the inner product of the upstream gradient and input, producing a scalar", "$W^\\top \\delta$ — the transposed weight matrix times the upstream gradient, an $n$-vector", "$x \\delta^\\top$ — the outer product of the input and upstream gradient, an $n \\times m$ matrix", "$\\delta x^\\top$ — the outer product of the upstream gradient and input, an $m \\times n$ matrix"],
      correct: 3,
      explanation: "$\\nabla_W L = \\delta x^\\top \\in \\mathbb{R}^{m \\times n}$. The gradient with respect to each weight $W_{ij}$ is $\\delta_i x_j$. This is the outer product of $\\delta$ (size $m$) and $x$ (size $n$), matching the shape of $W$. Note that $x \\delta^\\top$ would have the wrong shape ($n \\times m$ instead of $m \\times n$)."
    },
    {
      type: "info",
      title: "Gradient with Respect to the Input (Backprop Signal)",
      content: "The second gradient we need is $\\nabla_x L$ — this is what gets passed back to the previous layer during backpropagation.\n\nSince $z_i = \\sum_j W_{ij} x_j$, we have $\\frac{\\partial z_i}{\\partial x_j} = W_{ij}$. The Jacobian of $z = Wx$ with respect to $x$ is simply $W$ itself.\n\nBy the chain rule:\n\n$$(\\nabla_x L)_j = \\sum_i \\frac{\\partial L}{\\partial z_i} \\cdot \\frac{\\partial z_i}{\\partial x_j} = \\sum_i \\delta_i \\cdot W_{ij}$$\n\nIn matrix form:\n\n$$\\nabla_x L = W^\\top \\delta$$\n\nThe **transpose** of the weight matrix appears naturally. This is the key insight of backpropagation: the forward pass computes $z = Wx$, and the backward pass computes $\\nabla_x L = W^\\top \\delta$ — the same matrix, transposed. Every backward pass through a linear layer is just another matrix multiplication."
    },
    {
      type: "mc",
      question: "The gradient of the loss $L$ with respect to the input $x$ of a linear layer $z = Wx$ is:",
      options: ["$\\nabla_x L = W^\\top \\nabla_z L$ — the transposed weight matrix times the upstream gradient", "$\\nabla_x L = \\nabla_z L \\cdot W$ — the upstream gradient row vector times the weight matrix", "$\\nabla_x L = W \\nabla_z L$ — the weight matrix directly times the upstream gradient vector", "$\\nabla_x L = \\nabla_z L \\cdot W^\\top$ — the upstream gradient times the transposed weight matrix"],
      correct: 0,
      explanation: "$(\\nabla_x L)_j = \\sum_i (\\nabla_z L)_i W_{ij}$. In matrix form this is $W^\\top (\\nabla_z L)$. The transpose appears because backprop reverses the direction of information flow — the forward pass multiplies by $W$, and the backward pass multiplies by $W^\\top$. Option C ($W \\delta$) has the wrong shape: $W$ is $m \\times n$ and $\\delta$ is $m \\times 1$, so $W\\delta$ doesn't conform."
    },
    {
      type: "info",
      title: "The Hessian Matrix",
      content: "The **Hessian** $H$ is the matrix of second derivatives. For a scalar function $L: \\mathbb{R}^n \\to \\mathbb{R}$:\n\n$$H_{ij} = \\frac{\\partial^2 L}{\\partial x_i \\partial x_j}$$\n\nThe Hessian is always **symmetric** ($H_{ij} = H_{ji}$, since mixed partial derivatives commute), so it has a real eigendecomposition $H = Q \\Lambda Q^\\top$.\n\nThe Hessian captures the **curvature** of the loss landscape:\n- An eigenvalue $\\lambda_i > 0$ means the loss curves **upward** along eigenvector $q_i$ (a \"valley wall\")\n- An eigenvalue $\\lambda_i < 0$ means the loss curves **downward** along $q_i$ (a \"hill\")\n- An eigenvalue $\\lambda_i \\approx 0$ means the loss is nearly **flat** along $q_i$ (a \"plateau\")\n\nAt a local minimum, all eigenvalues must be $\\geq 0$ (the Hessian is PSD). A mix of positive and negative eigenvalues indicates a **saddle point**. In deep learning, saddle points are far more common than local minima — most critical points in high dimensions are saddles."
    },
    {
      type: "mc",
      question: "A loss function $L(\\theta)$ has Hessian $H = \\nabla^2 L$ at a critical point (where $\\nabla L = 0$). The Hessian has eigenvalues $\\{5, 3, 0.1, -0.5\\}$. What kind of critical point is this?",
      options: ["A local minimum, because most eigenvalues are positive", "A local maximum, because there is a negative eigenvalue present", "A saddle point, because there are both positive and negative eigenvalues", "Indeterminate, because the Hessian eigenvalues are insufficient to classify critical points"],
      correct: 2,
      explanation: "The presence of both positive ($5, 3, 0.1$) and negative ($-0.5$) eigenvalues means the loss curves upward in some directions and downward in others — this is the definition of a saddle point. A local minimum requires all eigenvalues $\\geq 0$; a local maximum requires all $\\leq 0$. The single negative eigenvalue means there exists a direction along which the loss decreases, so this cannot be a minimum."
    },
    {
      type: "info",
      title: "Summary: The Two Key Identities",
      content: "For a linear layer $z = Wx$ with scalar loss $L$ and upstream gradient $\\delta = \\nabla_z L$:\n\n$$\\nabla_W L = \\delta x^\\top \\quad \\text{(outer product — for weight update)}$$\n$$\\nabla_x L = W^\\top \\delta \\quad \\text{(matrix-vector product — for backprop to previous layer)}$$\n\nThese two identities are the **entire backward pass** for a linear layer. Every other layer's backward pass (attention, normalization, convolutions) builds on these by applying the chain rule through more complex expressions.\n\nThe key takeaways:\n- The Jacobian organizes all partial derivatives into a matrix that chains cleanly via matrix multiplication\n- Weight gradients are outer products (inherently low-rank)\n- Input gradients use the transposed weight matrix (forward = $W$, backward = $W^\\top$)\n- The Hessian captures curvature; its eigenvalues classify critical points"
    }
  ]
};
