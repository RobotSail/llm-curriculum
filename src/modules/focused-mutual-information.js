// Focused module: Mutual Information
// Covers MI definition, properties, representation learning, data processing
// inequality, chain rule of MI, contrastive learning bounds, and MI estimation.

export const mutualInformationLearning = {
  id: "0.2-mutual-information-learning-medium",
  sectionId: "0.2",
  title: "Mutual Information",
  difficulty: "medium",
  moduleType: "learning",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Mutual Information: Shared Uncertainty",
      content: "**Mutual information** quantifies the information shared between two random variables:\n\n$$I(X; Y) = H(X) - H(X \\mid Y) = H(Y) - H(Y \\mid X)$$\n\nEquivalently, it is the KL divergence between the joint and the product of marginals:\n\n$$I(X; Y) = \\text{KL}\\big(P(X, Y) \\,\\|\\, P(X) P(Y)\\big) = \\sum_{x, y} P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}$$\n\nKey properties:\n- **Non-negative**: $I(X; Y) \\geq 0$, with equality iff $X \\perp Y$.\n- **Symmetric**: $I(X; Y) = I(Y; X)$. Knowing $X$ reduces uncertainty about $Y$ by the same amount that knowing $Y$ reduces uncertainty about $X$.\n- **Bounded**: $I(X; Y) \\leq \\min(H(X), H(Y))$. You can't learn more about $X$ from $Y$ than there is uncertainty in $X$ to begin with.\n\n$I(X; Y) = 0$ means the variables are independent — knowing one tells you nothing about the other. $I(X; Y) = H(X)$ means $X$ is fully determined by $Y$."
    },
    {
      type: "mc",
      question: "If $X$ and $Y$ are independent, what is $I(X; Y)$?",
      options: ["$H(X) + H(Y)$, since MI equals the sum of individual entropies", "$H(X) \\cdot H(Y)$, since MI equals the product of individual entropies", "$-\\infty$, because the log-ratio in the KL formulation is undefined", "$0$, because $H(X \\mid Y) = H(X)$ when $X$ and $Y$ are independent"],
      correct: 3,
      explanation: "When $X \\perp Y$: $P(X, Y) = P(X)P(Y)$, so the KL divergence $\\text{KL}(P(X,Y) \\| P(X)P(Y)) = 0$. Equivalently, $H(X \\mid Y) = H(X)$ — observing $Y$ doesn't reduce uncertainty about $X$ — so $I(X;Y) = H(X) - H(X) = 0$. Independence means zero shared information."
    },
    {
      type: "info",
      title: "MI in Representation Learning",
      content: "In representation learning, an encoder maps input $X$ to representation $Z = f_\\theta(X)$. Two MI quantities matter:\n\n**$I(X; Z)$**: How much the representation remembers about the input. A lossless encoder has $I(X; Z) = H(X)$; a constant encoder has $I(X; Z) = 0$.\n\n**$I(Z; Y)$**: How useful the representation is for predicting target $Y$. This is what task performance depends on.\n\nThe **data processing inequality** (DPI) constrains what's possible: for any Markov chain $Y \\to X \\to Z$:\n\n$$I(Y; Z) \\leq I(Y; X)$$\n\nYou can never create information that wasn't in the input. The representation $Z$ can be *at most* as informative about $Y$ as the raw input $X$ is.\n\nThe **information bottleneck** objective formalizes the trade-off: minimize $I(X; Z)$ (compress) while maximizing $I(Z; Y)$ (preserve task signal). A good representation discards task-irrelevant information (low $I(X; Z)$ relative to $H(X)$) while retaining everything the task needs (high $I(Z; Y)$)."
    },
    {
      type: "mc",
      question: "A representation $Z = f(X)$ is a deterministic function of $X$. Can $I(X; Z)$ be zero?",
      options: ["Yes — if $f$ is a constant function that maps every input to the same value, then $I(X; Z) = 0$", "No — any deterministic function of $X$ always preserves at least some information about $X$", "Yes — if $f$ is a non-invertible hash function, the MI is always exactly zero", "It depends on the dimensionality of $Z$ relative to the dimensionality of $X$"],
      correct: 0,
      explanation: "$I(X; Z) = 0$ requires $X \\perp Z$, meaning $Z$ carries no information about $X$. For a deterministic $Z = f(X)$, this happens only if $f$ is constant — $Z$ takes the same value regardless of input. Any non-constant deterministic function has $I(X; Z) > 0$, because knowing $Z$ eliminates at least some uncertainty about $X$. A hash function that maps different inputs to different outputs would actually have high MI, even though it's not invertible."
    },
    {
      type: "mc",
      question: "Two random variables $X$ and $Y$ have correlation $\\rho = 0$ but are clearly dependent (e.g., $Y = X^2$ where $X \\sim \\text{Uniform}(-1, 1)$). What is $I(X; Y)$?",
      options: ["$I(X; Y) = 0$ because zero correlation always implies statistical independence between variables", "$I(X; Y) = 0$ because mutual information is proportional to the square of the correlation coefficient", "$I(X; Y) > 0$ because MI detects all dependencies including nonlinear ones that correlation misses", "$I(X; Y) < 0$ because the nonlinear relationship between the variables introduces negative information"],
      correct: 2,
      explanation: "MI is zero if and only if $X$ and $Y$ are independent. Zero correlation only rules out *linear* dependence. If $Y = X^2$ and $X$ is symmetric around 0, then $\\text{Corr}(X, Y) = 0$ but $Y$ is completely determined by $X$, so $I(X; Y) = H(Y) > 0$. This is a key advantage of MI over correlation: it captures arbitrary statistical dependencies, making it a more principled measure for representation learning."
    },
    {
      type: "mc",
      question: "In the information bottleneck framework, you want to find a representation $Z$ of input $X$ that maximizes $I(Z; Y) - \\beta \\cdot I(Z; X)$ for task label $Y$. What happens as $\\beta \\to 0$?",
      options: ["The representation becomes maximally compressed, collapsing $Z$ to a constant regardless of input", "The representation retains all input information, with $Z$ becoming a copy of $X$ and no compression", "The representation degenerates to random noise, since the objective has no compression penalty", "The optimization becomes infeasible, since without compression the search space is unbounded"],
      correct: 1,
      explanation: "As $\\beta \\to 0$, the compression penalty $\\beta \\cdot I(Z; X)$ vanishes, so the objective reduces to maximizing $I(Z; Y)$ alone. The optimal solution retains all information from $X$ (no compression) to maximize task-relevant information. As $\\beta$ increases, the compression penalty becomes stronger, forcing $Z$ to discard task-irrelevant details from $X$. This is the information bottleneck tradeoff: $\\beta$ controls the compression-relevance balance."
    },
    {
      type: "info",
      title: "Data Processing Inequality: Information Only Flows Downhill",
      content: "The **data processing inequality** (DPI) is one of the most powerful results in information theory:\n\nIf $X \\to Y \\to Z$ is a Markov chain (i.e., $Z$ depends on $X$ only through $Y$), then:\n\n$$I(X; Z) \\leq I(X; Y)$$\n\nEach processing step can only **lose** information, never create it.\n\nIn a neural network, the layers form a Markov chain:\n\n$$X \\to h_1 \\to h_2 \\to \\cdots \\to h_L \\to \\hat{Y}$$\n\nSo DPI gives us: $I(X; h_1) \\geq I(X; h_2) \\geq \\cdots \\geq I(X; h_L)$. Deeper layers can only have *less* mutual information with the input.\n\nThis has a striking interpretation: as information flows through the network, each layer acts as an information bottleneck. Early layers retain most input information; deeper layers progressively discard task-irrelevant details. The **information plane** plots $I(X; h_l)$ vs. $I(Y; h_l)$ for each layer $l$ — a well-trained network should show layers moving toward the bottom-right: low input MI (compressed) but high task MI (useful).\n\n*Caveat*: For deterministic networks with invertible activations (like ReLU on distinct inputs), DPI is technically satisfied with equality. The \"information compression\" story is nuanced and debated."
    },
    {
      type: "mc",
      question: "Layer 3 of a network has $I(X; h_3) = 5$ bits. What can you say about $I(X; h_5)$ for layer 5?",
      options: ["$I(X; h_5) = 5$ bits — information is conserved as it passes through deterministic layers", "$I(X; h_5) \\geq 5$ bits — deeper layers learn richer representations that capture more input detail", "$I(X; h_5) \\leq 5$ bits — the data processing inequality prevents information from increasing", "Nothing definitive — MI between non-adjacent layers is not constrained by the DPI at all"],
      correct: 2,
      explanation: "Since $X \\to h_3 \\to h_4 \\to h_5$ is a Markov chain, DPI gives $I(X; h_5) \\leq I(X; h_3) = 5$ bits. Each additional layer can only preserve or lose information about the input. The *useful* information $I(Y; h_5)$ might be concentrated and refined, but the *total* input information $I(X; h_5)$ cannot exceed what layer 3 retained."
    },
    {
      type: "info",
      title: "Chain Rule of Mutual Information",
      content: "Just as entropy has a chain rule, so does MI:\n\n$$I(X; Y, Z) = I(X; Y) + I(X; Z \\mid Y)$$\n\nThe information $X$ shares with $(Y, Z)$ jointly equals the information from $Y$ alone plus the **additional** information $Z$ provides once $Y$ is already known.\n\nHere, the conditional MI is:\n\n$$I(X; Z \\mid Y) = H(X \\mid Y) - H(X \\mid Y, Z)$$\n\nThis decomposition is essential for understanding what different parts of the context contribute in language modeling.\n\nConsider a language model with recent tokens $Y$ and distant tokens $Z$ in the context window. The chain rule tells us:\n\n$$I(W_{\\text{next}}; Y, Z) = I(W_{\\text{next}}; Y) + I(W_{\\text{next}}; Z \\mid Y)$$\n\nThe second term $I(W_{\\text{next}}; Z \\mid Y)$ measures how much *additional* prediction signal the distant context provides beyond what local context already gives. If this is near zero, the distant tokens are redundant for prediction — they add nothing beyond what nearby context already captures."
    },
    {
      type: "mc",
      question: "In a language model's context window, $Y$ = recent tokens and $Z$ = distant tokens. If $I(W_{\\text{next}}; Z \\mid Y) \\approx 0$, what does this mean?",
      options: [
        "The distant tokens $Z$ contain no useful information for predicting $W_{\\text{next}}$ in any context",
        "The distant context $Z$ adds no prediction signal beyond what the recent context $Y$ already provides",
        "The context window is too short to capture any meaningful long-range dependencies in the sequence",
        "The model should increase its attention weights on distant tokens to extract their unused signal"
      ],
      correct: 1,
      explanation: "$I(W_{\\text{next}}; Z \\mid Y) \\approx 0$ means $H(W_{\\text{next}} \\mid Y) \\approx H(W_{\\text{next}} \\mid Y, Z)$ — the residual uncertainty about the next token is the same whether or not you have distant context, *given that you already have recent context*. The distant tokens are redundant for prediction, not necessarily irrelevant in isolation. This is why many sequences can be predicted well with short contexts — and why efficient architectures that limit long-range attention (e.g., sliding window attention) often lose little performance."
    },
    {
      type: "info",
      title: "Contrastive Learning Maximizes MI Bounds",
      content: "Contrastive methods like **CLIP** and **SimCLR** learn representations by maximizing mutual information between paired views.\n\nCLIP trains image encoder $f$ and text encoder $g$ so that matching (image, caption) pairs have high similarity while non-matching pairs have low similarity. The **InfoNCE** loss for a batch of $N$ pairs is:\n\n$$\\mathcal{L}_{\\text{InfoNCE}} = -\\frac{1}{N} \\sum_{i=1}^N \\log \\frac{\\exp(\\text{sim}(f(x_i), g(t_i)) / \\tau)}{\\sum_{j=1}^N \\exp(\\text{sim}(f(x_i), g(t_j)) / \\tau)}$$\n\nOord et al. (2018) showed this loss satisfies:\n\n$$I(X; Y) \\geq \\log N - \\mathcal{L}_{\\text{InfoNCE}}$$\n\nSo minimizing InfoNCE maximizes a **lower bound** on $I(X; Y)$. The bound's tightness depends on batch size $N$: the maximum estimable MI is $\\log N$. This is why contrastive methods use large batches — a batch of 32,768 allows estimating up to $\\log(32768) = 15$ bits of MI, while a batch of 64 caps you at $\\log(64) = 6$ bits.\n\nThis also explains why CLIP's performance scales with batch size: larger batches give a tighter bound on the true MI, enabling the model to capture more fine-grained image-text correspondences."
    },
    {
      type: "mc",
      question: "In CLIP training with batch size $N = 1024$, what is the maximum mutual information $I(\\text{image}; \\text{text})$ that the InfoNCE loss can estimate?",
      options: ["$1024$ bits — the bound scales linearly with $N$", "$\\sqrt{1024} = 32$ bits — the bound scales as $\\sqrt{N}$", "$1024 \\cdot \\log_2(1024) = 10240$ bits — the bound scales as $N \\log N$", "$\\log_2(1024) = 10$ bits — the bound scales as $\\log N$"],
      correct: 3,
      explanation: "The InfoNCE bound is $I(X; Y) \\geq \\log N - \\mathcal{L}$. Even when the loss is driven to zero, the bound saturates at $\\log N$. With $N = 1024$: $\\log_2(1024) = 10$ bits. If the true MI exceeds 10 bits, the InfoNCE loss with this batch size simply cannot distinguish — the bound is too loose. This is why CLIP uses batch sizes of 32K+: $\\log_2(32768) = 15$ bits, allowing the model to capture finer-grained correspondences."
    },
    {
      type: "info",
      title: "The Challenge of MI Estimation",
      content: "Mutual information is conceptually clean but **computationally hard** to estimate in high dimensions.\n\nFor discrete distributions with small support, you can compute MI exactly from counts. But for continuous or high-dimensional variables (like neural network representations), MI estimation is a research problem in itself.\n\nThe core difficulty: MI involves a ratio of joint to marginal densities, $\\log \\frac{P(x,y)}{P(x)P(y)}$, and estimating densities in high dimensions is exponentially hard.\n\nVariational approaches provide bounds:\n- **InfoNCE** (lower bound): tractable but limited by batch size.\n- **MINE** (Mutual Information Neural Estimation): uses a learned critic network to estimate a tight lower bound, but has high variance.\n- **BA bound** (Barber-Agakov): provides an upper bound using a learned conditional $Q(X \\mid Z)$.\n\nA key result from McAllester & Statos (2020): **any MI estimator that provides a high-confidence lower bound on MI must have sample complexity exponential in the true MI value.** In other words, the harder the problem (higher true MI), the more data you need to confirm it — and the relationship is exponential, not polynomial."
    },
    {
      type: "mc",
      question: "Why is estimating MI between a 768-dimensional representation and its input fundamentally hard?",
      options: ["Reliable MI estimation requires sample complexity that grows exponentially with the true MI value", "MI is not well-defined for continuous random variables and requires discretization to compute", "768 dimensions exceeds the capacity of any neural network to process as a density estimation input", "The representation must first be discretized into bins before MI can be meaningfully computed"],
      correct: 0,
      explanation: "MI is well-defined for continuous variables (as a KL divergence between densities), but *estimating* it reliably is the problem. The McAllester-Statos impossibility result shows that any estimator providing a high-confidence lower bound needs exponentially many samples in the true MI. A 768-dim representation can have very high MI with its input (potentially hundreds of bits for a deterministic encoder), making reliable estimation require astronomical sample sizes. This is why empirical \"information plane\" analyses of deep networks should be interpreted with caution."
    }
  ]
};
