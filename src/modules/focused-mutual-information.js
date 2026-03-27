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
      title: "MI and Data Compression",
      content: "Suppose you want to compress a random variable $X$ into a summary $Z$ — for example, summarizing a patient's medical history ($X$) into a short risk score ($Z$) to predict disease outcome ($Y$).\n\nTwo MI quantities capture the quality of any such compression:\n\n**$I(X; Z)$**: How much $Z$ remembers about the original data. If $Z$ is a perfect copy of $X$, $I(X; Z) = H(X)$. If $Z$ is a constant (same output for every input), $I(X; Z) = 0$.\n\n**$I(Z; Y)$**: How useful the summary is for predicting the target. This is what task performance depends on.\n\nThe **data processing inequality** (DPI) constrains what's possible: for any chain $Y \\to X \\to Z$ where $Z$ depends on $Y$ only through $X$:\n\n$$I(Y; Z) \\leq I(Y; X)$$\n\nYou can never create information that wasn't in the input. The summary $Z$ can be *at most* as informative about $Y$ as the raw data $X$ is.\n\nThe **information bottleneck** objective formalizes the trade-off: minimize $I(X; Z)$ (compress aggressively) while maximizing $I(Z; Y)$ (preserve task-relevant signal). A good summary discards irrelevant details while retaining everything the task needs."
    },
    {
      type: "mc",
      question: "A summary $Z = f(X)$ is a deterministic function of $X$. Can $I(X; Z)$ be zero?",
      options: ["Yes — if $f$ is a constant function that maps every input to the same output value", "No — any deterministic function of $X$ always preserves at least some information about $X$", "Yes — if $f$ is a non-invertible function then the mutual information is always exactly zero", "It depends on the dimensionality of $Z$ relative to the dimensionality of $X$"],
      correct: 0,
      explanation: "$I(X; Z) = 0$ requires $X \\perp Z$, meaning $Z$ carries no information about $X$. For a deterministic $Z = f(X)$, this happens only if $f$ is constant — $Z$ takes the same value regardless of input. Any non-constant deterministic function has $I(X; Z) > 0$, because knowing $Z$ eliminates at least some uncertainty about $X$. A non-invertible function that maps different inputs to different outputs would still have positive MI."
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
      content: "The **data processing inequality** (DPI) is one of the most powerful results in information theory:\n\nIf $X \\to Y \\to Z$ is a Markov chain (i.e., $Z$ depends on $X$ only through $Y$), then:\n\n$$I(X; Z) \\leq I(X; Y)$$\n\nEach processing step can only **lose** information, never create it.\n\nConsider a pipeline that processes data in stages: raw data $X$, then a first summary $S_1$, then a further-compressed summary $S_2$. DPI tells us:\n\n$$I(X; S_1) \\geq I(X; S_2)$$\n\nThe second summary cannot contain more information about the original data than the first summary does — information can only be lost at each stage, never gained.\n\nThis has a practical consequence: if you compress data in multiple stages, the order matters. A lossy first step permanently limits what any subsequent step can recover. This is why, in machine learning pipelines, early transformations that discard information (e.g., aggressive feature selection or dimensionality reduction) cannot be compensated for by later processing."
    },
    {
      type: "mc",
      question: "A data pipeline has three stages: raw data $X$, intermediate summary $S_1$ with $I(X; S_1) = 5$ bits, and final summary $S_2$ computed from $S_1$. What can you say about $I(X; S_2)$?",
      options: ["$I(X; S_2) = 5$ bits — information is conserved through deterministic transformations", "$I(X; S_2) \\geq 5$ bits — the final summary could capture more input detail than the intermediate", "$I(X; S_2) \\leq 5$ bits — the data processing inequality prevents information from increasing", "Nothing definitive — MI between non-adjacent stages is not constrained by the DPI"],
      correct: 2,
      explanation: "Since $X \\to S_1 \\to S_2$ is a Markov chain, DPI gives $I(X; S_2) \\leq I(X; S_1) = 5$ bits. The final summary cannot contain more information about $X$ than the intermediate summary does. Equality is possible (if no information is lost), but information can never be *created* by further processing. This is why lossy compression early in a pipeline permanently limits downstream performance."
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
      explanation: "$I(W_{\\text{next}}; Z \\mid Y) \\approx 0$ means $H(W_{\\text{next}} \\mid Y) \\approx H(W_{\\text{next}} \\mid Y, Z)$ — the residual uncertainty about the next token is the same whether or not you have distant context, *given that you already have recent context*. The distant tokens are redundant for prediction, not necessarily irrelevant in isolation. This is why many sequences can be predicted well with short contexts — most of the prediction signal comes from nearby tokens."
    },
    {
      type: "info",
      title: "Estimating MI with Contrastive Bounds",
      content: "Directly computing MI is often intractable, but we can optimize **lower bounds** on it. The most widely used is the **InfoNCE** bound.\n\nThe setup: you have $N$ pairs of related observations $(x_1, y_1), \\ldots, (x_N, y_N)$ drawn from a joint distribution $P(X, Y)$. You want to estimate $I(X; Y)$. The InfoNCE loss scores each true pair $(x_i, y_i)$ against $N-1$ \"negative\" pairs $(x_i, y_j)$ where $j \\neq i$:\n\n$$\\mathcal{L}_{\\text{InfoNCE}} = -\\frac{1}{N} \\sum_{i=1}^N \\log \\frac{\\exp(s(x_i, y_i) / \\tau)}{\\sum_{j=1}^N \\exp(s(x_i, y_j) / \\tau)}$$\n\nwhere $s$ is a learned similarity function and $\\tau$ is a temperature parameter. Oord et al. (2018) showed:\n\n$$I(X; Y) \\geq \\log N - \\mathcal{L}_{\\text{InfoNCE}}$$\n\nSo minimizing InfoNCE maximizes a **lower bound** on $I(X; Y)$. Crucially, the bound saturates at $\\log N$ — you cannot estimate more than $\\log N$ bits of MI from $N$ negative samples. A batch of 64 caps you at $\\log(64) = 6$ bits; a batch of 32,768 allows up to $\\log(32768) = 15$ bits.\n\nThis bound is the foundation of many modern ML training objectives. You'll encounter it later in topics like contrastive learning, where models are trained to maximize MI between different views of the same data."
    },
    {
      type: "mc",
      question: "Using InfoNCE with batch size $N = 1024$, what is the maximum mutual information $I(X; Y)$ that the bound can estimate?",
      options: ["$1024$ bits — the bound scales linearly with $N$", "$\\sqrt{1024} = 32$ bits — the bound scales as $\\sqrt{N}$", "$1024 \\cdot \\log_2(1024) = 10240$ bits — the bound scales as $N \\log N$", "$\\log_2(1024) = 10$ bits — the bound scales as $\\log N$"],
      correct: 3,
      explanation: "The InfoNCE bound is $I(X; Y) \\geq \\log N - \\mathcal{L}$. Even when the loss is driven to zero, the bound saturates at $\\log N$. With $N = 1024$: $\\log_2(1024) = 10$ bits. If the true MI exceeds 10 bits, InfoNCE with this batch size simply cannot distinguish — the bound is too loose. Larger batch sizes give tighter bounds: $N = 32768$ allows estimating up to $\\log_2(32768) = 15$ bits."
    },
    {
      type: "info",
      title: "The Challenge of MI Estimation",
      content: "Mutual information is conceptually clean but **computationally hard** to estimate in high dimensions.\n\nFor discrete distributions with small support, you can compute MI exactly from counts. But for continuous or high-dimensional variables, MI estimation is a research problem in itself.\n\nThe core difficulty: MI involves a ratio of joint to marginal densities, $\\log \\frac{P(x,y)}{P(x)P(y)}$, and estimating densities in high dimensions is exponentially hard.\n\nSeveral approaches provide tractable bounds:\n- **InfoNCE** (lower bound): tractable but limited by batch size, as we just saw.\n- **MINE** (Mutual Information Neural Estimation): uses a learned function to estimate a tighter lower bound, but has high variance.\n- **Barber-Agakov bound**: provides an upper bound using a learned approximation $Q(X \\mid Z)$.\n\nA key impossibility result from McAllester & Statos (2020): **any MI estimator that provides a high-confidence lower bound must have sample complexity exponential in the true MI value.** In other words, the higher the true MI, the more data you need to confirm it — and the relationship is exponential, not polynomial. This is a fundamental limitation, not a matter of better algorithms."
    },
    {
      type: "mc",
      question: "A deterministic function $Z = f(X)$ maps a high-dimensional input $X$ to a high-dimensional output $Z$, with true MI $I(X; Z) = 200$ bits. Why is reliably estimating this MI value fundamentally hard?",
      options: ["Reliable MI estimation requires sample complexity that grows exponentially with the true MI value", "MI is not well-defined for continuous random variables and requires discretization before computing", "Deterministic functions always have infinite MI, so any finite-sample estimate would necessarily diverge", "The MI between high-dimensional variables is always zero due to the curse of dimensionality in estimation"],
      correct: 0,
      explanation: "MI is well-defined for continuous variables (as a KL divergence between densities), but *estimating* it reliably is the problem. The McAllester-Statos impossibility result shows that any estimator providing a high-confidence lower bound needs exponentially many samples in the true MI. With true MI of 200 bits, the required sample size grows as $\\sim 2^{200}$ — astronomical. Even InfoNCE with batch size $N$ can only estimate up to $\\log N$ bits. This fundamental limitation means that high MI values in complex systems are essentially impossible to verify empirically."
    }
  ]
};
