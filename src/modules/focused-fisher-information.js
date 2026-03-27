// Focused learning module: Fisher Information (single-parameter)
// Section 0.2: Probability & Information Theory
// ONE concept: Fisher Information as the variance of the score function,
// its interpretation as likelihood curvature, and the Cramér-Rao bound.

export const fisherInformationLearning = {
  id: "0.2-fisher-information-learning-easy",
  sectionId: "0.2",
  title: "Fisher Information",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // --- Step 1: Motivation ---
    {
      type: "info",
      title: "Why Fisher Information Matters",
      content: "Suppose you are training a language model with parameters $\\theta$ by maximizing $\\log p_\\theta(x)$. At some point during training, you want to know: **how informative is my data about the current parameters?** If you change $\\theta$ slightly, does the output distribution shift dramatically, or barely at all?\n\nThis question has a precise answer: **Fisher Information**. It quantifies how much a single observation tells you about the parameter $\\theta$. High Fisher Information means the data is highly informative — small parameter changes produce large, detectable shifts in the distribution. Low Fisher Information means the data is uninformative — the distribution looks nearly the same across a range of parameter values.\n\nFisher Information appears throughout modern ML:\n- It determines the **fundamental limit** on how accurately you can estimate $\\theta$ (the Cramér-Rao bound)\n- It defines the **natural gradient** used in second-order optimizers like K-FAC\n- It measures local curvature of the KL divergence, connecting information theory to optimization geometry\n\nWe will build up to Fisher Information from its most basic ingredient: the **score function**."
    },
    // --- Step 2: The score function ---
    {
      type: "info",
      title: "The Score Function",
      content: "Given a parametric model $p_\\theta(x)$, the **score function** is the gradient of the log-likelihood with respect to $\\theta$:\n\n$$s(x; \\theta) = \\frac{\\partial}{\\partial \\theta} \\log p_\\theta(x)$$\n\nThe score tells you: for this particular observation $x$, which direction should I move $\\theta$ to increase the likelihood?\n\nA crucial property of the score is that **its expected value is zero** under the model:\n\n$$\\mathbb{E}_{x \\sim p_\\theta}[s(x; \\theta)] = 0$$\n\nTo see why, note that $\\frac{\\partial}{\\partial \\theta} \\log p_\\theta(x) = \\frac{1}{p_\\theta(x)} \\frac{\\partial p_\\theta(x)}{\\partial \\theta}$. Taking the expectation:\n\n$$\\mathbb{E}_{x \\sim p_\\theta}\\left[\\frac{1}{p_\\theta(x)} \\frac{\\partial p_\\theta(x)}{\\partial \\theta}\\right] = \\int \\frac{\\partial p_\\theta(x)}{\\partial \\theta} dx = \\frac{\\partial}{\\partial \\theta} \\int p_\\theta(x) \\, dx = \\frac{\\partial}{\\partial \\theta} 1 = 0$$\n\nThe score fluctuates around zero — sometimes positive, sometimes negative — but on average it vanishes. The **spread** of these fluctuations is what Fisher Information captures."
    },
    // --- Step 3: Score quiz ---
    {
      type: "mc",
      question: "For a Bernoulli distribution $p_\\theta(x) = \\theta^x (1-\\theta)^{1-x}$ with $x \\in \\{0, 1\\}$, what is the score function $s(x; \\theta)$?",
      options: [
        "$s(x; \\theta) = \\frac{x}{\\theta} - \\frac{1-x}{1-\\theta}$, obtained by differentiating $x \\log \\theta + (1-x) \\log(1-\\theta)$",
        "$s(x; \\theta) = \\log \\theta - \\log(1 - \\theta)$, the log-odds regardless of the observed value $x$",
        "$s(x; \\theta) = \\theta^x (1-\\theta)^{1-x} \\log \\theta$, because the score is the likelihood times the log-parameter",
        "$s(x; \\theta) = \\frac{x - \\theta}{\\theta(1-\\theta)^2}$, obtained by differentiating $p_\\theta(x)$ directly without the log"
      ],
      correct: 0,
      explanation: "The log-likelihood is $\\log p_\\theta(x) = x \\log \\theta + (1-x) \\log(1-\\theta)$. Differentiating with respect to $\\theta$: $s(x; \\theta) = \\frac{x}{\\theta} - \\frac{1-x}{1-\\theta}$. For $x=1$, this gives $1/\\theta > 0$ (push $\\theta$ up). For $x=0$, it gives $-1/(1-\\theta) < 0$ (push $\\theta$ down). The expected score is $\\theta \\cdot \\frac{1}{\\theta} + (1-\\theta) \\cdot \\frac{-1}{1-\\theta} = 1 - 1 = 0$, confirming the zero-mean property."
    },
    // --- Step 4: Fisher Information definition ---
    {
      type: "info",
      title: "Defining Fisher Information",
      content: "Since the score has zero mean, its variance is simply the expected squared score. This variance is **Fisher Information**:\n\n$$I(\\theta) = \\text{Var}_{x \\sim p_\\theta}[s(x; \\theta)] = \\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\left(\\frac{\\partial \\log p_\\theta(x)}{\\partial \\theta}\\right)^{\\!2}\\right]$$\n\nFisher Information is always non-negative (it is an expected square). It equals zero only when $p_\\theta(x)$ does not depend on $\\theta$ at all — the data carries no information about the parameter.\n\nThere is an equivalent form that is often more convenient for computation:\n\n$$I(\\theta) = -\\mathbb{E}_{x \\sim p_\\theta}\\!\\left[\\frac{\\partial^2 \\log p_\\theta(x)}{\\partial \\theta^2}\\right]$$\n\nThis says Fisher Information equals the **negative expected curvature** (second derivative) of the log-likelihood. We will see in the next step why these two forms are equivalent and what the curvature interpretation means."
    },
    // --- Step 5: Curvature interpretation ---
    {
      type: "info",
      title: "Fisher Information as Curvature",
      content: "The equivalence $I(\\theta) = -\\mathbb{E}[\\partial^2 \\log p_\\theta / \\partial \\theta^2]$ reveals a geometric meaning.\n\nImagine plotting the log-likelihood $\\ell(\\theta) = \\log p_\\theta(x)$ as a function of $\\theta$ for a fixed observation $x$. Near the maximum likelihood estimate $\\hat{\\theta}$, this curve has a peak. The second derivative $\\ell''(\\hat{\\theta})$ measures how **sharply** the log-likelihood curves downward from that peak.\n\n- **Large $|\\ell''|$** (sharp peak): the log-likelihood drops steeply as you move away from $\\hat{\\theta}$. The data strongly pins down the parameter — even slight deviations from $\\hat{\\theta}$ make the data much less likely. This means **high Fisher Information**.\n\n- **Small $|\\ell''|$** (flat peak): the log-likelihood barely changes near $\\hat{\\theta}$. Many values of $\\theta$ give nearly the same likelihood. The data is uninformative about the exact value of $\\theta$. This means **low Fisher Information**.\n\nFisher Information is the expected sharpness, averaged over all possible data the model could generate. It answers: \"on average, how much does the log-likelihood curve when I wiggle $\\theta$?\"\n\nThis curvature interpretation is why Fisher Information shows up in optimization: it measures how sensitive the loss surface is to parameter changes, which is exactly the information a second-order optimizer needs."
    },
    // --- Step 6: Curvature quiz ---
    {
      type: "mc",
      question: "You are estimating the mean $\\mu$ of a Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$ with known variance. How does Fisher Information $I(\\mu)$ depend on $\\sigma^2$?",
      options: [
        "$I(\\mu) = \\sigma^2$, because higher variance means each observation spans a wider range of values and thus carries more information about $\\mu$",
        "$I(\\mu) = \\mu^2/\\sigma^2$, because Fisher Information depends on both the true parameter value and the noise level",
        "$I(\\mu) = 1/\\sigma$, because Fisher Information scales with the standard deviation, not the variance",
        "$I(\\mu) = 1/\\sigma^2$, because the log-likelihood $-\\frac{(x-\\mu)^2}{2\\sigma^2}$ has curvature $1/\\sigma^2$ with respect to $\\mu$"
      ],
      correct: 3,
      explanation: "The log-likelihood is $\\ell(\\mu) = -\\frac{(x-\\mu)^2}{2\\sigma^2} + \\text{const}$. The second derivative is $\\ell''(\\mu) = -1/\\sigma^2$, so $I(\\mu) = -\\mathbb{E}[\\ell''] = 1/\\sigma^2$. Low noise ($\\sigma^2$ small) means each observation precisely locates $\\mu$ — high Fisher Information. High noise ($\\sigma^2$ large) means observations are spread out and $\\mu$ is hard to pin down — low Fisher Information. Note that $I(\\mu)$ does not depend on the true value of $\\mu$ itself."
    },
    // --- Step 7: Bernoulli example ---
    {
      type: "info",
      title: "Example: Fisher Information for the Bernoulli",
      content: "Let us compute Fisher Information for the Bernoulli distribution $p_\\theta(x) = \\theta^x(1-\\theta)^{1-x}$.\n\nWe already found the score: $s(x; \\theta) = \\frac{x}{\\theta} - \\frac{1-x}{1-\\theta}$.\n\n**Method 1 (variance of score)**:\n$$s(1; \\theta) = \\frac{1}{\\theta}, \\quad s(0; \\theta) = \\frac{-1}{1-\\theta}$$\n\n$$I(\\theta) = \\theta \\cdot \\frac{1}{\\theta^2} + (1-\\theta) \\cdot \\frac{1}{(1-\\theta)^2} = \\frac{1}{\\theta} + \\frac{1}{1-\\theta} = \\frac{1}{\\theta(1-\\theta)}$$\n\n**Method 2 (negative expected curvature)**:\n$$\\frac{\\partial^2 \\log p_\\theta(x)}{\\partial \\theta^2} = -\\frac{x}{\\theta^2} - \\frac{1-x}{(1-\\theta)^2}$$\n\n$$I(\\theta) = -\\mathbb{E}\\left[-\\frac{x}{\\theta^2} - \\frac{1-x}{(1-\\theta)^2}\\right] = \\frac{1}{\\theta} + \\frac{1}{1-\\theta} = \\frac{1}{\\theta(1-\\theta)}$$\n\nBoth methods give $I(\\theta) = \\frac{1}{\\theta(1-\\theta)}$. This function is U-shaped: Fisher Information is **highest** when $\\theta$ is near 0 or 1 (extreme probabilities — each coin flip is very informative) and **lowest** when $\\theta = 0.5$ (a fair coin — each flip tells you the least about $\\theta$)."
    },
    // --- Step 8: Bernoulli quiz ---
    {
      type: "mc",
      question: "For a coin with true bias $\\theta = 0.01$ (almost always tails), the Fisher Information is $I(0.01) = 1/(0.01 \\times 0.99) \\approx 101$. For a fair coin $\\theta = 0.5$, $I(0.5) = 4$. Why is the biased coin more informative per flip?",
      options: [
        "Because a biased coin produces longer sequences of identical outcomes, which are easier to compress and therefore contain more information in the Shannon entropy sense",
        "Because the biased coin has lower entropy, and Fisher Information is always inversely proportional to Shannon entropy $H(\\theta)$",
        "Because observing heads from a $\\theta = 0.01$ coin is a rare, high-surprise event that forces a large update to $\\theta$, while any outcome from a fair coin is unsurprising",
        "Because the variance of the Bernoulli distribution is $\\theta(1-\\theta)$, and lower variance always implies higher Fisher Information regardless of the distribution family"
      ],
      correct: 2,
      explanation: "When $\\theta = 0.01$, seeing heads (probability 1%) is a dramatic event — the score $s(1; 0.01) = 1/0.01 = 100$ is huge, strongly pushing $\\theta$ upward. Even tails has a non-trivial score $s(0; 0.01) = -1/0.99 \\approx -1.01$. The expected squared score is large because of the occasional extreme score values. For a fair coin, both outcomes produce moderate scores ($\\pm 2$), so the expected squared score is small. Fisher Information measures how much each observation can move your estimate — rare but dramatic events contribute heavily."
    },
    // --- Step 9: Cramér-Rao bound ---
    {
      type: "info",
      title: "The Cramér-Rao Lower Bound",
      content: "Fisher Information sets a **fundamental limit** on estimation accuracy. The **Cramér-Rao bound** states that for any unbiased estimator $\\hat{\\theta}$ of $\\theta$ based on $n$ i.i.d. observations:\n\n$$\\text{Var}(\\hat{\\theta}) \\geq \\frac{1}{n \\cdot I(\\theta)}$$\n\nNo matter how clever your estimator is, you cannot beat this variance floor. The bound has three important implications:\n\n**1. More data reduces variance linearly**: With $n$ observations, the best achievable variance is $1/(n I(\\theta))$. This is the standard $1/n$ rate familiar from the Central Limit Theorem.\n\n**2. Higher Fisher Information means better estimation**: If each observation carries more information about $\\theta$, you need fewer observations to achieve the same precision.\n\n**3. The MLE is asymptotically optimal**: The maximum likelihood estimator achieves the Cramér-Rao bound as $n \\to \\infty$. Specifically, $\\hat{\\theta}_{\\text{MLE}} \\sim \\mathcal{N}(\\theta, \\, 1/(n I(\\theta)))$ for large $n$. This is one reason MLE (and cross-entropy training, which is MLE for language models) is so widely used — it extracts the maximum possible information from the data."
    },
    // --- Step 10: Cramér-Rao quiz ---
    {
      type: "mc",
      question: "You observe $n$ i.i.d. samples from $\\mathcal{N}(\\mu, \\sigma^2)$ with known $\\sigma^2 = 25$. The Fisher Information is $I(\\mu) = 1/25$. How many samples do you need for the Cramér-Rao bound to guarantee a standard error of at most 0.1 for any unbiased estimator?",
      options: [
        "$n \\geq 40$, because $\\text{Var} \\geq 25/n$ and we need $\\sqrt{25/n} \\leq 0.1$, giving $n \\geq 2500$ — wait, that does not match",
        "$n \\geq 2500$, because $\\text{Var} \\geq 25/n$ and standard error $\\leq 0.1$ requires $25/n \\leq 0.01$",
        "$n \\geq 250$, because $\\text{Var} \\geq 1/(n/25)$ and standard error $\\leq 0.1$ requires $n \\geq 25/0.01$",
        "$n \\geq 100$, because with $I(\\mu) = 1/25$, the bound gives $\\text{Var} \\geq 25/n$ and we need $25/n \\leq 0.1$"
      ],
      correct: 1,
      explanation: "The Cramér-Rao bound gives $\\text{Var}(\\hat{\\mu}) \\geq 1/(n \\cdot I(\\mu)) = 1/(n/25) = 25/n$. For standard error $\\leq 0.1$, we need variance $\\leq 0.01$, so $25/n \\leq 0.01$, giving $n \\geq 2500$. Note: the sample mean $\\bar{x}$ has variance exactly $\\sigma^2/n = 25/n$, achieving the Cramér-Rao bound — it is the minimum variance unbiased estimator for the Gaussian mean."
    },
    // --- Step 11: Additivity and sufficient statistics ---
    {
      type: "info",
      title: "Additivity and Sufficient Statistics",
      content: "Fisher Information has two additional properties that make it especially useful:\n\n**Additivity for independent observations**: If $x_1, \\dots, x_n$ are i.i.d. from $p_\\theta$, the total Fisher Information is:\n\n$$I_n(\\theta) = n \\cdot I(\\theta)$$\n\nEach observation contributes independently to your knowledge about $\\theta$. This is why the Cramér-Rao bound has $n$ in the denominator.\n\n**Connection to sufficient statistics**: A statistic $T(x)$ is **sufficient** for $\\theta$ if it captures all the information in the data about $\\theta$. The formal criterion: $T$ is sufficient if and only if the Fisher Information in $T$ equals the Fisher Information in the full data $x$.\n\nFor the Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$ with known $\\sigma^2$, the sample mean $\\bar{x} = \\frac{1}{n}\\sum x_i$ is sufficient for $\\mu$ — it carries the same Fisher Information as the full dataset of $n$ observations. No information is lost by compressing the data into this single number.\n\nThis connects directly to deep learning: when we compress inputs through a neural network to produce a representation, we can ask how much Fisher Information about the task-relevant parameters is preserved. The **information bottleneck** perspective on deep learning is closely related to this idea."
    },
    // --- Step 12: Sufficient statistics quiz ---
    {
      type: "mc",
      question: "For $n$ i.i.d. observations from a Bernoulli($\\theta$), the total count $T = \\sum_{i=1}^n x_i$ is a sufficient statistic. What is the Fisher Information carried by $T$ alone?",
      options: [
        "$I_T(\\theta) = \\frac{1}{\\theta(1-\\theta)}$, the same as a single observation, because summarizing the data into one number always loses information",
        "$I_T(\\theta) = \\frac{n^2}{\\theta(1-\\theta)}$, because $T$ ranges from 0 to $n$ and its Fisher Information scales with the square of the range",
        "$I_T(\\theta) = \\frac{\\sqrt{n}}{\\theta(1-\\theta)}$, because Fisher Information in a summary statistic grows as $\\sqrt{n}$ by the central limit theorem",
        "$I_T(\\theta) = \\frac{n}{\\theta(1-\\theta)}$, because $T$ is sufficient and therefore retains all $n \\cdot I(\\theta)$ of the total Fisher Information"
      ],
      correct: 3,
      explanation: "Sufficiency means $T$ retains all information: $I_T(\\theta) = n \\cdot I(\\theta) = n/(\\theta(1-\\theta))$. You can verify this directly: $T \\sim \\text{Binomial}(n, \\theta)$, so $\\log p(T; \\theta) = T \\log \\theta + (n-T) \\log(1-\\theta) + \\text{const}$. The second derivative is $-T/\\theta^2 - (n-T)/(1-\\theta)^2$, and taking $-\\mathbb{E}[\\cdot]$ gives $n/\\theta + n/(1-\\theta) = n/(\\theta(1-\\theta))$. No information is lost by reducing $n$ binary observations to their sum."
    },
    // --- Step 13: Connection to LLM training ---
    {
      type: "info",
      title: "Fisher Information in Language Model Training",
      content: "In language model training, we maximize $\\sum_t \\log p_\\theta(x_t | x_{<t})$ — the sum of per-token log-likelihoods. The score for a single token is $\\nabla_\\theta \\log p_\\theta(x_t | x_{<t})$, and the Fisher Information (now a matrix, since $\\theta$ is high-dimensional) is:\n\n$$F(\\theta) = \\mathbb{E}\\left[\\nabla_\\theta \\log p_\\theta(x_t | x_{<t}) \\, \\nabla_\\theta \\log p_\\theta(x_t | x_{<t})^\\top\\right]$$\n\nThis is the foundation for several practical techniques:\n\n**Natural gradient descent**: Instead of updating $\\theta \\leftarrow \\theta - \\eta \\nabla \\mathcal{L}$, the natural gradient uses $\\theta \\leftarrow \\theta - \\eta F^{-1} \\nabla \\mathcal{L}$. This accounts for the geometry of the output distribution, not just the parameter space. Adam's per-parameter scaling $g_i / \\sqrt{v_i}$ approximates the diagonal of $F^{-1}g$.\n\n**Elastic Weight Consolidation (EWC)**: When fine-tuning a pretrained model, EWC uses the diagonal Fisher to identify which parameters were important for the pretraining task. Parameters with high Fisher Information get a strong regularization penalty, preventing catastrophic forgetting.\n\n**Model merging**: Fisher-weighted averaging of fine-tuned models weights each parameter by its Fisher Information, giving more influence to the model that is most \"confident\" about each parameter.\n\nThe next module will extend Fisher Information to the multi-parameter setting (the Fisher Information Matrix) and explore its deep connection to KL divergence."
    },
    // --- Step 14: LLM application quiz ---
    {
      type: "mc",
      question: "Elastic Weight Consolidation (EWC) penalizes changes to parameters with high Fisher Information from the pretraining phase. A particular attention head's value projection has very high Fisher Information. What does this imply about fine-tuning that parameter?",
      options: [
        "The parameter should be fine-tuned aggressively because high Fisher Information means it is the most learnable and will adapt quickly to the new task",
        "The parameter was poorly learned during pretraining and has high variance, so EWC correctly identifies it as needing retraining on the fine-tuning data",
        "The parameter strongly influences the pretrained model's output distribution, so changing it risks destroying capabilities learned during pretraining",
        "The parameter is numerically unstable and high Fisher Information indicates it is near a saddle point where gradient noise is amplified"
      ],
      correct: 2,
      explanation: "High Fisher Information means the pretrained model's output distribution is very sensitive to this parameter — small changes produce large distribution shifts. EWC penalizes moving such parameters because they encode important learned structure: changing them would significantly alter the model's behavior on pretraining-era tasks. Parameters with low Fisher Information can be changed freely because the pretrained model's outputs barely depend on them. This is the core insight of EWC: use Fisher Information to distinguish \"load-bearing\" parameters from \"free\" ones."
    }
  ]
};
