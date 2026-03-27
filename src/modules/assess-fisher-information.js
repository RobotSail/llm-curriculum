// Test module: Fisher Information
// Section 0.2: Probability & Information Theory
// Covers both scalar Fisher Information and the Fisher Information Matrix.
// 10 MC questions, no info steps.

export const fisherInformationAssessment = {
  id: "0.2-assess-fisher-information",
  sectionId: "0.2",
  title: "Fisher Information Assessment",
  moduleType: "test",
  difficulty: "medium",
  estimatedMinutes: 15,
  steps: [
    // Q1 — correct: 2
    {
      type: "mc",
      question: "The score function $s(x; \\theta) = \\frac{\\partial}{\\partial \\theta} \\log p_\\theta(x)$ has a fundamental property under the model distribution. Which of the following is correct?",
      options: [
        "The score is always positive because the log-likelihood is maximized at the true $\\theta$",
        "The score has unit variance regardless of the distribution family, by the normalization of $p_\\theta$",
        "The expected score is zero: $\\mathbb{E}_{p_\\theta}[s(x; \\theta)] = 0$, because differentiating $\\int p_\\theta(x) dx = 1$ gives $\\int \\frac{\\partial p_\\theta}{\\partial \\theta} dx = 0$",
        "The score equals the log-likelihood ratio $\\log \\frac{p_\\theta(x)}{p_{\\theta_0}(x)}$ for a fixed reference $\\theta_0$"
      ],
      correct: 2,
      explanation: "The zero-mean property follows from differentiating the normalization constraint $\\int p_\\theta(x) dx = 1$. This gives $\\int \\frac{\\partial p_\\theta}{\\partial \\theta} dx = 0$, and since $\\frac{\\partial p_\\theta}{\\partial \\theta} = p_\\theta \\cdot s(x; \\theta)$, we get $\\mathbb{E}_{p_\\theta}[s] = 0$. The score fluctuates around zero for individual observations, and Fisher Information measures the magnitude of these fluctuations."
    },
    // Q2 — correct: 0
    {
      type: "mc",
      question: "For a Poisson distribution $p_\\lambda(k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$, the Fisher Information for the rate parameter $\\lambda$ is $I(\\lambda) = 1/\\lambda$. If you observe $n = 100$ i.i.d. counts from a Poisson with $\\lambda = 4$, what is the Cramér-Rao lower bound on the variance of any unbiased estimator of $\\lambda$?",
      options: [
        "$\\text{Var}(\\hat{\\lambda}) \\geq 4/100 = 0.04$, because the bound is $\\lambda / n$",
        "$\\text{Var}(\\hat{\\lambda}) \\geq 1/400 = 0.0025$, because the bound is $1/(n\\lambda^2)$ for Poisson",
        "$\\text{Var}(\\hat{\\lambda}) \\geq 1/100 = 0.01$, because each observation contributes unit information regardless of $\\lambda$",
        "$\\text{Var}(\\hat{\\lambda}) \\geq 16/100 = 0.16$, because the bound uses $\\lambda^2/n$ for the Poisson family"
      ],
      correct: 0,
      explanation: "The Cramér-Rao bound is $1/(n \\cdot I(\\lambda)) = 1/(100 \\cdot 1/4) = 4/100 = 0.04$. The sample mean $\\bar{k}$ has variance $\\lambda/n = 4/100 = 0.04$, achieving the bound exactly — it is the MVUE for the Poisson rate. Higher $\\lambda$ means more variance per observation (Poisson variance equals $\\lambda$), so more samples are needed for the same precision."
    },
    // Q3 — correct: 3
    {
      type: "mc",
      question: "The Fisher Information for the variance parameter $\\sigma^2$ of a Gaussian $\\mathcal{N}(\\mu, \\sigma^2)$ (with known $\\mu$) is $I(\\sigma^2) = \\frac{1}{2\\sigma^4}$. Why does it scale as $1/\\sigma^4$ rather than $1/\\sigma^2$?",
      options: [
        "Because the Gaussian likelihood involves $\\sigma^4$ in the normalizing constant when both $\\mu$ and $\\sigma^2$ are parameters",
        "Because Fisher Information for scale parameters always involves the fourth power by a general property of location-scale families",
        "Because the curvature of the log-likelihood with respect to $\\sigma^2$ involves the fourth moment of the Gaussian, which equals $3\\sigma^4$",
        "Because the log-likelihood $-\\frac{(x-\\mu)^2}{2\\sigma^2} - \\log \\sigma$ has a second derivative with respect to $\\sigma^2$ that involves $1/\\sigma^4$ from differentiating $1/\\sigma^2$ terms"
      ],
      correct: 3,
      explanation: "The log-likelihood in terms of $\\sigma^2$ is $\\ell = -\\frac{(x-\\mu)^2}{2\\sigma^2} - \\frac{1}{2}\\log \\sigma^2 + \\text{const}$. The first derivative with respect to $\\sigma^2$ is $\\frac{(x-\\mu)^2}{2\\sigma^4} - \\frac{1}{2\\sigma^2}$. The second derivative yields terms involving $1/\\sigma^4$ and $1/\\sigma^6$. Taking the negative expectation produces $I(\\sigma^2) = 1/(2\\sigma^4)$. The extra power of $\\sigma^2$ in the denominator comes from differentiating the $1/\\sigma^2$ coefficient in the exponential."
    },
    // Q4 — correct: 1
    {
      type: "mc",
      question: "The FIM for a model's parameters satisfies $\\text{KL}(p_\\theta \\| p_{\\theta + \\delta}) \\approx \\frac{1}{2}\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta}$ for small $\\boldsymbol{\\delta}$. A TRPO-style optimizer constrains $\\text{KL}(p_\\theta \\| p_{\\theta + \\delta}) \\leq \\epsilon$. In terms of the FIM, what shape is this trust region in parameter space?",
      options: [
        "A sphere $\\|\\boldsymbol{\\delta}\\|_2 \\leq \\sqrt{2\\epsilon}$, because the FIM is proportional to the identity for well-trained models",
        "An ellipsoid $\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta} \\leq 2\\epsilon$, elongated along directions where the output distribution is insensitive to parameter changes",
        "A hypercube $|\\delta_i| \\leq \\sqrt{2\\epsilon / F_{ii}}$ for each parameter $i$, because the FIM constraint decomposes independently per parameter",
        "A simplex $\\sum_i |\\delta_i| \\cdot \\sqrt{F_{ii}} \\leq \\sqrt{2\\epsilon}$, because the $L_1$ norm is the natural metric for probability distributions"
      ],
      correct: 1,
      explanation: "The constraint $\\frac{1}{2}\\boldsymbol{\\delta}^\\top F \\boldsymbol{\\delta} \\leq \\epsilon$ defines an ellipsoid in parameter space. The principal axes align with the eigenvectors of $F$. Along directions with large eigenvalues (high Fisher Information — the distribution is sensitive), the ellipsoid is narrow (small allowed steps). Along directions with small eigenvalues (low Fisher Information — the distribution is insensitive), the ellipsoid is elongated (large steps are safe). This is the geometric interpretation of trust region methods."
    },
    // Q5 — correct: 0
    {
      type: "mc",
      question: "In Elastic Weight Consolidation (EWC), a regularization term $\\sum_i \\frac{\\lambda}{2} F_{ii} (\\theta_i - \\theta^*_i)^2$ penalizes deviating from pretrained parameters $\\theta^*$. What is the key limitation of using only the diagonal $F_{ii}$?",
      options: [
        "It ignores parameter interactions — two parameters might be individually unimportant ($F_{ii}$ small) but jointly critical (large off-diagonal $F_{ij}$), allowing EWC to freely change both and destroy a capability that depends on their relationship",
        "It overestimates the Fisher Information because the diagonal of a PSD matrix is always larger than the corresponding eigenvalue",
        "It makes the regularization non-convex because diagonal approximations can have negative entries when the empirical Fisher is poorly estimated",
        "It prevents learning any new parameters because the diagonal Fisher is always large for pretrained models that have converged to a good solution"
      ],
      correct: 0,
      explanation: "The diagonal approximation treats parameters as independent. But in reality, two parameters might have small individual Fisher Information (the distribution is not very sensitive to either alone) while having large off-diagonal FIM entries (changing both together in a correlated way significantly affects the distribution). EWC's diagonal penalty would allow both to be freely modified, potentially destroying a learned capability that depends on their joint configuration. This is a fundamental limitation of all diagonal Fisher approximations, including Adam's implicit diagonal preconditioning."
    },
    // Q6 — correct: 3
    {
      type: "mc",
      question: "For $n$ i.i.d. observations from an exponential distribution $p_\\lambda(x) = \\lambda e^{-\\lambda x}$ with $x > 0$, the Fisher Information is $I(\\lambda) = 1/\\lambda^2$. The MLE is $\\hat{\\lambda} = 1/\\bar{x}$. For large $n$, the MLE is approximately distributed as:",
      options: [
        "$\\hat{\\lambda} \\sim \\mathcal{N}(\\lambda, \\, \\lambda/n)$, because the variance of the exponential is $1/\\lambda^2$ and the MLE variance scales with the distribution variance",
        "$\\hat{\\lambda} \\sim \\mathcal{N}(\\lambda, \\, 1/(n\\lambda))$, because Fisher Information $1/\\lambda^2$ gives CR bound $\\lambda^2/n$ but the square root enters the normal approximation",
        "$\\hat{\\lambda} \\sim \\mathcal{N}(\\lambda, \\, 1/(n^2 \\lambda^2))$, because the CLT applies to $\\bar{x}$ and the delta method squares the $n$ in the denominator",
        "$\\hat{\\lambda} \\sim \\mathcal{N}(\\lambda, \\, \\lambda^2/n)$, because the MLE achieves the CR bound $1/(nI(\\lambda)) = \\lambda^2/n$ asymptotically"
      ],
      correct: 3,
      explanation: "The MLE is asymptotically normal with variance equal to the Cramér-Rao bound: $\\text{Var}(\\hat{\\lambda}) \\to 1/(n \\cdot I(\\lambda)) = 1/(n/\\lambda^2) = \\lambda^2/n$. Note that larger $\\lambda$ (shorter-lived events, more concentrated near zero) actually gives higher variance for $\\hat{\\lambda}$ — but this is in absolute terms. The coefficient of variation $\\text{CV} = \\sqrt{\\text{Var}}/\\lambda = 1/\\sqrt{n}$ is the same for all $\\lambda$, reflecting constant relative precision."
    },
    // Q7 — correct: 2
    {
      type: "mc",
      question: "A neural network has two parameterizations: (A) raw weights $\\mathbf{W}$ and (B) $\\mathbf{W}' = \\alpha \\mathbf{W}$ for some scalar $\\alpha > 1$. How do the FIMs and natural gradients compare?",
      options: [
        "Both the FIM and natural gradient are identical because multiplying weights by a constant does not change the model's output distribution",
        "The FIM scales as $F_{W'} = \\alpha^2 F_W$ and the natural gradient scales as $F_{W'}^{-1}g_{W'} = \\alpha^{-2} F_W^{-1} g_W$, so the update is smaller",
        "The FIM scales as $F_{W'} = \\alpha^{-2} F_W$ (the Jacobian $J = \\alpha^{-1}I$ enters as $J^{-\\top}FJ^{-1}$), but the natural gradient $F_{W'}^{-1}g_{W'}$ produces the same change in the output distribution",
        "The FIM is unchanged but the gradient doubles, so the natural gradient step is $2\\times$ larger in parameterization B"
      ],
      correct: 2,
      explanation: "With $\\mathbf{W}' = \\alpha \\mathbf{W}$, the Jacobian is $J = \\partial \\mathbf{W}'/\\partial \\mathbf{W} = \\alpha I$, so $J^{-1} = \\alpha^{-1}I$. The FIM transforms as $F_{W'} = J^{-\\top}F_W J^{-1} = \\alpha^{-2}F_W$. The gradient transforms as $g_{W'} = J^{-\\top}g_W = \\alpha^{-1}g_W$. The natural gradient: $F_{W'}^{-1}g_{W'} = \\alpha^2 F_W^{-1} \\cdot \\alpha^{-1}g_W = \\alpha F_W^{-1}g_W$. Combined with the coordinate change back ($\\Delta W = \\alpha^{-1} \\Delta W'$), the resulting distribution change is identical. This is reparameterization invariance."
    },
    // Q8 — correct: 1
    {
      type: "mc",
      question: "You compute the empirical Fisher and the true Fisher for a language model that has been trained to near-convergence on its training set. At a held-out test input where the model is confident but wrong (assigns 90% probability to the wrong token), how do the two Fisher estimates compare for this input?",
      options: [
        "Both are approximately equal because the model is well-trained overall and the Fisher is a global property that does not depend on individual inputs",
        "The empirical Fisher will be large (the score for the true label is large since the model is confident and wrong), while the true Fisher will be moderate (model-sampled tokens would mostly come from the high-confidence wrong prediction, producing smaller scores)",
        "The true Fisher will be larger because it samples from the model distribution which includes rare tokens that produce extreme score values",
        "Both will be near zero because the model is confident, and confident predictions always produce small gradients regardless of correctness"
      ],
      correct: 1,
      explanation: "The empirical Fisher uses the true label: $\\nabla \\log p_\\theta(y^* | x)$. Since the model assigns only ~10% to $y^*$, this gradient is large (the model would need a big parameter change to increase $p(y^*)$). The true Fisher samples from $p_\\theta$: most samples will be the wrong token the model is confident about, giving $\\nabla \\log p_\\theta(y_{\\text{wrong}} | x)$ which is small (the model already assigns high probability here — small gradient). This divergence between the two Fisher estimates is largest precisely when the model is confidently wrong."
    },
    // Q9 — correct: 0
    {
      type: "mc",
      question: "Fisher Information for the Bernoulli is $I(\\theta) = 1/(\\theta(1-\\theta))$, which diverges as $\\theta \\to 0$ or $\\theta \\to 1$. The Cramér-Rao bound gives $\\text{Var}(\\hat{\\theta}) \\geq \\theta(1-\\theta)/n$. What is the practical implication for estimating a very rare event probability ($\\theta = 10^{-6}$)?",
      options: [
        "The CR bound is $\\approx 10^{-6}/n$, so the relative error $\\sqrt{\\text{Var}}/\\theta \\approx 1/\\sqrt{n\\theta}$ is huge — you need $n \\gg 1/\\theta = 10^6$ observations to achieve even moderate relative precision",
        "The CR bound is $\\approx 10^{-12}/n$, meaning the variance is extremely small and very few observations are needed to estimate $\\theta$ precisely in absolute terms",
        "The diverging Fisher Information means the MLE is inconsistent for rare events — it never converges to the true $\\theta$ regardless of sample size",
        "High Fisher Information guarantees fast convergence, so rare event probabilities are actually easier to estimate than moderate probabilities"
      ],
      correct: 0,
      explanation: "The CR bound $\\theta(1-\\theta)/n \\approx \\theta/n = 10^{-6}/n$ is small in absolute terms (option B is partially true). But the relative error matters more: $\\text{CV} = \\sqrt{\\text{Var}}/\\theta \\approx \\sqrt{1/(n\\theta)} = 1/\\sqrt{n \\cdot 10^{-6}}$. For 10% relative error (CV = 0.1), you need $n \\geq 10^8$. High Fisher Information means each observation that reveals a rare event is very informative, but such observations are themselves rare — you need many samples before you see enough events to get a precise estimate."
    },
    // Q10 — correct: 3
    {
      type: "mc",
      question: "A researcher uses Fisher Information to decide which layers of a pretrained LLM to freeze during fine-tuning. They compute the trace $\\text{tr}(F_\\ell)$ for each layer $\\ell$. Layers with $\\text{tr}(F_\\ell) > \\tau$ are frozen; others are fine-tuned. What is a potential flaw in this approach?",
      options: [
        "The trace is always zero for well-trained models because the expected Hessian vanishes at a loss minimum",
        "The trace depends on the number of parameters in the layer, so larger layers will always exceed the threshold regardless of their actual importance",
        "The Fisher Information is only defined for the final layer of the network, not for intermediate layers, making per-layer traces meaningless",
        "A layer could have high trace but concentrated in a few directions (high-rank structure), meaning most of its parameters are free to change — a better criterion would consider the spectrum of $F_\\ell$, not just the trace"
      ],
      correct: 3,
      explanation: "The trace $\\text{tr}(F_\\ell) = \\sum_i \\lambda_i$ sums all eigenvalues. A layer could have one huge eigenvalue (one critical direction) and many near-zero eigenvalues (many free directions). Freezing the entire layer wastes the free directions. A spectral analysis — examining the distribution of eigenvalues — would reveal that only a low-rank subspace needs protection, while the remaining dimensions can be fine-tuned. This insight is related to why LoRA works: most of the useful fine-tuning happens in a low-rank subspace, leaving the high-Fisher directions of the pretrained model untouched."
    }
  ]
};
