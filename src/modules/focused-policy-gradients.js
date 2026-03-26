// Focused module: Policy Gradients for Language Models
// Covers REINFORCE, the log-probability trick, baselines, and variance.

export const policyGradientsLearning = {
  id: "A.3-policy-gradients-learning-easy",
  sectionId: "A.3",
  title: "Policy Gradients for Language Models",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "The RL Objective for Language Models",
      content: "In supervised training, we minimize cross-entropy against target tokens. In RL, there are no target tokens — instead, the model generates complete responses and receives a **scalar reward** $r(y)$ for each response $y$.\n\nThe objective is to maximize expected reward:\n\n$$J(\\theta) = \\mathbb{E}_{y \\sim \\pi_\\theta}[r(y)]$$\n\nwhere $\\pi_\\theta$ is the language model viewed as a policy: given a prompt $x$, it generates a response $y = (y_1, y_2, \\dots, y_T)$ token by token, with each token sampled from $\\pi_\\theta(y_t | x, y_{<t})$.\n\nThe challenge: $J(\\theta)$ involves an expectation over **discrete samples** from $\\pi_\\theta$. We cannot backpropagate through discrete sampling. How do we get gradients?"
    },
    {
      type: "mc",
      question: "Why can't we compute $\\nabla_\\theta J(\\theta) = \\nabla_\\theta \\mathbb{E}_{y \\sim \\pi_\\theta}[r(y)]$ by simply backpropagating through the sampling process?",
      options: [
        "The reward function $r(y)$ is typically not differentiable with respect to the tokens, so gradients cannot flow through the reward computation",
        "Sampling discrete tokens creates a non-differentiable step — there is no gradient through categorical sampling, regardless of whether the reward is differentiable",
        "The expectation is over exponentially many possible responses, making exact gradient computation intractable even if each term were differentiable",
        "The language model's softmax output is not differentiable at the token boundaries, preventing standard backpropagation through the vocabulary"
      ],
      correct: 1,
      explanation: "The fundamental barrier is the discrete sampling step. Even with a differentiable reward and a manageable number of responses, you cannot backpropagate through the act of sampling a discrete token from a categorical distribution. The softmax itself IS differentiable — the problem is the argmax/sampling that selects one token. The log-probability trick (REINFORCE) sidesteps this by moving the gradient inside the expectation without differentiating through the sample."
    },
    {
      type: "info",
      title: "The Log-Probability Trick (REINFORCE)",
      content: "The key identity that makes policy gradients work:\n\n$$\\nabla_\\theta \\mathbb{E}_{y \\sim \\pi_\\theta}[r(y)] = \\mathbb{E}_{y \\sim \\pi_\\theta}[r(y) \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)]$$\n\nThis is the **REINFORCE** estimator. The derivation is a one-line application of the identity $\\nabla_\\theta \\pi_\\theta(y) = \\pi_\\theta(y) \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)$.\n\nWhat this says: sample a response $y$ from $\\pi_\\theta$, compute the reward $r(y)$, then compute the gradient of $\\log \\pi_\\theta(y)$ and scale it by $r(y)$. No need to differentiate through sampling.\n\nFor a language model, $\\log \\pi_\\theta(y) = \\sum_{t=1}^T \\log \\pi_\\theta(y_t | x, y_{<t})$ — the sum of log-probabilities of each token. This is exactly what the model already computes during a forward pass. The gradient $\\nabla_\\theta \\log \\pi_\\theta(y)$ is computed by standard backpropagation through these log-probabilities.\n\nThe reward $r(y)$ acts as a **scalar multiplier**: high reward → increase the probability of this response. Low reward → decrease it."
    },
    {
      type: "mc",
      question: "The REINFORCE gradient $r(y) \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)$ looks similar to the supervised cross-entropy gradient $\\nabla_\\theta \\log \\pi_\\theta(y^*)$ on a target $y^*$. What is the key difference?",
      options: [
        "REINFORCE uses the model's own generated $y$ rather than a fixed target $y^*$, and scales the gradient by the reward rather than treating all tokens equally",
        "REINFORCE computes gradients with respect to the reward model parameters, while cross-entropy computes gradients with respect to the language model parameters",
        "REINFORCE averages over all possible responses, while cross-entropy only uses one target per input",
        "There is no meaningful difference — supervised training is a special case where $r(y) = 1$ for the target"
      ],
      correct: 0,
      explanation: "Two key differences: (1) the response $y$ is sampled from the current policy, not fixed from a dataset — this makes it on-policy; (2) the gradient is scaled by reward, so good responses get reinforced and bad ones get suppressed. Option D is partially right (SFT can be viewed as $r=1$ for target, $r=0$ for others), but the on-policy sampling distinction is fundamental."
    },
    {
      type: "info",
      title: "Variance: The Achilles Heel",
      content: "REINFORCE is an **unbiased** estimator of the policy gradient, but it has extremely high **variance**. With a single sample:\n\n$$\\hat{g} = r(y) \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)$$\n\nThe variance of $\\hat{g}$ depends on:\n1. **Variance in $r(y)$**: If rewards range from -10 to +10, the scale of the gradient swings wildly between samples\n2. **Variance in $\\nabla_\\theta \\log \\pi_\\theta(y)$**: Different responses activate different parts of the model, producing very different gradient directions\n3. **The product**: Multiplying two high-variance quantities makes variance even worse\n\nWith a language model generating 512-token responses, the space of possible $y$ is enormous. A single sample gives one noisy estimate of the gradient across this vast space.\n\nHigh variance means the optimizer receives noisy, unreliable gradient estimates. It takes many samples (large batch sizes) or variance reduction techniques to get useful signal."
    },
    {
      type: "mc",
      question: "A team uses REINFORCE with batch size 1 (one response per gradient step) to fine-tune a language model. Training is extremely unstable. Which change would most directly reduce gradient variance?",
      options: [
        "Reducing the learning rate by 10x to compensate for noisy gradients",
        "Using a larger language model with more parameters",
        "Increasing the batch size to 64 and averaging the per-sample gradient estimates",
        "Switching from float32 to bfloat16 for gradient computation"
      ],
      correct: 2,
      explanation: "Averaging $N$ independent gradient estimates reduces variance by a factor of $N$. Going from batch 1 to 64 reduces variance by 64x. A lower learning rate doesn't reduce variance — it just shrinks the noisy steps. A larger model increases the gradient dimensionality but doesn't help per-sample noise. Precision affects rounding error, not statistical variance."
    },
    {
      type: "info",
      title: "Baselines: Subtracting the Mean",
      content: "A powerful variance reduction technique: subtract a **baseline** $b$ from the reward:\n\n$$\\hat{g} = (r(y) - b) \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)$$\n\nCrucially, this estimator is still **unbiased** for any baseline $b$ that does not depend on the action $y$. The proof: $\\mathbb{E}_{y \\sim \\pi_\\theta}[b \\cdot \\nabla_\\theta \\log \\pi_\\theta(y)] = b \\cdot \\nabla_\\theta \\sum_y \\pi_\\theta(y) = b \\cdot \\nabla_\\theta 1 = 0$.\n\nThe optimal baseline (minimizing variance) is approximately the **expected reward** $b \\approx \\mathbb{E}[r(y)]$. With this baseline:\n- Responses **better** than average get $r(y) - b > 0$ → probability increases\n- Responses **worse** than average get $r(y) - b < 0$ → probability decreases\n\nIn practice, the baseline is estimated by a learned **value function** $V_\\phi(x)$ that predicts expected reward given a prompt $x$. The quantity $r(y) - V_\\phi(x)$ is called the **advantage** — it measures how much better (or worse) this specific response was compared to what we expected."
    },
    {
      type: "mc",
      question: "All responses in a batch receive positive rewards between 7.0 and 9.0, with a baseline $b = 8.0$. Without the baseline, the gradient would increase the probability of ALL responses (since all rewards are positive). With the baseline, what happens?",
      options: [
        "All responses still get increased probability because $r(y) - b$ can be negative but the gradient direction doesn't change",
        "Responses with reward $> 8.0$ get increased probability, responses with reward $< 8.0$ get decreased probability — the model learns to distinguish better from worse even when all are positive",
        "The baseline cancels out the reward entirely, producing zero gradient",
        "The model only updates on the single response with the highest reward, ignoring the rest"
      ],
      correct: 1,
      explanation: "The baseline centers the rewards around zero. $r(y) = 9.0 \\to +1.0$ (increase), $r(y) = 7.0 \\to -1.0$ (decrease). Without the baseline, all responses get reinforced (some more, some less), which wastes gradient signal. With the baseline, the model actively learns which responses are above average (reinforce) vs below average (suppress). This is a much stronger learning signal."
    },
    {
      type: "info",
      title: "Per-Token vs Per-Sequence Rewards",
      content: "In the simplest setting, the reward model assigns a single scalar to the entire response: $r(y_1, y_2, \\dots, y_T)$. The REINFORCE gradient then assigns the **same reward** to every token's log-probability:\n\n$$\\hat{g} = (r(y) - b) \\sum_{t=1}^T \\nabla_\\theta \\log \\pi_\\theta(y_t | x, y_{<t})$$\n\nThis is inefficient: token $y_1$ gets the same credit as token $y_T$, even if the reward was entirely determined by the last few tokens. This is the **credit assignment** problem.\n\nA more refined approach uses per-token advantages from a value function:\n\n$$\\hat{g} = \\sum_{t=1}^T A_t \\cdot \\nabla_\\theta \\log \\pi_\\theta(y_t | x, y_{<t})$$\n\nwhere $A_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ assigns each token its own advantage. In LLM fine-tuning, this is often implemented via **Generalized Advantage Estimation (GAE)**, which interpolates between high-bias (single-step) and high-variance (full-return) estimates."
    },
    {
      type: "mc",
      question: "A model generates a 200-token response. A reward model gives it a high score because the final sentence (tokens 180-200) correctly answers the question, while tokens 1-179 are rambling filler. With per-sequence REINFORCE (same reward for all tokens), what happens?",
      options: [
        "All 200 tokens are equally reinforced, meaning the model also learns to produce the rambling filler, since it cannot distinguish which tokens caused the high reward",
        "The model correctly learns to generate the answer at the end, since the final tokens get the strongest gradient signal",
        "The gradient is zero because the positive reward on good tokens cancels with zero reward on filler tokens",
        "Only the last token gets updated because autoregressive models only backpropagate through the final position"
      ],
      correct: 0,
      explanation: "Per-sequence REINFORCE applies the same reward to all tokens' log-probabilities. It cannot distinguish which tokens were responsible for the reward. So the model reinforces the entire sequence — including the 179 filler tokens. This is why credit assignment matters, and why per-token advantages (via a value function) are important for efficient RL on long sequences."
    },
    {
      type: "info",
      title: "Why Gradients Are Shaped Differently Than in SFT",
      content: "Policy gradient updates have a fundamentally different structure than supervised updates, and this matters for optimizer choice.\n\n**SFT gradients** are computed on fixed (input, target) pairs. The gradient direction is determined by the target tokens, which are consistent across epochs. Gradient variance is relatively low because the same targets appear repeatedly.\n\n**Policy gradients** are computed on model-generated responses that change every iteration. The gradient direction depends on:\n1. Which response was sampled (random)\n2. What reward it received (variable)\n3. The advantage estimate (noisy)\n\nThis means policy gradients are **noisier and less consistent** than SFT gradients. The gradient at step $t$ may point in a very different direction than at step $t+1$, even for the same prompt.\n\nThis has implications for optimizers:\n- Adam's second-moment estimate $v_t$ may lag behind rapidly changing gradient distributions\n- Momentum $m_t$ may smooth out important directional changes\n- Muon's orthogonalization of the momentum buffer may help or hurt depending on whether the gradient noise is structured or random"
    },
    {
      type: "mc",
      question: "During SFT, Adam's second moment $v_t$ converges to a stable per-parameter estimate after a few hundred steps. During RL fine-tuning with the same model and optimizer settings, $v_t$ is observed to fluctuate significantly even after thousands of steps. What is the most likely cause?",
      options: [
        "The reward model introduces numerical instabilities that propagate through the gradient computation",
        "The KL penalty term in the RL objective adds an oscillating component to the gradient",
        "RL uses smaller batch sizes than SFT, causing higher per-batch gradient variance",
        "RL generates different responses each step, so the gradient distribution is non-stationary — $v_t$ tracks a moving target rather than converging to a fixed estimate"
      ],
      correct: 3,
      explanation: "The key difference is that RL's data distribution changes every step (the policy generates new responses from an evolving distribution). This makes the gradient distribution non-stationary — the second moment $v_t$ is estimating a quantity that keeps changing. SFT's fixed dataset means the gradient distribution stabilizes, allowing $v_t$ to converge. This non-stationarity is inherent to on-policy RL, not a batch size or numerical issue."
    }
  ]
};
