// Focused module: Actor-Critic Methods
// Covers how combining a learned value function (critic) with a policy (actor) reduces variance,
// the advantage function, TD error as advantage estimate, and the actor-critic training loop.

export const actorCriticLearning = {
  id: "A.3-actor-critic-learning-easy",
  sectionId: "A.3",
  title: "Actor-Critic Methods",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Variance Problem with REINFORCE",
      content: "REINFORCE gives us an unbiased policy gradient using the log-probability trick:\n\n$$\\nabla_\\theta J = \\mathbb{E}_{y \\sim \\pi_\\theta}\\left[R(y) \\nabla_\\theta \\log \\pi_\\theta(y)\\right]$$\n\nBut the return $R(y)$ for a complete response is the sum of all future rewards from that point. In LLM fine-tuning, reward is typically given **only at the end** of the response, so every token in the sequence gets the same scalar signal.\n\nThis causes **high variance**: two responses to the same prompt might score 0.3 and 0.9 from the reward model. REINFORCE treats these as equally informative — it scales gradients by the raw return. With a batch of 256 responses, the gradient estimate swings wildly between batches because each response's total return is noisy.\n\nA baseline $b$ helps by replacing $R(y)$ with $R(y) - b$, centering the signal. But a **constant** baseline ignores that some states are inherently high-value and others low-value. What if the baseline could **adapt to each state**?"
    },
    {
      type: "mc",
      question: "A REINFORCE agent generates two responses to the same prompt. Response A scores $R = 4.0$ and Response B scores $R = 4.5$. With a batch mean baseline $b = 4.25$, how does REINFORCE weight these responses?",
      options: [
        "Response A gets weight $-0.25$ and Response B gets weight $+0.25$, centering the signal but throwing away the absolute reward information",
        "Both responses get positive weight because both scored above zero, reinforcing both equally",
        "Response B gets weight $4.5$ and Response A gets weight $4.0$ because the baseline only affects the bias, not individual sample gradients",
        "The baseline makes both weights zero because they are symmetric around $b = 4.25$, producing no gradient signal"
      ],
      correct: 0,
      explanation: "With the baseline subtracted, Response A contributes $R - b = 4.0 - 4.25 = -0.25$ (pushed away) and Response B contributes $4.5 - 4.25 = +0.25$ (reinforced). This centering helps but the signal is weak — the gradient depends on the tiny difference between two noisy returns. A state-dependent baseline like $V(s)$ would give a stronger, more informative signal: 'was this action better than expected from this particular state?'"
    },
    {
      type: "info",
      title: "The Actor-Critic Idea",
      content: "The Actor-Critic framework introduces two components:\n\n**Actor** ($\\pi_\\theta$): The policy network that selects actions. In LLM fine-tuning, this is the language model itself.\n\n**Critic** ($V_\\phi$): A value function network that estimates the expected return from each state. It answers: \"how much total reward do I expect from this point onward, following the current policy?\"\n\nThe critic provides a **state-dependent baseline** — instead of comparing each return to a batch average, we compare it to the critic's prediction for that specific state:\n\n$$\\hat{A}(s_t, a_t) = R_t - V_\\phi(s_t)$$\n\nThis is the **advantage**: how much better was the actual outcome compared to what the critic expected. If the critic is accurate, the advantage has much lower variance than the raw return, because the critic absorbs the predictable part of the return.\n\nThe key insight: the actor learns **what to do**, the critic learns **how good each situation is**, and the advantage measures **how surprising the outcome was**."
    },
    {
      type: "mc",
      question: "A language model generates a response that scores $R = 6.0$ from the reward model. The critic predicts $V(s_0) = 5.8$ for the prompt state. What does the advantage $\\hat{A} = R - V(s_0) = 0.2$ tell us about this response?",
      options: [
        "The response is poor quality because the advantage is close to zero, indicating it barely exceeded random performance",
        "The critic is poorly calibrated because a high-quality response ($R = 6.0$) should produce a large advantage, not $0.2$",
        "The advantage is unreliable because $V(s_0)$ was estimated at the start of generation before any tokens were produced",
        "The response performed slightly better than the critic expected from this prompt, providing a small positive gradient for reinforcement"
      ],
      correct: 3,
      explanation: "The advantage $\\hat{A} = 0.2$ means this response was slightly better than expected — the critic predicted $5.8$ and it scored $6.0$. This is informative: the prompt was likely easy (high baseline value), and this response was marginally above average for this prompt. The gradient is small and positive, which is appropriate — we should slightly reinforce this behavior. A constant baseline would compare $6.0$ to the batch mean (say $4.0$), giving a misleadingly large gradient of $2.0$."
    },
    {
      type: "info",
      title: "Why the Critic Reduces Variance",
      content: "To see why the critic reduces variance, consider two prompts in a batch:\n\n**Prompt A** (easy math problem): Typical reward ~8.0. The critic learns $V(s_A) \\approx 8.0$.\n**Prompt B** (hard reasoning task): Typical reward ~2.0. The critic learns $V(s_B) \\approx 2.0$.\n\nWith a **constant baseline** $b = 5.0$ (batch mean), a response to Prompt A scoring $R = 7.5$ gets advantage $7.5 - 5.0 = 2.5$ (strongly reinforced), even though it was actually **below average for that prompt**. A response to Prompt B scoring $R = 3.0$ gets advantage $3.0 - 5.0 = -2.0$ (suppressed), even though it was **above average for that prompt**.\n\nWith the **critic baseline**, the same responses get advantage $7.5 - 8.0 = -0.5$ (correctly suppressed for Prompt A) and $3.0 - 2.0 = 1.0$ (correctly reinforced for Prompt B).\n\nThe critic separates the signal (\"was this action good?\") from the noise (\"was this prompt easy or hard?\"). Mathematically, $\\text{Var}[R - V(s)] < \\text{Var}[R - b]$ whenever the critic captures any meaningful state-dependent information."
    },
    {
      type: "mc",
      question: "A critic predicts $V(s) = 3.0$ for every state regardless of the prompt. Compared to using a constant baseline $b = 3.0$, what is the effect on training?",
      options: [
        "Training diverges because the critic's gradient signal conflicts with the actor's gradient direction",
        "Training is identical — a constant critic is exactly equivalent to a constant baseline with the same value",
        "Training is slower because the critic introduces bias by always predicting the same value regardless of the state",
        "Training improves because the critic's gradient updates will eventually break the constant prediction pattern"
      ],
      correct: 1,
      explanation: "If $V(s) = 3.0$ for all states, then $\\hat{A}_t = R_t - V(s_t) = R_t - 3.0$, which is identical to using baseline $b = 3.0$. The critic provides zero variance reduction because it captures no state-dependent information. The actor's gradient is still unbiased (any baseline preserves unbiasedness), but the variance is the same as REINFORCE with a constant baseline. The critic only helps when it learns to predict different values for different states."
    },
    {
      type: "info",
      title: "The TD Error as Advantage Estimate",
      content: "Using the full return $R_t - V(s_t)$ as the advantage still requires waiting until the episode ends. A more efficient approach uses the **temporal difference (TD) error**:\n\n$$\\delta_t = r_t + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)$$\n\nThis estimates the advantage using only the immediate reward $r_t$ and the critic's predictions at two consecutive states. The TD error asks: \"was the immediate outcome better or worse than the critic expected for one step?\"\n\nFor LLM generation, where reward is typically sparse (given only at the end), the TD error at most tokens is:\n\n$$\\delta_t = 0 + \\gamma V_\\phi(s_{t+1}) - V_\\phi(s_t)$$\n\nThis means the training signal for intermediate tokens comes entirely from changes in the critic's value estimate. If the critic's value increases after a token is generated ($V(s_{t+1}) > V(s_t)$), that token gets positive advantage — the model is \"on a good track.\"\n\nThe TD error has lower variance than Monte Carlo returns but introduces **bias** when the critic is imperfect. GAE (covered in a separate module) interpolates between these extremes."
    },
    {
      type: "mc",
      question: "In RLHF with sparse reward (reward only at the final token), the critic estimates $V(s_5) = 2.1$ after the 5th token and $V(s_6) = 2.8$ after the 6th token (with $\\gamma = 1$). What is the TD error at step 5, and what does it imply?",
      options: [
        "$\\delta_5 = 2.8 - 2.1 = 0.7$, suggesting token 6 moved the response toward higher expected reward according to the critic",
        "$\\delta_5 = 2.1 - 2.8 = -0.7$, suggesting the response quality decreased between steps 5 and 6",
        "$\\delta_5 = 0$ because no reward was received at step 5, so the TD error carries no information",
        "$\\delta_5 = (2.8 + 2.1)/2 = 2.45$, the average value across both states, used to weight the policy gradient"
      ],
      correct: 0,
      explanation: "With $r_5 = 0$ (sparse reward) and $\\gamma = 1$: $\\delta_5 = r_5 + \\gamma V(s_6) - V(s_5) = 0 + 2.8 - 2.1 = 0.7$. The positive TD error means the critic thinks the situation improved — the 6th token moved the response toward a region of higher expected reward. This gives token 6 a positive advantage signal even though no reward was received yet. The quality of this signal depends entirely on the accuracy of the critic."
    },
    {
      type: "info",
      title: "The Actor-Critic Training Loop",
      content: "The actor and critic are trained simultaneously but with different objectives:\n\n**Critic loss** (regression): Minimize the gap between predicted values and observed returns.\n$$L_{\\text{critic}} = \\mathbb{E}\\left[(V_\\phi(s_t) - R_t)^2\\right]$$\n\nwhere $R_t$ is the return (actual cumulative reward from state $s_t$). The critic learns by standard supervised regression.\n\n**Actor loss** (policy gradient): Maximize the policy gradient objective using advantages from the critic.\n$$L_{\\text{actor}} = -\\mathbb{E}\\left[\\hat{A}_t \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t)\\right]$$\n\nThe training alternates:\n1. **Collect data**: Roll out the current policy to generate responses\n2. **Compute returns**: Calculate $R_t$ (or use TD targets) from the collected rewards\n3. **Update critic**: Fit $V_\\phi$ to predict returns accurately\n4. **Compute advantages**: $\\hat{A}_t = R_t - V_\\phi(s_t)$ (or TD error, or GAE)\n5. **Update actor**: Apply the policy gradient weighted by advantages\n\nThis is the foundation of PPO's inner loop — PPO adds clipping and multiple epochs on top of this basic actor-critic structure."
    },
    {
      type: "mc",
      question: "If the critic is updated too aggressively (many gradient steps) between each actor update, what problem can arise?",
      options: [
        "The critic overfits to the current policy's returns, providing accurate advantages that correctly guide the actor",
        "The critic's value estimates become stale because the actor changes between critic updates",
        "The critic becomes too accurate for the current policy, so advantages shrink to near-zero and the actor receives almost no gradient signal",
        "The critic's loss diverges because the regression targets $R_t$ change faster than the critic can track"
      ],
      correct: 2,
      explanation: "A critic that perfectly predicts the current policy's returns gives $\\hat{A}_t \\approx 0$ for every action — the return matches the prediction, so there is no advantage signal. The actor receives near-zero gradients and stops improving. In practice, some approximation error in the critic is useful because it creates non-zero advantages that drive learning. The critic should be good enough to reduce variance but not so good that it kills the learning signal. This is related to the bias-variance tradeoff: a perfect critic eliminates variance but can also eliminate signal."
    },
    {
      type: "info",
      title: "Actor-Critic Architecture in LLM RLHF",
      content: "In PPO-based RLHF, the actor-critic architecture takes a specific form:\n\n**Actor**: The language model $\\pi_\\theta$ being fine-tuned (e.g., 7B parameters).\n\n**Critic (value model)**: Often initialized from the same pretrained checkpoint as the actor, with a **scalar head** replacing the language modeling head. It takes the same (prompt, partial response) input and outputs a single number: the predicted return.\n\nThis means the RLHF training loop has **four models in memory simultaneously**:\n1. $\\pi_\\theta$ — the policy (actor) being trained\n2. $V_\\phi$ — the value model (critic) being trained\n3. $\\pi_{\\text{ref}}$ — the frozen reference policy (for KL penalty)\n4. $r_\\psi$ — the frozen reward model\n\nFor a 7B model, each copy is ~14 GB in FP16. Four copies need ~56 GB just for parameters, plus optimizer states and activations. This is why RLHF is much more memory-intensive than SFT, and why methods like GRPO that eliminate the value model are attractive.\n\nThe value model is discarded after training — only the actor is deployed. The critic exists solely to provide better advantage estimates during training."
    },
    {
      type: "mc",
      question: "GRPO eliminates the critic by estimating advantages from groups of sampled responses. What is the fundamental tradeoff compared to using a learned critic?",
      options: [
        "GRPO introduces bias because group normalization systematically underestimates advantages for high-reward prompts",
        "GRPO cannot handle sparse rewards because it needs multiple reward signals per prompt to compute relative advantages",
        "GRPO requires more samples per prompt to achieve comparable advantage estimates, trading compute for memory savings",
        "GRPO loses the ability to assign per-token advantages, forcing uniform credit assignment across all tokens in a response"
      ],
      correct: 2,
      explanation: "Without a critic, GRPO estimates advantages by comparing responses within a group: $\\hat{A}_i = (r_i - \\mu_G) / \\sigma_G$. This requires sampling $G$ responses per prompt (typically 8-64) to get a reliable estimate, using more generation compute. A critic can estimate advantages from a single response by comparing to its learned value baseline. The tradeoff: GRPO saves the ~14 GB of value model memory and avoids critic training instabilities, but needs more samples. For large models where memory is the bottleneck, this trade favors GRPO."
    },
    {
      type: "info",
      title: "The Bias-Variance Spectrum",
      content: "Actor-Critic methods sit on a spectrum between two extremes:\n\n**Pure Monte Carlo (REINFORCE with baseline)**: Use actual returns as advantage. **Unbiased** — the expected advantage equals the true advantage. **High variance** — each estimate depends on the full trajectory.\n\n**Pure TD (1-step actor-critic)**: Use TD error $\\delta_t$ as advantage. **Low variance** — depends on only one step. **Biased** — accuracy depends on the critic quality.\n\nWhere you sit on this spectrum is controlled by how you estimate advantages:\n\n| Method | Advantage Estimate | Bias | Variance |\n|---|---|---|---|\n| REINFORCE | $R_t - b$ | None | High |\n| MC Actor-Critic | $R_t - V(s_t)$ | None | Medium |\n| N-step AC | $\\sum_{k=0}^{n-1} \\gamma^k r_{t+k} + \\gamma^n V(s_{t+n}) - V(s_t)$ | Low | Medium |\n| 1-step AC (TD) | $r_t + \\gamma V(s_{t+1}) - V(s_t)$ | Higher | Low |\n| GAE($\\lambda$) | $\\sum_{l=0}^{\\infty} (\\gamma\\lambda)^l \\delta_{t+l}$ | Tunable | Tunable |\n\nGAE (covered in its own module) provides a principled way to choose any point on this spectrum via a single parameter $\\lambda$. In practice, PPO uses GAE with $\\lambda = 0.95$ — close to Monte Carlo but with meaningful variance reduction."
    },
    {
      type: "mc",
      question: "A team switches their RLHF pipeline from REINFORCE with a constant baseline to a 1-step actor-critic (TD error advantages). Training becomes more stable but the final policy quality is slightly worse. What is the most likely explanation?",
      options: [
        "The constant baseline was accidentally closer to the optimal baseline than the learned critic, making REINFORCE more efficient",
        "The 1-step TD error cannot capture long-range dependencies in the response, causing it to misattribute credit to late tokens",
        "The actor-critic architecture doubled the memory usage, forcing a smaller batch size that offset the variance reduction",
        "The TD error has lower variance, which stabilizes training, but the critic introduces bias that pulls advantages away from their true values"
      ],
      correct: 3,
      explanation: "This is the classic bias-variance tradeoff. The 1-step TD error $\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$ has lower variance (only depends on one transition) but is biased whenever $V$ is imperfect — which it always is in practice. The bias can cause the policy to converge to a suboptimal solution. The fix is to use GAE with $\\lambda > 0$ to interpolate: $\\lambda = 0.95$ keeps most of the variance reduction while limiting bias."
    },
    {
      type: "mc",
      question: "In the LLM RLHF setup, the value model $V_\\phi$ is initialized from the same pretrained checkpoint as the actor $\\pi_\\theta$. What is the main reason for this initialization choice?",
      options: [
        "It allows parameter sharing between actor and critic, reducing memory usage by approximately half",
        "It ensures the actor and critic share the same tokenizer and vocabulary, which is required for the TD error computation",
        "The pretrained representations already encode language understanding needed to predict response quality from partial context",
        "The pretrained weights provide a regularization target that prevents the critic from overfitting to reward model artifacts"
      ],
      correct: 2,
      explanation: "The critic must predict expected reward from (prompt, partial response) inputs. A pretrained language model already has rich representations of language structure, coherence, and content — exactly the features needed to predict response quality. Training a critic from scratch would require learning these representations from reward signals alone, which is far slower and less sample-efficient. The actor and critic are separate models (no parameter sharing in standard RLHF) — they just share the same initialization point."
    }
  ]
};
