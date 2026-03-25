// Focused learning module: GAN training as divergence minimization.
// Covers JS divergence → GAN objective → f-GAN → Wasserstein → modern connections.

export const ganObjectivesLearning = {
  id: "0.2-gan-objectives-learning-medium",
  sectionId: "0.2",
  title: "GAN Training as Divergence Minimization",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 20,
  steps: [
    {
      type: "info",
      title: "Jensen-Shannon: A Symmetric, Bounded Divergence",
      content: "The Jensen-Shannon divergence symmetrizes KL by comparing both distributions to their average:\n\n$$\\text{JS}(P \\| Q) = \\frac{1}{2}\\text{KL}(P \\| M) + \\frac{1}{2}\\text{KL}(Q \\| M), \\quad M = \\frac{P + Q}{2}$$\n\nKey properties that made JS attractive as a training objective:\n\n**Symmetric**: $\\text{JS}(P \\| Q) = \\text{JS}(Q \\| P)$ — neither distribution is privileged, unlike KL where the direction matters enormously.\n\n**Bounded**: $0 \\leq \\text{JS}(P \\| Q) \\leq \\log 2$ — the divergence never explodes, even when distributions have disjoint support.\n\n**Well-defined everywhere**: Unlike KL, JS is finite even when $P$ and $Q$ don't overlap, because both are compared to the mixture $M$ which has support wherever either $P$ or $Q$ does.\n\nThese properties made JS the implicit objective of the original GAN — but as we'll see, boundedness is both a blessing and a curse."
    },
    {
      type: "mc",
      question: "If $P$ and $Q$ have completely disjoint support (no overlap at all), what is $\\text{JS}(P \\| Q)$?",
      options: [
        "$0$ — disjoint support means the distributions agree on placing zero mass in common regions",
        "$\\log 2$ — JS saturates at its finite upper bound when the two distributions share no support",
        "$\\infty$ — disjoint support causes the divergence to blow up, just as with KL",
        "It depends on the shape of $P$ and $Q$ within their respective supports"
      ],
      correct: 1,
      explanation: "When $P$ and $Q$ don't overlap: $M = (P+Q)/2$ has support everywhere either does. Then $\\text{KL}(P \\| M) = \\sum P \\log \\frac{P}{P/2} = \\log 2$, and similarly $\\text{KL}(Q \\| M) = \\log 2$. So $\\text{JS} = \\frac{1}{2}\\log 2 + \\frac{1}{2}\\log 2 = \\log 2$. JS saturates at its maximum — it tells you the distributions are maximally different, but the gradient of this constant is zero."
    },
    {
      type: "info",
      title: "The GAN Objective Is JS Minimization",
      content: "The original GAN (Goodfellow et al., 2014) training objective is a minimax game:\n\n$$\\min_G \\max_D \\;\\mathbb{E}_{x \\sim P_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{x \\sim P_G}[\\log(1 - D(x))]$$\n\nwhere $P_{\\text{data}}$ is the real data distribution, $P_G$ is the generator's distribution, and $D(x)$ outputs the probability that $x$ is real.\n\nAt the **optimal discriminator**, $D^*(x) = \\frac{P_{\\text{data}}(x)}{P_{\\text{data}}(x) + P_G(x)}$, the inner maximization evaluates to:\n\n$$2\\,\\text{JS}(P_{\\text{data}} \\| P_G) - 2\\log 2$$\n\nSo the generator minimizes Jensen-Shannon divergence between real and generated data. This connection explains both GAN successes and the notorious training instabilities:\n\n**Success**: JS is bounded and symmetric, so when distributions overlap, training proceeds smoothly.\n\n**Failure**: When $P_{\\text{data}}$ and $P_G$ don't overlap (common early in training), JS saturates at $\\log 2$ and $\\nabla_G \\text{JS} \\approx 0$. The generator receives no useful gradient signal."
    },
    {
      type: "mc",
      question: "Early in GAN training, the generator produces low-quality samples far from the data distribution. The discriminator easily distinguishes real from fake with near-perfect accuracy. What happens to the generator's gradient signal?",
      options: [
        "The gradient is very large because the discriminator's confidence amplifies the learning signal",
        "The gradient oscillates rapidly, causing instability but still providing useful direction",
        "The gradient is moderate — JS divergence provides stable gradients regardless of distribution overlap",
        "The gradient nearly vanishes because JS saturates at $\\log 2$ when distributions don't overlap"
      ],
      correct: 3,
      explanation: "When the discriminator is near-perfect, $P_{\\text{data}}$ and $P_G$ are effectively non-overlapping. JS saturates at its maximum $\\log 2$, which is a constant — its gradient with respect to the generator's parameters is approximately zero. The generator receives almost no learning signal. This vanishing gradient problem is fundamental to JS-based GANs and motivated the search for alternative divergence measures."
    },
    {
      type: "info",
      title: "The f-GAN Framework: Any f-Divergence Becomes a GAN",
      content: "The f-GAN framework (Nowozin et al., 2016) showed that the original GAN is just one instance of a general recipe: **any f-divergence can be turned into a GAN objective** using the Fenchel conjugate.\n\nRecall that every f-divergence has a variational lower bound:\n\n$$D_f(P \\| Q) \\geq \\sup_T \\left\\{ \\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[f^*(T(x))] \\right\\}$$\n\nwhere $f^*$ is the Fenchel conjugate of $f$, and $T$ is any function. The bound is tight when $T^*(x) = f'(P(x)/Q(x))$.\n\nIn the f-GAN recipe:\n- **The discriminator** plays the role of $T$ — it maximizes the lower bound, tightening the divergence estimate\n- **The generator** minimizes the resulting divergence estimate\n\nDifferent generators $f$ yield different GAN variants:\n- $f(t) = t \\log t$ → KL-GAN\n- JS generator → Original GAN\n- $f(t) = (t - 1)^2$ → Pearson GAN (closely related to Least-Squares GAN)\n\nThis unified view reveals that the choice of $f$-divergence determines the GAN's sensitivity to mode dropping, gradient quality, and training stability."
    },
    {
      type: "mc",
      question: "In the f-GAN variational bound $D_f(P \\| Q) \\geq \\sup_T\\{\\mathbb{E}_P[T] - \\mathbb{E}_Q[f^*(T)]\\}$, what does the discriminator network compute?",
      options: [
        "The density ratio $P(x)/Q(x)$ directly, which is then plugged into the f-divergence formula",
        "An unbiased Monte Carlo estimate of the f-divergence without any variational approximation",
        "A function $T(x)$ that maximizes the variational lower bound on the chosen f-divergence",
        "The gradient of the f-divergence with respect to the generator parameters"
      ],
      correct: 2,
      explanation: "The discriminator is the function $T$ in the variational bound. By maximizing $\\mathbb{E}_P[T(x)] - \\mathbb{E}_Q[f^*(T(x))]$, it tightens the lower bound on $D_f(P \\| Q)$. It does not directly compute the density ratio (though the optimal $T^*$ is related to it via $T^* = f'(P/Q)$). The generator then minimizes the discriminator's best estimate of the divergence."
    },
    {
      type: "info",
      title: "Wasserstein Distance: Fixing the Gradient Problem",
      content: "The Wasserstein-1 distance (Earth Mover's distance) takes a fundamentally different approach from f-divergences:\n\n$$W_1(P, Q) = \\inf_{\\gamma \\in \\Pi(P, Q)} \\mathbb{E}_{(x, y) \\sim \\gamma}[\\|x - y\\|]$$\n\nwhere $\\Pi(P, Q)$ is the set of all joint distributions (couplings) with marginals $P$ and $Q$. Intuitively, $W_1$ measures the minimum cost of transporting mass from $P$ to $Q$.\n\nThe critical advantage over JS divergence: **$W_1$ provides meaningful gradients even when distributions have disjoint support**. If $P$ is a point mass at $x_0$ and $Q$ is a point mass at $x_1$, then $W_1(P, Q) = \\|x_0 - x_1\\|$ — the distance varies smoothly as $Q$ moves toward $P$.\n\nCompare this to JS: for any two non-overlapping point masses, $\\text{JS}(P \\| Q) = \\log 2$ regardless of how close or far apart they are. JS gives no directional signal; $W_1$ tells you exactly which way to move.\n\nBy the Kantorovich-Rubinstein duality, the WGAN objective becomes:\n\n$$\\min_G \\max_{\\|D\\|_L \\leq 1} \\mathbb{E}_{x \\sim P_{\\text{data}}}[D(x)] - \\mathbb{E}_{x \\sim P_G}[D(x)]$$\n\nwhere $D$ must be 1-Lipschitz (the output can't change faster than the input)."
    },
    {
      type: "mc",
      question: "Consider two non-overlapping distributions: $P$ is supported on $[0, 1]$ and $Q_\\theta$ is supported on $[2 + \\theta, 3 + \\theta]$ for $\\theta > 0$. As $\\theta$ decreases (distributions move closer), what happens to each divergence?",
      options: [
        "$W_1$ decreases smoothly in $\\theta$, but JS remains constant at $\\log 2$ until the supports actually overlap",
        "Both JS and $W_1$ decrease smoothly, providing useful gradients in $\\theta$",
        "JS decreases smoothly in $\\theta$, but $W_1$ remains constant because it only depends on the support size",
        "Both JS and $W_1$ remain constant — neither detects changes in distance between non-overlapping distributions"
      ],
      correct: 0,
      explanation: "Since the supports don't overlap for any $\\theta > 0$, JS remains pinned at $\\log 2$ regardless of the gap — moving the distributions closer produces zero gradient. In contrast, $W_1$ is approximately $1 + \\theta$ (the cost of transporting mass across the gap), which decreases smoothly as $\\theta$ shrinks. This is why WGAN training is more stable early on: the generator always receives a gradient pointing toward the data."
    },
    {
      type: "info",
      title: "The Non-Saturating Loss: A Practical Fix Within JS",
      content: "Before Wasserstein GANs, practitioners discovered a simpler trick to combat vanishing gradients: the **non-saturating loss** (also called the \"$-\\log D$ trick\").\n\nIn the original (saturating) GAN, the generator minimizes:\n\n$$\\mathcal{L}_G^{\\text{sat}} = \\mathbb{E}_{z}[\\log(1 - D(G(z)))]$$\n\nWhen $D$ is confident that $G(z)$ is fake, $D(G(z)) \\approx 0$, so $\\log(1 - D(G(z))) \\approx \\log 1 = 0$ — the gradient is flat.\n\nThe non-saturating alternative flips the objective:\n\n$$\\mathcal{L}_G^{\\text{ns}} = -\\mathbb{E}_{z}[\\log D(G(z))]$$\n\nNow when $D(G(z)) \\approx 0$, we get $-\\log(D(G(z))) \\to \\infty$ — the gradient is **large** precisely when the generator needs it most.\n\nImportantly, the non-saturating loss has the **same fixed point** as the original: both reach equilibrium when $P_G = P_{\\text{data}}$. But the non-saturating loss no longer minimizes JS divergence exactly — it minimizes a different objective that is related to reverse KL: $\\text{KL}(P_G \\| P_{\\text{data}})$. This shifts the generator toward **mode-seeking** behavior."
    },
    {
      type: "mc",
      question: "A GAN uses the non-saturating loss $\\mathcal{L}_G = -\\mathbb{E}_z[\\log D(G(z))]$. Compared to the original saturating loss, which behavioral change does this introduce?",
      options: [
        "The generator now minimizes Wasserstein distance instead of JS divergence",
        "The generator and discriminator now converge to a Nash equilibrium in all cases",
        "The discriminator no longer needs to be trained — the generator can learn from the fixed-point discriminator",
        "The generator shifts toward mode-seeking behavior, preferring to produce high-quality samples from fewer modes"
      ],
      correct: 3,
      explanation: "The non-saturating loss is related to $\\text{KL}(P_G \\| P_{\\text{data}})$ — reverse KL, which is mode-seeking. The generator prefers to place mass where $P_{\\text{data}}$ is high rather than covering all modes. This provides strong gradients early in training but can exacerbate mode collapse. It does not minimize Wasserstein distance (that requires the Lipschitz constraint), and Nash equilibrium convergence is not guaranteed in practice."
    },
    {
      type: "info",
      title: "Mode Collapse as a Divergence Failure Mode",
      content: "**Mode collapse** occurs when the generator produces samples from only a subset of the data distribution's modes — e.g., a GAN trained on all digits only generates 3s and 7s.\n\nDivergence minimization explains why this happens:\n\n**Forward KL** $\\text{KL}(P_{\\text{data}} \\| P_G)$ is mode-covering: $P_G$ is penalized heavily for missing any mode of $P_{\\text{data}}$. This would resist mode collapse, but forward KL requires samples from $P_{\\text{data}}$ evaluated under $P_G$, which is intractable for implicit generative models like GANs.\n\n**Reverse KL** $\\text{KL}(P_G \\| P_{\\text{data}})$ is mode-seeking: $P_G$ is penalized for placing mass where $P_{\\text{data}}$ has none, but **not** for missing modes entirely. The non-saturating GAN loss inherits this tendency.\n\n**JS divergence** sits between these extremes but still permits mode collapse. When the generator concentrates on a few modes, the discriminator may be unable to provide gradient signal pointing toward the missing modes — it can only say \"these generated samples look real\" without indicating where else the generator should explore.\n\nMode collapse is thus an inherent risk of any GAN objective that doesn't explicitly enforce coverage of the full data distribution."
    },
    {
      type: "mc",
      question: "A GAN suffers severe mode collapse: it generates only faces of young women, despite the training set containing diverse faces. Which modification most directly addresses this problem through the lens of divergence minimization?",
      options: [
        "Switching to a Wasserstein objective — $W_1$ distance inherently prevents mode collapse by penalizing missing modes",
        "Using a larger discriminator network so it can detect missing modes more accurately",
        "Adding an auxiliary loss that maximizes entropy of the generator's output distribution to encourage coverage",
        "Training the discriminator for more steps per generator step to better approximate the optimal $D^*$"
      ],
      correct: 2,
      explanation: "Adding an entropy maximization term directly combats mode collapse by penalizing the generator for concentrating its probability mass. This moves the effective objective toward mode-covering behavior. Wasserstein distance alone does not prevent mode collapse — it fixes vanishing gradients but the generator can still concentrate on a subset of modes. A larger or better-trained discriminator helps quality but doesn't inherently push for diversity."
    },
    {
      type: "info",
      title: "From GANs to Modern LLM Training: Discriminator-Based Methods",
      content: "The GAN framework — a generator optimized against a learned discriminator — has deeply influenced modern LLM alignment:\n\n**GAIL (Generative Adversarial Imitation Learning)** directly applies the GAN framework to policy learning. A discriminator distinguishes expert demonstrations from the policy's behavior, and the policy (generator) is trained to fool it. The resulting objective minimizes JS divergence between the policy's state-action distribution and the expert's.\n\n**Discriminator-based RLHF** uses a similar structure: the reward model acts as a learned discriminator between preferred and dispreferred completions. The policy is then optimized to maximize this reward, subject to a KL penalty against a reference policy:\n\n$$\\max_\\pi \\; \\mathbb{E}_{x \\sim \\pi}[R(x)] - \\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$$\n\nThe KL penalty plays the same role as regularization in GAN training — it prevents the policy from collapsing to a degenerate solution that exploits the reward model (analogous to mode collapse exploiting the discriminator).\n\n**DPO (Direct Preference Optimization)** eliminates the explicit discriminator by showing that the optimal policy under the RLHF objective has a closed-form relationship to the reward, effectively folding the GAN-like adversarial structure into a single supervised loss. The divergence-minimization perspective remains — DPO implicitly minimizes reverse KL between the policy and the optimal RLHF solution."
    },
    {
      type: "mc",
      question: "In RLHF, the KL penalty $\\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$ prevents the policy from deviating too far from the reference. What GAN training failure mode is this most analogous to?",
      options: [
        "Vanishing gradients — without the KL penalty, the reward model would saturate like JS divergence",
        "Mode collapse — without the KL penalty, the policy would concentrate on a narrow set of reward-maximizing outputs, losing diversity",
        "Discriminator overfitting — without the KL penalty, the reward model would memorize the preference data",
        "Training oscillation — without the KL penalty, the policy and reward model would cycle without converging"
      ],
      correct: 1,
      explanation: "Without the KL penalty, the policy would exploit the reward model by concentrating all probability on a few high-reward outputs — directly analogous to mode collapse in GANs, where the generator produces only a few outputs that fool the discriminator. The KL penalty acts like an entropy regularizer, forcing the policy to maintain coverage of the reference distribution rather than collapsing to a degenerate mode-seeking solution."
    }
  ]
};
