// Section A.4: Direct Alignment Methods Assessment

export const directAlignmentAssessment = {
  id: "A.4-assess",
  sectionId: "A.4",
  title: "Assessment: Direct Alignment Methods",
  difficulty: "hard",
  estimatedMinutes: 16,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "DPO's key derivation insight starts from the KL-constrained RLHF objective $\\max_\\pi \\mathbb{E}[r(x,y)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$. The closed-form optimal policy is $\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp(r(y,x)/\\beta)$. DPO's crucial next step is to:",
      options: ["Approximate $Z(x)$ via Monte Carlo sampling from the reference policy, averaging over many completions to estimate the intractable partition function and plugging it into the reward expression", "Set $Z(x) = 1$ by assumption, arguing that the partition function is approximately constant across prompts when the reward model is well-calibrated and the reference policy is sufficiently broad", "Train a separate normalizer network to approximate $Z(x)$ as a learned function of the prompt, then use this auxiliary model's output during the policy optimization stage", "Rearrange to express $r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$, then substitute into Bradley-Terry — the $Z(x)$ cancels in the preference probability, yielding a policy-only loss"],
      correct: 3,
      explanation: "The full derivation: (1) Solve for optimal policy: $\\pi^*(y|x) \\propto \\pi_{\\text{ref}}(y|x) e^{r/\\beta}$. (2) Invert: $r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$. (3) Substitute into Bradley-Terry: $P(y_w \\succ y_l) = \\sigma(r_w - r_l) = \\sigma\\left(\\beta \\log \\frac{\\pi^*(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi^*(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)$. The $\\beta \\log Z(x)$ terms cancel in the difference. (4) Replace $\\pi^*$ with $\\pi_\\theta$ to get the training objective."
    },
    {
      type: "mc",
      question: "The DPO loss is $\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$. The gradient with respect to $\\theta$ is proportional to $(1 - \\sigma(\\hat{r}_w - \\hat{r}_l))$ where $\\hat{r}_i = \\beta \\log \\frac{\\pi_\\theta(y_i|x)}{\\pi_{\\text{ref}}(y_i|x)}$. This means the gradient:",
      options: ["Is constant regardless of how well the model has learned the preference, providing uniform gradient signal across all training pairs — the $(1 - \\sigma(\\cdot))$ factor reduces to a fixed scaling constant", "Is always zero for correctly ranked pairs ($\\hat{r}_w - \\hat{r}_l > 0$), meaning the model completely stops learning once it achieves correct ranking on any given example in the preference dataset", "Is large when the model's implicit reward difference is wrong ($\\hat{r}_w - \\hat{r}_l \\ll 0$) and vanishes when the model already correctly ranks the pair ($\\hat{r}_w - \\hat{r}_l \\gg 0$) — an adaptive weighting similar to the reward model loss", "Only depends on the winning response $y_w$ and ignores the losing response $y_l$, since the reference model's log-probability anchors the losing response's contribution and removes it from the gradient"],
      correct: 2,
      explanation: "The factor $(1 - \\sigma(\\hat{r}_w - \\hat{r}_l))$ acts as an implicit curriculum: when the model already assigns much higher implicit reward to the preferred response, $\\sigma(\\hat{r}_w - \\hat{r}_l) \\approx 1$ and the gradient vanishes. When the model incorrectly ranks the pair, $\\sigma(\\hat{r}_w - \\hat{r}_l) \\approx 0$ and the gradient is maximal. The gradient simultaneously pushes $\\pi_\\theta(y_w|x)$ up and $\\pi_\\theta(y_l|x)$ down relative to the reference."
    },
    {
      type: "mc",
      question: "A major failure mode of offline DPO is \"distribution shift.\" What specifically causes this?",
      options: ["DPO trains on pairs $(y_w, y_l)$ from a different policy (often the SFT model), so as $\\pi_\\theta$ diverges from that data-generating policy, $\\log \\pi_\\theta(y|x)$ becomes unreliable for out-of-distribution responses", "The tokenizer vocabulary changes between pretraining and DPO stages, causing a systematic mismatch in sequence segmentation that progressively corrupts all log-probability calculations during training", "The reference model is too large relative to the policy being trained, creating a capacity imbalance that amplifies the implicit KL penalty well beyond the level of useful regularization", "Distribution shift only affects models above 100B parameters, because smaller models lack sufficient capacity to diverge meaningfully from the data-generating distribution during optimization"],
      correct: 0,
      explanation: "Offline DPO uses a fixed dataset of preference pairs. As $\\pi_\\theta$ evolves during training, it may encounter responses $y$ that are very different from what it would generate — the log-probabilities $\\log \\pi_\\theta(y|x)$ become poorly calibrated for these out-of-distribution sequences. The implicit reward $\\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$ can become meaningless. This is analogous to off-policy RL issues. Online DPO mitigates this by regenerating responses with the current $\\pi_\\theta$."
    },
    {
      type: "mc",
      question: "IPO (Identity Preference Optimization) modifies the DPO loss by replacing the Bradley-Terry log-sigmoid with a squared loss: $\\mathcal{L}_{\\text{IPO}} = \\mathbb{E}\\left[\\left(\\hat{r}_w - \\hat{r}_l - \\frac{1}{2\\beta}\\right)^2\\right]$. What problem does this address?",
      options: [
        "IPO trains faster because squared losses are cheaper to compute than sigmoids, reducing per-step training cost by avoiding the expensive log-sigmoid evaluation",
        "DPO can overfit to deterministic preferences by pushing $\\hat{r}_w - \\hat{r}_l \\to \\infty$, while IPO's regression target $\\frac{1}{2\\beta}$ provides a finite anchor — it regularizes the implicit reward margin to a specific value rather than maximizing it without bound",
        "IPO eliminates the need for a reference model by using the squared loss to implicitly regularize the policy toward uniform output distributions across the vocabulary",
        "IPO is only applicable to reward models and not to policies, since the squared loss formulation requires explicit scalar reward targets rather than implicit reward differences"
      ],
      correct: 1,
      explanation: "In DPO, the loss $-\\log \\sigma(\\hat{r}_w - \\hat{r}_l)$ can be driven to zero by pushing the margin $\\hat{r}_w - \\hat{r}_l \\to \\infty$. This encourages the policy to be infinitely confident about preferences, leading to overfitting and degenerate solutions. IPO instead regresses the margin toward a fixed target $\\frac{1}{2\\beta}$, preventing the policy from becoming overconfident. This directly addresses the issue that DPO's loss has no natural stopping point for well-classified pairs."
    },
    {
      type: "mc",
      question: "KTO (Kahneman-Tversky Optimization) differs from DPO in a fundamental data requirement. What is this difference?",
      options: ["KTO requires roughly 10x more data than DPO to reach comparable alignment quality, since its per-example signal is less informative without the contrastive pairing structure", "KTO only works with numerical reward scores assigned by a trained reward model, and cannot use binary human feedback labels or thumbs-up/thumbs-down annotations", "KTO requires paired preferences for each prompt ($y_w \\succ y_l$ from the same input), while DPO works with independent point-wise quality labels on individual responses", "KTO works with unpaired binary feedback (each response independently labeled good/bad) rather than requiring paired preferences for the same prompt, greatly simplifying data collection"],
      correct: 3,
      explanation: "DPO requires *pairwise* comparisons: for the same prompt $x$, a preferred $y_w$ and dispreferred $y_l$. KTO only needs independent binary labels: \"this response is good\" or \"this response is bad,\" without pairing. This is inspired by Kahneman-Tversky prospect theory (loss aversion). KTO's loss treats good and bad examples asymmetrically, applying stronger penalties for producing bad outputs than rewards for producing good ones. This dramatically simplifies data collection — binary thumbs up/down is far cheaper than side-by-side comparisons."
    },
    {
      type: "mc",
      question: "Online DPO (also called \"iterative DPO\" or \"online RLHF with DPO loss\") modifies the standard DPO pipeline by:",
      options: ["Training on a much larger static dataset of preference pairs collected from multiple annotator pools, improving coverage of the response distribution and reducing sampling bias", "Using a different loss function that replaces the log-sigmoid with a hinge loss, creating a larger and more stable gradient margin between preferred and dispreferred responses", "Generating new responses from the current $\\pi_\\theta$ each iteration, scoring with a reward model, and constructing fresh on-policy preference pairs to mitigate distribution shift", "Removing the reference model entirely from the objective and relying on implicit regularization of the DPO loss structure to prevent the policy from diverging too far"],
      correct: 2,
      explanation: "Online DPO closes the distribution shift gap: (1) sample responses from current $\\pi_\\theta$, (2) score with RM to determine preferences, (3) train with DPO loss on these on-policy pairs, (4) repeat. This is conceptually similar to RLHF (generate, score, update) but uses the DPO loss instead of PPO. The result is that $\\log \\pi_\\theta(y|x)$ is always computed for responses the current policy would actually generate, making the implicit reward estimates accurate. Empirically, online DPO significantly outperforms offline DPO."
    },
    {
      type: "mc",
      question: "SimPO (Simple Preference Optimization) makes a key simplification compared to DPO by defining the implicit reward as the *length-normalized* log-probability: $r_{\\text{SimPO}}(x, y) = \\frac{1}{|y|} \\log \\pi_\\theta(y | x)$. What does this eliminate?",
      options: ["The reference model $\\pi_{\\text{ref}}$ — SimPO needs no frozen reference for log-probability computation, halving memory and simplifying the training pipeline", "The need for separate GPU memory for the reward model, since the length-normalized log-probability serves as the implicit reward signal throughout training", "The preference data entirely, since the length-normalized log-probability can be optimized via self-supervised objectives without any human comparison labels", "The need for gradient computation through the policy, since the length-normalized reward can be optimized through rejection sampling on the output distribution"],
      correct: 0,
      explanation: "DPO's implicit reward is $\\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$, requiring a frozen $\\pi_{\\text{ref}}$ alongside $\\pi_\\theta$ during training (2x memory). SimPO replaces this with the length-normalized log-probability $\\frac{1}{|y|}\\log \\pi_\\theta(y|x)$, eliminating $\\pi_{\\text{ref}}$ entirely. Length normalization prevents the model from trivially increasing reward by generating shorter sequences. SimPO adds a margin term $\\gamma$ to the loss, similar to IPO's finite target, to prevent overoptimization."
    },
    {
      type: "mc",
      question: "Rejection sampling (best-of-$N$) is sometimes used as an alternative to RL-based optimization. Given a prompt $x$, we sample $N$ responses from $\\pi_{\\text{ref}}$, score each with a reward model, and select the best. The effective KL divergence of this procedure scales as:",
      options: [
        "$\\text{KL} = N$, growing linearly with $N$ — each additional sample adds exactly one nat of KL divergence from the reference policy",
        "$\\text{KL} \\approx \\log N - \\frac{N-1}{N}$, growing logarithmically with $N$ — each doubling of $N$ adds roughly $\\log 2$ nats of KL",
        "$\\text{KL} = 0$ for all $N$, since selecting the best sample does not change the underlying generating distribution of the policy",
        "$\\text{KL} = N^2$, growing quadratically with $N$ — the pairwise comparisons between samples compound the divergence superlinearly"
      ],
      correct: 1,
      explanation: "Best-of-$N$ defines an implicit policy $\\pi_{\\text{BoN}}$ that samples the maximum-reward completion from $N$ i.i.d. draws. The KL from this implicit policy to the reference is $\\text{KL}(\\pi_{\\text{BoN}} \\| \\pi_{\\text{ref}}) \\approx \\log N - \\frac{N-1}{N}$. This is remarkably efficient at small $N$ — best-of-16 achieves significant quality improvements at only $\\sim 2.5$ nats of KL. However, the logarithmic scaling means diminishing returns: going from $N=16$ to $N=256$ costs another $\\sim 2.8$ nats but yields smaller gains. This makes rejection sampling competitive with PPO at moderate KL budgets."
    },
    {
      type: "mc",
      question: "Constitutional AI (CAI) / RLAIF replaces human preference labels with AI-generated feedback. The \"constitutional\" part refers to:",
      options: ["A legal compliance framework where regulators define mandatory safety constraints that the model must satisfy before deployment, encoding alignment as regulatory requirements", "A requirement that all training data come from government-vetted and approved sources, ensuring institutional quality control over the preference signal used for alignment", "The model's architecture and weights being permanently fixed (\"constituted\") during alignment, so only the decoding strategy and sampling parameters are adapted to preferences", "A set of natural-language principles that the AI uses to critique and revise its own outputs, then generates preference labels from these evaluations for RLHF/DPO training"],
      correct: 3,
      explanation: "CAI (Bai et al., 2022) works in two phases: (1) **Self-critique and revision**: the model generates a response, then is prompted to critique and revise it according to constitutional principles (e.g., \"Choose the response that is most helpful and least harmful\"). (2) **RLAIF**: the AI compares original vs. revised responses to generate preference labels, which train a reward model for RLHF. The constitution is a set of human-written principles that encode values, replacing per-example human annotation with scalable AI-based evaluation."
    },
    {
      type: "mc",
      question: "ORPO (Odds Ratio Preference Optimization) combines SFT and preference optimization into a single loss: $\\mathcal{L}_{\\text{ORPO}} = \\mathcal{L}_{\\text{SFT}}(y_w) + \\lambda \\cdot \\mathcal{L}_{\\text{OR}}$ where $\\mathcal{L}_{\\text{OR}} = -\\log \\sigma\\left(\\log \\frac{\\text{odds}_\\theta(y_w|x)}{\\text{odds}_\\theta(y_l|x)}\\right)$ and $\\text{odds}_\\theta(y|x) = \\frac{P_\\theta(y|x)}{1 - P_\\theta(y|x)}$. What is the key advantage of this unified approach?",
      options: ["ORPO achieves consistently higher reward scores than all other preference optimization methods, because the odds ratio provides a tighter variational bound on the true preference likelihood", "ORPO never overfits regardless of training duration, because the odds ratio loss self-regularizes and prevents the model from pushing preference margins beyond a natural equilibrium point", "It eliminates the separate SFT stage and reference model — the SFT term teaches format while the odds ratio encodes preferences, collapsing the multi-stage pipeline into one training phase", "The odds ratio is always numerically easier to compute than log-probabilities, providing identical gradient signal with lower precision requirements and substantially reduced risk of underflow"],
      correct: 2,
      explanation: "Standard alignment pipelines require: (1) SFT, (2) freeze as $\\pi_{\\text{ref}}$, (3) DPO/RLHF. ORPO merges steps 1 and 2–3: the SFT loss $\\mathcal{L}_{\\text{SFT}}(y_w)$ on the preferred response teaches format and content, while the odds ratio loss $\\mathcal{L}_{\\text{OR}}$ teaches preferences. No reference model is needed because the odds ratio $\\frac{\\text{odds}(y_w)}{\\text{odds}(y_l)}$ implicitly regularizes — it contrasts the chosen vs. rejected response within the same model. This simplifies the pipeline and reduces computational cost."
    }
  ]
};
