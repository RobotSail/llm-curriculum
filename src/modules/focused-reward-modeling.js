// Focused module: Reward Modeling for RLHF
// Covers: the Bradley-Terry preference model, pairwise comparison training,
// reward hacking / overoptimization, and practical RM design considerations.

export const rewardModelingLearning = {
  id: "A.2-rm-learning-medium",
  sectionId: "A.2",
  title: "Reward Modeling for RLHF",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 22,
  steps: [
    {
      type: "info",
      title: "The Reward Problem",
      content: "In reinforcement learning from human feedback (RLHF), we want to optimize a language model to produce responses that humans prefer. But we need a **reward signal** — a scalar number that tells the optimizer how good each response is.\n\nWe can't have humans score every response during training (too slow, too expensive). Instead, we train a **reward model (RM)** — a neural network that learns to predict human preferences from a dataset of pairwise comparisons.\n\nThe data collection process works like this:\n\n1. Given a prompt $x$, generate two candidate responses $y_1$ and $y_2$\n2. A human annotator indicates which response they prefer: $y_w \\succ y_l$\n3. Collect thousands of these $(x, y_w, y_l)$ triples\n\nThe reward model is then trained to assign higher reward to the preferred response: $r_\\theta(x, y_w) > r_\\theta(x, y_l)$. Once trained, the RM replaces the human in the loop — it provides instant reward feedback to the policy optimizer."
    },
    {
      type: "mc",
      question: "Why do we collect *pairwise comparisons* (\"A is better than B\") rather than *absolute ratings* (\"A is a 4/5\") for reward model training?",
      options: [
        "Absolute ratings require more annotators per example, making them prohibitively expensive at the scale needed for RM training",
        "Pairwise comparisons are more reliable because humans are much better at relative judgments than calibrated absolute scores — different annotators use rating scales inconsistently, but they largely agree on which of two responses is better",
        "Absolute ratings cannot be converted to a scalar reward signal, while pairwise comparisons directly produce the scalar used by the policy optimizer",
        "Pairwise comparisons are only used for efficiency during data collection and produce identical reward models to absolute rating approaches"
      ],
      correct: 1,
      explanation: "Human agreement on pairwise comparisons is substantially higher than on absolute ratings. One annotator's \"4/5\" may be another's \"3/5\" — the scale is poorly calibrated across people and even within a single person over time. But when shown two responses side by side, annotators agree on which is better at much higher rates. This is a well-established result in psychometrics. The Bradley-Terry model is specifically designed to extract a latent quality scale from these pairwise judgments."
    },
    {
      type: "info",
      title: "The Bradley-Terry Model",
      content: "The **Bradley-Terry model** provides the probabilistic framework for learning from pairwise comparisons. It assumes each response has a latent scalar quality $r(x, y)$, and the probability of preferring one response over another follows a logistic function of their quality difference:\n\n$$P(y_w \\succ y_l \\mid x) = \\sigma\\big(r(x, y_w) - r(x, y_l)\\big) = \\frac{1}{1 + e^{-(r(x, y_w) - r(x, y_l))}}$$\n\nThis model encodes several assumptions:\n\n**Transitivity**: If $A \\succ B$ and $B \\succ C$, then $A \\succ C$. This follows from the scalar representation — if $r(A) > r(B) > r(C)$, then $r(A) > r(C)$.\n\n**Scale invariance**: Adding a constant $c$ to all rewards doesn't change preferences: $\\sigma(r_w + c - r_l - c) = \\sigma(r_w - r_l)$. Only *differences* matter.\n\n**Logistic noise**: The stochasticity in preferences follows a logistic distribution. A quality difference of 1 unit maps to ~73% preference probability; a difference of 3 units maps to ~95%."
    },
    {
      type: "mc",
      question: "Under the Bradley-Terry model, a reward model assigns $r(x, y_A) = 2.5$ and $r(x, y_B) = 1.5$. The predicted preference probability $P(y_A \\succ y_B)$ is $\\sigma(1.0) \\approx 0.73$. Now suppose we add a constant, making $r(x, y_A) = 102.5$ and $r(x, y_B) = 101.5$. What happens to the predicted preference?",
      options: [
        "The preference probability increases to nearly 1.0 because the absolute reward values are much larger",
        "The preference probability remains $\\sigma(1.0) \\approx 0.73$ because the Bradley-Terry model depends only on the reward *difference*, which is still 1.0",
        "The preference probability decreases because the sigmoid saturates at large input values, reducing the discriminative power",
        "The model becomes numerically unstable and produces undefined outputs when rewards exceed 100"
      ],
      correct: 1,
      explanation: "The Bradley-Terry probability is $\\sigma(r(x, y_A) - r(x, y_B)) = \\sigma(102.5 - 101.5) = \\sigma(1.0) \\approx 0.73$ — identical to before. This is the scale invariance property: only the *difference* in rewards matters. This means the absolute magnitude of reward model outputs is meaningless for ranking — a reward of 100 is not \"better\" than a reward of 2 unless compared to another response on the same prompt. This has practical implications: you cannot compare rewards across different prompts."
    },
    {
      type: "info",
      title: "Training the Reward Model",
      content: "Given a dataset of pairwise comparisons $\\mathcal{D} = \\{(x_i, y_w^i, y_l^i)\\}$, we train the reward model by maximizing the Bradley-Terry likelihood — equivalently, minimizing the **negative log-likelihood**:\n\n$$\\mathcal{L}_{\\text{RM}}(\\theta) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}} \\left[\\log \\sigma\\big(r_\\theta(x, y_w) - r_\\theta(x, y_l)\\big)\\right]$$\n\nThis is binary cross-entropy with a twist: the \"logit\" is the reward difference $\\Delta r = r_\\theta(x, y_w) - r_\\theta(x, y_l)$, and the target is always 1 (the preferred response should score higher).\n\nThe gradient has an important property:\n\n$$\\nabla_\\theta \\mathcal{L} = -\\mathbb{E}\\left[\\big(1 - \\sigma(\\Delta r)\\big) \\cdot \\big(\\nabla_\\theta r_\\theta(x, y_w) - \\nabla_\\theta r_\\theta(x, y_l)\\big)\\right]$$\n\nThe factor $(1 - \\sigma(\\Delta r))$ acts as an **adaptive weight**: when the model already correctly ranks a pair by a large margin ($\\Delta r \\gg 0$), $(1 - \\sigma(\\Delta r)) \\approx 0$ and the gradient vanishes. Training automatically focuses on pairs the model finds difficult."
    },
    {
      type: "mc",
      question: "A reward model is being trained. For comparison pair A, the model scores the preferred response 3.0 points higher than the rejected one ($\\Delta r = 3.0$). For pair B, the scores are nearly equal ($\\Delta r = 0.1$). Which pair contributes more to the gradient, and why?",
      options: [
        "Pair A contributes more because the larger reward gap amplifies the gradient magnitude through the chain rule",
        "Both contribute equally because the loss function normalizes gradient contributions across all pairs in the batch",
        "Pair B contributes more because $(1 - \\sigma(0.1)) \\approx 0.475$ while $(1 - \\sigma(3.0)) \\approx 0.047$ — the model focuses on pairs it hasn't yet learned to rank confidently",
        "Pair A contributes more because the sigmoid function is steeper at larger input values, producing larger derivatives"
      ],
      correct: 2,
      explanation: "The adaptive weight $(1 - \\sigma(\\Delta r))$ is ~0.047 for pair A (already well-ranked) and ~0.475 for pair B (uncertain). Pair B contributes about 10x more gradient. This is the same mechanism as in logistic regression: correctly classified examples with high confidence have vanishing gradients. The model naturally implements a curriculum — first solving easy, clear-cut preferences, then spending training capacity on the harder, ambiguous pairs."
    },
    {
      type: "info",
      title: "Architecture: From Language Model to Reward Model",
      content: "A reward model is typically initialized from a **pretrained language model** — the same type of model used as the policy. The key architectural change is replacing the language modeling head (which projects to vocabulary size for next-token prediction) with a **scalar reward head** (which projects to a single number).\n\nConcretely, for a transformer with hidden dimension $d$:\n- Language model head: $h \\in \\mathbb{R}^d \\to \\text{logits} \\in \\mathbb{R}^{|V|}$\n- Reward head: $h \\in \\mathbb{R}^d \\to r \\in \\mathbb{R}^1$\n\nThe reward is typically computed from the **last token's** hidden state (or a pooled representation), since the full context of both prompt and response is available at that position through causal attention.\n\nWhy start from a pretrained LM? The RM needs to *understand* text to judge quality — reading comprehension, factual knowledge, stylistic awareness. These capabilities transfer from pretraining. Training a reward model from scratch on comparison data alone would require orders of magnitude more annotations."
    },
    {
      type: "mc",
      question: "A team trains two reward models: RM-3B (3 billion parameters, initialized from a pretrained 3B LM) and RM-scratch (3 billion parameters, randomly initialized, trained only on comparison data). Both see the same 100K comparison pairs. Which is likely to perform better, and why?",
      options: [
        "RM-scratch performs better because random initialization avoids the pretrained model's biases, allowing the reward model to learn reward-specific features from clean data",
        "They perform identically because 100K examples is sufficient for either initialization to converge to the same solution",
        "RM-3B performs dramatically better because the pretrained backbone provides language understanding, factual knowledge, and reasoning capabilities that 100K comparisons cannot teach from scratch — the scalar head is the only new component",
        "RM-scratch performs better initially but RM-3B catches up after extended training, converging to the same final performance"
      ],
      correct: 2,
      explanation: "To judge whether response A is better than response B, the RM must understand what both responses *say*. This requires reading comprehension, factual knowledge, reasoning, and stylistic judgment — all capabilities acquired during pretraining on trillions of tokens. 100K comparison pairs provide a thin supervised signal that teaches *what humans prefer*, but they cannot simultaneously teach the model to *read*. The pretrained backbone handles language understanding; the comparisons teach the preference function on top."
    },
    {
      type: "info",
      title: "Reward Hacking and Overoptimization",
      content: "The reward model is an *approximation* of human preferences — it has finite capacity and was trained on finite data. When the policy optimizer pushes hard against this approximation, it discovers inputs that score high reward but are actually low quality. This is **reward hacking**.\n\nGao et al. (2023) quantified this with a rigorous experiment. They trained policies with varying optimization pressure (measured by KL divergence from the initial SFT model) and evaluated with both the proxy RM and a separate \"gold\" evaluator:\n\n- **Proxy reward** (from the trained RM): increases monotonically with optimization pressure\n- **Gold reward** (from humans or a much larger RM): increases initially, peaks, then **decreases**\n\nThe peak of the gold reward curve is where the policy extracts genuine improvement from the RM. Beyond that peak, additional optimization exploits RM artifacts — the proxy and gold rewards diverge.\n\nThis is a form of **Goodhart's Law**: \"When a measure becomes a target, it ceases to be a good measure.\" The RM is a measure of quality; when the policy targets it directly, it ceases to reliably measure quality."
    },
    {
      type: "mc",
      question: "During RLHF training, the proxy reward score has been increasing steadily for 1,000 steps. But manual inspection shows the model now produces excessively verbose responses with redundant caveats. What is the most likely explanation?",
      options: [
        "The model has catastrophically forgotten its SFT training and is generating random text that happens to score well on a broken reward model",
        "The training has found the optimal verbosity level that maximizes genuine helpfulness, and the manual assessment is too subjective",
        "The RM learned a spurious correlation between length and quality from the training data (longer responses were more often preferred), and the policy is exploiting this shortcut to maximize proxy reward without improving actual quality",
        "The KL penalty coefficient is too large, forcing the model to pad responses to maintain high probability under the reference model"
      ],
      correct: 2,
      explanation: "Length bias is one of the most common reward hacking modes. In human comparison data, longer responses often win — they tend to be more thorough and detailed. The RM picks up on this correlation. The policy then discovers that adding more text (caveats, rephrasing, hedging) reliably increases reward, even when the additional text adds no information. This is reward hacking: proxy reward goes up, but true quality goes down. Mitigations include length normalization, length-controlled comparisons in training data, and monitoring the gold reward curve."
    },
    {
      type: "info",
      title: "The KL Penalty: Controlling Overoptimization",
      content: "The standard RLHF objective includes a **KL divergence penalty** that limits how far the policy can stray from the reference model:\n\n$$\\max_\\theta \\; \\mathbb{E}_{x \\sim \\mathcal{D}, \\, y \\sim \\pi_\\theta(\\cdot|x)} \\left[ r_\\phi(x, y) - \\beta \\, \\text{KL}\\big(\\pi_\\theta(\\cdot|x) \\| \\pi_{\\text{ref}}(\\cdot|x)\\big) \\right]$$\n\nThe coefficient $\\beta$ controls the tradeoff:\n\n**Large $\\beta$**: Strong regularization. The policy stays close to the SFT reference. Reward improvement is modest but reliable — we stay in the region where the RM is accurate.\n\n**Small $\\beta$**: Weak regularization. The policy can deviate far from the reference to chase high reward. This allows larger improvements but risks reward hacking — venturing into regions where the RM is unreliable.\n\nGao et al. showed that the optimal $\\beta$ depends on RM quality: better reward models (larger, trained on more data) tolerate lower $\\beta$ because they are accurate over a wider region of the response space. This creates a direct link between **RM quality and achievable policy improvement**."
    },
    {
      type: "mc",
      question: "Two teams run RLHF with different KL penalties. Team A uses $\\beta = 0.01$ and achieves a proxy reward of 5.0. Team B uses $\\beta = 0.5$ and achieves a proxy reward of 2.0. The reward model was trained on only 10K comparison pairs. Which team is more likely to have a better final policy?",
      options: [
        "Team A — the higher proxy reward directly translates to better outputs since the reward model was validated on comparison data",
        "Team B — with only 10K training pairs the RM has significant blind spots, and $\\beta = 0.5$ keeps the policy in the region where the RM is most reliable, avoiding overoptimization",
        "Neither — the final policy quality depends entirely on the RM quality, not on the KL penalty, so both teams achieve identical gold reward",
        "Team A — the low KL penalty allows the policy to explore more diverse responses, which always leads to better generalization"
      ],
      correct: 1,
      explanation: "With only 10K comparisons, the RM is accurate near the SFT distribution but unreliable further away. Team A's low $\\beta$ allows the policy to stray far from the reference (proxy reward of 5.0 indicates large KL divergence), likely entering regions where the RM is inaccurate — high proxy reward but potentially low gold reward. Team B's high $\\beta$ constrains the policy to stay near the reference, where the RM's predictions are trustworthy. The lower proxy reward likely reflects genuine, reliable improvement."
    },
    {
      type: "info",
      title: "Annotator Disagreement and Data Quality",
      content: "Human preferences are inherently noisy. Two annotators shown the same pair of responses may disagree — one prefers A, the other prefers B. This isn't a bug in the annotation process; it reflects genuine ambiguity in what makes a \"better\" response.\n\nThe standard approach: collect multiple annotations per pair and use **majority vote** to determine the preferred response. Optionally, the loss can be **weighted by agreement level** — a 5/5 unanimous pair provides clearer signal than a 3/2 split.\n\nSome important data quality practices:\n\n**Diverse annotators**: A homogeneous annotator pool encodes a narrow view of quality. Diverse annotators better represent the range of user preferences.\n\n**Calibration examples**: Include \"gold\" pairs with known correct preferences to monitor annotator quality and filter unreliable annotators.\n\n**Difficulty stratification**: Very easy comparisons (one response is clearly wrong) teach the RM less than moderately difficult ones (both responses are decent but one is better). Some teams oversample hard pairs.\n\nThe quality of the comparison dataset is the ceiling for RM performance — no architecture or training trick can recover from systematically biased or noisy annotations."
    },
    {
      type: "mc",
      question: "Your annotation team collects 50,000 comparison pairs, but analysis reveals that 40% of the pairs have 2/3 annotator agreement (barely above random). How should you handle this for RM training?",
      options: [
        "Discard the 40% entirely — only unambiguous preferences should train the RM, and ambiguous pairs inject noise that degrades accuracy",
        "Keep all pairs with uniform weight — the RM will learn to output moderate confidence on ambiguous pairs, which is the correct behavior",
        "Keep all pairs but down-weight the ambiguous ones in the loss function — they still contain signal (the majority-vote direction is slightly better), but with lower confidence, reflecting the genuine uncertainty",
        "Flip the labels on the ambiguous pairs to create a noise-robust training signal through data augmentation"
      ],
      correct: 2,
      explanation: "Ambiguous pairs still contain signal — even a 2/3 split means one response is preferred ~67% of the time. But treating them with the same weight as 3/3 unanimous pairs overstates confidence. Weighting by agreement level (e.g., weight = agreement fraction) gives the RM appropriate supervision: strong gradient on clear preferences, weak gradient on ambiguous ones. Discarding 40% of data wastes annotations and biases toward easy pairs. The RM learns to output calibrated confidence — high for clear pairs, moderate for ambiguous ones."
    },
    {
      type: "mc",
      question: "A reward model achieves 72% accuracy on a held-out comparison test set. The policy trained with this RM at $\\beta = 0.1$ produces outputs rated slightly better than the SFT baseline by humans. The team considers two interventions: (A) doubling the comparison data to 200K pairs, or (B) doubling the RM size from 7B to 13B parameters. Which is more likely to improve the final policy, given the Gao et al. scaling results?",
      options: [
        "Intervention A — more data always dominates model size for reward modeling because the comparison data is the binding constraint",
        "Intervention B — parameter count is the dominant factor because the RM needs capacity to represent the complex preference function over all possible responses",
        "Both interventions improve RM accuracy, but the key insight from Gao et al. is that a better RM allows a lower $\\beta$ (more optimization pressure) before overoptimization kicks in, so either intervention expands the usable optimization range",
        "Neither intervention matters because 72% accuracy is already sufficient for RLHF to converge to the optimal policy"
      ],
      correct: 2,
      explanation: "Gao et al.'s central finding is that RM quality determines the peak of the gold reward curve. A more accurate RM (whether from more data or more parameters) pushes the peak further out — the policy can be optimized more aggressively before the proxy and gold rewards diverge. So the impact of better RM isn't just higher accuracy on a test set; it's a larger *usable optimization budget*. Both more data and larger models improve RM quality, and both translate to better final policies through this mechanism."
    }
  ]
};
