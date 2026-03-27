// Focused module: Maximal Update Parameterization (μP)
// Teaches how μP rescales initialization, learning rates, and output layers
// so that hyperparameters transfer across model widths — enabling cheap HP tuning
// on small proxy models and stable feature learning at any scale.
//
// Grounded in: Yang et al. "Tensor Programs V" (2022), EleutherAI/Cerebras
// practitioner's guide, and Microsoft mup library documentation.

export const mupLearning = {
  id: "B.4-mup-learning-easy",
  sectionId: "B.4",
  title: "Maximal Update Parameterization (μP)",
  moduleType: "learning",
  difficulty: "easy",
  estimatedMinutes: 22,
  steps: [
    // Step 1: Info — The hyperparameter transfer problem
    {
      type: "info",
      title: "The Scaling Problem: Hyperparameters Don't Transfer",
      content: "Training a frontier LLM costs millions of dollars. Before committing that budget, you need to choose hyperparameters: learning rate, initialization scale, weight decay, and more. The standard approach is to tune these on small models and hope they work at scale.\n\nUnder **standard parameterization (SP)** — the default in PyTorch and most frameworks — this hope is often misplaced. The optimal learning rate for a 125M-parameter model is typically very different from the optimal learning rate for a 7B-parameter model. Practitioners have observed empirically that the best learning rate tends to **decrease** as model width increases.\n\nThis means every time you scale up, you need to re-tune hyperparameters on the large model itself — which is prohibitively expensive. A single training run of a 70B model might cost \\$1M+. Running a grid search over 20 learning rates would cost \\$20M+.\n\n**Maximal Update Parameterization (μP)**, introduced by Yang et al. (2022) in *Tensor Programs V*, solves this problem. Under μP, the optimal hyperparameters remain **stable across model widths**. You can tune on a tiny proxy model (e.g., 13M parameters) and transfer those hyperparameters directly to the full-scale model — a technique called **μTransfer**."
    },
    // Step 2: MC — Motivating the problem
    {
      type: "mc",
      question: "A lab wants to find the optimal learning rate for a 70B-parameter model. Under standard parameterization, the most common approach is to run a learning rate sweep at the target scale. Why is this impractical?",
      options: [
        "Learning rate has no effect on final model quality — only the total number of training tokens determines the final loss achieved",
        "Standard parameterization makes the loss landscape convex, so any learning rate works equally well and sweeping provides no benefit",
        "The optimal learning rate for a 70B model is always exactly $10^{-4}$ regardless of other hyperparameters, so no sweep is needed",
        "Each full training run at 70B scale costs millions of dollars in compute, making a sweep over multiple learning rates prohibitively expensive"
      ],
      correct: 3,
      explanation: "A single pretraining run of a 70B model can cost $1M+ in compute. A learning rate sweep requires many such runs (typically 10-30), making the total cost astronomical. This is the central motivation for μP: if you can tune hyperparameters on a cheap 13M-parameter proxy and transfer them to 70B, you reduce tuning cost by orders of magnitude."
    },
    // Step 3: Info — What goes wrong with standard parameterization
    {
      type: "info",
      title: "Why Standard Parameterization Breaks at Scale",
      content: "To understand μP, we first need to understand what goes wrong with SP as you increase model width $d$.\n\nIn a standard Transformer layer, the output of a linear projection is:\n$$y = Wx, \\quad W \\in \\mathbb{R}^{d \\times d}$$\n\nUnder SP (e.g., Xavier/Kaiming initialization), each entry of $W$ is initialized with variance $\\sim 1/d$, which ensures the output $y$ has $O(1)$ magnitude at initialization. So far, so good.\n\nThe problem appears during **training**. After one gradient step with learning rate $\\eta$, the weight update is $\\Delta W = -\\eta \\nabla_W L$. The change in the layer's output is:\n$$\\Delta y = (\\Delta W) x$$\n\nUnder SP with a fixed learning rate, the magnitude of $\\Delta y$ depends on $d$. For hidden layers, $\\|\\Delta y\\|$ scales as $\\Theta(1/\\sqrt{d})$ — the update **shrinks** as the model gets wider.\n\nThis means wider models under SP make smaller and smaller updates to their internal representations. In the limit of infinite width, the representations freeze at their random initialization — the model enters the **kernel (lazy) regime** where it behaves like a linear model over fixed random features. Feature learning ceases."
    },
    // Step 4: MC — SP failure mode
    {
      type: "mc",
      question: "Under standard parameterization with a fixed global learning rate, what happens to the magnitude of weight updates' effect on hidden layer activations as model width $d$ increases?",
      options: [
        "The effect shrinks as $\\Theta(1/\\sqrt{d})$ — wider models effectively stop learning because updates become negligible",
        "The effect remains $\\Theta(1)$ — SP automatically preserves update magnitudes across widths by design",
        "The effect oscillates unpredictably — some layers update more while others update less, depending on the gradient distribution",
        "The effect grows as $\\Theta(\\sqrt{d})$ — wider models make proportionally larger updates, risking divergence at scale"
      ],
      correct: 0,
      explanation: "Under SP, the update to activations $\\Delta y = (\\Delta W)x$ shrinks as $\\Theta(1/\\sqrt{d})$ for hidden layers. This is because the gradient of each weight entry shrinks with width while the learning rate stays fixed. As $d \\to \\infty$, representation changes vanish — the model enters the kernel regime where features are frozen at initialization and only the output layer learns."
    },
    // Step 5: Info — The μP solution: maximal updates
    {
      type: "info",
      title: "The μP Principle: Make Every Layer Update Maximally",
      content: "μP is built on two desiderata that together uniquely determine the parameterization:\n\n1. **Outputs stay $O(1)$**: The model's logits should have bounded magnitude at initialization (otherwise softmax saturates).\n2. **Updates are maximal**: Each layer's parameters should be updated **as much as possible** without causing the output to diverge.\n\n\"Maximal\" here has a precise meaning: the change in each layer's output $\\Delta y$ should be $\\Theta(1)$ — neither shrinking to zero (underfitting) nor growing unbounded (diverging). This is the largest stable update rate.\n\nTo achieve this, μP adjusts three things per layer, as a function of width $d$ relative to a base width $d_0$:\n\n- **Initialization variance**: controls the scale of weights at $t=0$\n- **Learning rate multiplier**: controls how fast each layer learns\n- **Output multiplier**: a scalar applied to the layer's output\n\nThe key insight is that different layer types (input embeddings, hidden layers, output/readout layer) need **different scaling rules** to achieve maximal updates."
    },
    // Step 6: MC — Core μP principle
    {
      type: "mc",
      question: "μP is defined by two desiderata: (1) model outputs remain $O(1)$ at initialization, and (2) weight updates are \"maximal.\" What does \"maximal\" mean in this context?",
      options: [
        "All parameters in the network receive identical gradient magnitudes, ensuring uniform learning speed across every layer",
        "Each layer's weight update changes its output by $\\Theta(1)$ — the largest magnitude that keeps the network stable as width grows",
        "Each layer's gradient norm is the largest value that doesn't trigger the gradient clipping threshold during training",
        "The learning rate is set to the maximum value before the training loss starts oscillating or diverging"
      ],
      correct: 1,
      explanation: "\"Maximal update\" means the change in each layer's output $\\Delta y = (\\Delta W)x$ is $\\Theta(1)$ with respect to width — the largest stable scaling. If updates are smaller (e.g., $\\Theta(1/\\sqrt{d})$ as in SP), the layer under-learns. If larger, the network diverges. μP uniquely achieves this $\\Theta(1)$ update scaling for every layer simultaneously."
    },
    // Step 7: Info — The μP scaling table
    {
      type: "info",
      title: "μP Scaling Rules: SP vs μP",
      content: "Here are the concrete scaling rules for a model with width $d$, compared to a base model with width $d_0$. Let $m = d / d_0$ be the **width multiplier**.\n\n**Input embeddings:**\n- SP: init variance $\\sim 1$, LR = $\\eta$\n- μP: init variance $\\sim 1$, LR = $\\eta$ (same — embeddings are \"finite\" in width)\n\n**Hidden layers** ($W \\in \\mathbb{R}^{d \\times d}$):\n- SP: init variance $\\sim 1/d$, LR = $\\eta$\n- μP: init variance $\\sim 1/d$, LR = $\\eta / m$\n\n**Output/readout layer** ($W_{\\text{out}} \\in \\mathbb{R}^{V \\times d}$):\n- SP: init variance $\\sim 1/d$, LR = $\\eta$\n- μP: init zero (or $\\sim 1/d$), LR = $\\eta / m$, **output multiplied by $1/m$**\n\n**Attention logits** ($QK^T / \\text{scale}$):\n- SP: scale = $1/\\sqrt{d_h}$ (standard softmax temperature)\n- μP: scale = $1/d_h$ (divide by head dimension, not its square root)\n\nThe critical difference: μP scales the hidden layer learning rate inversely with the width multiplier ($\\eta / m$) and applies a $1/m$ multiplier to the output layer. These two changes ensure that activations change by $\\Theta(1)$ per step at every width."
    },
    // Step 8: MC — Scaling rules
    {
      type: "mc",
      question: "A team trains a base model with width $d_0 = 256$ using learning rate $\\eta = 3 \\times 10^{-4}$ under μP. They now scale to width $d = 2048$ ($m = 8$). What learning rate should they use for the hidden layers?",
      options: [
        "$3 \\times 10^{-4}$ — μP means the same learning rate works at any width, that's the whole point of the parameterization",
        "$3 \\times 10^{-4} / 8 = 3.75 \\times 10^{-5}$ — hidden layer LR scales as $\\eta / m$ in μP",
        "$3 \\times 10^{-4} \\times \\sqrt{8} \\approx 8.5 \\times 10^{-4}$ — LR increases with the square root of the width multiplier",
        "$3 \\times 10^{-4} \\times 8 = 2.4 \\times 10^{-3}$ — LR scales linearly with width to compensate for smaller per-weight gradients"
      ],
      correct: 1,
      explanation: "In μP, the hidden layer learning rate scales as $\\eta / m$ where $m = d/d_0$ is the width multiplier. With $m = 8$, the hidden LR becomes $3 \\times 10^{-4} / 8 = 3.75 \\times 10^{-5}$. This $1/m$ scaling exactly compensates for the way gradient magnitudes change with width, ensuring $\\Theta(1)$ activation updates. Note: the global learning rate $\\eta$ that you *tune* on the base model stays the same — the μP framework applies the $1/m$ scaling automatically."
    },
    // Step 9: Info — Attention scaling in μP
    {
      type: "info",
      title: "Attention Scaling: $1/d_h$ Instead of $1/\\sqrt{d_h}$",
      content: "One of μP's most surprising prescriptions concerns the attention mechanism. In standard Transformers:\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_h}}\\right) V$$\n\nThe $1/\\sqrt{d_h}$ scaling keeps the pre-softmax logits $O(1)$ at **initialization**. But μP requires considering what happens during **training** as well.\n\nAs the model trains, the queries and keys develop correlations. Under the standard $1/\\sqrt{d_h}$ scaling, the attention logits grow with width during training — wider heads produce larger logits, causing softmax to saturate and gradients to vanish. This is the \"attention entropy collapse\" phenomenon.\n\nμP prescribes $1/d_h$ scaling instead:\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{d_h}\\right) V$$\n\nAt initialization, this makes the logits smaller (more uniform attention). But as training proceeds and $Q$, $K$ develop structure, the logits grow to useful magnitudes. Crucially, this growth is now **independent of head dimension** $d_h$, so attention behavior transfers across scales.\n\nThis $1/d_h$ scaling is often the single most important change when adopting μP in practice."
    },
    // Step 10: MC — Attention scaling
    {
      type: "mc",
      question: "μP changes the attention scaling from $1/\\sqrt{d_h}$ to $1/d_h$. At initialization this makes attention patterns more uniform (higher entropy). Why is this acceptable?",
      options: [
        "The $1/d_h$ scaling is immediately compensated by the output multiplier $1/m$, so the effective scaling is still $1/\\sqrt{d_h}$ in practice",
        "Higher entropy at init is irrelevant because attention patterns are never used during the forward pass in the first few hundred training steps",
        "Uniform attention is always optimal — the model should attend equally to all positions, and any deviation from uniformity indicates overfitting",
        "As training develops correlations in $Q$ and $K$, logit magnitudes grow to produce sharp attention — the $1/d_h$ scaling ensures this growth is width-independent"
      ],
      correct: 3,
      explanation: "At initialization, $Q$ and $K$ are random, so their dot products are noise — uniform attention is actually the correct behavior (no spurious patterns). As training progresses, $Q$ and $K$ develop meaningful correlations, and the logit magnitudes grow. The $1/d_h$ scaling ensures this growth is $\\Theta(1)$ regardless of head dimension, so the same training dynamics play out at every width. Under $1/\\sqrt{d_h}$, the logit growth depends on $d_h$, breaking width transfer."
    },
    // Step 11: Info — μTransfer in practice
    {
      type: "info",
      title: "μTransfer: Tuning at Small Scale, Training at Large Scale",
      content: "The practical payoff of μP is the **μTransfer** recipe:\n\n1. **Define a base model**: a small proxy (e.g., 13M or 40M parameters) that is cheap to train.\n2. **Parameterize it in μP**: apply the scaling rules for init, LR, attention, and output.\n3. **Tune hyperparameters** on the base model: run a grid search over learning rate, weight decay, batch size, etc. This is cheap — each run takes minutes or hours.\n4. **Transfer to the target model**: scale up to the target width (e.g., 7B, 70B). The μP framework automatically adjusts per-layer LR and multipliers based on the width ratio $m = d/d_0$. The global HP values found in step 3 are used **as-is**.\n\nYang et al. demonstrated this dramatically:\n- Tuned HPs on a **40M** proxy model → transferred to **6.7B** GPT-3 → **matched published GPT-3 performance** with tuning cost only **7% of one full pretraining run**.\n- Tuned on **13M** → transferred to **350M** BERT-large → **exceeded published BERT-large results** with tuning cost equivalent to pretraining BERT-large once.\n\nThe key requirement is that the base model and target model share the same **architecture** (number of layers, head structure) — μP transfers across **width** (hidden dimension $d$), not across depth or architectural changes."
    },
    // Step 12: MC — μTransfer
    {
      type: "mc",
      question: "A team uses μTransfer to tune HPs on a 50M-parameter proxy and wants to transfer to a 10B-parameter target. Which statement about this transfer is correct?",
      options: [
        "The proxy and target should share the same depth and architecture but differ in width — μP adjusts per-layer LR and multipliers based on the width ratio automatically",
        "μTransfer works by scaling all hyperparameters (LR, weight decay, batch size, dropout) proportionally to the parameter count ratio",
        "The target model must have the same width, depth, and head count as the proxy — μP only transfers learning rate, not architecture",
        "μTransfer requires training the target model for at least 10% of full training to verify the transferred HPs before committing to a complete run"
      ],
      correct: 0,
      explanation: "μP guarantees HP stability across width changes. The proxy and target share the same number of layers and architectural structure, but differ in hidden dimension $d$. The μP framework uses the width ratio $m = d_{\\text{target}}/d_{\\text{base}}$ to automatically scale per-layer learning rates and output multipliers. The global HP values (base LR, weight decay, etc.) found on the proxy are used directly. Depth, number of heads, and other structural HPs do not transfer automatically."
    },
    // Step 13: Info — The coordinate check
    {
      type: "info",
      title: "Verifying μP: The Coordinate Check",
      content: "Implementing μP correctly is subtle — a single missed scaling factor can break the transfer guarantee. The **coordinate check** is a diagnostic tool to verify your implementation.\n\nThe procedure is:\n1. Create models at several widths (e.g., $d$ = 128, 256, 512, 1024, 2048).\n2. Train each for a few steps on the same data.\n3. For each layer, measure the **L1 norm of activations** (average absolute value of each coordinate).\n4. Plot these norms against width.\n\n**Under correct μP**, the activation norms should be **flat** — approximately the same at every width, both at initialization and after training steps. The curves are horizontal lines.\n\n**Under SP**, the activation norms change with width: some grow (output layer activations inflate), some shrink (hidden representations freeze). After a few training steps, the curves clearly diverge.\n\nThe coordinate check catches common implementation bugs:\n- Forgetting to apply $1/m$ to the output layer\n- Using $1/\\sqrt{d_h}$ instead of $1/d_h$ for attention\n- Not scaling the hidden layer learning rate\n- Applying μP scaling to embeddings (which should use base LR)\n\nIf any layer's activation norm varies systematically with width, something is wrong."
    },
    // Step 14: MC — Coordinate check
    {
      type: "mc",
      question: "A researcher implements μP and runs the coordinate check: they train models at widths 256, 512, 1024, and 2048 for 100 steps and plot the L1 norm of hidden layer activations vs. width. The plot shows a downward slope — norms decrease as width increases. What is the most likely bug?",
      options: [
        "The output layer multiplier $1/m$ was applied twice, causing the output signal to shrink and propagating smaller gradients back to hidden layers",
        "The attention scaling was set to $1/d_h^2$ instead of $1/d_h$, making attention too uniform at larger widths and reducing learned correlations",
        "The hidden layer learning rate was not scaled by $1/m$ — it's still using the global LR, so updates shrink as $\\Theta(1/\\sqrt{d})$ under SP behavior",
        "The initialization variance was set to $1/d^2$ instead of $1/d$, making initial weights too small at large widths and suppressing forward-pass activations"
      ],
      correct: 2,
      explanation: "If hidden layer LR is not scaled by $1/m$ (i.e., it's still the fixed global LR), then the update magnitude $\\|\\Delta y\\|$ shrinks as $\\Theta(1/\\sqrt{d})$ — this is exactly the SP failure mode. Activations change less at larger widths, so after 100 training steps, wider models have norms closer to initialization while narrower models have diverged more. The fix is to apply the $1/m$ LR multiplier to hidden layers."
    },
    // Step 15: MC — Big-picture synthesis
    {
      type: "mc",
      question: "A startup has a \\$2M compute budget for pretraining a 13B-parameter LLM. They allocate \\$50K to HP tuning. Under the μTransfer paradigm with μP, they should:",
      options: [
        "Train the full 13B model for 2.5% of the planned tokens to identify good HPs, then restart from scratch with the best settings",
        "Train a width-scaled proxy (same depth, smaller $d$) under μP, sweep HPs cheaply, then transfer the best global HPs to the 13B model with μP-adjusted per-layer rates",
        "Use published HPs from a similar-sized model trained by another lab, since the optimal LR for 13B models is approximately the same regardless of architecture",
        "Skip HP tuning entirely and use the default Adam LR of $10^{-3}$ — at 13B scale the loss landscape is smooth enough that any reasonable LR converges to the same minimum"
      ],
      correct: 1,
      explanation: "μTransfer's workflow: (1) build a small proxy with the same architecture but smaller width, (2) parameterize with μP, (3) run an extensive HP sweep at low cost, (4) transfer the best global HPs (LR, weight decay, etc.) to the 13B model — μP automatically adjusts per-layer LR multipliers via the width ratio. The \\$50K buys hundreds of small runs, giving thorough coverage of the HP space. This is far more informative than a few expensive partial runs at full scale."
    }
  ]
};
