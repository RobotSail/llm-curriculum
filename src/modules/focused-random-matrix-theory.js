export const randomMatrixTheoryLearning = {
  id: "0.1-random-matrix-theory-learning-medium",
  sectionId: "0.1",
  title: "Random Matrix Theory for Weight Analysis",
  moduleType: "learning",
  difficulty: "medium",
  estimatedMinutes: 18,
  optional: true,
  steps: [
    {
      type: "info",
      title: "Random Matrices at Initialization",
      content: "When we initialize a neural network layer, we typically draw each weight independently from a distribution like $\\mathcal{N}(0, \\sigma^2)$. The result is a **random matrix** $W \\in \\mathbb{R}^{m \\times n}$ with i.i.d. entries.\n\nA natural question: what do the singular values of $W$ look like before any training? If you compute $W^\\top W / m$ and look at its eigenvalues, they don't scatter uniformly — they follow a specific, predictable distribution. This is the starting point of **random matrix theory (RMT)**: even \"structureless\" random matrices have highly structured spectra.\n\nUnderstanding this baseline spectrum is critical because it tells us what the weights look like when they encode **no learned information**. Any deviation from this baseline after training is a fingerprint of learning."
    },
    {
      type: "mc",
      question: "You initialize a weight matrix $W \\in \\mathbb{R}^{512 \\times 1024}$ with i.i.d. Gaussian entries and compute the eigenvalues of $W^\\top W / 512$. Before any training, what should you expect?",
      options: [
        "The eigenvalues are unpredictable because the entries are random, so no structure emerges until training",
        "The eigenvalues are all approximately equal, forming a flat spectrum around $\\sigma^2$",
        "The eigenvalues are uniformly distributed between 0 and the largest possible singular value",
        "The eigenvalues follow a specific predictable distribution determined by the matrix dimensions and entry variance"
      ],
      correct: 3,
      explanation: "Random matrix theory shows that even i.i.d. random matrices produce eigenvalue spectra with a specific, predictable shape determined by the aspect ratio $m/n$ and the entry variance $\\sigma^2$. The spectrum is not flat, not uniform, and not unpredictable — it follows the Marchenko-Pastur distribution (the Marchenko-Pastur law)."
    },
    {
      type: "info",
      title: "The Marchenko-Pastur Law",
      content: "The **Marchenko-Pastur (MP) law** describes the eigenvalue distribution of $S = W^\\top W / m$ when $W$ is an $m \\times n$ random matrix with i.i.d. entries of variance $\\sigma^2$, in the limit where both $m$ and $n$ grow large with fixed ratio $\\gamma = n / m$.\n\nWhen $\\gamma \\leq 1$, the eigenvalues concentrate in a **bulk** supported on the interval:\n\n$$\\left[ \\sigma^2 (1 - \\sqrt{\\gamma})^2,\\; \\sigma^2 (1 + \\sqrt{\\gamma})^2 \\right]$$\n\nThe density within this interval is not uniform — it has a characteristic shape that peaks and vanishes at the endpoints. The key takeaway is that the **aspect ratio $\\gamma$** controls the width of the bulk. A square matrix ($\\gamma = 1$) has the widest spread, with the lower edge touching zero. A very rectangular matrix ($\\gamma \\ll 1$) has a narrow bulk clustered near $\\sigma^2$.\n\nThis law gives us a precise **null model**: the spectrum we expect when the matrix contains no learned structure."
    },
    {
      type: "mc",
      question: "A weight matrix $W \\in \\mathbb{R}^{4096 \\times 1024}$ has entries drawn i.i.d. with variance $\\sigma^2 = 1$. Using the Marchenko-Pastur law with $\\gamma = n/m = 1/4$, what is the approximate upper edge of the bulk eigenvalue distribution of $W^\\top W / m$?",
      options: [
        "$(1 + \\sqrt{1/4})^2 = 2.25$, so the bulk extends up to about 2.25",
        "$(1 + \\sqrt{4})^2 = 9$, because $\\gamma$ should be $m/n$ not $n/m$",
        "$(1 + 1/4)^2 = 1.5625$, by substituting $\\gamma$ directly without the square root",
        "$(\\sqrt{4096} + \\sqrt{1024})^2 / 4096 \\approx 4$, from the raw dimension formula"
      ],
      correct: 0,
      explanation: "With $\\gamma = n/m = 1024/4096 = 1/4$ and $\\sigma^2 = 1$, the upper edge is $\\sigma^2(1 + \\sqrt{\\gamma})^2 = (1 + \\sqrt{0.25})^2 = (1 + 0.5)^2 = 2.25$. The other options reflect common mistakes: confusing $\\gamma$ with $m/n$, forgetting the square root, or using a non-standard formula."
    },
    {
      type: "info",
      title: "Outlier Eigenvalues as Learned Signal",
      content: "After training, the eigenvalue spectrum of a weight matrix typically has two components: the **bulk** (which still roughly follows the MP distribution) and a set of **outlier eigenvalues** that emerge above the upper MP edge.\n\nThese outliers correspond to directions in weight space where the network has learned **structured, low-rank signal**. Intuitively, gradient-based training reinforces specific input-output correlations, pushing certain singular values well above the random baseline.\n\nThe **BBP (Baik-Ben Arous-Peche) phase transition** formalizes this: a rank-1 signal added to a random matrix only produces a detectable outlier eigenvalue when the signal strength exceeds $\\sigma^2 \\sqrt{\\gamma}$. Below this threshold, the signal is \"lost\" in the bulk noise and cannot be recovered from the spectrum alone.\n\nIn practice, if you see 20 eigenvalues clearly separated above the MP bulk in a trained weight matrix, those 20 directions carry most of the learned information."
    },
    {
      type: "mc",
      question: "A trained weight matrix has eigenvalue spectrum with a clear MP bulk up to $\\lambda_{+} = 3.0$ and five eigenvalues at 15, 22, 35, 48, and 71. You then examine an untrained matrix of the same shape and find its bulk also ends near 3.0. What is the most accurate interpretation?",
      options: [
        "The five large eigenvalues indicate numerical instability during training that should be corrected with gradient clipping",
        "Both trained and untrained matrices have the same bulk edge, so the five large eigenvalues are likely statistical fluctuations consistent with the MP distribution",
        "The five eigenvalues above the bulk reflect the intrinsic rank of the data distribution, which would appear regardless of training",
        "The five outlier eigenvalues represent learned low-rank structure, encoding the dominant input-output correlations found during training"
      ],
      correct: 3,
      explanation: "Eigenvalues well above the MP bulk edge are the hallmark of learned structure. The untrained matrix confirms that the bulk edge at 3.0 is the random baseline, so eigenvalues at 15-71 are far too large to be MP fluctuations. They represent specific directions where training concentrated signal into a low-rank component. They are not instability (they are structured, not erratic), not fluctuations (they are orders of magnitude above the edge), and not intrinsic to the data (they require training to emerge)."
    },
    {
      type: "info",
      title: "The Spectral Gap and Low-Rank Approximation",
      content: "A **spectral gap** is a clear separation between outlier eigenvalues and the MP bulk. When this gap is large and sharp, it tells you something practically useful: there is a natural rank at which to truncate the matrix.\n\nSuppose a weight matrix $W$ has singular value decomposition $W = U \\Sigma V^\\top$, with $k$ singular values clearly above the bulk and the rest inside. The rank-$k$ approximation $W_k = U_k \\Sigma_k V_k^\\top$ captures the **learned signal** while discarding the random-looking component.\n\nThe quality of this approximation depends on how sharp the gap is. A **sharp spectral gap** means:\n- The choice of $k$ is unambiguous\n- The discarded singular values are well-described by MP noise\n- The rank-$k$ approximation retains almost all task-relevant information\n\nThis is the spectral basis for **weight compression**: if a $1024 \\times 1024$ matrix has only 50 outlier singular values, you can store it as two matrices of size $1024 \\times 50$ and $50 \\times 1024$, reducing parameters from $\\sim 10^6$ to $\\sim 10^5$."
    },
    {
      type: "mc",
      question: "You plot the singular values of a trained $2048 \\times 2048$ weight matrix in descending order. You see 80 values ranging from 12 to 45, then a gap, then 1968 values between 0.8 and 3.1 forming a smooth MP-like distribution. A colleague suggests compressing this layer. What is the most appropriate approach?",
      options: [
        "Keep the top 1024 singular values (half the rank) as a standard 50% compression baseline regardless of the spectral structure",
        "Keep all singular values above 1.0 since anything below 1.0 is negligible and can be safely zeroed out",
        "Use a rank-80 approximation, since the sharp spectral gap at that point separates learned signal from MP noise",
        "Apply uniform quantization to all singular values rather than truncation, since truncation destroys the bulk structure needed for generalization"
      ],
      correct: 2,
      explanation: "The sharp spectral gap after 80 singular values provides a natural, principled truncation point. The 80 outlier values carry the learned signal, while the remaining 1968 values in the MP bulk represent random-looking structure. A rank-80 approximation compresses from ~4M to ~328K parameters while retaining the task-relevant information. Arbitrary rank-1024 ignores the spectral structure; a threshold of 1.0 would cut into the bulk mid-way; and quantization doesn't exploit the low-rank structure at all."
    },
    {
      type: "info",
      title: "Weight Decay as Spectral Shrinkage",
      content: "Weight decay adds a term $\\frac{\\lambda}{2} \\|W\\|_F^2$ to the loss, which applies a gradient update of $-\\lambda W$ at each step. This is equivalent to multiplying all weights by $(1 - \\lambda \\eta)$ per step, where $\\eta$ is the learning rate.\n\nThe effect on the **singular value spectrum** is a proportional shrinkage: each singular value $\\sigma_i$ is pulled toward zero by a factor proportional to its current magnitude. In absolute terms, large singular values shrink by more per step ($\\lambda \\eta \\sigma_i$ is larger when $\\sigma_i$ is large), but in relative terms, all singular values shrink by the **same fraction**.\n\nThis has a crucial spectral consequence: weight decay compresses the **dynamic range** of the spectrum. Small singular values near the MP bulk can be driven to near-zero, effectively increasing the spectral gap. Meanwhile, large outlier singular values shrink in absolute terms but remain dominant.\n\nThe net result: weight decay acts as an implicit **spectral regularizer**, encouraging low effective rank by suppressing the bulk while preserving the relative ordering of signal directions."
    },
    {
      type: "mc",
      question: "A weight matrix has two singular values: $\\sigma_1 = 40$ (learned signal) and $\\sigma_2 = 2$ (in the MP bulk). After one step of weight decay with $\\lambda \\eta = 0.01$, what happens to the ratio $\\sigma_1 / \\sigma_2$?",
      options: [
        "The ratio stays exactly the same at 20, because multiplicative shrinkage preserves all ratios between singular values",
        "The ratio increases substantially because weight decay removes a fixed absolute amount from each singular value, shrinking $\\sigma_2$ proportionally more",
        "The ratio decreases because weight decay penalizes large values more heavily, acting like a progressive tax on singular value magnitude",
        "The ratio changes unpredictably because weight decay interacts with the gradient signal, not just the current singular values"
      ],
      correct: 0,
      explanation: "Weight decay multiplies all weights by $(1 - \\lambda\\eta) = 0.99$. After one step: $\\sigma_1 = 39.6$, $\\sigma_2 = 1.98$, and the ratio is $39.6 / 1.98 = 20$, unchanged. The shrinkage is multiplicative, so it preserves ratios exactly. In absolute terms $\\sigma_1$ loses 0.4 while $\\sigma_2$ loses only 0.02, but the proportional effect is identical. The spectral gap widens only when weight decay interacts with gradient updates over many steps, pushing bulk values toward zero faster than the gradient replenishes them."
    },
    {
      type: "info",
      title: "Monitoring Weight Spectra During Training",
      content: "In practice, tracking the eigenvalue spectrum of key weight matrices during training provides a diagnostic window into what the network is learning and when.\n\n**Early training**: The spectrum starts as pure MP bulk. As the network begins fitting structure, outlier eigenvalues emerge above the bulk edge. The number of outliers grows as the network discovers more independent signal directions.\n\n**Healthy convergence**: The outlier eigenvalues stabilize, the spectral gap sharpens, and the bulk remains well-described by the MP law. The effective rank (number of significant outliers) plateaus.\n\n**Overfitting signals**: If the number of outlier eigenvalues keeps growing without plateauing, the network may be fitting noise — it is \"learning\" structure in the training set that has no signal above the MP threshold. The bulk also distorts, losing its MP shape.\n\n**Layer comparison**: Deeper layers in well-trained networks tend to have fewer outlier eigenvalues (lower effective rank) because they represent more abstract, compressed features. If an early layer has very low rank while a deep layer has very high rank, this may indicate a bottleneck or an architectural problem.\n\nPractically, you can compute the SVD periodically (e.g., every 1000 steps) and log the top-$k$ singular values and the MP upper edge to track these trends."
    },
    {
      type: "mc",
      question: "You are training a language model and logging the singular value spectrum of each layer every 1000 steps. At step 50,000, layer 12 shows 30 outlier eigenvalues above the MP edge. By step 200,000, it shows 400 outlier eigenvalues with a blurred boundary between outliers and bulk. The validation loss stopped improving at step 120,000. What does the spectral evidence suggest?",
      options: [
        "The layer is becoming more expressive by learning additional signal directions, which is healthy and should continue with a longer training run",
        "The growing number of outliers after validation loss plateaued suggests the layer is fitting training noise, encoding spurious structure above the MP threshold",
        "The blurred spectral gap indicates the learning rate is too high, causing eigenvalues to oscillate between the bulk and outlier regions randomly",
        "The increase from 30 to 400 outliers reflects the MP bulk shifting due to changes in the effective aspect ratio as other layers update"
      ],
      correct: 1,
      explanation: "The key signal is that outlier eigenvalues kept proliferating (30 to 400) long after validation loss plateaued at step 120K. This means the network is encoding training-set-specific patterns that don't generalize — exactly what overfitting looks like in spectral terms. A blurred gap further confirms that the \"outliers\" are not strong, clean signal but marginal patterns near the noise floor. A high learning rate would cause instability across the spectrum, not specifically at the bulk-outlier boundary, and the MP bulk shift explanation doesn't account for the correlation with validation loss stagnation."
    }
  ]
};
