// Branch E Assessments: Multimodal Models
// E.1: Vision-Language Models, E.2: Image Generation, E.3: Audio & Speech, E.4: Video & Beyond
// Pure assessment — no info steps

// ============================================================================
// E.1: Vision-Language Models
// ============================================================================
export const vlmAssessment = {
  id: "E.1-assess",
  sectionId: "E.1",
  title: "Assessment: Vision-Language Models",
  difficulty: "easy",
  estimatedMinutes: 12,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "The Vision Transformer (ViT) processes images by splitting them into fixed-size patches and linearly embedding each patch. If an image is $224 \\times 224$ pixels and the patch size is $16 \\times 16$, how many visual tokens are produced (excluding the [CLS] token)?",
      options: [
        "16",
        "784",
        "224",
        "196"
      ],
      correct: 3,
      explanation: "The image is divided into a grid of $(224/16) \\times (224/16) = 14 \\times 14 = 196$ non-overlapping patches. Each patch is flattened and linearly projected to produce one token embedding. This is a key design choice: ViT converts spatial structure into a sequence, allowing the standard transformer self-attention to operate over visual tokens just as it would over text tokens."
    },
    {
      type: "mc",
      question: "CLIP (Contrastive Language-Image Pretraining) trains paired image and text encoders using a contrastive objective. Given a batch of $N$ image-text pairs, what does the training loss optimize?",
      options: ["Reconstruct the original image pixels from the text embedding via a learned decoder, using an $L_2$ reconstruction loss to align the visual and textual modalities at the pixel level", "Classify each image into one of $N$ predefined categories derived from the batch, using the text embeddings as class-specific weight vectors in a standard softmax classification head", "Predict randomly masked patches in the image conditioned on the paired text, using a masked image modeling objective that forces the text encoder to capture fine-grained visual semantics", "Maximize cosine similarity between matching image-text pairs and minimize it between all $N^2 - N$ non-matching pairs in the batch, using a symmetric cross-entropy loss over the similarity matrix"],
      correct: 3,
      explanation: "CLIP computes an $N \\times N$ similarity matrix of cosine similarities between all image and text embeddings in a batch. The diagonal entries are the positive pairs. Two cross-entropy losses are computed: one treating each row as a classification problem (which text matches this image?) and one treating each column similarly (which image matches this text?). The final loss is their average. This requires large batch sizes (CLIP used 32,768) to provide enough hard negatives."
    },
    {
      type: "mc",
      question: "SigLIP replaces CLIP's softmax-based contrastive loss with a sigmoid loss applied independently to each pair in the similarity matrix. What practical advantage does this provide?",
      options: ["It produces uniformly higher-quality image embeddings across all downstream benchmarks by learning a sharper decision boundary between matching and non-matching pairs in the similarity space", "It eliminates the need for an all-gather across devices to compute the full $N \\times N$ similarity matrix, enabling better scaling to very large batch sizes without cross-device communication overhead", "It allows training on image-only datasets without any paired text data by using the sigmoid loss as a self-supervised clustering objective over visual features in the embedding space", "It reduces the image encoder's parameter count by approximately half through weight sharing between the similarity computation and the embedding projection layers"],
      correct: 1,
      explanation: "CLIP's softmax normalization requires the full $N \\times N$ matrix to compute the partition function, demanding an all-gather across all GPUs. SigLIP treats each of the $N^2$ pairs independently with a binary sigmoid cross-entropy: $-\\log \\sigma(z_{ij} \\cdot (2y_{ij} - 1))$, where $z_{ij}$ is the scaled similarity and $y_{ij}$ indicates if the pair matches. Each device can compute its local portion without global synchronization. This is critical at scale: SigLIP trains with batch sizes up to 1M."
    },
    {
      type: "mc",
      question: "LLaVA-style vision-language models connect a frozen visual encoder to a frozen LLM via a learned projection layer. What is the primary role of this projection?",
      options: ["It compresses all visual patches into a single aggregated token to reduce context length consumption, using a learned weighted average over patch embeddings", "It fine-tunes the visual encoder end-to-end to produce outputs that already lie in the LLM's token embedding space, eliminating the need for any separate alignment stage", "It maps visual encoder embeddings from the vision embedding space into the LLM's token embedding space so visual tokens can be processed by the LLM alongside text tokens", "It replaces the LLM's self-attention mechanism with cross-attention layers that attend over the raw image features at every transformer block in the decoder"],
      correct: 2,
      explanation: "The visual encoder (e.g., CLIP ViT) and the LLM operate in different embedding spaces with potentially different dimensionalities. The projection layer (often a simple MLP or linear layer) learns a mapping $f: \\mathbb{R}^{d_{\\text{vision}}} \\to \\mathbb{R}^{d_{\\text{LLM}}}$ so that projected visual tokens are \"in-distribution\" for the LLM. This is the core insight of the LLaVA architecture: rather than complex cross-attention, a lightweight projection suffices when combined with visual instruction tuning."
    },
    {
      type: "mc",
      question: "Cross-attention and projection-based fusion represent two paradigms for integrating visual and textual information. Which statement best captures their tradeoff?",
      options: [
        "Projection-based fusion (LLaVA-style) is simpler and cheaper but treats visual tokens identically to text tokens in self-attention, while cross-attention (Flamingo-style) adds dedicated layers that let text tokens attend into visual features, enabling more fine-grained grounding at the cost of additional parameters and compute",
        "Cross-attention is strictly superior in all settings because it provides explicit grounding by attending to all visual tokens at every decoder layer, and this additional computational cost always translates to proportionally better multimodal understanding",
        "Both approaches converge to identical downstream performance given sufficient paired training data and compute budget, so the choice between projection-based and cross-attention fusion is purely one of implementation convenience and engineering preference",
        "Projection-based fusion requires modifying the visual encoder's internal architecture to produce text-compatible features through structural changes to its attention layers, while cross-attention keeps both the visual encoder and LLM backbone entirely frozen during training"
      ],
      correct: 0,
      explanation: "In projection-based fusion, visual tokens are concatenated with text tokens and processed by the LLM's self-attention -- simple but the model must learn to ground text in vision entirely through self-attention. Cross-attention (as in Flamingo) inserts gated cross-attention layers where text tokens explicitly attend to visual features via $\\text{Attention}(Q_{\\text{text}}, K_{\\text{vision}}, V_{\\text{vision}})$. This provides a structured inductive bias for grounding but adds parameters and FLOPs per layer. Many recent models favor projection for its simplicity, relying on instruction tuning to teach grounding."
    },
    {
      type: "mc",
      question: "Visual instruction tuning (as introduced in LLaVA) involves fine-tuning on data that pairs images with multi-turn conversations. Why is this stage critical for VLM capability?",
      options: ["Without it, the model can embed visual tokens but has not learned to reason about, describe, or follow instructions involving visual content -- the projection layer alone provides spatial alignment but not task competence", "It is only necessary for projection-based models that lack cross-attention, since cross-attention models already learn visual grounding implicitly through their architectural inductive bias", "It replaces the contrastive pretraining stage of CLIP entirely, providing a more task-relevant visual representation learned end-to-end from conversational image-text pairs", "It trains the visual encoder from scratch to recognize objects, since the frozen CLIP encoder lacks the fine-grained spatial understanding needed for conversational visual tasks"],
      correct: 0,
      explanation: "After pretraining the projection layer on image-caption pairs, the model can roughly align vision and language spaces. But image captioning is a narrow task. Visual instruction tuning exposes the model to diverse tasks -- VQA, spatial reasoning, OCR, visual dialogue -- teaching it to *follow instructions* that reference visual content. This mirrors how text-only SFT unlocks a pretrained LLM's latent abilities: the knowledge is in the encoders, but instruction tuning teaches the model how to deploy it on demand."
    },
    {
      type: "mc",
      question: "Increasing the input image resolution from $224 \\times 224$ to $448 \\times 448$ with a patch size of $14 \\times 14$ changes the number of visual tokens from 256 to 1024. What is the computational consequence for self-attention over these tokens?",
      options: ["Attention cost increases by $8\\times$ -- the token count quadruples but the attention mechanism only computes pairwise interactions within each row of the spatial grid, not across the full sequence", "Attention cost increases by $2\\times$ -- self-attention scales linearly with token count since each token attends to a fixed-size local window determined by the patch size, not the full sequence", "Attention cost remains unchanged because the patch size is the same and each patch undergoes identical computation regardless of how many other patches exist in the sequence", "Attention cost increases by $16\\times$ -- self-attention has $O(n^2)$ complexity in sequence length, and the visual token count itself scales quadratically with resolution, so doubling resolution yields a $4\\times$ token increase and $16\\times$ attention cost relative to the original"],
      correct: 3,
      explanation: "At $224 \\times 224$ with patch size $14$: $(224/14)^2 = 256$ tokens. At $448 \\times 448$: $(448/14)^2 = 1024$ tokens, a $4\\times$ increase. Self-attention over $n$ tokens costs $O(n^2)$, so attention FLOPs go from $O(256^2)$ to $O(1024^2)$ -- a $16\\times$ increase. This quadratic-in-resolution scaling is why high-resolution VLMs require tiling strategies, visual token compression, or efficient attention to remain tractable."
    },
    {
      type: "mc",
      question: "Tiling strategies (used in models like LLaVA-NeXT and Monkey) handle high-resolution images by dividing them into crops. A $1344 \\times 1344$ image tiled into $6$ crops of $448 \\times 448$ each (plus a downsampled global view) presents what challenge?",
      options: ["The total visual token count becomes very large (e.g., $6 \\times 1024 + 256 \\approx 6400$ tokens), consuming a significant fraction of the LLM's context window and proportionally increasing inference cost", "Tiling requires retraining the visual encoder from scratch on cropped image distributions, since a ViT pretrained on full images produces degraded features when applied to partial crops", "Each crop loses all global context since crops are processed independently, making holistic scene understanding impossible even with the global view providing only a low-resolution summary", "The crops overlap at their boundaries by design, creating redundant tokens at shared edges that confuse the model's spatial reasoning and introduce inconsistent duplicate representations"],
      correct: 0,
      explanation: "With 6 high-res crops at 1024 tokens each plus a 256-token global view, the model processes ~6400 visual tokens. For an LLM with an 8K context window, this leaves only ~1600 tokens for the text conversation. Each generated text token must also attend to all 6400 visual tokens, dominating prefill cost. This is why visual token compression (e.g., pooling, resampler modules like Perceiver) is important: reduce the 6400 tokens to a few hundred while retaining spatial detail."
    },
    {
      type: "mc",
      question: "A Perceiver Resampler (as used in Flamingo) processes a variable number of visual tokens into a fixed number of learned latent queries. Formally, it applies cross-attention where queries are learned embeddings, and keys/values come from visual tokens. Why is this preferable to simply average-pooling visual tokens?",
      options: [
        "Average pooling is computationally more expensive than cross-attention with learned queries, since it requires materializing attention weights over all visual tokens",
        "A learned resampler can selectively attend to and preserve the most informative spatial features, while average pooling collapses all spatial information into a single vector, destroying localization needed for tasks like OCR or counting",
        "Average pooling introduces a dimensionality mismatch between the visual encoder output and the LLM input, requiring an additional linear projection layer to resolve",
        "The resampler is needed only because different visual encoders output variable-length sequences, and a fixed-size output is required for batched processing in the LLM"
      ],
      correct: 1,
      explanation: "Average pooling maps all visual tokens to a single vector, irreversibly discarding spatial relationships. A Perceiver with $M$ learned queries produces $M$ output tokens, each of which can specialize (e.g., one might attend to text regions, another to object boundaries). The cross-attention scores $\\text{softmax}(Q_{\\text{learned}} K_{\\text{visual}}^T / \\sqrt{d})$ enable this selective aggregation. Tasks requiring spatial precision (OCR, counting, referring expression grounding) degrade severely under average pooling but are preserved by learned resampling."
    },
    {
      type: "mc",
      question: "A VLM trained on images at $336 \\times 336$ resolution is evaluated on a document understanding task requiring reading 10pt text. The model fails despite high performance on natural image VQA. What is the most likely root cause?",
      options: ["The language model's pretraining corpus lacked structured documents and forms, so it has no internal knowledge about document layouts, table structures, or reading order conventions", "The visual encoder's convolutional patch embedding cannot encode text characters at all, since character shapes require dedicated OCR-specific feature extractors not present in standard ViT architectures", "At $336 \\times 336$ with $14 \\times 14$ patches, each patch covers roughly $14 \\times 14$ pixels of the original image -- too coarse to resolve individual glyphs of small text, so the textual information is lost before it even reaches the LLM", "Document understanding requires a separate pretrained OCR module to extract text regions first, since end-to-end learning cannot jointly optimize character recognition and semantic comprehension objectives"],
      correct: 2,
      explanation: "At $336 \\times 336$, a document image that was originally $2000 \\times 2000$ is downsampled by roughly $6\\times$. After ViT patching at $14 \\times 14$, each patch corresponds to roughly $84 \\times 84$ pixels in the original image. For 10pt text (~13 pixels tall at 96 DPI), multiple characters are collapsed into a single patch where they become indistinguishable. This is a resolution bottleneck, not a capability bottleneck. Higher resolution or tiling strategies (LLaVA-NeXT, InternVL) directly address this by preserving fine-grained pixel detail."
    }
  ]
};

// ============================================================================
// E.2: Image Generation
// ============================================================================
export const imageGenAssessment = {
  id: "E.2-assess",
  sectionId: "E.2",
  title: "Assessment: Image Generation",
  difficulty: "medium",
  estimatedMinutes: 14,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "In the forward (diffusion) process of a denoising diffusion probabilistic model (DDPM), noise is added to the data according to a fixed schedule $q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t}\\, x_{t-1},\\, \\beta_t I)$. A key property is that $x_t$ can be sampled directly from $x_0$ without iterating through intermediate steps. What enables this?",
      options: ["The forward process follows a deterministic ODE trajectory rather than a stochastic SDE, so the path from $x_0$ to $x_t$ is uniquely determined without requiring intermediate samples", "The noise added at each step is correlated with previous steps via a shared random seed, enabling recursive cancellation of intermediate noise terms through the Markov property", "The composition of Gaussian transitions is itself Gaussian: $q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t}\\, x_0,\\, (1-\\bar{\\alpha}_t)I)$ where $\\bar{\\alpha}_t = \\prod_{s=1}^{t}(1-\\beta_s)$, so any timestep can be sampled in closed form", "Each step adds exactly the same amount of noise $\\beta$ regardless of the timestep, making the total noise after $t$ steps simply $t \\cdot \\beta$ and the closed-form expression trivial"],
      correct: 2,
      explanation: "Since each forward step is a linear Gaussian transformation of $x_{t-1}$ and Gaussians are closed under linear combinations, the marginal $q(x_t | x_0)$ has an analytical form. Defining $\\alpha_t = 1 - \\beta_t$ and $\\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$, we get $x_t = \\sqrt{\\bar{\\alpha}_t}\\, x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\, \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, I)$. This reparameterization is essential for efficient training: we sample a random $t$, compute $x_t$ directly from $x_0$, and train the network to predict $\\epsilon$."
    },
    {
      type: "mc",
      question: "The reverse process in DDPM learns $p_\\theta(x_{t-1} | x_t)$ to denoise from $x_T \\sim \\mathcal{N}(0, I)$ back to $x_0$. The network is typically trained to predict the noise $\\epsilon_\\theta(x_t, t)$. What loss function is used?",
      options: [
        "Adversarial loss where a discriminator distinguishes between the model's predicted clean image $\\hat{x}_0$ and the true data sample $x_0$, pushing the denoiser to produce realistic outputs",
        "Cross-entropy between the predicted per-pixel color distributions $p_\\theta(x_0 | x_t)$ and the one-hot ground truth pixel values, treating each pixel as an independent classification problem",
        "A simplified MSE: $\\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon}\\left[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2\\right]$, which is a reweighted variational lower bound on the log-likelihood",
        "Perceptual loss computed by comparing intermediate VGG feature activations of the predicted clean image $\\hat{x}_0$ against the true image $x_0$ at multiple network layers"
      ],
      correct: 2,
      explanation: "Ho et al. (2020) showed that the variational lower bound (ELBO) decomposes into per-timestep KL terms, each comparing the learned reverse step to the tractable posterior $q(x_{t-1}|x_t, x_0)$. Through reparameterization, this simplifies to predicting the noise $\\epsilon$ added at each step. The \"simple\" loss drops the per-timestep weighting from the full ELBO, yielding $\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2$. This unweighted version empirically produces better samples despite being a looser bound."
    },
    {
      type: "mc",
      question: "Classifier-free guidance (CFG) generates images using $\\tilde{\\epsilon}_\\theta(x_t, t, c) = (1 + w)\\,\\epsilon_\\theta(x_t, t, c) - w\\,\\epsilon_\\theta(x_t, t, \\varnothing)$, where $c$ is the conditioning signal and $w$ is the guidance scale. What tradeoff does increasing $w$ control?",
      options: ["Higher $w$ accelerates convergence during the reverse diffusion process, reducing the wall-clock time per sample at the cost of decreased numerical stability in the noise prediction estimates", "Higher $w$ increases the effective output resolution of the generated image by sharpening high-frequency spatial details, while simultaneously reducing the dynamic range of color values in low-frequency regions", "Higher $w$ reduces the number of denoising steps required for convergence by amplifying the signal-to-noise ratio at each step, enabling faster sampling with fewer function evaluations of the noise prediction network", "Higher $w$ pushes samples toward modes of the conditional distribution, increasing text-image alignment and perceived quality at the cost of reduced diversity -- extreme $w$ produces saturated, over-sharpened images"],
      correct: 3,
      explanation: "CFG extrapolates in the direction away from the unconditional prediction toward the conditional prediction. Geometrically, it amplifies the \"signal\" that the conditioning provides. At $w=0$, output equals the conditional model. At moderate $w$ (typically 5--15), samples become sharper and more aligned with the prompt. At very high $w$, the model over-commits to the most likely mode: colors saturate, details become exaggerated, and diversity collapses. This is the quality-diversity tradeoff -- analogous to temperature in language models but operating in score-function space."
    },
    {
      type: "mc",
      question: "During classifier-free guidance training, the conditioning signal $c$ is randomly replaced with the null embedding $\\varnothing$ with some probability $p_{\\text{uncond}}$ (typically 10--20%). Why is this dropout necessary?",
      options: [
        "It trains a single network to model both $\\epsilon_\\theta(x_t, t, c)$ and $\\epsilon_\\theta(x_t, t, \\varnothing)$, so that at inference time the guidance formula can compute both the conditional and unconditional score estimates from one model",
        "It acts as regularization that prevents overfitting to the paired text-image training data by forcing the model to reconstruct noisy images without relying on the text signal, improving generalization to unseen prompts",
        "It reduces peak GPU memory consumption during training by periodically skipping the text encoder's forward pass, allowing larger batch sizes that improve the quality of the learned noise predictions",
        "It forces the model to master unconditional image generation before learning conditional generation, establishing a strong prior over natural images that the conditioning signal can then steer toward specific outputs"
      ],
      correct: 0,
      explanation: "Classifier-free guidance requires both conditional and unconditional noise predictions at each denoising step. Rather than training two separate models, Ho & Salimans (2022) train a single model that can operate in both modes by randomly dropping the conditioning during training. When $c = \\varnothing$, the model learns the unconditional score $\\nabla_{x_t} \\log p(x_t)$; when $c$ is present, it learns the conditional score. The drop rate $p_{\\text{uncond}}$ trades off unconditional model quality against conditional model quality."
    },
    {
      type: "mc",
      question: "Latent Diffusion Models (LDMs), as used in Stable Diffusion, perform the diffusion process in a compressed latent space rather than pixel space. What is the primary motivation for this?",
      options: ["Latent spaces produce inherently higher-fidelity images than pixel space diffusion because the autoencoder's decoder adds perceptual sharpening that compensates for any denoising artifacts in the latent representation", "Working in a compressed latent space eliminates the need for classifier-free guidance because the latent bottleneck already constrains the output distribution to match the conditioning signal without explicit score extrapolation", "Pixel-space diffusion models cannot be effectively conditioned on text prompts because the high dimensionality of pixel space dilutes the text signal, making the conditional and unconditional scores nearly indistinguishable", "Diffusion in pixel space is computationally prohibitive at high resolutions -- a $512 \\times 512 \\times 3$ image has 786K dimensions, but a pretrained autoencoder compresses this to e.g. $64 \\times 64 \\times 4$ (16K dimensions), reducing the U-Net's compute by orders of magnitude while the autoencoder handles perceptual detail"],
      correct: 3,
      explanation: "The key insight of Rombach et al. (2022) is separating perceptual compression (handled by a pretrained autoencoder) from semantic generation (handled by the diffusion model). The autoencoder compresses images with a large spatial downsampling factor (typically $8\\times$), learning a compact latent representation that captures perceptually important features. The diffusion U-Net then operates on this much smaller tensor, making training and inference tractable at high resolutions. The decoder converts the denoised latent back to pixel space."
    },
    {
      type: "mc",
      question: "The autoencoder in Stable Diffusion is trained with a combination of reconstruction loss, perceptual loss, and a KL or VQ regularization term on the latent space. Why is regularization of the latent space critical?",
      options: ["Without regularization the latent space can have arbitrary scale and structure -- the diffusion model assumes latents are well-behaved (approximately unit Gaussian after noising), so an unregularized latent space causes the diffusion model to fail to learn a consistent denoising process", "Without regularization the autoencoder learns latent codes that decode to perceptually blurry images, because the encoder distributes information too evenly across all latent dimensions rather than concentrating it in a structured low-dimensional manifold", "Regularization is necessary only when conditioning on text, because the cross-attention between text embeddings and latent features requires the latent space to have a specific geometric structure compatible with the text encoder's output distribution", "Regularization prevents the decoder from memorizing individual training images by forcing the latent codes of similar images to overlap, ensuring the decoder must learn generalizable reconstruction patterns rather than image-specific lookup tables"],
      correct: 0,
      explanation: "An unregularized autoencoder can learn latent spaces with widely varying scales, dead dimensions, or complex multi-modal structure. The diffusion forward process adds Gaussian noise and assumes $x_T \\approx \\mathcal{N}(0, I)$. If latent values span $[-100, 100]$ in some dimensions and $[-0.01, 0.01]$ in others, the fixed noise schedule $\\beta_t$ is miscalibrated. KL regularization (toward $\\mathcal{N}(0,1)$) or VQ regularization ensures latents are compact and well-scaled. In practice, a small KL weight produces the best balance between reconstruction quality and latent regularity."
    },
    {
      type: "mc",
      question: "VQ-VAE based autoregressive image generation (as in DALL-E 1) first encodes images into a grid of discrete codebook tokens, then trains a transformer to predict these tokens autoregressively. What is a fundamental limitation of this discrete approach compared to continuous diffusion?",
      options: [
        "Discrete codebooks cannot faithfully represent the continuous color gradients and subtle lighting variations in natural images, introducing visible banding artifacts that diffusion models avoid entirely",
        "Autoregressive generation over discrete tokens is inherently faster than iterative diffusion sampling, since it requires only a single forward pass per token rather than hundreds of denoising steps across the full spatial grid",
        "VQ-VAE models cannot be conditioned on text because the discrete codebook indices are not differentiable, preventing gradient flow from a text encoder through the quantization step during training",
        "The fixed codebook size creates an information bottleneck: the image must be represented by selecting from $K$ codes at each spatial position, so reconstruction quality is bounded by codebook expressiveness, and increasing $K$ makes the autoregressive modeling problem harder (larger vocabulary, sparser distribution)"
      ],
      correct: 3,
      explanation: "With a codebook of size $K$ and a $32 \\times 32$ token grid, the image is compressed to 1024 discrete tokens from a vocabulary of $K$ (typically 8192--16384). This is lossy: fine details not captured by the nearest codebook entry are permanently lost. Increasing $K$ improves reconstruction but makes the transformer's job harder -- it must model a categorical distribution over more choices at each position, and the codebook may suffer from index collapse (many codes going unused). Continuous diffusion avoids this discretization bottleneck entirely."
    },
    {
      type: "mc",
      question: "Comparing discrete (VQ-based autoregressive) and continuous (diffusion) representations for image generation: which statement accurately characterizes their different inductive biases?",
      options: [
        "Discrete VQ-based models produce superior results across all image generation benchmarks because the categorical bottleneck acts as a natural regularizer that prevents mode collapse and overfitting",
        "Continuous diffusion models operate over the full continuous data manifold and naturally capture smooth variations (gradients, lighting), while discrete VQ models impose a categorical bottleneck but benefit from the well-understood autoregressive framework (exact log-likelihood, straightforward scaling with transformer architectures)",
        "Continuous diffusion models cannot be scaled to the same parameter counts as autoregressive transformer models because the U-Net architecture has fundamental scaling limitations that prevent efficient parallelization",
        "VQ-based autoregressive models consistently achieve better FID scores than diffusion models at every scale, because the discrete token prediction objective provides a sharper learning signal than the MSE denoising loss"
      ],
      correct: 1,
      explanation: "Diffusion models predict in continuous space, naturally representing subtle variations in color, texture, and lighting without quantization artifacts. VQ models must snap every spatial feature to the nearest codebook entry, potentially introducing block artifacts. However, VQ + autoregressive models inherit the LLM scaling playbook: they use standard transformer architectures, benefit from next-token prediction infrastructure, and provide tractable likelihoods. This is why some recent unified models (e.g., Chameleon, Emu) choose discrete visual tokens for seamless integration with text generation."
    },
    {
      type: "mc",
      question: "DDIM (Denoising Diffusion Implicit Models) accelerates sampling by allowing larger step sizes in the reverse process. It achieves this by:",
      options: ["Training a lightweight distilled network to approximate multiple U-Net denoising steps in a single forward pass, amortizing the cost of iterative refinement into a faster student model", "Redefining the reverse process as a non-Markovian deterministic mapping that shares the same marginals $q(x_t|x_0)$ as DDPM but allows skipping timesteps, producing a deterministic ODE trajectory from noise to image", "Incorporating a GAN discriminator that evaluates intermediate denoised images and skips remaining denoising steps once the sample is judged to be of sufficient perceptual quality by the discriminator", "Progressively reducing the spatial resolution of the image at intermediate denoising timesteps and upsampling only at the final step, so the U-Net operates on smaller tensors for most of the reverse process"],
      correct: 1,
      explanation: "Song et al. (2020) showed that the DDPM forward marginals $q(x_t|x_0)$ can be shared by a family of non-Markovian reverse processes with tunable stochasticity $\\eta$. At $\\eta = 0$, the reverse process becomes a deterministic ODE: given $x_t$, $x_{t-1}$ is uniquely determined. This determinism means the trajectory is smooth, allowing large step sizes (e.g., 50 steps instead of 1000) with minimal quality loss. It also enables latent interpolation: two noise vectors $x_T$ produce meaningfully interpolable images."
    },
    {
      type: "mc",
      question: "Flow matching (Lipman et al., 2023) and rectified flows provide an alternative to the DDPM noise schedule by learning a velocity field $v_\\theta(x_t, t)$ that transports a noise distribution to the data distribution along straight paths. What advantage does the straight-path formulation offer?",
      options: ["Straight-path transport eliminates the need for a neural network entirely, since the linear interpolation between noise and data can be computed analytically without any learned parameters", "Straight paths minimize transport cost and enable high-quality generation in very few steps (sometimes 1--4), because the learned ODE trajectories have low curvature and the Euler discretization error is small", "The straight-path formulation is only applicable to images smaller than $256 \\times 256$, because the linear transport assumption breaks down at higher resolutions where the data manifold has more complex curvature", "Straight-path transport replaces the U-Net with a single linear transformation from noise to data space, since the velocity field along straight trajectories is constant and requires no nonlinear function approximation"],
      correct: 1,
      explanation: "DDPM reverse trajectories are curved in data space because the noise schedule creates non-linear paths. Curved trajectories require many small Euler steps to follow accurately. Rectified flows / flow matching learn to transport along straight lines: $x_t = (1-t)\\epsilon + t\\, x_0$, and the model predicts the constant velocity $v = x_0 - \\epsilon$. Since straight-line ODE trajectories have zero curvature, even a single Euler step can produce reasonable results. This is the core principle behind models like Stable Diffusion 3 and FLUX, which use flow matching for efficient few-step generation."
    }
  ]
};

// ============================================================================
// E.3: Audio & Speech
// ============================================================================
export const audioAssessment = {
  id: "E.3-assess",
  sectionId: "E.3",
  title: "Assessment: Audio & Speech",
  difficulty: "medium",
  estimatedMinutes: 14,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Encodec (Meta) and SpeechTokenizer convert continuous audio waveforms into discrete token sequences using residual vector quantization (RVQ). In RVQ with $Q$ codebooks, the first codebook captures:",
      options: ["The highest-frequency components of the audio waveform, with subsequent codebooks capturing progressively lower frequency bands in a spectral decomposition hierarchy", "The coarsest approximation of the signal, with each subsequent codebook encoding the residual error from previous quantization stages, progressively refining the reconstruction", "Only the silence and non-silence boundary transitions in the audio, with subsequent codebooks filling in the spectral content within each detected speech segment", "Randomly selected non-overlapping frequency bands from the mel spectrogram, with each codebook independently responsible for a disjoint portion of the frequency spectrum"],
      correct: 1,
      explanation: "RVQ works hierarchically: codebook 1 quantizes the original embedding to its nearest codeword, producing a coarse approximation. Codebook 2 then quantizes the *residual* (the difference between the original and the codebook-1 approximation). Codebook 3 quantizes the residual of the residual, and so on. Each layer captures progressively finer details. For speech, the first codebook typically captures phonemic/semantic content, while later codebooks encode prosodic detail, speaker timbre, and acoustic texture."
    },
    {
      type: "mc",
      question: "A key design difference between Encodec and SpeechTokenizer is how they handle the separation of semantic and acoustic information. SpeechTokenizer specifically ensures that:",
      options: [
        "All codebook levels are trained to contain identical redundant copies of the full audio information, providing robustness to quantization errors through distributed encoding",
        "Only the last codebook level is used during inference because it captures the most refined representation, while earlier levels are discarded as intermediate training scaffolds",
        "The first RVQ level is trained with a semantic distillation loss (e.g., from HuBERT) to capture linguistic content, while subsequent levels capture paralinguistic and acoustic details, enabling disentangled manipulation of content vs. style",
        "Semantic and acoustic information are intentionally entangled across all codebook levels to maximize reconstruction quality, since disentanglement would reduce the expressiveness of each level"
      ],
      correct: 2,
      explanation: "SpeechTokenizer adds a semantic teacher loss on the first RVQ level, forcing its tokens to align with representations from a self-supervised speech model like HuBERT. This creates a natural disentanglement: level-1 tokens are approximately semantic (content, phonemes), while levels 2+ encode residual acoustic details (speaker identity, pitch contour, recording conditions). This factored representation is powerful for applications like voice conversion (swap level-1 tokens, keep levels 2+) or TTS (generate level-1 from text, then fill in acoustic details)."
    },
    {
      type: "mc",
      question: "Whisper (OpenAI) is trained on 680,000 hours of weakly supervised audio-transcript pairs scraped from the internet. What does \"weakly supervised\" mean in this context, and why is it significant?",
      options: ["The model is trained without any transcript labels at all, using a contrastive self-supervised objective that learns speech representations by predicting masked audio segments from surrounding context", "\"Weakly supervised\" means the model first learns from unlabeled audio via self-supervised pretraining, then is fine-tuned using reinforcement learning from human feedback on transcription quality scores", "The model is trained exclusively on English audio-transcript pairs but learns to transcribe other languages through zero-shot cross-lingual transfer enabled by shared phonetic representations across languages", "The transcripts are not manually verified -- they come from existing subtitles, captions, and ASR outputs of varying quality, but the sheer scale and diversity of this noisy supervision produces a model that generalizes far better than prior systems trained on smaller, cleaner datasets"],
      correct: 3,
      explanation: "Prior ASR systems trained on carefully transcribed corpora (e.g., LibriSpeech, 960 hours) achieved high accuracy on matched domains but degraded on out-of-distribution audio. Whisper trades label precision for scale and diversity: internet subtitles are noisy (mistimed, paraphrased, or machine-generated), but covering 680K hours across 96+ languages and countless acoustic conditions teaches the model robust generalization. This mirrors the pretraining philosophy of LLMs: massive, noisy, diverse data outperforms small, curated data."
    },
    {
      type: "mc",
      question: "AudioLM (Google) generates audio by first predicting semantic tokens (from w2v-BERT), then using those to condition the generation of acoustic tokens (from SoundStream). Why is this two-stage coarse-to-fine approach necessary?",
      options: ["Jointly modeling semantic and acoustic tokens in a single flat sequence would require the model to simultaneously maintain long-range linguistic coherence and fine-grained acoustic detail across thousands of tokens per second -- the hierarchical approach lets each stage focus on its respective timescale and level of abstraction", "The two-stage approach reduces the total model parameter count by factoring the generation problem into two smaller models, each with half the parameters of what a single joint model would require", "The two stages must use fundamentally different neural architectures (a recurrent model for semantics and a convolutional model for acoustics), since transformers cannot handle both token types within a single architecture", "Semantic tokens from w2v-BERT are only necessary for generating non-English speech, while English generation can proceed directly from acoustic tokens due to the simpler phonemic structure of English"],
      correct: 0,
      explanation: "At 24 kHz with Encodec-style tokenization, one second of audio can produce 75+ tokens per RVQ level across 8 levels = 600+ tokens/second. A flat autoregressive model over all tokens would need to maintain coherence over thousands of tokens for a few seconds of speech. AudioLM's hierarchy solves this: the semantic stage models linguistic structure (what words/sounds to produce) at a compressed timescale, then the acoustic stage fills in the fine-grained details conditioned on those semantic decisions. Each stage's context length is manageable."
    },
    {
      type: "mc",
      question: "VALL-E (Microsoft) frames text-to-speech as a language modeling problem: given text and a 3-second audio prompt, it generates speech in the target speaker's voice. What role does the 3-second prompt serve?",
      options: [
        "It provides the phonemic transcription that guides the generated speech, since the model requires both written text and a spoken reference to resolve grapheme-to-phoneme ambiguities",
        "It is used exclusively to determine the target language and accent of the generated output, with the actual voice characteristics being controlled by a separate speaker embedding vector",
        "Its acoustic tokens serve as a prefix/context that defines the speaker's voice characteristics -- the model continues generating tokens that are linguistically guided by the text input but acoustically consistent with the prompt's speaker identity, enabling zero-shot voice cloning",
        "It establishes the maximum duration and speaking rate of the generated audio, since the model uses the prompt's token rate to calibrate how many acoustic tokens to produce per phoneme"
      ],
      correct: 2,
      explanation: "VALL-E encodes the 3-second prompt into Encodec tokens that capture the speaker's timbre, pitch range, speaking style, and recording conditions. These tokens are prepended to the generation context, acting as an in-context example. The model has learned during training that the acoustic properties of the prefix should be maintained in the continuation. Combined with the phoneme sequence from the target text, the model generates speech that says the right words in the right voice -- without ever having been fine-tuned on that speaker."
    },
    {
      type: "mc",
      question: "Real-time streaming speech recognition (e.g., for live captions) faces a fundamental challenge compared to offline transcription. What is the core difficulty?",
      options: [
        "The model must emit transcription tokens with low latency while only having access to limited future audio context -- it cannot \"look ahead\" at the full utterance to resolve ambiguities, forcing a tradeoff between latency and accuracy that offline models avoid by processing the complete audio",
        "Streaming models require dedicated GPU hardware for real-time inference while offline models can run efficiently on CPUs, making streaming deployment significantly more expensive at scale",
        "Streaming models fundamentally cannot handle overlapping speech from multiple speakers, since the causal attention mask prevents attending to future frames needed to separate concurrent audio sources",
        "Audio quality in streaming scenarios is inherently degraded by network packet loss and compression artifacts, which corrupt the input signal in ways that offline recording pipelines avoid entirely"
      ],
      correct: 0,
      explanation: "In offline mode, the model processes the entire utterance bidirectionally, using future context to disambiguate homophones, word boundaries, and disfluencies. In streaming mode, the model must commit to output tokens after seeing only a small lookahead window (e.g., 240ms). The word \"recognize\" and \"wreck a nice\" are acoustically similar -- offline models use sentence-level context to disambiguate, but streaming models may have emitted \"wreck\" before \"recognize\" becomes clear. This latency-accuracy tradeoff is managed via techniques like triggered attention, chunked processing, and dynamic latency control."
    },
    {
      type: "mc",
      question: "Prosody (intonation, stress, rhythm) carries significant meaning in speech. The sentence \"You're going?\" vs. \"You're going.\" differs only in pitch contour. Why is preserving prosody particularly challenging in discrete speech tokenization?",
      options: ["Prosody is a phenomenon exclusive to tonal languages like Mandarin and Cantonese, so tokenizers trained primarily on English speech discard pitch variation as noise rather than encoding it as a meaningful feature", "Prosody is fully determined by the text content itself and does not need to be separately encoded in the audio tokens, since a competent language model can reconstruct the correct intonation pattern from the words alone", "Discrete codebooks with limited size force prosodic variation into a finite set of categories, quantizing continuous $F_0$ (fundamental frequency) contours and energy envelopes into coarse bins -- subtle distinctions like sarcasm, hesitation, or emphasis that depend on fine-grained pitch and timing modulation may fall within the same quantization bucket", "Modern discrete speech tokenizers with sufficiently large codebooks perfectly capture all prosodic features including pitch, stress, and rhythm, since the codebook entries span the full space of acoustic variation"],
      correct: 2,
      explanation: "Pitch ($F_0$) is a continuous signal varying from roughly 80-400 Hz in speech. A codebook entry must represent a region of this space. If the quantization is too coarse, the difference between a rising intonation (question) and a falling one (statement) may be captured, but subtler cues -- the slight pitch rise indicating sarcasm, the micro-pauses signaling hesitation, the emphasis pattern conveying focus (\"I didn't say HE stole it\" vs. \"I didn't say he STOLE it\") -- get collapsed. This is why multi-level RVQ helps: coarse levels capture broad contour, fine levels capture nuance."
    },
    {
      type: "mc",
      question: "Continuous speech representations (e.g., from wav2vec 2.0 or HuBERT encoder outputs before quantization) vs. discrete speech tokens offer different tradeoffs. Which statement best characterizes the advantage of continuous representations?",
      options: ["They preserve the full information density of the encoder output without quantization loss, enabling higher reconstruction fidelity and smoother interpolation -- but they cannot be directly consumed by standard autoregressive language models, which require categorical distributions over a finite vocabulary", "Continuous representations are always smaller in memory footprint than discrete token sequences, since floating-point vectors compress more efficiently than sequences of integer codebook indices across multiple RVQ levels", "Continuous representations can be extracted directly from the raw audio waveform without requiring any neural network encoder, using classical signal processing techniques like mel spectrograms and linear predictive coding coefficients", "Continuous representations are faster to generate autoregressively than discrete tokens, since predicting a continuous vector requires only a single regression step rather than sampling from a large categorical distribution"],
      correct: 0,
      explanation: "Continuous representations retain all information from the encoder -- no codebook bottleneck. This matters for high-fidelity applications (music generation, emotional speech synthesis). However, the standard LLM next-token prediction framework uses softmax over a discrete vocabulary. To use continuous representations, you need either: (1) diffusion-based decoders that operate on continuous vectors, (2) flow-matching that maps between distributions, or (3) regression heads with continuous loss functions. Each adds architectural complexity compared to the simplicity of predicting discrete token IDs."
    },
    {
      type: "mc",
      question: "Emotion and affect in speech are conveyed through a combination of pitch variation, speaking rate, energy dynamics, and voice quality (breathiness, tenseness). A speech tokenizer trained primarily on ASR-oriented objectives (e.g., CTC loss for transcription) tends to:",
      options: ["Perfectly preserve all emotional cues in the learned representations, since recognizing the speaker's emotional state is essential for correctly transcribing words that are pronounced differently under different affects", "Discard emotional information because ASR objectives reward tokens that are invariant to speaker affect -- the same phoneme should map to the same token whether spoken angrily or sadly, so the encoder actively learns to suppress paralinguistic variation that does not aid transcription", "Capture emotional information exclusively in the first RVQ level alongside the semantic content, since emotion and phonemic identity are inherently entangled in the acoustic signal and cannot be separated by any training objective", "Produce emotional representations that are more perceptually accurate than human listeners, since neural speech encoders trained at scale learn to detect subtle affective cues in pitch and timing that are below the threshold of conscious human perception"],
      correct: 1,
      explanation: "ASR-oriented objectives define success as correct word sequence output regardless of how the words are spoken. The ideal ASR representation is one where \"hello\" maps to the same latent whether whispered, shouted, or spoken sarcastically. This means the encoder learns to become invariant to exactly the paralinguistic features that carry emotion. Conversely, speech synthesis models need to preserve these features. This tension drives the design of disentangled representations where semantic (ASR-friendly) and paralinguistic (synthesis-friendly) information are explicitly separated."
    },
    {
      type: "mc",
      question: "A speech-to-speech translation system must preserve the source speaker's voice characteristics in the target language output. Using a pipeline approach (ASR $\\to$ MT $\\to$ TTS) vs. a direct speech-to-speech model, what information is lost in the pipeline?",
      options: [
        "No information is lost in the pipeline approach; each specialized stage preserves and passes through all relevant features from the source audio to the target synthesis",
        "The pipeline approach preserves more overall information than a direct model because each specialized stage can be independently optimized to retain its relevant features with minimal loss",
        "Only background noise and recording artifacts are lost at the text boundary, which is actually desirable since the TTS stage can synthesize cleaner audio than the original recording contained",
        "The text bottleneck in the pipeline discards all paralinguistic information -- speaker identity, prosody, emotion, speaking rate, and acoustic environment are lost at the ASR $\\to$ text boundary and must be artificially reconstructed by the TTS stage, which has no access to the original audio"
      ],
      correct: 3,
      explanation: "The text transcript is a severe information bottleneck: it encodes *what* was said but not *how*. A transcript of \"I'm fine\" is identical whether spoken cheerfully or through tears. The TTS stage receives only text, so it cannot reproduce the source speaker's voice, emotional state, or speaking style. Direct speech-to-speech models (e.g., Translatotron, SeamlessM4T) map source audio features to target audio features without passing through text, potentially preserving paralinguistic information. However, this requires end-to-end training data with parallel speech, which is scarce."
    }
  ]
};

// ============================================================================
// E.4: Video & Beyond
// ============================================================================
export const videoAssessment = {
  id: "E.4-assess",
  sectionId: "E.4",
  title: "Assessment: Video & Beyond",
  difficulty: "hard",
  estimatedMinutes: 15,
  moduleType: "test",
  steps: [
    {
      type: "mc",
      question: "Temporal consistency is a central challenge in video generation. A diffusion model generates each frame conditioned on a text prompt, but without explicit temporal modeling, frames exhibit flickering and identity drift. What is the primary architectural approach to enforcing consistency?",
      options: ["Extending the 2D spatial attention of image diffusion models with temporal attention layers that attend across frames at corresponding spatial positions, allowing the model to enforce coherence of objects, lighting, and motion across the time dimension", "Limiting generation to very short clips under 0.5 seconds, since temporal coherence degrades proportionally with duration and short clips stay within the model's consistency radius", "Applying a separate post-processing neural network trained on optical flow estimation to smooth out temporal inconsistencies by warping each frame to align with its neighbors after generation", "Generating each frame independently at very high resolution so that the increased per-frame detail provides enough visual consistency cues for human perception to interpolate smooth motion"],
      correct: 0,
      explanation: "Models like Video Diffusion Models, AnimateDiff, and Sora insert temporal attention blocks (or 3D convolutions) between the existing spatial attention layers. In temporal attention, a token at spatial position $(h, w)$ attends to tokens at the same position across all frames, learning that an object should maintain consistent appearance and that motion should be smooth. Some architectures use full 3D self-attention (each token attends across both space and time), trading higher compute for stronger consistency. The key insight: temporal coherence is not a post-processing problem but must be baked into the generation process."
    },
    {
      type: "mc",
      question: "A 10-second video at 24 FPS and $256 \\times 256$ resolution, tokenized with a spatial patch size of $16 \\times 16$ and temporal patch size of 1 (every frame), produces how many tokens?",
      options: ["$(256/16) \\times (256/16) \\times 24 = 256 \\times 24 = 6{,}144$ tokens, since only one frame per second requires spatial tokenization", "$(10 \\times 24) \\times (16 \\times 16) = 240 \\times 256 = 61{,}440$ tokens, since each patch contributes $16 \\times 16$ sub-tokens to the sequence", "$10 \\times 24 \\times 256 = 61{,}440$ tokens, where 256 is the raw pixel count per row and each row is treated as a single spatial token", "$(10 \\times 24) \\times (256/16)^2 = 240 \\times 256 = 61{,}440$ tokens, since each of the 240 frames contributes 256 spatial patch tokens"],
      correct: 3,
      explanation: "The calculation: 10 seconds $\\times$ 24 FPS = 240 frames. Each frame has $(256/16) \\times (256/16) = 16 \\times 16 = 256$ spatial patches. Total tokens: $240 \\times 256 = 61{,}440$. With self-attention at $O(n^2)$, this means $\\sim 3.8 \\times 10^9$ attention entries -- already straining memory at this modest resolution. At $1024 \\times 1024$ resolution, the same video would produce $240 \\times 4096 = 983{,}040$ tokens. This token explosion is the fundamental computational bottleneck of video generation."
    },
    {
      type: "mc",
      question: "Temporal patch size (the number of consecutive frames compressed into a single token) directly trades off temporal resolution against token count. Increasing temporal patch size from 1 to 4 (compressing 4 frames per token) reduces the token count by $4\\times$, but introduces what limitation?",
      options: ["The model can no longer generate coherent video sequences and instead produces independent images, because compressing multiple frames into one token destroys all inter-frame relationships", "Color accuracy and tonal consistency decrease proportionally with the compression ratio, since averaging across 4 frames blends different lighting conditions into a single muddied color representation", "It reduces the model's ability to represent fast motion and rapid scene changes -- events shorter than 4 frames are blurred within a single token's representation, and the model cannot generate frame-level variations within a temporal patch", "The spatial resolution within each frame is also reduced by $4\\times$ as a side effect, because the temporal and spatial dimensions share the same compression budget in the autoencoder's bottleneck layer"],
      correct: 2,
      explanation: "With a temporal patch size of 4 at 24 FPS, each token spans $4/24 \\approx 167$ ms. Any motion or change occurring within that window is compressed into a single latent vector. Fast hand gestures, eye blinks, or rapid scene transitions (~100ms events) get averaged or blurred. The model also cannot specify per-frame details within a temporal patch -- it generates a \"summary\" that the decoder must expand to 4 frames. This is directly analogous to spatial patching: larger patches reduce cost but lose fine detail."
    },
    {
      type: "mc",
      question: "Frame sampling strategies for video understanding (not generation) balance temporal coverage against token budget. Uniform sampling of $N$ frames from a long video fails when:",
      options: [
        "The video contains predominantly static scenes with minimal motion, where uniform sampling captures redundant near-identical frames that waste the token budget",
        "The video resolution exceeds the visual encoder's maximum supported input size, requiring spatial downsampling that destroys fine-grained details regardless of temporal sampling strategy",
        "Important events are brief relative to the video duration -- in a 5-minute video sampled to 16 frames, each frame represents ~19 seconds, so a 2-second key event may fall entirely between sampled frames and be invisible to the model",
        "The video contains more than one distinct scene with different backgrounds, causing the model to confuse object identities across the sampled frames from different scene contexts"
      ],
      correct: 2,
      explanation: "Uniform sampling assumes events are distributed evenly across the video. For a 5-minute video with 16 uniformly sampled frames, the sampling interval is ~19 seconds. A critical 2-second event (e.g., a decisive goal in a soccer match) has a roughly $2/19 \\approx 10\\%$ chance of being captured by any given frame. Adaptive strategies include: keyframe detection (sample more densely around motion/scene changes), hierarchical sampling (coarse pass to identify important segments, then fine sampling within them), or dynamic frame allocation based on query relevance."
    },
    {
      type: "mc",
      question: "Omni-modal models (e.g., GPT-4o, Gemini) aim to process and generate across text, image, audio, and video within a single model. What is the fundamental alignment challenge when integrating modalities that were pretrained separately?",
      options: ["Each modality's encoder was trained with different objectives (contrastive, reconstructive, predictive) and maps data to embedding spaces with different geometric structures, scales, and semantic granularities -- forcing these into a shared representation space risks either destroying modality-specific information or creating a space where cross-modal reasoning is superficial", "The tokenizers for each modality must produce vocabularies of identical size so that the shared transformer can allocate equal attention capacity to each modality without introducing distributional bias toward the larger vocabulary", "Omni-modal models inherently require more total training data than the sum of what each single-modality model needs, because cross-modal alignment creates an additional data requirement proportional to the product of modality pairs", "Different modalities require fundamentally different GPU hardware configurations for efficient processing, making it impractical to run vision, audio, and text encoders on the same accelerator during joint training"],
      correct: 0,
      explanation: "A CLIP vision encoder learns a space optimized for image-text retrieval (contrastive, normalized embeddings on a hypersphere). A speech encoder like HuBERT learns representations optimized for phonemic discrimination. A video encoder might use reconstruction objectives. These spaces have fundamentally different geometries: contrastive spaces are hyperspherical, reconstruction spaces may be approximately Gaussian, predictive spaces may have complex manifold structure. Naively projecting all into a shared space either requires the projections to be extremely expressive (adding parameters) or accepts lossy alignment. Successful omni-modal models typically use extensive joint training to co-adapt representations."
    },
    {
      type: "mc",
      question: "A 30-second video at 720p (1280$\\times$720), 30 FPS, with spatial patch size 16 and temporal patch size 2 produces tokens for an LLM backbone. Approximately how many visual tokens result, and how does this compare to a typical LLM context window?",
      options: ["~1,000,000 tokens (each pixel contributes roughly one token even after patching), requiring a completely new architectural paradigm beyond what current transformers can handle", "~2,000 tokens (aggressive spatial and temporal pooling reduces the representation to roughly 4--5 tokens per frame), fitting comfortably within any standard context window without compression", "~101,250 tokens (450 temporal slots $\\times$ 225 spatial tokens per frame pair), which exceeds the 32K--128K context windows of most LLMs, requiring either aggressive compression, sparse attention, or hierarchical processing", "~10,000 tokens (moderate spatial compression yields roughly 22 tokens per temporal slot), which is manageable with standard full self-attention and does not require specialized efficiency techniques"],
      correct: 2,
      explanation: "Frames: $30 \\times 30 = 900$. With temporal patch size 2: $900/2 = 450$ temporal slots. Spatial tokens per frame pair: $(1280/16) \\times (720/16) = 80 \\times 45 = 3600$. Total: $450 \\times 3600 = 1{,}620{,}000$ -- actually even larger than option C. Even with aggressive spatial compression to, say, 225 tokens (via a Perceiver or $4\\times$ spatial downsampling to 320$\\times$180 effective), you get $450 \\times 225 \\approx 101{,}250$ tokens. This far exceeds standard context windows, illustrating why video models must use aggressive spatial-temporal compression, memory-efficient attention, or streaming approaches."
    },
    {
      type: "mc",
      question: "Video generation models often use a cascaded approach: generate at low resolution (e.g., $64 \\times 64$), then apply spatial and temporal super-resolution models. Compared to directly generating at high resolution, what tradeoff does cascading introduce?",
      options: ["Each cascade stage can introduce its own artifacts and inconsistencies -- spatial upsampling may hallucinate fine details that are temporally inconsistent, temporal interpolation may create ghosting between keyframes, and errors accumulate across stages, but the approach is far more computationally tractable than direct high-resolution generation", "Cascading requires exactly 3 fixed stages (base, spatial upsampling, temporal interpolation) to work correctly, and deviating from this specific number of stages causes training instability", "Cascading always produces better results than direct generation with no downsides, since each stage can be independently optimized on its specific subtask with specialized loss functions and training data", "The low-resolution base model does not need temporal attention because at $64 \\times 64$ resolution, adjacent frames are similar enough that spatial self-attention alone captures inter-frame coherence"],
      correct: 0,
      explanation: "At $64 \\times 64$, the base model has a manageable $16$ spatial tokens per frame (with patch size 16) or $64$ tokens (patch size 8). Spatial super-resolution $64 \\to 256 \\to 1024$ adds details but must invent fine-grained textures that are consistent across frames. Temporal super-resolution (generating intermediate frames) must produce smooth motion but may create ghosting when objects move non-linearly between keyframes. Each stage is independently trained and may have different error modes. Despite these challenges, cascading is the dominant approach because direct $1024 \\times 1024 \\times 30\\text{fps}$ generation remains prohibitively expensive."
    },
    {
      type: "mc",
      question: "Modality-specific tokenization rates vary enormously. Text produces ~1 token per word, images produce ~256-1024 tokens, and audio at 24kHz with Encodec produces ~75 tokens/second across multiple RVQ levels. For a unified autoregressive model processing a 30-second video with audio, this asymmetry creates what problem?",
      options: [
        "The shared vocabulary runs out of codebook space because video, audio, and text tokens must all share a single discrete vocabulary, and the combined modality requirements exceed practical codebook sizes",
        "The audio and video token streams overwhelm the text tokens -- a 30-second clip might produce ~150K+ video tokens and ~2250+ audio tokens but only ~50 text tokens for a caption, creating a severe imbalance where the model's capacity and context window are dominated by dense perceptual modalities, leaving minimal room for textual reasoning",
        "Text tokens become disproportionately influential in the attention mechanism because their lower frequency makes each individual text token carry more semantic density, causing the model to over-weight textual reasoning",
        "The model fundamentally cannot process more than one modality at a time within a single forward pass, requiring separate sequential passes for each modality with an explicit fusion step between them"
      ],
      correct: 1,
      explanation: "The token rate asymmetry is extreme: video can produce 5000+ tokens/second, audio 75+ tokens/second, but text ~2-3 tokens/second of speech. In a unified sequence, the model's attention is dominated by perceptual tokens. This has several consequences: (1) the context window fills up with vision/audio before meaningful text can be included, (2) compute is spent mostly on perceptual self-attention, (3) the model may learn to \"ignore\" the sparse text tokens. Solutions include aggressive perceptual compression (Perceiver, Q-Former), modality-specific encoders that run outside the main transformer, or hierarchical architectures that process modalities at different rates."
    },
    {
      type: "mc",
      question: "Sora (OpenAI) generates videos using a diffusion transformer (DiT) operating on spacetime patches. The use of a transformer (rather than U-Net) architecture for video diffusion offers which key advantage?",
      options: ["Transformers are always computationally faster than U-Nets for diffusion-based image generation because self-attention avoids the redundant multi-scale feature computation inherent in the encoder-decoder structure", "Transformer-based diffusion eliminates the need for a noise schedule entirely, since the self-attention mechanism implicitly learns the optimal denoising trajectory from data rather than relying on a predefined variance schedule", "U-Nets cannot process video data at any resolution because their fixed spatial downsampling/upsampling structure does not accommodate the temporal dimension required for video frame sequences", "Transformers treat spatial and temporal dimensions uniformly as a sequence of tokens, enabling flexible attention patterns (spatial-only, temporal-only, or full spacetime attention), seamless scaling via standard transformer scaling laws, and natural handling of variable resolutions and durations without architectural changes"],
      correct: 3,
      explanation: "U-Nets have a fixed hierarchical structure with hard-coded spatial downsampling/upsampling stages. Adapting them for video requires inserting temporal layers at each scale, creating architectural complexity. DiTs flatten everything into a 1D sequence of spacetime patch tokens, then apply standard transformer blocks. This offers: (1) uniform treatment of space and time -- attention can be spatial, temporal, or full 3D based on compute budget; (2) known scaling laws from LLMs transfer to guide capacity planning; (3) variable-length sequences handle different resolutions and durations naturally; (4) infrastructure optimizations (FlashAttention, tensor parallelism) apply directly."
    },
    {
      type: "mc",
      question: "A researcher wants to build a model that understands and generates across text, images, audio, and video. They consider two architectures: (A) separate specialized encoders/decoders per modality connected via a shared latent space, vs. (B) a single tokenizer that maps all modalities to a shared discrete vocabulary and a single autoregressive transformer. What is the core tradeoff?",
      options: ["Architecture B is always superior because a single unified model is simpler to train end-to-end, avoids the integration complexity of connecting separate encoders, and benefits from positive transfer between modalities that always outweighs any capacity competition", "Architecture A preserves modality-specific representations and allows each component to be optimized independently, but requires complex routing and loses the emergent cross-modal reasoning that may arise from joint training; Architecture B offers elegant simplicity and enables cross-modal in-context learning, but suffers from the vocabulary and tokenization-rate mismatches discussed earlier, and the shared capacity must serve all modalities", "Architecture A is always superior because independently optimized specialized components consistently outperform general-purpose ones, and the modular design allows each encoder to use the ideal architecture for its modality without compromise", "There is no meaningful tradeoff between the two approaches; given sufficient training data and compute both architectures converge to identical capabilities, making the choice purely a matter of engineering convenience and implementation preference"],
      correct: 1,
      explanation: "This is one of the defining architectural debates in multimodal AI. Architecture A (modular, e.g., Flamingo, LLaVA) benefits from plug-and-play components: upgrade the vision encoder without retraining the LLM. But the integration points (projection layers, cross-attention) may be information bottlenecks. Architecture B (monolithic, e.g., Chameleon, Gemini-style) treats everything as tokens in one sequence, potentially enabling emergent behaviors like \"reasoning about what an image would sound like.\" But it faces tokenization-rate mismatches (video dominates the sequence), codebook design challenges (one vocabulary for all modalities?), and requires enormous training data covering all modality combinations. Current frontier models increasingly lean toward hybrid approaches."
    }
  ]
};
