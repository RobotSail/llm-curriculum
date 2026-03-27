// Assessment: Image Generation (E.2)
// Pure assessment — no info steps

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

