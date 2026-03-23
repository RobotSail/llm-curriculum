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
        "196",
        "224",
        "784"
      ],
      correct: 1,
      explanation: "The image is divided into a grid of $(224/16) \\times (224/16) = 14 \\times 14 = 196$ non-overlapping patches. Each patch is flattened and linearly projected to produce one token embedding. This is a key design choice: ViT converts spatial structure into a sequence, allowing the standard transformer self-attention to operate over visual tokens just as it would over text tokens."
    },
    {
      type: "mc",
      question: "CLIP (Contrastive Language-Image Pretraining) trains paired image and text encoders using a contrastive objective. Given a batch of $N$ image-text pairs, what does the training loss optimize?",
      options: ["Reconstruct the image pixels from the text embedding via a decoder", "Classify each image into one of $N$ predefined categories using the text as labels", "Predict masked patches in the image conditioned on the text", "Maximize cosine similarity between matching image-text pairs and minimize it between all $N^2 - N$ non-matching pairs in the batch, using a symmetric cross-entropy loss over the similarity matrix"],
      correct: 3,
      explanation: "CLIP computes an $N \\times N$ similarity matrix of cosine similarities between all image and text embeddings in a batch. The diagonal entries are the positive pairs. Two cross-entropy losses are computed: one treating each row as a classification problem (which text matches this image?) and one treating each column similarly (which image matches this text?). The final loss is their average. This requires large batch sizes (CLIP used 32,768) to provide enough hard negatives."
    },
    {
      type: "mc",
      question: "SigLIP replaces CLIP's softmax-based contrastive loss with a sigmoid loss applied independently to each pair in the similarity matrix. What practical advantage does this provide?",
      options: ["It eliminates the need for an all-gather across devices to compute the full $N \\times N$ similarity matrix, enabling better scaling to very large batch sizes without cross-device communication", "It produces higher-quality image embeddings on all benchmarks", "It allows training without any text data", "It reduces the image encoder size by half"],
      correct: 0,
      explanation: "CLIP's softmax normalization requires the full $N \\times N$ matrix to compute the partition function, demanding an all-gather across all GPUs. SigLIP treats each of the $N^2$ pairs independently with a binary sigmoid cross-entropy: $-\\log \\sigma(z_{ij} \\cdot (2y_{ij} - 1))$, where $z_{ij}$ is the scaled similarity and $y_{ij}$ indicates if the pair matches. Each device can compute its local portion without global synchronization. This is critical at scale: SigLIP trains with batch sizes up to 1M."
    },
    {
      type: "mc",
      question: "LLaVA-style vision-language models connect a frozen visual encoder to a frozen LLM via a learned projection layer. What is the primary role of this projection?",
      options: ["It compresses the image into a single token to save context length", "It fine-tunes the visual encoder to produce text-like outputs", "It maps visual encoder embeddings from the vision embedding space into the LLM's token embedding space so visual tokens can be processed by the LLM alongside text tokens", "It replaces the LLM's self-attention with cross-attention over image features"],
      correct: 2,
      explanation: "The visual encoder (e.g., CLIP ViT) and the LLM operate in different embedding spaces with potentially different dimensionalities. The projection layer (often a simple MLP or linear layer) learns a mapping $f: \\mathbb{R}^{d_{\\text{vision}}} \\to \\mathbb{R}^{d_{\\text{LLM}}}$ so that projected visual tokens are \"in-distribution\" for the LLM. This is the core insight of the LLaVA architecture: rather than complex cross-attention, a lightweight projection suffices when combined with visual instruction tuning."
    },
    {
      type: "mc",
      question: "Cross-attention and projection-based fusion represent two paradigms for integrating visual and textual information. Which statement best captures their tradeoff?",
      options: [
        "Cross-attention is always superior because it attends to all visual tokens at every layer",
        "Projection-based fusion (LLaVA-style) is simpler and cheaper but treats visual tokens identically to text tokens in self-attention, while cross-attention (Flamingo-style) adds dedicated layers that let text tokens attend into visual features, enabling more fine-grained grounding at the cost of additional parameters and compute",
        "Both approaches produce identical results given enough training data",
        "Projection-based fusion requires modifying the visual encoder, while cross-attention does not"
      ],
      correct: 1,
      explanation: "In projection-based fusion, visual tokens are concatenated with text tokens and processed by the LLM's self-attention -- simple but the model must learn to ground text in vision entirely through self-attention. Cross-attention (as in Flamingo) inserts gated cross-attention layers where text tokens explicitly attend to visual features via $\\text{Attention}(Q_{\\text{text}}, K_{\\text{vision}}, V_{\\text{vision}})$. This provides a structured inductive bias for grounding but adds parameters and FLOPs per layer. Many recent models favor projection for its simplicity, relying on instruction tuning to teach grounding."
    },
    {
      type: "mc",
      question: "Visual instruction tuning (as introduced in LLaVA) involves fine-tuning on data that pairs images with multi-turn conversations. Why is this stage critical for VLM capability?",
      options: ["It trains the visual encoder to recognize objects for the first time", "It is only needed for models that lack a cross-attention mechanism", "It replaces the contrastive pretraining of CLIP entirely", "Without it, the model can embed visual tokens but has not learned to reason about, describe, or follow instructions involving visual content -- the projection layer alone provides spatial alignment but not task competence"],
      correct: 3,
      explanation: "After pretraining the projection layer on image-caption pairs, the model can roughly align vision and language spaces. But image captioning is a narrow task. Visual instruction tuning exposes the model to diverse tasks -- VQA, spatial reasoning, OCR, visual dialogue -- teaching it to *follow instructions* that reference visual content. This mirrors how text-only SFT unlocks a pretrained LLM's latent abilities: the knowledge is in the encoders, but instruction tuning teaches the model how to deploy it on demand."
    },
    {
      type: "mc",
      question: "Increasing the input image resolution from $224 \\times 224$ to $448 \\times 448$ with a patch size of $14 \\times 14$ changes the number of visual tokens from 256 to 1024. What is the computational consequence for self-attention over these tokens?",
      options: ["Attention cost quadruples -- self-attention has $O(n^2)$ complexity in sequence length, and the visual token count itself scales quadratically with resolution, so doubling resolution yields a $4\\times$ token increase and $16\\times$ attention cost relative to the original", "Attention cost doubles (linear scaling with token count)", "Attention cost remains the same because patch size is unchanged", "Attention cost increases by $8\\times$"],
      correct: 0,
      explanation: "At $224 \\times 224$ with patch size $14$: $(224/14)^2 = 256$ tokens. At $448 \\times 448$: $(448/14)^2 = 1024$ tokens, a $4\\times$ increase. Self-attention over $n$ tokens costs $O(n^2)$, so attention FLOPs go from $O(256^2)$ to $O(1024^2)$ -- a $16\\times$ increase. This quadratic-in-resolution scaling is why high-resolution VLMs require tiling strategies, visual token compression, or efficient attention to remain tractable."
    },
    {
      type: "mc",
      question: "Tiling strategies (used in models like LLaVA-NeXT and Monkey) handle high-resolution images by dividing them into crops. A $1344 \\times 1344$ image tiled into $6$ crops of $448 \\times 448$ each (plus a downsampled global view) presents what challenge?",
      options: ["Each crop loses all global context, making scene understanding impossible", "Tiling requires retraining the visual encoder from scratch", "The total visual token count becomes very large (e.g., $6 \\times 1024 + 256 \\approx 6400$ tokens), consuming a significant fraction of the LLM's context window and proportionally increasing inference cost", "The crops overlap, creating redundant tokens that confuse the model"],
      correct: 2,
      explanation: "With 6 high-res crops at 1024 tokens each plus a 256-token global view, the model processes ~6400 visual tokens. For an LLM with an 8K context window, this leaves only ~1600 tokens for the text conversation. Each generated text token must also attend to all 6400 visual tokens, dominating prefill cost. This is why visual token compression (e.g., pooling, resampler modules like Perceiver) is important: reduce the 6400 tokens to a few hundred while retaining spatial detail."
    },
    {
      type: "mc",
      question: "A Perceiver Resampler (as used in Flamingo) processes a variable number of visual tokens into a fixed number of learned latent queries. Formally, it applies cross-attention where queries are learned embeddings, and keys/values come from visual tokens. Why is this preferable to simply average-pooling visual tokens?",
      options: [
        "Average pooling is computationally more expensive than cross-attention",
        "A learned resampler can selectively attend to and preserve the most informative spatial features, while average pooling collapses all spatial information into a single vector, destroying localization needed for tasks like OCR or counting",
        "Average pooling changes the embedding dimensionality",
        "The resampler is needed only because visual encoders output variable-length sequences"
      ],
      correct: 1,
      explanation: "Average pooling maps all visual tokens to a single vector, irreversibly discarding spatial relationships. A Perceiver with $M$ learned queries produces $M$ output tokens, each of which can specialize (e.g., one might attend to text regions, another to object boundaries). The cross-attention scores $\\text{softmax}(Q_{\\text{learned}} K_{\\text{visual}}^T / \\sqrt{d})$ enable this selective aggregation. Tasks requiring spatial precision (OCR, counting, referring expression grounding) degrade severely under average pooling but are preserved by learned resampling."
    },
    {
      type: "mc",
      question: "A VLM trained on images at $336 \\times 336$ resolution is evaluated on a document understanding task requiring reading 10pt text. The model fails despite high performance on natural image VQA. What is the most likely root cause?",
      options: ["The language model lacks knowledge about documents", "The visual encoder cannot encode text characters at all", "Document understanding requires a separate OCR module that cannot be learned end-to-end", "At $336 \\times 336$ with $14 \\times 14$ patches, each patch covers roughly $14 \\times 14$ pixels of the original image -- too coarse to resolve individual glyphs of small text, so the textual information is lost before it even reaches the LLM"],
      correct: 3,
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
      options: ["The composition of Gaussian transitions is itself Gaussian: $q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t}\\, x_0,\\, (1-\\bar{\\alpha}_t)I)$ where $\\bar{\\alpha}_t = \\prod_{s=1}^{t}(1-\\beta_s)$, so any timestep can be sampled in closed form", "The noise at each step is correlated, enabling recursive cancellation", "The forward process uses a deterministic ODE rather than an SDE", "Each step adds the same amount of noise, making the sum trivial"],
      correct: 0,
      explanation: "Since each forward step is a linear Gaussian transformation of $x_{t-1}$ and Gaussians are closed under linear combinations, the marginal $q(x_t | x_0)$ has an analytical form. Defining $\\alpha_t = 1 - \\beta_t$ and $\\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$, we get $x_t = \\sqrt{\\bar{\\alpha}_t}\\, x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\, \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, I)$. This reparameterization is essential for efficient training: we sample a random $t$, compute $x_t$ directly from $x_0$, and train the network to predict $\\epsilon$."
    },
    {
      type: "mc",
      question: "The reverse process in DDPM learns $p_\\theta(x_{t-1} | x_t)$ to denoise from $x_T \\sim \\mathcal{N}(0, I)$ back to $x_0$. The network is typically trained to predict the noise $\\epsilon_\\theta(x_t, t)$. What loss function is used?",
      options: [
        "Adversarial loss between the generated and real images",
        "A simplified MSE: $\\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon}\\left[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2\\right]$, which is a reweighted variational lower bound on the log-likelihood",
        "Cross-entropy between predicted and true pixel values",
        "Perceptual loss computed in VGG feature space"
      ],
      correct: 1,
      explanation: "Ho et al. (2020) showed that the variational lower bound (ELBO) decomposes into per-timestep KL terms, each comparing the learned reverse step to the tractable posterior $q(x_{t-1}|x_t, x_0)$. Through reparameterization, this simplifies to predicting the noise $\\epsilon$ added at each step. The \"simple\" loss drops the per-timestep weighting from the full ELBO, yielding $\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2$. This unweighted version empirically produces better samples despite being a looser bound."
    },
    {
      type: "mc",
      question: "Classifier-free guidance (CFG) generates images using $\\tilde{\\epsilon}_\\theta(x_t, t, c) = (1 + w)\\,\\epsilon_\\theta(x_t, t, c) - w\\,\\epsilon_\\theta(x_t, t, \\varnothing)$, where $c$ is the conditioning signal and $w$ is the guidance scale. What tradeoff does increasing $w$ control?",
      options: ["Training speed vs. convergence stability", "Higher $w$ increases image resolution while reducing color depth", "Higher $w$ pushes samples toward modes of the conditional distribution, increasing text-image alignment and perceived quality at the cost of reduced diversity -- extreme $w$ produces saturated, over-sharpened images", "Higher $w$ reduces the number of denoising steps required"],
      correct: 2,
      explanation: "CFG extrapolates in the direction away from the unconditional prediction toward the conditional prediction. Geometrically, it amplifies the \"signal\" that the conditioning provides. At $w=0$, output equals the conditional model. At moderate $w$ (typically 5--15), samples become sharper and more aligned with the prompt. At very high $w$, the model over-commits to the most likely mode: colors saturate, details become exaggerated, and diversity collapses. This is the quality-diversity tradeoff -- analogous to temperature in language models but operating in score-function space."
    },
    {
      type: "mc",
      question: "During classifier-free guidance training, the conditioning signal $c$ is randomly replaced with the null embedding $\\varnothing$ with some probability $p_{\\text{uncond}}$ (typically 10--20%). Why is this dropout necessary?",
      options: [
        "It acts as regularization to prevent overfitting to the training set",
        "It trains a single network to model both $\\epsilon_\\theta(x_t, t, c)$ and $\\epsilon_\\theta(x_t, t, \\varnothing)$, so that at inference time the guidance formula can compute both the conditional and unconditional score estimates from one model",
        "It reduces GPU memory usage during training by skipping the text encoder",
        "It forces the model to learn unconditional generation first before conditional generation"
      ],
      correct: 1,
      explanation: "Classifier-free guidance requires both conditional and unconditional noise predictions at each denoising step. Rather than training two separate models, Ho & Salimans (2022) train a single model that can operate in both modes by randomly dropping the conditioning during training. When $c = \\varnothing$, the model learns the unconditional score $\\nabla_{x_t} \\log p(x_t)$; when $c$ is present, it learns the conditional score. The drop rate $p_{\\text{uncond}}$ trades off unconditional model quality against conditional model quality."
    },
    {
      type: "mc",
      question: "Latent Diffusion Models (LDMs), as used in Stable Diffusion, perform the diffusion process in a compressed latent space rather than pixel space. What is the primary motivation for this?",
      options: ["Latent spaces produce higher-fidelity images than pixel space diffusion", "Working in latent space eliminates the need for classifier-free guidance", "Pixel-space diffusion cannot be conditioned on text", "Diffusion in pixel space is computationally prohibitive at high resolutions -- a $512 \\times 512 \\times 3$ image has 786K dimensions, but a pretrained autoencoder compresses this to e.g. $64 \\times 64 \\times 4$ (16K dimensions), reducing the U-Net's compute by orders of magnitude while the autoencoder handles perceptual detail"],
      correct: 3,
      explanation: "The key insight of Rombach et al. (2022) is separating perceptual compression (handled by a pretrained autoencoder) from semantic generation (handled by the diffusion model). The autoencoder compresses images with a large spatial downsampling factor (typically $8\\times$), learning a compact latent representation that captures perceptually important features. The diffusion U-Net then operates on this much smaller tensor, making training and inference tractable at high resolutions. The decoder converts the denoised latent back to pixel space."
    },
    {
      type: "mc",
      question: "The autoencoder in Stable Diffusion is trained with a combination of reconstruction loss, perceptual loss, and a KL or VQ regularization term on the latent space. Why is regularization of the latent space critical?",
      options: ["Without regularization the latent space can have arbitrary scale and structure -- the diffusion model assumes latents are well-behaved (approximately unit Gaussian after noising), so an unregularized latent space causes the diffusion model to fail to learn a consistent denoising process", "Without regularization the autoencoder produces blurry images", "Regularization is only needed for text conditioning", "It prevents the decoder from memorizing training images"],
      correct: 0,
      explanation: "An unregularized autoencoder can learn latent spaces with widely varying scales, dead dimensions, or complex multi-modal structure. The diffusion forward process adds Gaussian noise and assumes $x_T \\approx \\mathcal{N}(0, I)$. If latent values span $[-100, 100]$ in some dimensions and $[-0.01, 0.01]$ in others, the fixed noise schedule $\\beta_t$ is miscalibrated. KL regularization (toward $\\mathcal{N}(0,1)$) or VQ regularization ensures latents are compact and well-scaled. In practice, a small KL weight produces the best balance between reconstruction quality and latent regularity."
    },
    {
      type: "mc",
      question: "VQ-VAE based autoregressive image generation (as in DALL-E 1) first encodes images into a grid of discrete codebook tokens, then trains a transformer to predict these tokens autoregressively. What is a fundamental limitation of this discrete approach compared to continuous diffusion?",
      options: [
        "Discrete codebooks cannot represent color images",
        "Autoregressive generation is always faster than diffusion",
        "The fixed codebook size creates an information bottleneck: the image must be represented by selecting from $K$ codes at each spatial position, so reconstruction quality is bounded by codebook expressiveness, and increasing $K$ makes the autoregressive modeling problem harder (larger vocabulary, sparser distribution)",
        "VQ-VAE models cannot use text conditioning"
      ],
      correct: 2,
      explanation: "With a codebook of size $K$ and a $32 \\times 32$ token grid, the image is compressed to 1024 discrete tokens from a vocabulary of $K$ (typically 8192--16384). This is lossy: fine details not captured by the nearest codebook entry are permanently lost. Increasing $K$ improves reconstruction but makes the transformer's job harder -- it must model a categorical distribution over more choices at each position, and the codebook may suffer from index collapse (many codes going unused). Continuous diffusion avoids this discretization bottleneck entirely."
    },
    {
      type: "mc",
      question: "Comparing discrete (VQ-based autoregressive) and continuous (diffusion) representations for image generation: which statement accurately characterizes their different inductive biases?",
      options: [
        "Discrete models are better at all image generation tasks",
        "Continuous diffusion models operate over the full continuous data manifold and naturally capture smooth variations (gradients, lighting), while discrete VQ models impose a categorical bottleneck but benefit from the well-understood autoregressive framework (exact log-likelihood, straightforward scaling with transformer architectures)",
        "Diffusion models cannot be scaled as large as autoregressive models",
        "VQ-based models always achieve higher FID scores than diffusion models"
      ],
      correct: 1,
      explanation: "Diffusion models predict in continuous space, naturally representing subtle variations in color, texture, and lighting without quantization artifacts. VQ models must snap every spatial feature to the nearest codebook entry, potentially introducing block artifacts. However, VQ + autoregressive models inherit the LLM scaling playbook: they use standard transformer architectures, benefit from next-token prediction infrastructure, and provide tractable likelihoods. This is why some recent unified models (e.g., Chameleon, Emu) choose discrete visual tokens for seamless integration with text generation."
    },
    {
      type: "mc",
      question: "DDIM (Denoising Diffusion Implicit Models) accelerates sampling by allowing larger step sizes in the reverse process. It achieves this by:",
      options: ["Training a faster neural network to replace the U-Net", "Reducing the image resolution during intermediate denoising steps", "Using a GAN discriminator to skip denoising steps that are \"good enough\"", "Redefining the reverse process as a non-Markovian deterministic mapping that shares the same marginals $q(x_t|x_0)$ as DDPM but allows skipping timesteps, producing a deterministic ODE trajectory from noise to image"],
      correct: 3,
      explanation: "Song et al. (2020) showed that the DDPM forward marginals $q(x_t|x_0)$ can be shared by a family of non-Markovian reverse processes with tunable stochasticity $\\eta$. At $\\eta = 0$, the reverse process becomes a deterministic ODE: given $x_t$, $x_{t-1}$ is uniquely determined. This determinism means the trajectory is smooth, allowing large step sizes (e.g., 50 steps instead of 1000) with minimal quality loss. It also enables latent interpolation: two noise vectors $x_T$ produce meaningfully interpolable images."
    },
    {
      type: "mc",
      question: "Flow matching (Lipman et al., 2023) and rectified flows provide an alternative to the DDPM noise schedule by learning a velocity field $v_\\theta(x_t, t)$ that transports a noise distribution to the data distribution along straight paths. What advantage does the straight-path formulation offer?",
      options: ["Straight paths minimize transport cost and enable high-quality generation in very few steps (sometimes 1--4), because the learned ODE trajectories have low curvature and the Euler discretization error is small", "It eliminates the need for a neural network entirely", "It only works for images smaller than $256 \\times 256$", "It replaces the U-Net with a linear transformation"],
      correct: 0,
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
      options: ["The highest-frequency components of the audio", "Only the silence/non-silence boundaries", "The coarsest approximation of the signal, with each subsequent codebook encoding the residual error from previous quantization stages, progressively refining the reconstruction", "Randomly selected frequency bands"],
      correct: 2,
      explanation: "RVQ works hierarchically: codebook 1 quantizes the original embedding to its nearest codeword, producing a coarse approximation. Codebook 2 then quantizes the *residual* (the difference between the original and the codebook-1 approximation). Codebook 3 quantizes the residual of the residual, and so on. Each layer captures progressively finer details. For speech, the first codebook typically captures phonemic/semantic content, while later codebooks encode prosodic detail, speaker timbre, and acoustic texture."
    },
    {
      type: "mc",
      question: "A key design difference between Encodec and SpeechTokenizer is how they handle the separation of semantic and acoustic information. SpeechTokenizer specifically ensures that:",
      options: [
        "All codebook levels contain identical information",
        "The first RVQ level is trained with a semantic distillation loss (e.g., from HuBERT) to capture linguistic content, while subsequent levels capture paralinguistic and acoustic details, enabling disentangled manipulation of content vs. style",
        "Only the last codebook level is used during inference",
        "Semantic and acoustic information are never separated"
      ],
      correct: 1,
      explanation: "SpeechTokenizer adds a semantic teacher loss on the first RVQ level, forcing its tokens to align with representations from a self-supervised speech model like HuBERT. This creates a natural disentanglement: level-1 tokens are approximately semantic (content, phonemes), while levels 2+ encode residual acoustic details (speaker identity, pitch contour, recording conditions). This factored representation is powerful for applications like voice conversion (swap level-1 tokens, keep levels 2+) or TTS (generate level-1 from text, then fill in acoustic details)."
    },
    {
      type: "mc",
      question: "Whisper (OpenAI) is trained on 680,000 hours of weakly supervised audio-transcript pairs scraped from the internet. What does \"weakly supervised\" mean in this context, and why is it significant?",
      options: ["The model is trained without any labels using contrastive learning", "\"Weakly supervised\" means the model is fine-tuned with reinforcement learning", "The model only learns to transcribe English", "The transcripts are not manually verified -- they come from existing subtitles, captions, and ASR outputs of varying quality, but the sheer scale and diversity of this noisy supervision produces a model that generalizes far better than prior systems trained on smaller, cleaner datasets"],
      correct: 3,
      explanation: "Prior ASR systems trained on carefully transcribed corpora (e.g., LibriSpeech, 960 hours) achieved high accuracy on matched domains but degraded on out-of-distribution audio. Whisper trades label precision for scale and diversity: internet subtitles are noisy (mistimed, paraphrased, or machine-generated), but covering 680K hours across 96+ languages and countless acoustic conditions teaches the model robust generalization. This mirrors the pretraining philosophy of LLMs: massive, noisy, diverse data outperforms small, curated data."
    },
    {
      type: "mc",
      question: "AudioLM (Google) generates audio by first predicting semantic tokens (from w2v-BERT), then using those to condition the generation of acoustic tokens (from SoundStream). Why is this two-stage coarse-to-fine approach necessary?",
      options: ["Jointly modeling semantic and acoustic tokens in a single flat sequence would require the model to simultaneously maintain long-range linguistic coherence and fine-grained acoustic detail across thousands of tokens per second -- the hierarchical approach lets each stage focus on its respective timescale and level of abstraction", "It reduces the total model size", "The two stages use different programming languages", "Semantic tokens are only needed for non-English languages"],
      correct: 0,
      explanation: "At 24 kHz with Encodec-style tokenization, one second of audio can produce 75+ tokens per RVQ level across 8 levels = 600+ tokens/second. A flat autoregressive model over all tokens would need to maintain coherence over thousands of tokens for a few seconds of speech. AudioLM's hierarchy solves this: the semantic stage models linguistic structure (what words/sounds to produce) at a compressed timescale, then the acoustic stage fills in the fine-grained details conditioned on those semantic decisions. Each stage's context length is manageable."
    },
    {
      type: "mc",
      question: "VALL-E (Microsoft) frames text-to-speech as a language modeling problem: given text and a 3-second audio prompt, it generates speech in the target speaker's voice. What role does the 3-second prompt serve?",
      options: [
        "It provides the text transcription for the generated speech",
        "It is used only to determine the language of the output",
        "Its acoustic tokens serve as a prefix/context that defines the speaker's voice characteristics -- the model continues generating tokens that are linguistically guided by the text input but acoustically consistent with the prompt's speaker identity, enabling zero-shot voice cloning",
        "It sets the maximum duration of the generated audio"
      ],
      correct: 2,
      explanation: "VALL-E encodes the 3-second prompt into Encodec tokens that capture the speaker's timbre, pitch range, speaking style, and recording conditions. These tokens are prepended to the generation context, acting as an in-context example. The model has learned during training that the acoustic properties of the prefix should be maintained in the continuation. Combined with the phoneme sequence from the target text, the model generates speech that says the right words in the right voice -- without ever having been fine-tuned on that speaker."
    },
    {
      type: "mc",
      question: "Real-time streaming speech recognition (e.g., for live captions) faces a fundamental challenge compared to offline transcription. What is the core difficulty?",
      options: [
        "Streaming models require GPUs while offline models can run on CPUs",
        "The model must emit transcription tokens with low latency while only having access to limited future audio context -- it cannot \"look ahead\" at the full utterance to resolve ambiguities, forcing a tradeoff between latency and accuracy that offline models avoid by processing the complete audio",
        "Streaming models cannot handle multiple speakers",
        "Audio quality is always worse in streaming scenarios"
      ],
      correct: 1,
      explanation: "In offline mode, the model processes the entire utterance bidirectionally, using future context to disambiguate homophones, word boundaries, and disfluencies. In streaming mode, the model must commit to output tokens after seeing only a small lookahead window (e.g., 240ms). The word \"recognize\" and \"wreck a nice\" are acoustically similar -- offline models use sentence-level context to disambiguate, but streaming models may have emitted \"wreck\" before \"recognize\" becomes clear. This latency-accuracy tradeoff is managed via techniques like triggered attention, chunked processing, and dynamic latency control."
    },
    {
      type: "mc",
      question: "Prosody (intonation, stress, rhythm) carries significant meaning in speech. The sentence \"You're going?\" vs. \"You're going.\" differs only in pitch contour. Why is preserving prosody particularly challenging in discrete speech tokenization?",
      options: ["Prosody only exists in tonal languages like Mandarin", "Prosody is fully determined by the text content and does not need to be encoded", "Modern tokenizers perfectly capture all prosodic features", "Discrete codebooks with limited size force prosodic variation into a finite set of categories, quantizing continuous $F_0$ (fundamental frequency) contours and energy envelopes into coarse bins -- subtle distinctions like sarcasm, hesitation, or emphasis that depend on fine-grained pitch and timing modulation may fall within the same quantization bucket"],
      correct: 3,
      explanation: "Pitch ($F_0$) is a continuous signal varying from roughly 80-400 Hz in speech. A codebook entry must represent a region of this space. If the quantization is too coarse, the difference between a rising intonation (question) and a falling one (statement) may be captured, but subtler cues -- the slight pitch rise indicating sarcasm, the micro-pauses signaling hesitation, the emphasis pattern conveying focus (\"I didn't say HE stole it\" vs. \"I didn't say he STOLE it\") -- get collapsed. This is why multi-level RVQ helps: coarse levels capture broad contour, fine levels capture nuance."
    },
    {
      type: "mc",
      question: "Continuous speech representations (e.g., from wav2vec 2.0 or HuBERT encoder outputs before quantization) vs. discrete speech tokens offer different tradeoffs. Which statement best characterizes the advantage of continuous representations?",
      options: ["They preserve the full information density of the encoder output without quantization loss, enabling higher reconstruction fidelity and smoother interpolation -- but they cannot be directly consumed by standard autoregressive language models, which require categorical distributions over a finite vocabulary", "Continuous representations are always smaller in memory than discrete tokens", "Continuous representations do not require a neural network to produce", "They are faster to generate autoregressively"],
      correct: 0,
      explanation: "Continuous representations retain all information from the encoder -- no codebook bottleneck. This matters for high-fidelity applications (music generation, emotional speech synthesis). However, the standard LLM next-token prediction framework uses softmax over a discrete vocabulary. To use continuous representations, you need either: (1) diffusion-based decoders that operate on continuous vectors, (2) flow-matching that maps between distributions, or (3) regression heads with continuous loss functions. Each adds architectural complexity compared to the simplicity of predicting discrete token IDs."
    },
    {
      type: "mc",
      question: "Emotion and affect in speech are conveyed through a combination of pitch variation, speaking rate, energy dynamics, and voice quality (breathiness, tenseness). A speech tokenizer trained primarily on ASR-oriented objectives (e.g., CTC loss for transcription) tends to:",
      options: ["Perfectly preserve all emotional cues since they are essential for transcription", "Only capture emotion in the first RVQ level", "Discard emotional information because ASR objectives reward tokens that are invariant to speaker affect -- the same phoneme should map to the same token whether spoken angrily or sadly, so the encoder actively learns to suppress paralinguistic variation that does not aid transcription", "Produce emotional representations that are more accurate than human perception"],
      correct: 2,
      explanation: "ASR-oriented objectives define success as correct word sequence output regardless of how the words are spoken. The ideal ASR representation is one where \"hello\" maps to the same latent whether whispered, shouted, or spoken sarcastically. This means the encoder learns to become invariant to exactly the paralinguistic features that carry emotion. Conversely, speech synthesis models need to preserve these features. This tension drives the design of disentangled representations where semantic (ASR-friendly) and paralinguistic (synthesis-friendly) information are explicitly separated."
    },
    {
      type: "mc",
      question: "A speech-to-speech translation system must preserve the source speaker's voice characteristics in the target language output. Using a pipeline approach (ASR $\\to$ MT $\\to$ TTS) vs. a direct speech-to-speech model, what information is lost in the pipeline?",
      options: [
        "No information is lost; pipelines are strictly superior",
        "The text bottleneck in the pipeline discards all paralinguistic information -- speaker identity, prosody, emotion, speaking rate, and acoustic environment are lost at the ASR $\\to$ text boundary and must be artificially reconstructed by the TTS stage, which has no access to the original audio",
        "Only background noise is lost, which is desirable",
        "The pipeline approach preserves more information because each stage is specialized"
      ],
      correct: 1,
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
      options: ["Generating each frame independently at very high resolution", "Limiting generation to very short clips (under 0.5 seconds)", "Using a separate post-processing neural network to smooth out inconsistencies", "Extending the 2D spatial attention of image diffusion models with temporal attention layers that attend across frames at corresponding spatial positions, allowing the model to enforce coherence of objects, lighting, and motion across the time dimension"],
      correct: 3,
      explanation: "Models like Video Diffusion Models, AnimateDiff, and Sora insert temporal attention blocks (or 3D convolutions) between the existing spatial attention layers. In temporal attention, a token at spatial position $(h, w)$ attends to tokens at the same position across all frames, learning that an object should maintain consistent appearance and that motion should be smooth. Some architectures use full 3D self-attention (each token attends across both space and time), trading higher compute for stronger consistency. The key insight: temporal coherence is not a post-processing problem but must be baked into the generation process."
    },
    {
      type: "mc",
      question: "A 10-second video at 24 FPS and $256 \\times 256$ resolution, tokenized with a spatial patch size of $16 \\times 16$ and temporal patch size of 1 (every frame), produces how many tokens?",
      options: ["$(10 \\times 24) \\times (256/16)^2 = 240 \\times 256 = 61,440$ tokens", "$240 \\times (16 \\times 16) = 61,440$ tokens", "$240 \\times 256 = 61,440$ tokens where 256 is the number of spatial patches per frame", "$240 \\times 256 = 61,440$ tokens"],
      correct: 0,
      explanation: "The calculation: 10 seconds $\\times$ 24 FPS = 240 frames. Each frame has $(256/16) \\times (256/16) = 16 \\times 16 = 256$ spatial patches. Total tokens: $240 \\times 256 = 61{,}440$. With self-attention at $O(n^2)$, this means $\\sim 3.8 \\times 10^9$ attention entries -- already straining memory at this modest resolution. At $1024 \\times 1024$ resolution, the same video would produce $240 \\times 4096 = 983{,}040$ tokens. This token explosion is the fundamental computational bottleneck of video generation."
    },
    {
      type: "mc",
      question: "Temporal patch size (the number of consecutive frames compressed into a single token) directly trades off temporal resolution against token count. Increasing temporal patch size from 1 to 4 (compressing 4 frames per token) reduces the token count by $4\\times$, but introduces what limitation?",
      options: ["The model can no longer generate video, only images", "Color accuracy decreases proportionally", "It reduces the model's ability to represent fast motion and rapid scene changes -- events shorter than 4 frames are blurred within a single token's representation, and the model cannot generate frame-level variations within a temporal patch", "The spatial resolution is also reduced by $4\\times$"],
      correct: 2,
      explanation: "With a temporal patch size of 4 at 24 FPS, each token spans $4/24 \\approx 167$ ms. Any motion or change occurring within that window is compressed into a single latent vector. Fast hand gestures, eye blinks, or rapid scene transitions (~100ms events) get averaged or blurred. The model also cannot specify per-frame details within a temporal patch -- it generates a \"summary\" that the decoder must expand to 4 frames. This is directly analogous to spatial patching: larger patches reduce cost but lose fine detail."
    },
    {
      type: "mc",
      question: "Frame sampling strategies for video understanding (not generation) balance temporal coverage against token budget. Uniform sampling of $N$ frames from a long video fails when:",
      options: [
        "The video contains only static scenes",
        "Important events are brief relative to the video duration -- in a 5-minute video sampled to 16 frames, each frame represents ~19 seconds, so a 2-second key event may fall entirely between sampled frames and be invisible to the model",
        "The video resolution is too high",
        "The video contains more than one scene"
      ],
      correct: 1,
      explanation: "Uniform sampling assumes events are distributed evenly across the video. For a 5-minute video with 16 uniformly sampled frames, the sampling interval is ~19 seconds. A critical 2-second event (e.g., a decisive goal in a soccer match) has a roughly $2/19 \\approx 10\\%$ chance of being captured by any given frame. Adaptive strategies include: keyframe detection (sample more densely around motion/scene changes), hierarchical sampling (coarse pass to identify important segments, then fine sampling within them), or dynamic frame allocation based on query relevance."
    },
    {
      type: "mc",
      question: "Omni-modal models (e.g., GPT-4o, Gemini) aim to process and generate across text, image, audio, and video within a single model. What is the fundamental alignment challenge when integrating modalities that were pretrained separately?",
      options: ["Different modalities require different GPU types", "The tokenizers for different modalities must have identical vocabulary sizes", "Omni-modal models always require more training data than single-modality models", "Each modality's encoder was trained with different objectives (contrastive, reconstructive, predictive) and maps data to embedding spaces with different geometric structures, scales, and semantic granularities -- forcing these into a shared representation space risks either destroying modality-specific information or creating a space where cross-modal reasoning is superficial"],
      correct: 3,
      explanation: "A CLIP vision encoder learns a space optimized for image-text retrieval (contrastive, normalized embeddings on a hypersphere). A speech encoder like HuBERT learns representations optimized for phonemic discrimination. A video encoder might use reconstruction objectives. These spaces have fundamentally different geometries: contrastive spaces are hyperspherical, reconstruction spaces may be approximately Gaussian, predictive spaces may have complex manifold structure. Naively projecting all into a shared space either requires the projections to be extremely expressive (adding parameters) or accepts lossy alignment. Successful omni-modal models typically use extensive joint training to co-adapt representations."
    },
    {
      type: "mc",
      question: "A 30-second video at 720p (1280$\\times$720), 30 FPS, with spatial patch size 16 and temporal patch size 2 produces tokens for an LLM backbone. Approximately how many visual tokens result, and how does this compare to a typical LLM context window?",
      options: ["~101,250 tokens (450 temporal slots $\\times$ 225 spatial tokens per frame pair), which exceeds the 32K--128K context windows of most LLMs, requiring either aggressive compression, sparse attention, or hierarchical processing", "~2,000 tokens, fitting easily in any context window", "~1,000,000 tokens, requiring a completely new architecture", "~10,000 tokens, manageable with standard attention"],
      correct: 0,
      explanation: "Frames: $30 \\times 30 = 900$. With temporal patch size 2: $900/2 = 450$ temporal slots. Spatial tokens per frame pair: $(1280/16) \\times (720/16) = 80 \\times 45 = 3600$. Total: $450 \\times 3600 = 1{,}620{,}000$ -- actually even larger than option C. Even with aggressive spatial compression to, say, 225 tokens (via a Perceiver or $4\\times$ spatial downsampling to 320$\\times$180 effective), you get $450 \\times 225 \\approx 101{,}250$ tokens. This far exceeds standard context windows, illustrating why video models must use aggressive spatial-temporal compression, memory-efficient attention, or streaming approaches."
    },
    {
      type: "mc",
      question: "Video generation models often use a cascaded approach: generate at low resolution (e.g., $64 \\times 64$), then apply spatial and temporal super-resolution models. Compared to directly generating at high resolution, what tradeoff does cascading introduce?",
      options: ["Cascading always produces better results with no downsides", "Cascading requires exactly 3 stages to work", "Each cascade stage can introduce its own artifacts and inconsistencies -- spatial upsampling may hallucinate fine details that are temporally inconsistent, temporal interpolation may create ghosting between keyframes, and errors accumulate across stages, but the approach is far more computationally tractable than direct high-resolution generation", "The low-resolution base model does not need temporal attention"],
      correct: 2,
      explanation: "At $64 \\times 64$, the base model has a manageable $16$ spatial tokens per frame (with patch size 16) or $64$ tokens (patch size 8). Spatial super-resolution $64 \\to 256 \\to 1024$ adds details but must invent fine-grained textures that are consistent across frames. Temporal super-resolution (generating intermediate frames) must produce smooth motion but may create ghosting when objects move non-linearly between keyframes. Each stage is independently trained and may have different error modes. Despite these challenges, cascading is the dominant approach because direct $1024 \\times 1024 \\times 30\\text{fps}$ generation remains prohibitively expensive."
    },
    {
      type: "mc",
      question: "Modality-specific tokenization rates vary enormously. Text produces ~1 token per word, images produce ~256-1024 tokens, and audio at 24kHz with Encodec produces ~75 tokens/second across multiple RVQ levels. For a unified autoregressive model processing a 30-second video with audio, this asymmetry creates what problem?",
      options: [
        "The model runs out of vocabulary space",
        "The audio and video token streams overwhelm the text tokens -- a 30-second clip might produce ~150K+ video tokens and ~2250+ audio tokens but only ~50 text tokens for a caption, creating a severe imbalance where the model's capacity and context window are dominated by dense perceptual modalities, leaving minimal room for textual reasoning",
        "Text tokens become more important than audiovisual tokens",
        "The model cannot process more than one modality at a time"
      ],
      correct: 1,
      explanation: "The token rate asymmetry is extreme: video can produce 5000+ tokens/second, audio 75+ tokens/second, but text ~2-3 tokens/second of speech. In a unified sequence, the model's attention is dominated by perceptual tokens. This has several consequences: (1) the context window fills up with vision/audio before meaningful text can be included, (2) compute is spent mostly on perceptual self-attention, (3) the model may learn to \"ignore\" the sparse text tokens. Solutions include aggressive perceptual compression (Perceiver, Q-Former), modality-specific encoders that run outside the main transformer, or hierarchical architectures that process modalities at different rates."
    },
    {
      type: "mc",
      question: "Sora (OpenAI) generates videos using a diffusion transformer (DiT) operating on spacetime patches. The use of a transformer (rather than U-Net) architecture for video diffusion offers which key advantage?",
      options: ["Transformers are always faster than U-Nets for image generation", "Transformers eliminate the need for a noise schedule", "U-Nets cannot process video data at all", "Transformers treat spatial and temporal dimensions uniformly as a sequence of tokens, enabling flexible attention patterns (spatial-only, temporal-only, or full spacetime attention), seamless scaling via standard transformer scaling laws, and natural handling of variable resolutions and durations without architectural changes"],
      correct: 3,
      explanation: "U-Nets have a fixed hierarchical structure with hard-coded spatial downsampling/upsampling stages. Adapting them for video requires inserting temporal layers at each scale, creating architectural complexity. DiTs flatten everything into a 1D sequence of spacetime patch tokens, then apply standard transformer blocks. This offers: (1) uniform treatment of space and time -- attention can be spatial, temporal, or full 3D based on compute budget; (2) known scaling laws from LLMs transfer to guide capacity planning; (3) variable-length sequences handle different resolutions and durations naturally; (4) infrastructure optimizations (FlashAttention, tensor parallelism) apply directly."
    },
    {
      type: "mc",
      question: "A researcher wants to build a model that understands and generates across text, images, audio, and video. They consider two architectures: (A) separate specialized encoders/decoders per modality connected via a shared latent space, vs. (B) a single tokenizer that maps all modalities to a shared discrete vocabulary and a single autoregressive transformer. What is the core tradeoff?",
      options: ["Architecture A preserves modality-specific representations and allows each component to be optimized independently, but requires complex routing and loses the emergent cross-modal reasoning that may arise from joint training; Architecture B offers elegant simplicity and enables cross-modal in-context learning, but suffers from the vocabulary and tokenization-rate mismatches discussed earlier, and the shared capacity must serve all modalities", "Architecture B is always better because a single model is simpler to train", "Architecture A is always better because specialized components outperform general ones", "There is no meaningful tradeoff; both architectures produce identical results"],
      correct: 0,
      explanation: "This is one of the defining architectural debates in multimodal AI. Architecture A (modular, e.g., Flamingo, LLaVA) benefits from plug-and-play components: upgrade the vision encoder without retraining the LLM. But the integration points (projection layers, cross-attention) may be information bottlenecks. Architecture B (monolithic, e.g., Chameleon, Gemini-style) treats everything as tokens in one sequence, potentially enabling emergent behaviors like \"reasoning about what an image would sound like.\" But it faces tokenization-rate mismatches (video dominates the sequence), codebook design challenges (one vocabulary for all modalities?), and requires enormous training data covering all modality combinations. Current frontier models increasingly lean toward hybrid approaches."
    }
  ]
};
