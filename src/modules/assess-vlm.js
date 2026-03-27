// Assessment: Vision-Language Models (Section E.1)
// 10 MC questions covering ViT, CLIP, SigLIP, LLaVA, cross-attention vs projection fusion,
// visual instruction tuning, resolution scaling, tiling, Perceiver resampling, resolution bottlenecks.

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
