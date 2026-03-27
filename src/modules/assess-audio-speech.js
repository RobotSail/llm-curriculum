// Assessment: Audio & Speech (E.3)
// Pure assessment — no info steps

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
        "All codebook levels are trained to contain identical redundant copies of the full audio information, providing robustness through distributed encoding so that any single codebook failure can be compensated by the others",
        "Only the final codebook level is used during inference because it captures the most complete and refined representation, while all earlier levels serve purely as intermediate training scaffolds discarded after model convergence",
        "The first RVQ level is distilled against a semantic teacher (e.g., HuBERT) to capture linguistic content, while subsequent levels encode paralinguistic details like timbre, prosody, and recording conditions",
        "Semantic and acoustic information are intentionally entangled across all codebook levels to maximize reconstruction quality, since disentanglement would reduce per-level expressiveness and degrade overall fidelity significantly"
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
      options: ["A flat sequence over all token types would need to maintain linguistic coherence and acoustic detail across thousands of tokens per second -- the hierarchy lets each stage focus on its own timescale and level of abstraction", "The two-stage approach reduces total parameter count by factoring generation into two smaller models, each handling half the representational burden, which yields equivalent quality at significantly lower total compute and memory cost", "The two stages require fundamentally different architectures -- a recurrent network for semantic modeling and a convolutional network for acoustic synthesis -- since transformers cannot handle both token types within a unified framework", "Semantic tokens from w2v-BERT are only needed for non-English speech generation, while English can proceed directly from acoustic tokens alone because of its comparatively simpler phonemic inventory and more regular structure"],
      correct: 0,
      explanation: "At 24 kHz with Encodec-style tokenization, one second of audio can produce 75+ tokens per RVQ level across 8 levels = 600+ tokens/second. A flat autoregressive model over all tokens would need to maintain coherence over thousands of tokens for a few seconds of speech. AudioLM's hierarchy solves this: the semantic stage models linguistic structure (what words/sounds to produce) at a compressed timescale, then the acoustic stage fills in the fine-grained details conditioned on those semantic decisions. Each stage's context length is manageable."
    },
    {
      type: "mc",
      question: "VALL-E (Microsoft) frames text-to-speech as a language modeling problem: given text and a 3-second audio prompt, it generates speech in the target speaker's voice. What role does the 3-second prompt serve?",
      options: [
        "It provides the phonemic transcription that guides the generated speech, since the model requires both written text and a spoken audio reference to correctly resolve grapheme-to-phoneme ambiguities present in the target utterance",
        "It is used exclusively to determine the target language and regional accent of the generated output, with the actual voice characteristics, speaker identity, and timbre being controlled by a separate learned embedding vector",
        "Its acoustic tokens serve as a prefix that defines the speaker's voice -- the model generates continuation tokens guided by the text input but acoustically consistent with the prompt's identity and timbre",
        "It establishes the duration and speaking rate of the generated output, since the model calibrates how many acoustic tokens to produce per phoneme by matching the temporal pacing observed in the prompt's token sequence"
      ],
      correct: 2,
      explanation: "VALL-E encodes the 3-second prompt into Encodec tokens that capture the speaker's timbre, pitch range, speaking style, and recording conditions. These tokens are prepended to the generation context, acting as an in-context example. The model has learned during training that the acoustic properties of the prefix should be maintained in the continuation. Combined with the phoneme sequence from the target text, the model generates speech that says the right words in the right voice -- without ever having been fine-tuned on that speaker."
    },
    {
      type: "mc",
      question: "Real-time streaming speech recognition (e.g., for live captions) faces a fundamental challenge compared to offline transcription. What is the core difficulty?",
      options: [
        "The model must emit tokens with low latency using only limited future context -- it cannot look ahead at the full utterance to resolve ambiguities, forcing a fundamental latency-accuracy tradeoff that offline models avoid entirely",
        "Streaming models require dedicated GPU hardware for real-time inference while offline models run efficiently on CPUs alone, making streaming deployment at scale significantly more expensive and operationally complex to manage and maintain",
        "Streaming models fundamentally cannot handle overlapping speech from multiple concurrent speakers, since the causal attention mask prevents attending to future audio frames that would be needed to properly separate sources",
        "Audio quality in streaming scenarios is inherently degraded by network packet loss, jitter, and codec compression artifacts, which corrupt the input signal in systematic ways that offline recording and processing pipelines avoid"
      ],
      correct: 0,
      explanation: "In offline mode, the model processes the entire utterance bidirectionally, using future context to disambiguate homophones, word boundaries, and disfluencies. In streaming mode, the model must commit to output tokens after seeing only a small lookahead window (e.g., 240ms). The word \"recognize\" and \"wreck a nice\" are acoustically similar -- offline models use sentence-level context to disambiguate, but streaming models may have emitted \"wreck\" before \"recognize\" becomes clear. This latency-accuracy tradeoff is managed via techniques like triggered attention, chunked processing, and dynamic latency control."
    },
    {
      type: "mc",
      question: "Prosody (intonation, stress, rhythm) carries significant meaning in speech. The sentence \"You're going?\" vs. \"You're going.\" differs only in pitch contour. Why is preserving prosody particularly challenging in discrete speech tokenization?",
      options: ["Prosody is a phenomenon exclusive to tonal languages like Mandarin and Cantonese, so tokenizers trained primarily on Indo-European language data learn to systematically discard pitch variation as irrelevant noise during encoding", "Prosody is fully determined by the text content and syntactic structure, so it does not need separate encoding in audio tokens -- a competent language model can reconstruct the correct intonation patterns from words alone", "Discrete codebooks force prosodic variation into finite categories, quantizing continuous $F_0$ contours and energy envelopes into coarse bins -- subtle cues like sarcasm or emphasis may collapse into the same token", "Modern discrete tokenizers with sufficiently large codebooks perfectly capture all prosodic features including pitch, stress, rhythm, and emphasis, since enough codebook entries can span the full acoustic space"],
      correct: 2,
      explanation: "Pitch ($F_0$) is a continuous signal varying from roughly 80-400 Hz in speech. A codebook entry must represent a region of this space. If the quantization is too coarse, the difference between a rising intonation (question) and a falling one (statement) may be captured, but subtler cues -- the slight pitch rise indicating sarcasm, the micro-pauses signaling hesitation, the emphasis pattern conveying focus (\"I didn't say HE stole it\" vs. \"I didn't say he STOLE it\") -- get collapsed. This is why multi-level RVQ helps: coarse levels capture broad contour, fine levels capture nuance."
    },
    {
      type: "mc",
      question: "Continuous speech representations (e.g., from wav2vec 2.0 or HuBERT encoder outputs before quantization) vs. discrete speech tokens offer different tradeoffs. Which statement best characterizes the advantage of continuous representations?",
      options: ["They preserve full information density without quantization loss, enabling higher fidelity and smoother interpolation -- but cannot be directly consumed by autoregressive language models requiring categorical distributions over a finite vocabulary", "Continuous representations are always smaller in memory footprint than discrete token sequences, since floating-point vectors compress more efficiently than sequences of integer codebook indices spanning multiple residual quantization levels", "Continuous representations can be extracted directly from raw audio waveforms without any neural network encoder, relying instead on classical signal processing techniques like mel spectrograms, MFCCs, and linear predictive coding coefficients", "Continuous representations are faster to generate autoregressively than discrete tokens, since predicting a continuous vector requires only a single regression forward pass rather than iteratively sampling from a large categorical token distribution"],
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
        "No information is lost in the pipeline approach; each specialized stage preserves and forwards all relevant features from the source audio through text and into the final target language synthesis",
        "The pipeline preserves more total information than a direct model because each independently optimized stage can retain its domain-specific features with minimal loss across the translation process",
        "Only background noise and recording artifacts are lost at the text boundary, which is actually beneficial since the downstream TTS stage can synthesize cleaner and more intelligible audio than the original",
        "The text bottleneck discards all paralinguistic information -- speaker identity, prosody, emotion, and speaking rate are lost at the ASR $\\to$ text boundary and must be artificially reconstructed by TTS without access to the source"
      ],
      correct: 3,
      explanation: "The text transcript is a severe information bottleneck: it encodes *what* was said but not *how*. A transcript of \"I'm fine\" is identical whether spoken cheerfully or through tears. The TTS stage receives only text, so it cannot reproduce the source speaker's voice, emotional state, or speaking style. Direct speech-to-speech models (e.g., Translatotron, SeamlessM4T) map source audio features to target audio features without passing through text, potentially preserving paralinguistic information. However, this requires end-to-end training data with parallel speech, which is scarce."
    }
  ]
};

