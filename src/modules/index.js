import { svdLearning } from './focused-svd';
import { eigendecompositionLearning } from './focused-eigendecomposition';
import { positiveDefinitenessLearning } from './focused-positive-definiteness';
import { spectralNormLearning } from './focused-spectral-norm';
import { matrixCalculusLearning } from './focused-matrix-calculus';
import { einsumLearning } from './focused-einsum';
import { easyModule, hardModule } from './info-theory-f-divergences';
import { chiSquaredLearning } from './focused-chi-squared-divergence';
import { ganObjectivesLearning } from './focused-gan-objectives';
import { entropyLearning } from './focused-entropy';
import { crossEntropyLearning } from './focused-cross-entropy';
import { perplexityLearning } from './focused-perplexity';
import { mutualInformationLearning } from './focused-mutual-information';
import { labelSmoothingLearning } from './focused-label-smoothing';
import { calibrationLearning } from './focused-calibration';
import { probabilityFoundationsAssessment } from './prob-assessment-foundations';
import { exponentialFamilyAssessment } from './prob-assessment-exponential-family';
import { entropyAssessment } from './prob-assessment-entropy';
import { divergencesAssessment } from './prob-assessment-divergences';
import { bayesianAssessment } from './prob-assessment-bayesian';
import { samplingAssessment } from './prob-assessment-sampling';
import { concentrationAssessment } from './prob-assessment-concentration';
import { appliedInfoTheoryAssessment } from './prob-assessment-applied';
import { transformerAssessment } from './assess-transformer-architecture';
import { tokenizationAssessment } from './assess-tokenization';
import { pretrainingAssessment } from './assess-pretraining';
import { dataAssessment } from './assess-data';
import { evaluationAssessment } from './assess-evaluation';
import { distributedTrainingAssessment } from './assess-distributed-training';
import { sftAssessment } from './assess-sft';
import { rewardModelingAssessment } from './assess-reward-modeling';
import { rlhfAssessment } from './assess-rlhf';
import { directAlignmentAssessment } from './assess-direct-alignment';
import { frontierAlignmentAssessment } from './assess-frontier-alignment';
import { scalingLawsAssessment } from './assess-scaling-laws';
import { architectureAssessment } from './assess-architecture';
import { dataCentricAssessment } from './assess-data-centric';
import { trainingDynamicsAssessment } from './assess-training-dynamics';
import { novelObjectivesAssessment } from './assess-novel-objectives';
import { quantizationAssessment } from './assess-quantization';
import { decodingAssessment } from './assess-decoding';
import { servingAssessment } from './assess-serving';
import { compressionAssessment } from './assess-compression';
import { cotAssessment } from './assess-cot';
import { testTimeComputeAssessment } from './assess-ttc';
import { toolUseAssessment } from './assess-tool-use';
import { agenticAssessment } from './assess-agentic';
import { quantizationLearning } from './focused-quantization';
import { vlmAssessment } from './assess-vlm';
import { imageGenAssessment } from './assess-image-generation';
import { audioAssessment } from './assess-audio-speech';
import { videoAssessment } from './assess-video';
import { probingAssessment } from './assess-probing';
import { mechInterpAssessment } from './assess-mech-interp';
import { trainingInterpAssessment } from './assess-training-dynamics-interp';
import { formalTheoryAssessment } from './assess-formal-theory';
import { peftAssessment } from './assess-peft';
import { memoryEfficientAssessment } from './assess-memory-efficient';
import { hardwareAwareAssessment } from './assess-hardware-aware';
import { optimizationAssessment } from './assess-optimization';
import { systemsAssessment } from './assess-systems';

// Focused first-principles modules
import { forwardKLLearning, reverseKLLearning } from './focused-kl-divergence';
import { adamLearning } from './focused-adam-optimizer';
import { weightDecayLearning } from './focused-weight-decay';
import { muonOptimizerFundamentals } from './muon-optimizer-fundamentals';
import { policyGradientsLearning } from './focused-policy-gradients';
import { ppoMechanicsLearning } from './focused-ppo';
import { onOffPolicyLearning } from './focused-on-off-policy';
import { selfAttentionLearning } from './focused-self-attention';
import { multiHeadAttentionLearning } from './focused-multi-head-attention';
import { bpeLearning } from './focused-bpe';
import { positionalEncodingLearning } from './focused-positional-encoding';
import { residualConnectionsLearning } from './focused-residual-connections';
import { layerNormalizationLearning } from './focused-layer-normalization';
import { dataParallelismLearning } from './focused-data-parallelism';
import { dpoLearning } from './focused-dpo';
import { nextTokenPredictionLearning } from './focused-next-token-prediction';
import { sftMechanicsLearning } from './focused-sft-mechanics';
import { rewardModelingLearning } from './focused-reward-modeling';
import { dataQualityLearning } from './focused-data-quality';
import { benchmarkDesignLearning } from './focused-benchmark-design';
import { scalingLawsLearning } from './focused-scaling-laws';
import { loraLearning } from './focused-lora';
import { trainingInstabilitiesLearning } from './focused-training-instabilities';
import { mixedPrecisionLearning } from './focused-mixed-precision';
import { tensorParallelismLearning } from './focused-tensor-parallelism';
import { pipelineParallelismLearning } from './focused-pipeline-parallelism';
import { zeroFsdpLearning } from './focused-zero-fsdp';
import { mixtureOfExpertsLearning } from './focused-mixture-of-experts';
import { chainOfThoughtLearning } from './focused-chain-of-thought';
import { dataMixingLearning } from './focused-data-mixing';
import { kvCacheLearning } from './focused-kv-cache';
import { testTimeComputeLearning } from './focused-test-time-compute';
import { lrSchedulesLearning } from './focused-lr-schedules';
import { gradientClippingLearning } from './focused-gradient-clipping';
import { batchSizeScalingLearning } from './focused-batch-size-scaling';
import { secondOrderMethodsLearning } from './focused-second-order-methods';
import { emaAveragingLearning } from './focused-ema-averaging';
import { constitutionalAILearning } from './focused-constitutional-ai';

// Modules with optional: true are deep-theory / tangential content.
// They appear in the UI with an "Optional" badge and are excluded
// from daily warmup unless the user has started exploring them.
function markOptional(...mods) {
  return mods.map(m => ({ ...m, optional: true }));
}

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  // Tier 0 — Prerequisites
  "0.1": [matrixCalculusLearning, eigendecompositionLearning, positiveDefinitenessLearning, spectralNormLearning, svdLearning, einsumLearning],
  "0.3": [optimizationAssessment, adamLearning, weightDecayLearning, gradientClippingLearning, lrSchedulesLearning, batchSizeScalingLearning, secondOrderMethodsLearning, emaAveragingLearning, muonOptimizerFundamentals],
  "0.4": [systemsAssessment],
  "0.2": [
    // 1. Foundations — gauge starting level
    probabilityFoundationsAssessment,
    // 2. KL divergence from first principles
    forwardKLLearning,
    reverseKLLearning,
    // 3. Entropy, cross-entropy, perplexity — the core of LLM training
    entropyLearning,
    crossEntropyLearning,
    perplexityLearning,
    entropyAssessment,
    // 4. Divergences — KL, JS, f-divergences
    easyModule,
    divergencesAssessment,
    // 5. Intermediate — MI, chi-squared/IS, GANs
    mutualInformationLearning,
    chiSquaredLearning,
    ganObjectivesLearning,
    // 6. Bayesian & sampling methods
    bayesianAssessment,
    samplingAssessment,
    // 7. Advanced — label smoothing, calibration, variational bounds
    labelSmoothingLearning,
    calibrationLearning,
    hardModule,
    appliedInfoTheoryAssessment,
    // 8. Optional deep theory
    ...markOptional(exponentialFamilyAssessment),
    ...markOptional(concentrationAssessment),
  ],

  // Tier 1 — Foundational core
  "1.1": [selfAttentionLearning, multiHeadAttentionLearning, positionalEncodingLearning, residualConnectionsLearning, layerNormalizationLearning, transformerAssessment],
  "1.2": [bpeLearning, tokenizationAssessment],
  "1.3": [nextTokenPredictionLearning, trainingInstabilitiesLearning, pretrainingAssessment],
  "1.4": [dataQualityLearning, dataAssessment],
  "1.5": [benchmarkDesignLearning, evaluationAssessment],
  "1.6": [dataParallelismLearning, zeroFsdpLearning, tensorParallelismLearning, pipelineParallelismLearning, mixedPrecisionLearning, distributedTrainingAssessment],

  // Branch A — Post-training & alignment
  "A.1": [sftMechanicsLearning, sftAssessment],
  "A.2": [rewardModelingLearning, rewardModelingAssessment],
  "A.3": [rlhfAssessment, policyGradientsLearning, ppoMechanicsLearning, onOffPolicyLearning],
  "A.4": [dpoLearning, directAlignmentAssessment],
  "A.5": [constitutionalAILearning, frontierAlignmentAssessment],

  // Branch B — Pretraining & architecture research
  "B.1": [scalingLawsLearning, scalingLawsAssessment],
  "B.2": [mixtureOfExpertsLearning, architectureAssessment],
  "B.3": [dataMixingLearning, dataCentricAssessment],
  "B.4": [trainingDynamicsAssessment],
  "B.5": [novelObjectivesAssessment],

  // Branch C — Inference & deployment
  "C.1": [quantizationLearning, quantizationAssessment],
  "C.2": [kvCacheLearning, decodingAssessment],
  "C.3": [servingAssessment],
  "C.4": [compressionAssessment],

  // Branch D — Reasoning, agents & test-time compute
  "D.1": [chainOfThoughtLearning, cotAssessment],
  "D.2": [testTimeComputeLearning, testTimeComputeAssessment],
  "D.3": [toolUseAssessment],
  "D.4": [agenticAssessment],

  // Branch E — Multimodality
  "E.1": [vlmAssessment],
  "E.2": [imageGenAssessment],
  "E.3": [audioAssessment],
  "E.4": [videoAssessment],

  // Branch F — Interpretability & mechanistic understanding
  "F.1": [probingAssessment],
  "F.2": [mechInterpAssessment],
  "F.3": [trainingInterpAssessment],
  "F.4": [formalTheoryAssessment],

  // Branch G — Efficient training & parameter-efficient methods
  "G.1": [loraLearning, peftAssessment],
  "G.2": [memoryEfficientAssessment],
  "G.3": [hardwareAwareAssessment],
};
