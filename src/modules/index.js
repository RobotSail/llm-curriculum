import { svdLearning } from './focused-svd';
import { eigendecompositionLearning } from './focused-eigendecomposition';
import { matrixCalculusLearning } from './focused-matrix-calculus';
import { einsumLearning } from './focused-einsum';
import { randomMatrixTheoryLearning } from './focused-random-matrix-theory';
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
import { sftAssessment, rewardModelingAssessment, rlhfAssessment, directAlignmentAssessment, frontierAlignmentAssessment } from './assess-branch-a';
import { scalingLawsAssessment, architectureAssessment, dataCentricAssessment, trainingDynamicsAssessment, novelObjectivesAssessment } from './assess-branch-b';
import { quantizationAssessment, decodingAssessment, servingAssessment, compressionAssessment, cotAssessment, testTimeComputeAssessment, toolUseAssessment, agenticAssessment } from './assess-branch-cd';
import { vlmAssessment, imageGenAssessment, audioAssessment, videoAssessment } from './assess-branch-e';
import { probingAssessment, mechInterpAssessment, trainingInterpAssessment, formalTheoryAssessment } from './assess-branch-f';
import { peftAssessment, memoryEfficientAssessment, hardwareAwareAssessment, optimizationAssessment, systemsAssessment } from './assess-branch-g-and-tier0';

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

// Modules with optional: true are deep-theory / tangential content.
// They appear in the UI with an "Optional" badge and are excluded
// from daily warmup unless the user has started exploring them.
function markOptional(...mods) {
  return mods.map(m => ({ ...m, optional: true }));
}

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  // Tier 0 — Prerequisites
  "0.1": [matrixCalculusLearning, eigendecompositionLearning, svdLearning, einsumLearning, ...markOptional(randomMatrixTheoryLearning)],
  "0.3": [optimizationAssessment, adamLearning, weightDecayLearning, muonOptimizerFundamentals],
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
  "1.1": [selfAttentionLearning, multiHeadAttentionLearning, transformerAssessment],
  "1.2": [tokenizationAssessment],
  "1.3": [pretrainingAssessment],
  "1.4": [dataAssessment],
  "1.5": [evaluationAssessment],
  "1.6": [distributedTrainingAssessment],

  // Branch A — Post-training & alignment
  "A.1": [sftAssessment],
  "A.2": [rewardModelingAssessment],
  "A.3": [rlhfAssessment, policyGradientsLearning, ppoMechanicsLearning, onOffPolicyLearning],
  "A.4": [directAlignmentAssessment],
  "A.5": [frontierAlignmentAssessment],

  // Branch B — Pretraining & architecture research
  "B.1": [scalingLawsAssessment],
  "B.2": [architectureAssessment],
  "B.3": [dataCentricAssessment],
  "B.4": [trainingDynamicsAssessment],
  "B.5": [novelObjectivesAssessment],

  // Branch C — Inference & deployment
  "C.1": [quantizationAssessment],
  "C.2": [decodingAssessment],
  "C.3": [servingAssessment],
  "C.4": [compressionAssessment],

  // Branch D — Reasoning, agents & test-time compute
  "D.1": [cotAssessment],
  "D.2": [testTimeComputeAssessment],
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
  "G.1": [peftAssessment],
  "G.2": [memoryEfficientAssessment],
  "G.3": [hardwareAwareAssessment],
};
