import { easyModule as linAlgEasy, mediumModule as linAlgMedium, hardModule as linAlgHard } from './linear-algebra';
import { easyModule, mediumModule, hardModule } from './info-theory-f-divergences';
import { entropyEasy, entropyMedium, entropyHard } from './entropy-cross-entropy-mi';
import { probabilityFoundationsAssessment } from './prob-assessment-foundations';
import { exponentialFamilyAssessment } from './prob-assessment-exponential-family';
import { entropyAssessment } from './prob-assessment-entropy';
import { divergencesAssessment } from './prob-assessment-divergences';
import { bayesianAssessment } from './prob-assessment-bayesian';
import { samplingAssessment } from './prob-assessment-sampling';
import { concentrationAssessment } from './prob-assessment-concentration';
import { appliedInfoTheoryAssessment } from './prob-assessment-applied';
import { transformerAssessment, tokenizationAssessment, pretrainingAssessment } from './assess-tier1-part1';
import { dataAssessment, evaluationAssessment, distributedTrainingAssessment } from './assess-tier1-part2';
import { sftAssessment, rewardModelingAssessment, rlhfAssessment, directAlignmentAssessment, frontierAlignmentAssessment } from './assess-branch-a';
import { scalingLawsAssessment, architectureAssessment, dataCentricAssessment, trainingDynamicsAssessment, novelObjectivesAssessment } from './assess-branch-b';
import { quantizationAssessment, decodingAssessment, servingAssessment, compressionAssessment, cotAssessment, testTimeComputeAssessment, toolUseAssessment, agenticAssessment } from './assess-branch-cd';
import { vlmAssessment, imageGenAssessment, audioAssessment, videoAssessment } from './assess-branch-e';
import { probingAssessment, mechInterpAssessment, trainingInterpAssessment, formalTheoryAssessment } from './assess-branch-f';
import { peftAssessment, memoryEfficientAssessment, hardwareAwareAssessment, optimizationAssessment, systemsAssessment } from './assess-branch-g-and-tier0';

// Modules with optional: true are deep-theory / tangential content.
// They appear in the UI with an "Optional" badge and are excluded
// from daily warmup unless the user has started exploring them.
function markOptional(...mods) {
  return mods.map(m => ({ ...m, optional: true }));
}

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  // Tier 0 — Prerequisites
  "0.1": [linAlgEasy, linAlgMedium, linAlgHard],
  "0.3": [optimizationAssessment],
  "0.4": [systemsAssessment],
  "0.2": [
    probabilityFoundationsAssessment,
    entropyAssessment,
    ...markOptional(exponentialFamilyAssessment),
    divergencesAssessment,
    bayesianAssessment,
    samplingAssessment,
    ...markOptional(concentrationAssessment),
    appliedInfoTheoryAssessment,
    easyModule, mediumModule, hardModule,
    entropyEasy, entropyMedium, entropyHard,
  ],

  // Tier 1 — Foundational core
  "1.1": [transformerAssessment],
  "1.2": [tokenizationAssessment],
  "1.3": [pretrainingAssessment],
  "1.4": [dataAssessment],
  "1.5": [evaluationAssessment],
  "1.6": [distributedTrainingAssessment],

  // Branch A — Post-training & alignment
  "A.1": [sftAssessment],
  "A.2": [rewardModelingAssessment],
  "A.3": [rlhfAssessment],
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
