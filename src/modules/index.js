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

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  // Tier 0 — Prerequisites
  "0.1": [linAlgEasy, linAlgMedium, linAlgHard],
  "0.2": [
    probabilityFoundationsAssessment,
    entropyAssessment,
    exponentialFamilyAssessment,
    divergencesAssessment,
    bayesianAssessment,
    samplingAssessment,
    concentrationAssessment,
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
};
