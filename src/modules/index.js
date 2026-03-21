import { easyModule as linAlgEasy, mediumModule as linAlgMedium, hardModule as linAlgHard } from './linear-algebra';
import { easyModule, mediumModule, hardModule } from './info-theory-f-divergences';
import { probabilityFoundationsAssessment } from './prob-assessment-foundations';
import { exponentialFamilyAssessment } from './prob-assessment-exponential-family';
import { entropyAssessment } from './prob-assessment-entropy';
import { divergencesAssessment } from './prob-assessment-divergences';
import { bayesianAssessment } from './prob-assessment-bayesian';
import { samplingAssessment } from './prob-assessment-sampling';
import { concentrationAssessment } from './prob-assessment-concentration';
import { appliedInfoTheoryAssessment } from './prob-assessment-applied';

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  "0.1": [linAlgEasy, linAlgMedium, linAlgHard],
  "0.2": [
    // Diagnostic assessments — take these first to identify gaps
    probabilityFoundationsAssessment,
    entropyAssessment,
    exponentialFamilyAssessment,
    divergencesAssessment,
    bayesianAssessment,
    samplingAssessment,
    concentrationAssessment,
    appliedInfoTheoryAssessment,
    // Deep-dive teaching modules
    easyModule,
    mediumModule,
    hardModule,
  ],
};
