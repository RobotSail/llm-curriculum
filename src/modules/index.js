import { easyModule as linAlgEasy, mediumModule as linAlgMedium, hardModule as linAlgHard } from './linear-algebra';
import { easyModule, mediumModule, hardModule } from './info-theory-f-divergences';

// Registry: maps curriculum section IDs to available modules
export const MODULES = {
  "0.1": [linAlgEasy, linAlgMedium, linAlgHard],
  "0.2": [easyModule, mediumModule, hardModule],
};
