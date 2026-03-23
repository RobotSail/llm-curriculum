/**
 * Seeded shuffle utilities for randomizing quiz questions and options.
 * Ensures each attempt presents questions and options in a different order,
 * preventing pattern-matching and memorization of button sequences.
 */

// Linear congruential PRNG (same as used in WarmupView)
export function createRng(seed) {
  let s = Math.abs(seed) % 2147483647 || 1;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// Fisher-Yates shuffle using seeded RNG
export function shuffleArray(arr, rng) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Shuffle a MC step's options and remap the correct index.
 * Returns a new step object with shuffled options and updated correct index.
 */
export function shuffleMcStep(step, seed) {
  if (step.type !== 'mc') return step;
  const rng = createRng(seed);
  const indices = step.options.map((_, i) => i);
  const shuffled = shuffleArray(indices, rng);
  return {
    ...step,
    options: shuffled.map(i => step.options[i]),
    correct: shuffled.indexOf(step.correct),
  };
}

/**
 * Shuffle all MC steps in a module. For test modules, also shuffles question order.
 * Info steps in learning modules stay in their original positions relative to MC steps.
 * Returns a new steps array.
 */
export function shuffleModuleSteps(steps, moduleType, seed) {
  const rng = createRng(seed);

  // Shuffle option order for every MC step (unique sub-seed per step)
  const optionRng = createRng(seed + 31337);
  const withShuffledOptions = steps.map((step, i) => {
    if (step.type !== 'mc') return step;
    const stepSeed = Math.floor(optionRng() * 2147483646) + 1;
    return shuffleMcStep(step, stepSeed);
  });

  // For test modules, shuffle the order of MC questions
  // Keep any leading info/intro steps in place
  if (moduleType === 'test') {
    const leadingInfo = [];
    const rest = [];
    let pastIntro = false;
    for (const step of withShuffledOptions) {
      if (!pastIntro && step.type === 'info') {
        leadingInfo.push(step);
      } else {
        pastIntro = true;
        rest.push(step);
      }
    }
    return [...leadingInfo, ...shuffleArray(rest, rng)];
  }

  return withShuffledOptions;
}

/**
 * Generate or retrieve a per-attempt seed for a module.
 * Stored in sessionStorage so it persists across re-renders but changes per session/retry.
 */
export function getAttemptSeed(moduleId) {
  const key = `llm-curriculum-shuffle-${moduleId}`;
  try {
    const stored = sessionStorage.getItem(key);
    if (stored) return Number(stored);
  } catch {}
  return resetAttemptSeed(moduleId);
}

/**
 * Reset the attempt seed (called on Retry or new attempt).
 */
export function resetAttemptSeed(moduleId) {
  const key = `llm-curriculum-shuffle-${moduleId}`;
  const seed = Math.floor(Math.random() * 2147483646) + 1;
  try { sessionStorage.setItem(key, String(seed)); } catch {}
  return seed;
}
