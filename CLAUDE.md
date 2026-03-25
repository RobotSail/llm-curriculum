# LLM Curriculum

Interactive learning app for mastering topics required for frontier LLM research and development. Built with React + Vite, KaTeX math rendering, localStorage persistence.

## Content Architecture

Content is organized in a strict hierarchy. Each level has clear rules about what it contains and how it relates to the levels above and below it.

### Hierarchy: Subject → Chapter (optional) → Topic → Module

#### Subject
A subject is a broad field of study (e.g., "Probability & Information Theory", "Reinforcement Learning", "Linear Algebra"). Subjects are the top-level groupings in the curriculum.

#### Chapter (optional)
A subject can be divided into **chapters** (also called subtopics) that group related topics together. For example, within Probability & Information Theory:
- "Divergences" chapter → KL divergence, Jensen-Shannon divergence, f-divergences
- "Entropy Measures" chapter → entropy, perplexity, cross-entropy, mutual information

Chapters are optional — a subject with few topics can use a flat list instead. **Every chapter must end with a test** (see below).

#### Topic (atomic unit)
A **topic** is the atomic unit of learning. It represents a single, named concept that the student either has or has not learned. Topics are the leaves of the content tree.

**The single-concept rule**: A topic must cover exactly ONE concept. The name test: if a topic title contains a conjunction ("and", "&", "vs"), it likely contains multiple topics and must be split. Examples:
- **Good**: "KL Divergence", "PPO Clipping", "Newton-Schulz Iteration"
- **Bad**: "KL Divergence and Cross-Entropy" → split into "KL Divergence" and "Cross-Entropy"
- **Bad**: "Adam & Weight Decay" → split into "Adam Optimizer" and "Weight Decay Regularization"
- **Exception**: "Forward vs Reverse KL" is acceptable when the topic IS the comparison itself (the concept being taught is the relationship between two things the student already knows individually)

A topic consists of:
- **One or more learning modules** that teach the concept
- **One or more quiz/test modules** that verify understanding

Content within a topic must be **self-contained** — it should not try to teach other topics inline. It may reference prerequisites ("recall that KL divergence is...") but should not derive or explain them from scratch.

#### Module
A module is a single interactive unit within a topic. Every module has a `moduleType`:

**Learning modules** (`moduleType: "learning"`):
- Teach the topic through alternating info steps and inline quiz questions
- Info steps explain a concept; MC steps immediately test comprehension
- Each module: 10-15 steps, 15-25 minutes, one difficulty track (easy/medium/hard)
- A topic may have multiple learning modules at different difficulties covering the same concept at increasing depth

**Test modules** (`moduleType: "test"`):
- Standalone MC assessments with NO info steps (one brief intro step is acceptable)
- 10 questions per test, covering the topic's or chapter's full breadth
- Questions range from conceptual to applied within the test
- Used to measure readiness and feed the daily warmup question pool

### Test Requirements

Tests are mandatory at specific levels of the hierarchy:

1. **Every topic** should have at least one test module to verify the student learned the concept
2. **Every chapter** (if chapters are used) must end with a **chapter test** that integrates across the chapter's topics — students should need to combine knowledge from multiple topics to answer questions
3. **Every curriculum / recommended study path** must end with a **capstone test** that is genuinely challenging and integrative

### Capstone & Chapter Test Standards

Chapter-ending and curriculum-ending tests are held to a **higher standard** than per-topic quizzes:
- Questions must **tie multiple topics together** — e.g., "Given a PPO setup with KL penalty β and a reward model with known failure modes, what happens when..."
- Questions should require **multi-step reasoning**, not single-fact recall
- Like good physics exams: give a scenario, require the student to apply several concepts to reach the answer
- The test should be **difficult to pass without genuine understanding** — pattern-matching or partial knowledge should not suffice
- All Question Quality Standards (below) apply with extra strictness

### New content must be one of these types (learning or test). Every topic should have at least one test. Learning modules are added when a topic needs teaching content.

## Module Data Format

```js
export const myModule = {
  id: "SECTION_ID-TYPE-DIFFICULTY",  // e.g., "1.1-test", "A.3-learning-easy"
  sectionId: "1.1",                  // matches a section in CURRICULUM
  title: "Human-readable title",
  moduleType: "learning" | "test",   // REQUIRED
  difficulty: "easy" | "medium" | "hard",
  estimatedMinutes: 15,
  optional: false,                   // true for deep-theory/tangential content
  steps: [
    // Info step (learning modules only)
    {
      type: "info",
      title: "Step Title",
      content: "Paragraphs separated by \\n\\n. Use $inline math$ and $$display math$$. Use **bold**."
    },
    // MC question (both learning and test modules)
    {
      type: "mc",
      question: "Question text with $math$ support",
      options: ["Option A", "Option B", "Option C", "Option D"],
      correct: 2,        // 0-indexed
      explanation: "Shown after answering. Explain WHY the answer is correct."
    }
  ]
};
```

## Question Quality Standards (CRITICAL)

These rules prevent questions from being "hackable" by pattern-matching rather than understanding.

### Answer Position
- Distribute correct answers **evenly across positions 0-3** within each module
- In a 10-question test: aim for 2-3 correct answers per position
- NEVER cluster correct answers at position 1 or 2

### Distractor Quality
- Every wrong option must be a **plausible misconception**, partial truth, or common error
- Distractors should be things a student with partial understanding would genuinely consider
- Use: adjacent concepts, off-by-one errors, reversed relationships, correct-sounding but subtly wrong statements
- NEVER use obviously absurd options as filler

### Option Similarity
- All 4 options must be **similar in length, specificity, and style**
- The correct answer must NOT be the most detailed, the most qualified, or the most "textbook-sounding"
- If the correct answer is long, make distractors equally long
- If the correct answer is short, make distractors equally short

### No Giveaways
- Never use "all of the above" or "none of the above"
- Never have one option that is stylistically different from the rest
- Never use negatives in only one option ("does NOT" when others are positive)
- The correct answer should not be identifiable without domain knowledge

### Test Understanding
- Questions should require **reasoning**, not recall
- Include numerical reasoning where natural (e.g., "A 70B model with Adam — how much optimizer memory?")
- Frame questions around scenarios, not definitions
- Good: "What happens when you increase β in KL(π||π_ref)?"
- Bad: "What is the definition of KL divergence?"

## Math Rendering
- `$...$` for inline math (rendered by KaTeX)
- `$$...$$` for display math
- `**text**` for bold emphasis
- `\n\n` for paragraph breaks in content strings
- In JavaScript strings, backslashes must be doubled: `\\frac`, `\\text`, `\\mathbb`, etc.

## File Organization

```
src/modules/
  index.js                    # Module registry — ALL modules registered here
  focused-[topic-name].js     # Single-topic module file (preferred pattern)
  assess-[scope].js           # Chapter or capstone test files
  [legacy-name].js            # Older multi-topic files (avoid creating new ones)
```

### Naming Conventions
- **Single-topic files** (preferred): `focused-[topic].js` (e.g., `focused-kl-divergence.js`, `focused-ppo.js`)
- **Chapter/capstone tests**: `assess-[scope].js` (e.g., `assess-branch-a.js`, `assess-divergences.js`)
- **Export names**: camelCase, descriptive (e.g., `forwardKLLearning`, `rlhfAssessment`)
- **One concept per file** whenever possible — avoid bundling unrelated topics into a single file

### Module Registry (`index.js`)
- Every module must be imported and registered in the `MODULES` object
- Keys are section IDs (e.g., `"0.2"`, `"A.3"`)
- Learning modules and tests for the same section go in the same array
- Optional modules are wrapped with `markOptional()`:
  ```js
  "0.2": [
    essentialModule,
    ...markOptional(tangentialModule),
  ]
  ```

## Optional Content Policy
Mark a module `optional: true` when the topic is **tangential to core LLM R&D**:
- Content that is good mathematical background but rarely needed in practice
- Deep theory that most working LLM researchers don't use day-to-day
- Example: concentration inequalities (Hoeffding, Chernoff) — useful for theory papers, not for training models

Optional modules:
- Show an "Optional" badge in the UI
- Are excluded from daily warmup UNLESS the user has started exploring them
- Still appear in the section's module list for users who want depth

## Daily Warmup System
- Pulls 10 random MC questions from across all registered modules
- 60% from topics the user has explored, 40% from new topics
- Excludes optional modules the user hasn't started
- Uses a daily seed for consistency within the same day
- Sorted easy → hard for natural warmup progression

## Adding New Content — Checklist

1. **Identify the topic**: What single concept does this module teach? Apply the name test — if the title has a conjunction, split it into separate topics.
2. **Decide the module type**: Learning module or test?
3. **Check the hierarchy**: Does this topic belong to an existing chapter? Does the chapter already have an end-of-chapter test? If you're creating a new chapter, plan its test too.
4. Create or update the appropriate file in `src/modules/` (prefer `focused-[topic].js` for new single-topic content)
5. Follow the module data format exactly (include `moduleType`)
6. Verify question quality against all standards above
7. Register in `src/modules/index.js`
8. Run `npm run build` to verify no errors
9. The Stop hook will auto-commit and push on completion

## Tech Stack
- React 18 + Vite 5
- KaTeX for math rendering
- localStorage for progress, gaps, and warmup state
- macOS LaunchAgent auto-starts dev server on login (`scripts/start-dev.sh`)
- GitHub: https://github.com/RobotSail/llm-curriculum
