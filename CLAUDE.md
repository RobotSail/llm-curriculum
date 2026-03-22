# LLM Curriculum

Interactive learning app for mastering topics required for frontier LLM research and development. Built with React + Vite, KaTeX math rendering, localStorage persistence.

## Content Architecture

All content follows a strict three-part structure per section:

### 1. Learning Modules (`moduleType: "learning"`)
- Teach concepts through alternating info steps and inline quiz questions
- Info steps explain a concept; MC steps immediately test comprehension
- Each module: 10-15 steps, 15-25 minutes, one difficulty track (easy/medium/hard)
- A section typically has 3 learning modules (easy, medium, hard) covering the same topic area at increasing depth

### 2. Tests (`moduleType: "test"`)
- Standalone MC assessments with NO info steps (one brief intro step is acceptable)
- 10 questions per test, covering the section's full topic breadth
- Questions range from conceptual to applied within the test
- Used to measure readiness and feed the daily warmup question pool

### New content must be one of these types. Every section should have at least one test. Learning modules are added when a section needs teaching content.

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
  [topic-name].js             # Module files, one per topic area or section group
```

### Naming Conventions
- Learning module files: descriptive topic name (e.g., `entropy-cross-entropy-mi.js`)
- Test/assessment files: `assess-[scope].js` (e.g., `assess-branch-a.js`)
- Export names: camelCase, descriptive (e.g., `entropyEasy`, `rlhfAssessment`)

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

1. Decide: is this a learning module or a test?
2. Create or update the appropriate file in `src/modules/`
3. Follow the module data format exactly (include `moduleType`)
4. Verify question quality against all standards above
5. Register in `src/modules/index.js`
6. Run `npm run build` to verify no errors
7. The Stop hook will auto-commit and push on completion

## Tech Stack
- React 18 + Vite 5
- KaTeX for math rendering
- localStorage for progress, gaps, and warmup state
- macOS LaunchAgent auto-starts dev server on login (`scripts/start-dev.sh`)
- GitHub: https://github.com/RobotSail/llm-curriculum
