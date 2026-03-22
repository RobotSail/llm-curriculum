import { useState, useCallback, useEffect, useMemo } from 'react';
import MathText from './MathText';
import { MODULES } from '../modules';
import { getModuleProgress } from './ModuleView';

const WARMUP_SIZE = 10;

// Collect MC questions. Optional modules are excluded unless the user has explored them.
function collectAllQuestions(exploredModuleIds) {
  const questions = [];
  for (const [sectionId, mods] of Object.entries(MODULES)) {
    for (const mod of mods) {
      // Skip optional modules the user hasn't started
      if (mod.optional && !exploredModuleIds.has(mod.id)) continue;
      mod.steps.forEach((step, idx) => {
        if (step.type === 'mc') {
          questions.push({
            ...step,
            sectionId,
            moduleId: mod.id,
            moduleTitle: mod.title,
            difficulty: mod.difficulty,
            stepIndex: idx,
          });
        }
      });
    }
  }
  return questions;
}

// Seeded shuffle (Fisher-Yates) for daily consistency
function shuffle(arr, seed) {
  const a = [...arr];
  let s = seed;
  const rand = () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildWarmup() {
  const progress = getModuleProgress();

  // Find modules the user has interacted with
  const exploredModuleIds = new Set(
    Object.entries(progress)
      .filter(([, p]) => p.currentStep > 0 || p.completed)
      .map(([id]) => id)
  );

  // Collect questions, filtering out unexplored optional modules
  const allQ = collectAllQuestions(exploredModuleIds);

  // Split into explored vs unexplored (among non-optional + explored-optional)
  const explored = allQ.filter(q => exploredModuleIds.has(q.moduleId));
  const unexplored = allQ.filter(q => !exploredModuleIds.has(q.moduleId));

  // Use today's date as seed for daily consistency
  const today = new Date();
  const seed = today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate();

  const shuffledExplored = shuffle(explored, seed);
  const shuffledUnexplored = shuffle(unexplored, seed + 1);

  // Aim for ~60% explored topics, ~40% new topics (if available)
  const exploredCount = Math.min(Math.ceil(WARMUP_SIZE * 0.6), shuffledExplored.length);
  const unexploredCount = Math.min(WARMUP_SIZE - exploredCount, shuffledUnexplored.length);
  const remaining = WARMUP_SIZE - exploredCount - unexploredCount;

  let selected = [
    ...shuffledExplored.slice(0, exploredCount),
    ...shuffledUnexplored.slice(0, unexploredCount),
  ];

  // Fill remainder from whichever pool has more
  if (remaining > 0) {
    const extra = shuffledExplored.length > exploredCount
      ? shuffledExplored.slice(exploredCount, exploredCount + remaining)
      : shuffledUnexplored.slice(unexploredCount, unexploredCount + remaining);
    selected = [...selected, ...extra];
  }

  // Sort by difficulty for easy→hard warmup progression
  const diffOrder = { easy: 0, medium: 1, hard: 2 };
  selected.sort((a, b) => (diffOrder[a.difficulty] ?? 1) - (diffOrder[b.difficulty] ?? 1));

  return selected.slice(0, WARMUP_SIZE);
}

const DIFF_COLORS = { easy: '#1D9E75', medium: '#BA7517', hard: '#D85A30' };

export default function WarmupView({ onClose, getSectionTitle }) {
  const questions = useMemo(() => buildWarmup(), []);
  const total = questions.length;

  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState(null);
  const [checked, setChecked] = useState(false);
  const [score, setScore] = useState(0);

  const current = step < total ? questions[step] : null;
  const isComplete = step >= total;

  const handleCheck = useCallback(() => {
    if (selected === null) return;
    setChecked(true);
    if (selected === current.correct) setScore(s => s + 1);
  }, [selected, current]);

  const handleContinue = useCallback(() => {
    setSelected(null);
    setChecked(false);
    setStep(s => s + 1);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        if (isComplete) return;
        if (checked) handleContinue();
        else if (selected !== null) handleCheck();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isComplete, checked, selected, handleCheck, handleContinue]);

  useEffect(() => { window.scrollTo(0, 0); }, [step]);

  if (total === 0) {
    return (
      <div style={S.container}>
        <div style={S.inner}>
          <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 24 }}>No questions available yet. Complete some modules first!</p>
            <button onClick={onClose} style={{ ...S.btn, background: '#378ADD' }}>Back to Curriculum</button>
          </div>
        </div>
      </div>
    );
  }

  if (isComplete) {
    const pct = Math.round((score / total) * 100);
    const emoji = pct >= 80 ? 'Excellent' : pct >= 60 ? 'Good effort' : 'Keep practicing';
    return (
      <div style={S.container}>
        <div style={S.inner}>
          <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
            <div style={{ width: 72, height: 72, borderRadius: '50%', background: pct >= 80 ? '#1D9E75' : pct >= 60 ? '#BA7517' : '#D85A30', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: 24, fontWeight: 700 }}>
              {score}/{total}
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6 }}>Warmup Complete</h2>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 4, fontSize: 15 }}>{emoji} &mdash; {pct}% correct</p>
            <p style={{ color: 'var(--color-text-tertiary)', marginBottom: 32, fontSize: 13 }}>
              {pct >= 80 ? 'Strong recall across topics.' : pct >= 60 ? 'Some areas could use review.' : 'Consider revisiting flagged topics.'}
            </p>
            <button onClick={onClose} style={{ ...S.btn, background: '#378ADD' }}>
              Back to Curriculum
            </button>
          </div>
        </div>
      </div>
    );
  }

  const dc = DIFF_COLORS[current.difficulty] || '#378ADD';

  return (
    <div style={S.container}>
      {/* Header */}
      <div style={S.header}>
        <button onClick={onClose} style={S.linkBtn}>&larr; Skip</button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ flex: 1, height: 4, background: 'var(--color-border-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${((step + 1) / total) * 100}%`, background: '#378ADD', borderRadius: 2, transition: 'width 0.3s' }} />
          </div>
          <span style={{ fontSize: 13, color: 'var(--color-text-tertiary)', flexShrink: 0 }}>{step + 1}/{total}</span>
        </div>
        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: '#378ADD18', color: '#378ADD', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Warmup
        </span>
      </div>

      {/* Content */}
      <div style={S.inner} key={step}>
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4, background: dc + '18', color: dc, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {current.difficulty}
            </span>
            <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>
              {current.moduleTitle}
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', opacity: 0.7 }}>
            {current.sectionId} &mdash; {getSectionTitle ? getSectionTitle(current.sectionId) : current.sectionId}
          </div>
        </div>

        <MathText style={{ fontSize: 15, lineHeight: 1.75, color: 'var(--color-text-secondary)', marginBottom: 20 }}>
          {current.question}
        </MathText>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {current.options.map((opt, i) => {
            const isSel = selected === i;
            const isCorrect = i === current.correct;
            let bg = 'var(--color-background-secondary)';
            let border = '1.5px solid var(--color-border-tertiary)';
            if (checked) {
              if (isCorrect) { bg = '#1D9E7514'; border = '1.5px solid #1D9E75'; }
              else if (isSel) { bg = '#D85A3014'; border = '1.5px solid #D85A30'; }
            } else if (isSel) {
              border = '1.5px solid #378ADD';
            }
            return (
              <div key={i} onClick={() => !checked && setSelected(i)} style={{
                padding: '12px 16px', borderRadius: 'var(--border-radius-md)',
                background: bg, border, cursor: checked ? 'default' : 'pointer',
                transition: 'all 0.15s', display: 'flex', alignItems: 'flex-start', gap: 10,
              }}>
                <div style={{
                  width: 20, height: 20, borderRadius: '50%', flexShrink: 0, marginTop: 2,
                  border: `2px solid ${isSel ? (checked ? (isCorrect ? '#1D9E75' : '#D85A30') : '#378ADD') : 'var(--color-border-secondary)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {isSel && <div style={{ width: 10, height: 10, borderRadius: '50%', background: checked ? (isCorrect ? '#1D9E75' : '#D85A30') : '#378ADD' }} />}
                </div>
                <MathText as="span" style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--color-text-primary)' }}>{opt}</MathText>
              </div>
            );
          })}
        </div>

        {checked && current.explanation && (
          <div style={{
            marginTop: 16, padding: '14px 16px', borderRadius: 'var(--border-radius-md)',
            background: selected === current.correct ? '#1D9E750C' : '#D85A300C',
            borderLeft: `3px solid ${selected === current.correct ? '#1D9E75' : '#D85A30'}`,
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6, color: selected === current.correct ? '#1D9E75' : '#D85A30' }}>
              {selected === current.correct ? 'Correct!' : 'Not quite.'}
            </div>
            <MathText style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--color-text-secondary)' }}>
              {current.explanation}
            </MathText>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 24 }}>
          {!checked ? (
            <button onClick={handleCheck} disabled={selected === null} style={{
              ...S.btn, background: selected !== null ? '#378ADD' : 'var(--color-border-tertiary)',
              cursor: selected !== null ? 'pointer' : 'not-allowed', opacity: selected !== null ? 1 : 0.6,
            }}>Check</button>
          ) : (
            <button onClick={handleContinue} style={{ ...S.btn, background: '#378ADD' }}>
              {step + 1 < total ? 'Next \u2192' : 'Finish'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const S = {
  container: {
    position: 'fixed', inset: 0, background: 'var(--color-background-primary)',
    zIndex: 1000, overflowY: 'auto',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '12px 20px', borderBottom: '0.5px solid var(--color-border-tertiary)',
    position: 'sticky', top: 0, background: 'var(--color-background-primary)', zIndex: 10,
  },
  inner: {
    maxWidth: 640, margin: '0 auto', padding: '32px 20px 80px',
  },
  btn: {
    padding: '10px 24px', borderRadius: 'var(--border-radius-md)',
    border: 'none', color: 'white', fontSize: 14, fontWeight: 500,
    cursor: 'pointer', fontFamily: 'inherit',
  },
  linkBtn: {
    background: 'transparent', border: 'none', color: 'var(--color-text-tertiary)',
    cursor: 'pointer', fontSize: 14, fontFamily: 'inherit', padding: '4px 0',
  },
};
