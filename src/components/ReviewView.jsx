import { useState, useCallback, useEffect, useMemo } from 'react';
import MathText from './MathText';
import { getMistakes, removeMistake, recordMistake } from './ModuleView';
import { MODULES } from '../modules';
import { shuffleMcStep, createRng } from '../utils/shuffle';

const REVIEW_SIZE = 15;
const ACCENT = '#BA7517';

function buildReviewQuestions() {
  const mistakes = getMistakes();
  const questions = [];

  for (const [key, mistake] of Object.entries(mistakes)) {
    const mods = MODULES[mistake.sectionId];
    if (!mods) continue;
    const mod = mods.find(m => m.id === mistake.moduleId);
    if (!mod) continue;
    const step = mod.steps.find(s => s.type === 'mc' && s.question === mistake.question);
    if (!step) continue;

    questions.push({
      ...step,
      sectionId: mistake.sectionId,
      moduleId: mistake.moduleId,
      moduleTitle: mistake.moduleTitle,
      difficulty: mistake.difficulty,
      mistakeKey: key,
      mistakeCount: mistake.count,
    });
  }

  // Most-missed first
  questions.sort((a, b) => b.mistakeCount - a.mistakeCount);
  const selected = questions.slice(0, REVIEW_SIZE);

  // Shuffle options
  const rng = createRng(Date.now());
  return selected.map(q => {
    const stepSeed = Math.floor(rng() * 2147483646) + 1;
    const shuffled = shuffleMcStep(q, stepSeed);
    return { ...shuffled, mistakeKey: q.mistakeKey, mistakeCount: q.mistakeCount };
  });
}

export default function ReviewView({ onClose, getSectionTitle }) {
  const questions = useMemo(() => buildReviewQuestions(), []);
  const total = questions.length;

  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState(null);
  const [checked, setChecked] = useState(false);
  const [resolved, setResolved] = useState(0);

  const current = step < total ? questions[step] : null;
  const isComplete = step >= total;

  const handleCheck = useCallback(() => {
    if (selected === null) return;
    setChecked(true);
    if (selected === current.correct) {
      setResolved(r => r + 1);
      removeMistake(current.mistakeKey);
    } else {
      recordMistake({
        sectionId: current.sectionId, moduleId: current.moduleId,
        moduleTitle: current.moduleTitle, difficulty: current.difficulty,
        question: current.question,
      });
    }
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
            <div style={{ width: 64, height: 64, borderRadius: '50%', background: '#1D9E75', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: 28, fontWeight: 700 }}>
              {'\u2713'}
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6 }}>No Mistakes to Review</h2>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 24, fontSize: 15 }}>
              You haven't missed any questions yet, or all past mistakes have been resolved.
            </p>
            <button onClick={onClose} style={{ ...S.btn, background: ACCENT }}>Back to Curriculum</button>
          </div>
        </div>
      </div>
    );
  }

  if (isComplete) {
    const remaining = Object.keys(getMistakes()).length;
    const pct = Math.round((resolved / total) * 100);
    return (
      <div style={S.container}>
        <div style={S.inner}>
          <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
            <div style={{ width: 72, height: 72, borderRadius: '50%', background: pct >= 80 ? '#1D9E75' : pct >= 50 ? ACCENT : '#D85A30', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: 24, fontWeight: 700 }}>
              {resolved}/{total}
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6 }}>Review Complete</h2>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 4, fontSize: 15 }}>
              {resolved === total ? 'All mistakes resolved!' : `${resolved} resolved, ${total - resolved} still need work.`}
            </p>
            {remaining > 0 && (
              <p style={{ color: 'var(--color-text-tertiary)', marginBottom: 4, fontSize: 13 }}>
                {remaining} total mistake{remaining !== 1 ? 's' : ''} remaining across all topics.
              </p>
            )}
            <p style={{ color: 'var(--color-text-tertiary)', marginBottom: 32, fontSize: 13 }}>
              {pct >= 80 ? 'Great improvement \u2014 keep reinforcing these concepts.' : pct >= 50 ? 'Making progress \u2014 consider reviewing the related modules.' : 'These areas need more study \u2014 revisit the learning modules.'}
            </p>
            <button onClick={onClose} style={{ ...S.btn, background: ACCENT }}>
              Back to Curriculum
            </button>
          </div>
        </div>
      </div>
    );
  }

  const dc = { easy: '#1D9E75', medium: '#BA7517', hard: '#D85A30' }[current.difficulty] || ACCENT;

  return (
    <div style={S.container}>
      <div style={S.header}>
        <button onClick={onClose} style={S.linkBtn}>&larr; Back</button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ flex: 1, height: 4, background: 'var(--color-border-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${((step + 1) / total) * 100}%`, background: ACCENT, borderRadius: 2, transition: 'width 0.3s' }} />
          </div>
          <span style={{ fontSize: 13, color: 'var(--color-text-tertiary)', flexShrink: 0 }}>{step + 1}/{total}</span>
        </div>
        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: ACCENT + '18', color: ACCENT, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Review
        </span>
      </div>

      <div style={S.inner} key={step}>
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 4, background: dc + '18', color: dc, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {current.difficulty}
            </span>
            <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>
              {current.moduleTitle}
            </span>
            {current.mistakeCount > 1 && (
              <span style={{ fontSize: 10, color: '#D85A30', fontWeight: 500 }}>
                missed {current.mistakeCount}x
              </span>
            )}
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
              border = `1.5px solid ${ACCENT}`;
            }
            return (
              <div key={i} onClick={() => !checked && setSelected(i)} style={{
                padding: '12px 16px', borderRadius: 'var(--border-radius-md)',
                background: bg, border, cursor: checked ? 'default' : 'pointer',
                transition: 'all 0.15s', display: 'flex', alignItems: 'flex-start', gap: 10,
              }}>
                <div style={{
                  width: 20, height: 20, borderRadius: '50%', flexShrink: 0, marginTop: 2,
                  border: `2px solid ${isSel ? (checked ? (isCorrect ? '#1D9E75' : '#D85A30') : ACCENT) : 'var(--color-border-secondary)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {isSel && <div style={{ width: 10, height: 10, borderRadius: '50%', background: checked ? (isCorrect ? '#1D9E75' : '#D85A30') : ACCENT }} />}
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
              {selected === current.correct ? 'Correct! Mistake resolved.' : 'Not quite \u2014 this one stays in your review list.'}
            </div>
            <MathText style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--color-text-secondary)' }}>
              {current.explanation}
            </MathText>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 24 }}>
          {!checked ? (
            <button onClick={handleCheck} disabled={selected === null} style={{
              ...S.btn, background: selected !== null ? ACCENT : 'var(--color-border-tertiary)',
              cursor: selected !== null ? 'pointer' : 'not-allowed', opacity: selected !== null ? 1 : 0.6,
            }}>Check</button>
          ) : (
            <button onClick={handleContinue} style={{ ...S.btn, background: ACCENT }}>
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
