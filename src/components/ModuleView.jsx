import { useState, useCallback, useEffect } from 'react';
import MathText from './MathText';

const MODULE_PROGRESS_KEY = 'llm-curriculum-modules-v1';

function loadProgress() {
  try { return JSON.parse(localStorage.getItem(MODULE_PROGRESS_KEY)) || {}; }
  catch { return {}; }
}

function saveProgress(data) {
  try { localStorage.setItem(MODULE_PROGRESS_KEY, JSON.stringify(data)); } catch {}
}

export function getModuleProgress() {
  return loadProgress();
}

// Gap tracking — "need to learn this" items
const GAPS_KEY = 'llm-curriculum-gaps-v1';

function loadGaps() {
  try { return JSON.parse(localStorage.getItem(GAPS_KEY)) || []; }
  catch { return []; }
}

function saveGapsData(gaps) {
  try { localStorage.setItem(GAPS_KEY, JSON.stringify(gaps)); } catch {}
}

export function getGaps() { return loadGaps(); }

export function removeGap(id) {
  const gaps = loadGaps().filter(g => g.id !== id);
  saveGapsData(gaps);
  return gaps;
}

export function clearAllGaps() {
  saveGapsData([]);
}

const DIFF_COLORS = { easy: '#1D9E75', medium: '#BA7517', hard: '#D85A30' };
const DIFF_LABELS = { easy: 'Easy', medium: 'Medium', hard: 'Hard' };

export default function ModuleView({ module, tierColor, onClose }) {
  const accentColor = tierColor || DIFF_COLORS[module.difficulty] || '#378ADD';
  const totalSteps = module.steps.length;

  const [step, setStep] = useState(() => {
    const p = loadProgress();
    const saved = p[module.id]?.currentStep || 0;
    return saved >= totalSteps ? 0 : saved;
  });
  const [selected, setSelected] = useState(null);
  const [checked, setChecked] = useState(false);
  const [answers, setAnswers] = useState(() => {
    const p = loadProgress();
    return p[module.id]?.answers || {};
  });

  const current = step < totalSteps ? module.steps[step] : null;
  const isComplete = step >= totalSteps;

  const [flaggedSteps, setFlaggedSteps] = useState(() => {
    return new Set(loadGaps().filter(g => g.moduleId === module.id).map(g => g.id));
  });

  const isFlagged = current ? flaggedSteps.has(`${module.id}-${step}`) : false;

  const handleFlag = useCallback(() => {
    if (!current) return;
    const id = `${module.id}-${step}`;
    const gaps = loadGaps();
    if (flaggedSteps.has(id)) {
      saveGapsData(gaps.filter(g => g.id !== id));
      setFlaggedSteps(prev => { const next = new Set(prev); next.delete(id); return next; });
    } else {
      const label = current.title || (current.question ? current.question.substring(0, 150) : `Step ${step + 1}`);
      gaps.push({
        id, sectionId: module.sectionId, moduleId: module.id,
        moduleTitle: module.title, difficulty: module.difficulty,
        stepIndex: step, label, timestamp: Date.now(),
      });
      saveGapsData(gaps);
      setFlaggedSteps(prev => new Set(prev).add(id));
    }
  }, [module, step, current, flaggedSteps]);

  useEffect(() => {
    const p = loadProgress();
    p[module.id] = { currentStep: step, answers, completed: isComplete };
    saveProgress(p);
  }, [step, answers, isComplete, module.id]);

  const handleCheck = useCallback(() => {
    if (selected === null) return;
    setChecked(true);
    setAnswers(prev => ({ ...prev, [step]: selected }));
  }, [selected, step]);

  const handleContinue = useCallback(() => {
    setSelected(null);
    setChecked(false);
    setStep(s => s + 1);
  }, []);

  const handleBack = useCallback(() => {
    if (step > 0) {
      setSelected(null);
      setChecked(false);
      setStep(s => s - 1);
    }
  }, [step]);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        if (!current) return;
        if (current.type === 'info') handleContinue();
        else if (current.type === 'mc') {
          if (checked) handleContinue();
          else if (selected !== null) handleCheck();
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [current, checked, selected, handleCheck, handleContinue]);

  // Scroll to top on step change
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [step]);

  if (isComplete) {
    const correct = Object.entries(answers).reduce((n, [idx, ans]) => {
      const s = module.steps[Number(idx)];
      return s?.type === 'mc' && ans === s.correct ? n + 1 : n;
    }, 0);
    const totalMc = module.steps.filter(s => s.type === 'mc').length;
    return (
      <div style={S.container}>
        <div style={S.inner}>
          <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
            <div style={{ width: 64, height: 64, borderRadius: '50%', background: accentColor, color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: 28, fontWeight: 700 }}>
              {correct === totalMc ? '\u2713' : `${correct}/${totalMc}`}
            </div>
            <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 6 }}>Module Complete</h2>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 4, fontSize: 15 }}>{module.title}</p>
            <p style={{ color: 'var(--color-text-tertiary)', marginBottom: 32, fontSize: 13 }}>
              {correct}/{totalMc} questions correct
            </p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
              <button onClick={() => { setStep(0); setAnswers({}); }} style={{ ...S.btn, background: 'transparent', border: `1.5px solid ${accentColor}`, color: accentColor }}>
                Retry
              </button>
              <button onClick={onClose} style={{ ...S.btn, background: accentColor }}>
                Back to Curriculum
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={S.container}>
      {/* Header */}
      <div style={S.header}>
        <button onClick={onClose} style={S.linkBtn}>&larr; Back</button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ flex: 1, height: 4, background: 'var(--color-border-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${((step + 1) / totalSteps) * 100}%`, background: accentColor, borderRadius: 2, transition: 'width 0.3s' }} />
          </div>
          <span style={{ fontSize: 13, color: 'var(--color-text-tertiary)', flexShrink: 0 }}>{step + 1}/{totalSteps}</span>
        </div>
        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: DIFF_COLORS[module.difficulty] + '18', color: DIFF_COLORS[module.difficulty], textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {DIFF_LABELS[module.difficulty]}
        </span>
        <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: module.moduleType === 'test' ? '#D4537E18' : '#378ADD18', color: module.moduleType === 'test' ? '#D4537E' : '#378ADD', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {module.moduleType === 'test' ? 'Test' : 'Learn'}
        </span>
      </div>

      {/* Content */}
      <div style={S.inner} key={step}>
        {current.title && (
          <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 20, lineHeight: 1.4 }}>
            <MathText as="span">{current.title}</MathText>
          </h3>
        )}

        {current.type === 'info' && (
          <>
            {current.content.split('\n\n').map((para, i) => (
              <MathText key={i} style={{ marginBottom: 14, lineHeight: 1.8, fontSize: 15, color: 'var(--color-text-secondary)' }}>
                {para}
              </MathText>
            ))}
            <button onClick={handleFlag} style={{ ...S.flagBtn, color: isFlagged ? '#BA7517' : 'var(--color-text-tertiary)', borderColor: isFlagged ? '#BA751744' : 'var(--color-border-tertiary)' }}>
              {isFlagged ? '\u2605 Flagged for review' : '\u2606 Need to learn this'}
            </button>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12 }}>
              {step > 0 ? <button onClick={handleBack} style={S.linkBtn}>&larr; Previous</button> : <span />}
              <button onClick={handleContinue} style={{ ...S.btn, background: accentColor }}>Continue &rarr;</button>
            </div>
          </>
        )}

        {current.type === 'mc' && (
          <>
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
                  border = `1.5px solid ${accentColor}`;
                }
                return (
                  <div key={i} onClick={() => !checked && setSelected(i)} style={{
                    padding: '12px 16px', borderRadius: 'var(--border-radius-md)',
                    background: bg, border, cursor: checked ? 'default' : 'pointer',
                    transition: 'all 0.15s', display: 'flex', alignItems: 'flex-start', gap: 10,
                  }}>
                    <div style={{
                      width: 20, height: 20, borderRadius: '50%', flexShrink: 0, marginTop: 2,
                      border: `2px solid ${isSel ? (checked ? (isCorrect ? '#1D9E75' : '#D85A30') : accentColor) : 'var(--color-border-secondary)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      {isSel && <div style={{ width: 10, height: 10, borderRadius: '50%', background: checked ? (isCorrect ? '#1D9E75' : '#D85A30') : accentColor }} />}
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

            <button onClick={handleFlag} style={{ ...S.flagBtn, color: isFlagged ? '#BA7517' : 'var(--color-text-tertiary)', borderColor: isFlagged ? '#BA751744' : 'var(--color-border-tertiary)' }}>
              {isFlagged ? '\u2605 Flagged for review' : '\u2606 Need to learn this'}
            </button>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12 }}>
              {step > 0 ? <button onClick={handleBack} style={S.linkBtn}>&larr; Previous</button> : <span />}
              {!checked ? (
                <button onClick={handleCheck} disabled={selected === null} style={{
                  ...S.btn, background: selected !== null ? accentColor : 'var(--color-border-tertiary)',
                  cursor: selected !== null ? 'pointer' : 'not-allowed', opacity: selected !== null ? 1 : 0.6,
                }}>Check</button>
              ) : (
                <button onClick={handleContinue} style={{ ...S.btn, background: accentColor }}>Continue &rarr;</button>
              )}
            </div>
          </>
        )}
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
  flagBtn: {
    background: 'transparent', border: '1px solid var(--color-border-tertiary)',
    cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
    padding: '5px 12px', borderRadius: 'var(--border-radius-md)',
    marginTop: 20, transition: 'all 0.15s',
  },
};
