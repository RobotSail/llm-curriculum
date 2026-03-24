import { useMemo } from 'react';
import { buildModuleLookup } from '../data/recommended-curriculums';
import { getModuleProgress } from './ModuleView';
import MathText from './MathText';

const DIFF_COLORS = { easy: '#1D9E75', medium: '#BA7517', hard: '#D85A30' };

export default function CurriculumDetailView({ curriculum, onClose, onOpenModule }) {
  const lookup = useMemo(() => buildModuleLookup(), []);
  const progress = useMemo(() => getModuleProgress(), []);

  const items = curriculum.items;
  const completedCount = items.filter(item => {
    const p = progress[item.moduleId];
    return p && p.completed;
  }).length;
  const pct = items.length > 0 ? Math.round(completedCount / items.length * 100) : 0;

  // Group items by phase
  const phases = [];
  let currentPhase = null;
  for (const item of items) {
    if (item.phase !== (currentPhase && currentPhase.name)) {
      currentPhase = { name: item.phase, items: [] };
      phases.push(currentPhase);
    }
    currentPhase.items.push(item);
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'var(--color-background-primary)',
      zIndex: 1000, overflowY: 'auto',
    }}>
      <div style={{maxWidth:560,margin:'0 auto',padding:'2rem 1rem'}}>
        {/* Header */}
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:'8px'}}>
          <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
            <div style={{width:8,height:8,borderRadius:'50%',background:curriculum.color,flexShrink:0,marginTop:3}}/>
            <h2 style={{fontSize:16,fontWeight:600,margin:0,color:'var(--color-text-primary)'}}>{curriculum.title}</h2>
          </div>
          <button onClick={onClose} style={{background:'transparent',border:'none',fontSize:20,cursor:'pointer',color:'var(--color-text-tertiary)',fontFamily:'inherit',padding:'0 4px',flexShrink:0}}>&times;</button>
        </div>

        <p style={{fontSize:11,color:'var(--color-text-tertiary)',lineHeight:1.5,marginBottom:'16px',marginLeft:16}}>
          {curriculum.description}
        </p>

        {/* Progress bar */}
        <div style={{marginBottom:'20px'}}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'4px'}}>
            <span style={{fontSize:12,color:'var(--color-text-secondary)'}}>
              {completedCount} of {items.length} modules completed
            </span>
            <span style={{fontSize:14,fontWeight:500,color:curriculum.color}}>{pct}%</span>
          </div>
          <div style={{height:5,background:'var(--color-background-secondary)',borderRadius:3,overflow:'hidden'}}>
            <div style={{height:'100%',width:`${pct}%`,background:curriculum.color,borderRadius:3,transition:'width 0.3s ease'}}/>
          </div>
        </div>

        {/* Module list grouped by phase */}
        {phases.map((phase, pi) => (
          <div key={pi} style={{marginBottom:'16px'}}>
            <div style={{fontSize:10,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.06em',color:curriculum.color,marginBottom:'8px'}}>
              {phase.name}
            </div>
            {phase.items.map((item) => {
              const mod = lookup[item.moduleId];
              const isTbd = item.tbd || !mod;
              const isCompleted = !isTbd && progress[item.moduleId]?.completed;
              const globalIdx = items.indexOf(item);
              const isLast = globalIdx === items.length - 1;

              return (
                <div
                  key={item.moduleId}
                  onClick={() => {
                    if (!isTbd && mod) onOpenModule(mod);
                  }}
                  style={{
                    display:'flex',gap:'10px',
                    cursor: isTbd ? 'default' : 'pointer',
                    opacity: isTbd ? 0.5 : 1,
                  }}
                >
                  {/* Step indicator + connecting line */}
                  <div style={{display:'flex',flexDirection:'column',alignItems:'center',width:22,flexShrink:0}}>
                    <div style={{
                      width:20,height:20,borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',
                      fontSize:10,fontWeight:600,flexShrink:0,
                      background: isCompleted ? curriculum.color : 'var(--color-background-secondary)',
                      color: isCompleted ? 'white' : 'var(--color-text-tertiary)',
                      border: isCompleted ? 'none' : '1.5px solid var(--color-border-tertiary)',
                    }}>
                      {isCompleted ? '\u2713' : globalIdx + 1}
                    </div>
                    {!isLast && (
                      <div style={{width:1.5,flex:1,minHeight:8,background:'var(--color-border-tertiary)'}}/>
                    )}
                  </div>

                  {/* Content */}
                  <div style={{flex:1,paddingBottom: isLast ? 0 : '8px',minWidth:0}}>
                    <div style={{
                      padding:'8px 12px',borderRadius:'var(--border-radius-md)',
                      background:'var(--color-background-secondary)',
                      border: isCompleted
                        ? `1px solid ${curriculum.color}44`
                        : '0.5px solid var(--color-border-tertiary)',
                      transition:'border-color 0.15s',
                    }}
                    onMouseEnter={e => { if (!isTbd) e.currentTarget.style.borderColor = curriculum.color + '88'; }}
                    onMouseLeave={e => { if (!isTbd) e.currentTarget.style.borderColor = isCompleted ? curriculum.color + '44' : 'var(--color-border-tertiary)'; }}
                    >
                      <div style={{display:'flex',alignItems:'center',gap:'5px',flexWrap:'wrap',marginBottom: item.note ? 3 : 0}}>
                        <span style={{fontSize:12,fontWeight:500,color:'var(--color-text-primary)'}}>
                          {mod ? mod.title : (item.tbd ? item.moduleId.replace(/-/g, ' ') : item.moduleId)}
                        </span>
                        {isTbd && (
                          <span style={{fontSize:9,fontWeight:600,padding:'1px 5px',borderRadius:3,background:'#6B728018',color:'#6B7280',textTransform:'uppercase',letterSpacing:'0.04em'}}>
                            Coming Soon
                          </span>
                        )}
                        {mod && (
                          <span style={{fontSize:9,fontWeight:600,padding:'1px 5px',borderRadius:3,
                            background: mod.moduleType === 'learning' ? '#378ADD18' : '#8B5CF618',
                            color: mod.moduleType === 'learning' ? '#378ADD' : '#8B5CF6',
                            textTransform:'uppercase',letterSpacing:'0.04em'
                          }}>
                            {mod.moduleType === 'learning' ? 'Learn' : 'Test'}
                          </span>
                        )}
                        {mod && mod.difficulty && (
                          <span style={{fontSize:9,fontWeight:600,padding:'1px 5px',borderRadius:3,
                            background: DIFF_COLORS[mod.difficulty] + '18',
                            color: DIFF_COLORS[mod.difficulty],
                            textTransform:'uppercase',letterSpacing:'0.04em'
                          }}>
                            {mod.difficulty}
                          </span>
                        )}
                        {mod && mod.estimatedMinutes && (
                          <span style={{fontSize:10,color:'var(--color-text-tertiary)'}}>
                            {mod.estimatedMinutes} min
                          </span>
                        )}
                        {isCompleted && (
                          <span style={{fontSize:10,color:curriculum.color,fontWeight:500}}>Done</span>
                        )}
                      </div>
                      {item.note && (
                        <MathText as="div" style={{fontSize:11,color:'var(--color-text-tertiary)',lineHeight:1.4}}>
                          {item.note}
                        </MathText>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
