import { useMemo } from 'react';
import { RECOMMENDED_CURRICULUMS, buildModuleLookup } from '../data/recommended-curriculums';
import { getModuleProgress } from './ModuleView';

export default function CurriculumListView({ onClose, onSelect }) {
  const lookup = useMemo(() => buildModuleLookup(), []);
  const progress = useMemo(() => getModuleProgress(), []);

  return (
    <div style={{position:'fixed',inset:0,zIndex:2000,display:'flex',justifyContent:'center',alignItems:'flex-start'}}>
      <div onClick={onClose} style={{position:'absolute',inset:0,background:'rgba(0,0,0,0.4)'}}/>
      <div style={{position:'relative',background:'var(--color-background-primary)',borderRadius:'var(--border-radius-lg)',maxWidth:600,width:'100%',margin:'48px 16px',maxHeight:'calc(100vh - 96px)',overflow:'auto',border:'0.5px solid var(--color-border-tertiary)',padding:'24px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'20px'}}>
          <h2 style={{fontSize:16,fontWeight:600,margin:0,color:'var(--color-text-primary)'}}>Recommended Curriculums</h2>
          <button onClick={onClose} style={{background:'transparent',border:'none',fontSize:20,cursor:'pointer',color:'var(--color-text-tertiary)',fontFamily:'inherit',padding:'0 4px'}}>&times;</button>
        </div>
        <p style={{fontSize:12,color:'var(--color-text-tertiary)',marginBottom:'16px',lineHeight:1.5}}>
          Focused study paths for specific research areas. Each curriculum is an ordered sequence of modules — work through them day by day.
        </p>

        {RECOMMENDED_CURRICULUMS.map(curriculum => {
          const totalItems = curriculum.items.length;
          const completedItems = curriculum.items.filter(item => {
            const mod = lookup[item.moduleId];
            if (!mod) return false;
            const p = progress[item.moduleId];
            return p && p.completed;
          }).length;
          const pct = totalItems > 0 ? Math.round(completedItems / totalItems * 100) : 0;
          const tbdCount = curriculum.items.filter(i => i.tbd).length;

          return (
            <div
              key={curriculum.id}
              onClick={() => onSelect(curriculum)}
              style={{
                padding:'16px',marginBottom:'8px',borderRadius:'var(--border-radius-md)',
                background:'var(--color-background-secondary)',
                border:`1px solid ${curriculum.color}33`,
                cursor:'pointer',transition:'border-color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = curriculum.color + '88'}
              onMouseLeave={e => e.currentTarget.style.borderColor = curriculum.color + '33'}
            >
              <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'6px'}}>
                <div style={{width:8,height:8,borderRadius:'50%',background:curriculum.color,flexShrink:0}}/>
                <h3 style={{fontSize:14,fontWeight:600,margin:0,color:'var(--color-text-primary)',flex:1}}>{curriculum.title}</h3>
              </div>
              <p style={{fontSize:11,color:'var(--color-text-secondary)',lineHeight:1.5,margin:'0 0 10px 16px'}}>
                {curriculum.description}
              </p>
              <div style={{marginLeft:16}}>
                <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'4px'}}>
                  <span style={{fontSize:11,color:'var(--color-text-tertiary)'}}>
                    {completedItems} of {totalItems} modules{tbdCount > 0 ? ` (${tbdCount} coming soon)` : ''}
                  </span>
                  <span style={{fontSize:12,fontWeight:500,color:curriculum.color}}>{pct}%</span>
                </div>
                <div style={{height:4,background:'var(--color-background-primary)',borderRadius:2,overflow:'hidden'}}>
                  <div style={{height:'100%',width:`${pct}%`,background:curriculum.color,borderRadius:2,transition:'width 0.3s ease'}}/>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
