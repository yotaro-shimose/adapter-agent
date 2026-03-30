import React from 'react';
import './GraphCanvas.css';

interface ControlPanelProps {
  experiments: string[];
  selectedExperiment: string | null;
  onSelectExperiment: (exp: string) => void;
  onRefresh: () => void;
  showKnowledge: boolean;
  onToggleKnowledge: () => void;
  viewMode: 'global' | 'local';
  onSetGlobalView: () => void;
  focusDepth: number;
  onSetFocusDepth: (depth: number) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  experiments,
  selectedExperiment,
  onSelectExperiment,
  onRefresh,
  showKnowledge,
  onToggleKnowledge,
  viewMode,
  onSetGlobalView,
  focusDepth,
  onSetFocusDepth,
}) => {
  return (
    <div className="control-panel">
      {experiments.length > 0 && (
        <select 
          value={selectedExperiment || ''} 
          onChange={(e) => onSelectExperiment(e.target.value)}
        >
          {experiments.map(exp => (
            <option key={exp} value={exp} style={{ background: '#222' }}>{exp}</option>
          ))}
        </select>
      )}

      <button onClick={onRefresh}>Refresh</button>

      <button 
        onClick={onToggleKnowledge}
        className={showKnowledge ? 'active' : ''}
      >
        {showKnowledge ? 'Hide' : 'Show'} Knowledge
      </button>

      {viewMode === 'local' && (
        <>
          <button 
            onClick={onSetGlobalView}
            className="danger"
          >
            Global View
          </button>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            padding: '0 10px',
            color: '#888',
            fontSize: '12px',
            borderLeft: '1px solid #333'
          }}>
            <span>Depth: {focusDepth}</span>
            <input 
              type="range" 
              min="1" 
              max="5" 
              value={focusDepth} 
              onChange={(e) => onSetFocusDepth(parseInt(e.target.value))} 
              style={{ width: '80px', cursor: 'pointer' }}
            />
          </div>
        </>
      )}
    </div>
  );
};
