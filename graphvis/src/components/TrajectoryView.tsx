import React from 'react';
import ReactMarkdown from 'react-markdown';
import { getTrajectoryContent } from '../utils';
import type { TrajectoryData } from '../types';
import './GraphCanvas.css';

interface TrajectoryViewProps {
  trajectories: TrajectoryData[];
  selectedAttemptIndex: number;
  onSelectAttempt: (index: number) => void;
  loading: boolean;
  error: string | null;
}

export const TrajectoryView: React.FC<TrajectoryViewProps> = ({
  trajectories,
  selectedAttemptIndex,
  onSelectAttempt,
  loading,
  error,
}) => {
  if (loading) return (
    <div style={{ padding: '30px', textAlign: 'center' }}>
      <div className="spinner-small" style={{ margin: '0 auto 10px auto' }}></div>
      <div style={{ color: '#888', fontStyle: 'italic', fontSize: '13px' }}>Loading trajectories...</div>
    </div>
  );
  
  if (error) return (
    <div style={{ margin: '20px 0', padding: '16px', background: 'rgba(231, 76, 60, 0.1)', border: '1px solid rgba(231, 76, 60, 0.3)', borderRadius: '8px', color: '#e74c3c', fontSize: '13px' }}>
      <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>⚠️ Failed to load trajectories</div>
      <div style={{ opacity: 0.8 }}>{error}</div>
    </div>
  );

  if (!trajectories || trajectories.length === 0) return (
    <div style={{ padding: '40px 20px', textAlign: 'center', color: '#666', border: '1px dashed #333', borderRadius: '12px', marginTop: '20px' }}>
      <div style={{ fontSize: '24px', marginBottom: '10px' }}>📁</div>
      <div style={{ fontSize: '14px' }}>No trajectories found</div>
    </div>
  );

  const currentTrajectory = trajectories[selectedAttemptIndex];
  if (!currentTrajectory || !currentTrajectory.trials) return null;

  return (
    <div style={{ marginTop: '30px', borderTop: '1px solid #333', paddingTop: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 'bold' }}>
          Attempts ({trajectories.length})
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {trajectories.map((t, idx) => (
            <button
              key={t.id}
              onClick={() => onSelectAttempt(idx)}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                background: selectedAttemptIndex === idx ? 'rgba(97, 218, 251, 0.2)' : 'rgba(255,255,255,0.05)',
                color: selectedAttemptIndex === idx ? '#61dafb' : '#ccc',
                border: `1px solid ${selectedAttemptIndex === idx ? '#61dafb' : '#444'}`,
                cursor: 'pointer'
              }}
            >
              #{idx + 1} {t.reward > 0 ? '✅' : '❌'}
            </button>
          ))}
        </div>
      </div>

      <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '15px', fontWeight: 'bold' }}>
          Reasoning Trace ({currentTrajectory.trials.length} turns)
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {currentTrajectory.trials.map((turn, i) => {
          const contentStr = getTrajectoryContent(turn);

          const isThought = turn.role === 'assistant' && contentStr.includes('<think>');
          const thoughtMatch = contentStr.match(/<think>([\s\S]*?)<\/think>/);
          const restContent = contentStr.replace(/<think>[\s\S]*?<\/think>/, '').trim();

          return (
            <div key={i} className="trajectory-turn">
              {isThought && thoughtMatch && (
                <div className="trajectory-card thought">
                  <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase' }}>Thought</div>
                  <ReactMarkdown>{thoughtMatch[1].trim()}</ReactMarkdown>
                </div>
              )}
              {(turn.role === 'assistant' && (restContent || (turn.tool_calls && turn.tool_calls.length > 0))) && (
                <div className="trajectory-card assistant">
                  <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase' }}>Assistant</div>
                  {restContent && <ReactMarkdown>{restContent}</ReactMarkdown>}
                  
                  {turn.tool_calls && turn.tool_calls.length > 0 && (
                    <div style={{ marginTop: restContent ? '12px' : '0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {turn.tool_calls.map((tc, idx) => (
                        <div key={idx} style={{ background: 'rgba(97, 218, 251, 0.1)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(97, 218, 251, 0.2)' }}>
                          <div style={{ fontSize: '10px', color: '#61dafb', fontWeight: 'bold', marginBottom: '4px' }}>🛠 CALL: {tc.function?.name}</div>
                          <pre style={{ margin: 0 }}>{typeof tc.function?.arguments === 'string' ? tc.function.arguments : JSON.stringify(tc.function?.arguments, null, 2)}</pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {turn.role === 'tool' && (() => {
                const cit = currentTrajectory.citations.find(c => c.turn_index === i);
                return (
                  <div className="trajectory-card tool">
                    <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase', color: '#28a745' }}>
                      {cit ? `Knowledge Acquired (#${cit.knowledge_id})` : 'Tool Result'}
                    </div>
                    {cit ? <ReactMarkdown>{cit.content}</ReactMarkdown> : <pre>{contentStr}</pre>}
                  </div>
                );
              })()}
              {(turn.role === 'user' || turn.role === 'system') && (
                <div className="trajectory-card user">
                  <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase' }}>{turn.role === 'system' ? 'System' : 'User'}</div>
                  <ReactMarkdown>{contentStr}</ReactMarkdown>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
