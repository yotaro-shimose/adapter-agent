import React from 'react';
import ReactMarkdown from 'react-markdown';
import { TrajectoryView } from './TrajectoryView';
import type { CustomNode, CustomLink, TrajectoryData } from '../types';
import './GraphCanvas.css';

interface DetailPanelProps {
  selectedNode: CustomNode;
  onClose: () => void;
  onFocus: (nodeId: string) => void;
  handleNodeSelect: (node: CustomNode) => void; // For impacted tasks navigation
  trajectories: TrajectoryData[];
  selectedAttemptIndex: number;
  onSelectAttempt: (idx: number) => void;
  loadingTraj: boolean;
  trajError: string | null;
  allNodes: CustomNode[];
  allLinks: CustomLink[];
}

export const DetailPanel: React.FC<DetailPanelProps> = ({
  selectedNode,
  onClose,
  onFocus,
  handleNodeSelect,
  trajectories,
  selectedAttemptIndex,
  onSelectAttempt,
  loadingTraj,
  trajError,
  allNodes,
  allLinks,
}) => {
  const m = selectedNode.metadata;
  const isKnowledge = selectedNode.type === 'knowledge';

  return (
    <div className="detail-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '12px', gap: '10px' }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 'bold', color: '#61dafb', fontSize: '20px' }}>
            {isKnowledge ? 'Verified Knowledge' : 'Task Analysis'}
          </div>
        </div>
        <button
          onClick={() => onFocus(selectedNode.id)}
          style={{ padding: '6px 12px', background: 'rgba(97, 218, 251, 0.2)', color: '#61dafb', border: '1px solid #61dafb', borderRadius: '6px', fontSize: '12px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Focus
        </button>
        <button
          onClick={onClose}
          style={{ background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', cursor: 'pointer', width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px' }}
        >
          ×
        </button>
      </div>

      <div className="markdown-content" style={{ fontSize: '15px', color: '#fff', background: 'rgba(97, 218, 251, 0.1)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(97, 218, 251, 0.2)', marginBottom: '24px' }}>
        {isKnowledge && m.knowledge_title && (
          <div style={{ fontWeight: 'bold', fontSize: '18px', marginBottom: '10px', color: '#61dafb', borderBottom: '1px solid rgba(97, 218, 251, 0.3)', paddingBottom: '8px' }}>
            {m.knowledge_title}
          </div>
        )}
        <ReactMarkdown>{isKnowledge && m.knowledge_content ? m.knowledge_content : m.instruction}</ReactMarkdown>
      </div>

      {!isKnowledge && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '25px' }}>
          <div style={{ background: 'rgba(40, 167, 69, 0.15)', padding: '12px', borderRadius: '10px', borderLeft: '4px solid #28a745' }}>
            <div style={{ fontSize: '11px', color: '#aaa', textTransform: 'uppercase', fontWeight: 'bold' }}>Success Rate</div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#28a745', marginTop: '4px' }}>{m.success_count}/{m.total_count}</div>
          </div>
          <div style={{ background: 'rgba(241, 196, 15, 0.15)', padding: '12px', borderRadius: '10px', borderLeft: '4px solid #f1c40f' }}>
            <div style={{ fontSize: '11px', color: '#aaa', textTransform: 'uppercase', fontWeight: 'bold' }}>Status</div>
            <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#f1c40f', marginTop: '4px' }}>{m.is_solved ? '✅ Solved' : m.is_executing ? '⚡ Active' : '⏳ Queued'}</div>
          </div>
        </div>
      )}

      {selectedNode.type === 'task' && (
        <TrajectoryView
          trajectories={trajectories}
          selectedAttemptIndex={selectedAttemptIndex}
          onSelectAttempt={onSelectAttempt}
          loading={loadingTraj}
          error={trajError}
        />
      )}

      {isKnowledge && (
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
            Impacted Tasks
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {allLinks
              .filter(l => (typeof l.target === 'string' ? l.target : (l.target as any).id) === selectedNode.id)
              .map((l, i) => {
                const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id;
                const sourceNode = allNodes.find(n => n.id === sourceId);
                return (
                  <div 
                    key={i} 
                    onClick={() => sourceNode && handleNodeSelect(sourceNode)}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      padding: '10px',
                      borderRadius: '8px',
                      fontSize: '13px',
                      borderLeft: '3px solid #61dafb',
                      cursor: sourceNode ? 'pointer' : 'default',
                      transition: 'background 0.2s',
                      textOverflow: 'ellipsis',
                      overflow: 'hidden'
                    }}
                  >
                    {sourceNode?.name || sourceId}
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {m.slices.length > 0 && (
        <div>
          <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
            Knowledge Slices ({m.slices.length})
          </div>
          {m.slices.map((s, i) => (
            <div key={i} style={{ borderBottom: '1px solid #333', padding: '16px 0' }}>
              <div style={{ fontSize: '14px', marginBottom: '8px', fontWeight: '500' }}>
                <span style={{ color: '#61dafb', fontWeight: 'bold' }}>Q:</span> {s.question}
              </div>
              <div style={{ fontSize: '13px', color: '#eee', background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '6px' }}>
                <span style={{ color: '#28a745', fontWeight: '600' }}>A:</span> {s.answer}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
