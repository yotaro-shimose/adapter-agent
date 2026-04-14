import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { API_BASE } from '../constants';
import type { SimpleKnowledgeAggr, SimpleTrajectory, SimpleSftQna } from '../types';


export const SimplePipelineVis: React.FC = () => {
  const [trainIds, setTrainIds] = useState<string[]>([]);
  const [selectedTrain, setSelectedTrain] = useState<string>('');
  const [knowledgeList, setKnowledgeList] = useState<SimpleKnowledgeAggr[]>([]);
  const [selectedKnowledge, setSelectedKnowledge] = useState<SimpleKnowledgeAggr | null>(null);
  const [rollouts, setRollouts] = useState<SimpleTrajectory[]>([]);
  const [sftQnas, setSftQnas] = useState<SimpleSftQna[]>([]);
  const [activeTab, setActiveTab] = useState<'trajectories' | 'sft' | 'granular'>('trajectories');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);


  const loadTrains = () => {
    return fetch(`${API_BASE}/api/simple_trains`)
      .then(r => r.json())
      .then(data => {
        setTrainIds(data);
        if (data.length > 0 && !selectedTrain) {
          setSelectedTrain(data[0]);
        }
        return data;
      })
      .catch(e => console.error("Failed to load simple trains:", e));
  };

  useEffect(() => {
    loadTrains();
  }, []);

  const loadKnowledge = (trainId: string) => {
    setLoading(true);
    return fetch(`${API_BASE}/api/simple_train/${trainId}/knowledge`)
      .then(r => r.json())
      .then(data => {
        setKnowledgeList(data);
        // If we previously selected a knowledge item, try to update it in state to get fresh stats
        if (selectedKnowledge) {
          const updated = data.find((k: SimpleKnowledgeAggr) => k.knowledge_id === selectedKnowledge.knowledge_id);
          if (updated) setSelectedKnowledge(updated);
        }
        setLoading(false);
        return data;
      })
      .catch(e => {
        console.error("Failed to load knowledge:", e);
        setLoading(false);
      });
  };

  useEffect(() => {
    if (!selectedTrain) return;
    loadKnowledge(selectedTrain);
  }, [selectedTrain]);

  const loadRollouts = (trainId: string, knowledgeId: string) => {
    return fetch(`${API_BASE}/api/simple_train/${trainId}/knowledge/${knowledgeId}/rollouts`)
      .then(r => r.json())
      .then(data => {
        setRollouts(data);
        return data;
      })
      .catch(e => console.error("Failed to load rollouts:", e));
  };

  const loadSftQnas = (trainId: string, knowledgeId: string) => {
    return fetch(`${API_BASE}/api/simple_train/${trainId}/knowledge/${knowledgeId}/sft_qnas`)
      .then(r => r.json())
      .then(data => {
        setSftQnas(data);
        return data;
      })
      .catch(e => console.error("Failed to load SFT QRAs:", e));
  };

  const handleSelectKnowledge = (k: SimpleKnowledgeAggr) => {
    setSelectedKnowledge(k);
    setRollouts([]); // clear old
    setSftQnas([]);   // clear old
    loadRollouts(selectedTrain, k.knowledge_id);
    loadSftQnas(selectedTrain, k.knowledge_id);
  };


  const handleRefresh = async () => {
    setRefreshing(true);
    await loadTrains();
    if (selectedTrain) {
      await loadKnowledge(selectedTrain);
      if (selectedKnowledge) {
        await loadRollouts(selectedTrain, selectedKnowledge.knowledge_id);
        await loadSftQnas(selectedTrain, selectedKnowledge.knowledge_id);
      }

    }
    setRefreshing(false);
  };

  return (
    <div style={{ display: 'flex', height: '100%', fontFamily: '"Inter", sans-serif', textAlign: 'left' }}>
      {/* Left Sidebar: Knowledge Grid */}
      <div style={{ width: '400px', borderRight: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', background: 'rgba(15, 17, 26, 0.5)' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#ccc' }}>Select Training Run</h3>
            <button 
              onClick={handleRefresh}
              disabled={refreshing}
              style={{ 
                background: 'transparent', 
                border: '1px solid #444', 
                color: refreshing ? '#555' : '#888', 
                borderRadius: '4px', 
                padding: '4px 8px', 
                fontSize: '0.75rem', 
                cursor: refreshing ? 'default' : 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          <select 
            value={selectedTrain} 
            onChange={e => setSelectedTrain(e.target.value)}
            style={{ width: '100%', padding: '10px', background: '#1a1d27', color: '#fff', border: '1px solid #333', borderRadius: '6px', outline: 'none' }}
          >
            {trainIds.map(id => <option key={id} value={id}>{id}</option>)}
          </select>
        </div>
        
        <div style={{ flex: 1, overflowY: 'auto', padding: '15px' }}>
          {loading ? (
            <div style={{ color: '#888', textAlign: 'center', padding: '20px' }}>Loading Knowledge Base...</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {knowledgeList.map(k => (
                <div 
                  key={k.knowledge_id} 
                  onClick={() => handleSelectKnowledge(k)}
                  style={{ 
                    padding: '15px', 
                    background: selectedKnowledge?.knowledge_id === k.knowledge_id ? 'rgba(97, 218, 251, 0.1)' : 'rgba(255,255,255,0.03)', 
                    border: `1px solid ${selectedKnowledge?.knowledge_id === k.knowledge_id ? '#61dafb' : 'transparent'}`,
                    borderRadius: '8px', 
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={e => { if (selectedKnowledge?.knowledge_id !== k.knowledge_id) e.currentTarget.style.background = 'rgba(255,255,255,0.06)' }}
                  onMouseLeave={e => { if (selectedKnowledge?.knowledge_id !== k.knowledge_id) e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
                >
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem', color: '#fff', lineHeight: 1.4 }}>{k.knowledge_title}</h4>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#aaa', marginTop: '5px' }}>
                    <span>Success: <strong style={{ color: k.total_success > 0 ? '#28a745' : '#ff4d4d' }}>{k.total_success}</strong> / {k.total_rollouts}</span>
                    <span style={{ color: '#61dafb' }}>{k.sft_count || 0} SFT</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '5px' }}>
                    {k.steps.length} Steps
                  </div>

                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right Content Area: Rollouts */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '30px', background: 'radial-gradient(circle at top right, rgba(97, 218, 251, 0.05), transparent 40%)' }}>
        {!selectedKnowledge ? (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#555', fontSize: '1.2rem' }}>
            Select a knowledge item from the sidebar to view rollouts
          </div>
        ) : (
          <div style={{ maxWidth: '900px', margin: '0 auto', paddingBottom: '100px' }}>
            <h2 style={{ fontSize: '1.5rem', marginBottom: '10px' }}>{selectedKnowledge.knowledge_title}</h2>
            <div style={{ marginBottom: '30px', display: 'flex', gap: '15px', alignItems: 'center' }}>
              <span style={{ padding: '4px 10px', background: 'rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '0.85rem' }}>ID: {selectedKnowledge.knowledge_id}</span>
              <span style={{ padding: '4px 10px', background: 'rgba(97, 218, 251, 0.1)', color: '#61dafb', borderRadius: '12px', fontSize: '0.85rem' }}>Total Rollouts: {selectedKnowledge.total_rollouts}</span>
              <span style={{ padding: '4px 10px', background: 'rgba(155, 89, 182, 0.1)', color: '#9b59b6', borderRadius: '12px', fontSize: '0.85rem' }}>SFT Records: {selectedKnowledge.sft_count || 0}</span>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.1)', marginBottom: '30px', gap: '20px' }}>
              <button 
                onClick={() => setActiveTab('trajectories')}
                style={{
                  padding: '10px 20px',
                  background: 'none',
                  border: 'none',
                  color: activeTab === 'trajectories' ? '#61dafb' : '#888',
                  borderBottom: activeTab === 'trajectories' ? '2px solid #61dafb' : 'none',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  fontWeight: activeTab === 'trajectories' ? 600 : 400
                }}
              >
                Trajectories (RL)
              </button>
              <button 
                onClick={() => setActiveTab('sft')}
                style={{
                  padding: '10px 20px',
                  background: 'none',
                  border: 'none',
                  color: activeTab === 'sft' ? '#9b59b6' : '#888',
                  borderBottom: activeTab === 'sft' ? '2px solid #9b59b6' : 'none',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  fontWeight: activeTab === 'sft' ? 600 : 400
                }}
              >
                SFT Dataset (Static)
              </button>
              <button 
                onClick={() => setActiveTab('granular')}
                style={{
                  padding: '10px 20px',
                  background: 'none',
                  border: 'none',
                  color: activeTab === 'granular' ? '#f1c40f' : '#888',
                  borderBottom: activeTab === 'granular' ? '2px solid #f1c40f' : 'none',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  fontWeight: activeTab === 'granular' ? 600 : 400
                }}
              >
                Granular Insight (Article)
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {activeTab === 'trajectories' ? (
                <>
                  {rollouts.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#555' }}>No RL trajectories recorded yet.</div>
                  ) : (
                    rollouts.map(r => <RolloutCard key={r.id} rollout={r} />)
                  )}
                </>
              ) : activeTab === 'sft' ? (
                <>
                  {sftQnas.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#555' }}>No SFT data found for this knowledge item.</div>
                  ) : (
                    sftQnas.map(q => <SftQnaCard key={q.id} qna={q} />)
                  )}
                </>
              ) : (
                <div style={{ 
                  background: 'rgba(25, 28, 41, 0.7)', 
                  border: '1px solid rgba(241, 196, 15, 0.3)', 
                  borderRadius: '10px', 
                  padding: '30px',
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
                  textAlign: 'left'
                }}>
                   <h5 style={{ margin: '0 0 20px 0', color: '#f1c40f', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Granular Knowledge Content</h5>
                   <div className="markdown-body" style={{ color: '#fff', fontSize: '0.95rem', lineHeight: 1.6 }}>
                      {selectedKnowledge.content ? (
                        <ReactMarkdown>{selectedKnowledge.content}</ReactMarkdown>
                      ) : (
                        <div style={{ color: '#555', fontStyle: 'italic' }}>No granular content available for this node.</div>
                      )}
                   </div>
                </div>
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
};

const RolloutCard = ({ rollout }: { rollout: SimpleTrajectory }) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div style={{ 
      background: 'rgba(25, 28, 41, 0.7)', 
      border: `1px solid ${rollout.success ? 'rgba(40, 167, 69, 0.3)' : 'rgba(255, 77, 77, 0.3)'}`, 
      borderRadius: '10px', 
      overflow: 'hidden',
      backdropFilter: 'blur(10px)',
      boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
    }}>
      {/* Header bar */}
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{ 
          padding: '15px 20px', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid rgba(255,255,255,0.05)' : 'none',
          background: rollout.success ? 'linear-gradient(90deg, rgba(40, 167, 69, 0.1), transparent)' : 'linear-gradient(90deg, rgba(255, 77, 77, 0.1), transparent)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ 
            width: '40px', height: '40px', 
            borderRadius: '50%', 
            background: rollout.success ? '#28a745' : '#ff4d4d', 
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 'bold', color: '#fff', fontSize: '0.8rem',
            boxShadow: 'inset 0 2px 4px rgba(255,255,255,0.2)'
          }}>
            S{rollout.step}
          </div>
          <div>
            <span style={{ fontWeight: 500, display: 'block' }}>Step {rollout.step} Evaluation</span>
            <span style={{ fontSize: '0.75rem', color: '#888' }}>
              {rollout.created_at ? new Date(rollout.created_at).toLocaleString() : 'Processing...'}
            </span>
          </div>
        </div>
        <div>
          <span style={{ 
            padding: '6px 14px', 
            borderRadius: '20px', 
            fontSize: '0.85rem', 
            fontWeight: 700,
            background: rollout.success ? 'rgba(40, 167, 69, 0.2)' : 'rgba(255, 77, 77, 0.2)',
            color: rollout.success ? '#2dd35c' : '#ff7b7b'
          }}>
            {rollout.success ? 'SUCCESS' : 'FAILED'}
          </span>
        </div>
      </div>
      
      {/* Expanded content */}
      {expanded && (
        <div style={{ padding: '0px' }}>
          <div style={{ padding: '20px', background: 'rgba(0,0,0,0.2)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#aaa', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Task Question</h5>
            <div style={{ color: '#e0e0e0', lineHeight: 1.5, fontSize: '0.95rem', whiteSpace: 'pre-wrap', textAlign: 'left' }}>
              {rollout.question ? rollout.question.replace(/\\n/g, '\n') : ''}
            </div>
          </div>
          
          {rollout.reasoning && (
            <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <h5 style={{ margin: '0 0 10px 0', color: '#9b59b6', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Reasoning (Think)</h5>
              <div style={{ color: '#b3b3b3', fontSize: '0.9rem', fontStyle: 'italic', whiteSpace: 'pre-wrap', maxHeight: '300px', overflowY: 'auto', paddingRight: '10px', textAlign: 'left' }}>
                {rollout.reasoning.replace(/\\n/g, '\n')}
              </div>
            </div>
          )}
          
          <div style={{ padding: '20px' }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#61dafb', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Answer / Code</h5>
            <div className="markdown-body" style={{ color: '#fff', fontSize: '0.95rem', overflowX: 'auto', textAlign: 'left' }}>
              <ReactMarkdown>{rollout.answer ? rollout.answer.replace(/\\n/g, '\n') : ''}</ReactMarkdown>
            </div>
          </div>

          {rollout.execution_output && (
            <div style={{ padding: '20px', background: 'rgba(0,0,0,0.3)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              <h5 style={{ margin: '0 0 10px 0', color: '#f1c40f', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Execution Output</h5>
              <pre style={{ 
                margin: 0, 
                padding: '15px', 
                background: '#0a0c14', 
                color: '#d1d1d1', 
                fontSize: '0.85rem', 
                borderRadius: '6px', 
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
                fontFamily: 'source-code-pro, Menlo, Monaco, Consolas, "Courier New", monospace',
                border: '1px solid rgba(255,255,255,0.05)',
                textAlign: 'left'
              }}>
                {rollout.execution_output.replace(/\\n/g, '\n')}
              </pre>
            </div>
          )}

          {rollout.verification_output && (
            <div style={{ padding: '20px', background: 'rgba(155, 89, 182, 0.05)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
              <h5 style={{ margin: '0 0 10px 0', color: '#9b59b6', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Verifier Reasoning</h5>
              <div style={{ color: '#ccc', fontSize: '0.9rem', lineHeight: 1.5, textAlign: 'left' }}>
                {rollout.verification_output.replace(/\\n/g, '\n')}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
const SftQnaCard = ({ qna }: { qna: SimpleSftQna }) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div style={{ 
      background: 'rgba(25, 28, 41, 0.7)', 
      border: '1px solid rgba(155, 89, 182, 0.3)', 
      borderRadius: '10px', 
      overflow: 'hidden',
      backdropFilter: 'blur(10px)',
      boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
    }}>
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{ 
          padding: '15px 20px', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid rgba(255,255,255,0.05)' : 'none',
          background: 'linear-gradient(90deg, rgba(155, 89, 182, 0.1), transparent)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ 
            width: '40px', height: '40px', 
            borderRadius: '50%', 
            background: '#9b59b6', 
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 'bold', color: '#fff', fontSize: '0.8rem',
            boxShadow: 'inset 0 2px 4px rgba(255,255,255,0.2)'
          }}>
            SFT
          </div>
          <div>
            <span style={{ fontWeight: 500, display: 'block' }}>SFT Seed Triple</span>
            <span style={{ fontSize: '0.75rem', color: '#888' }}>
              {qna.created_at ? new Date(qna.created_at).toLocaleString() : ''}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.8rem', color: '#9b59b6', fontWeight: 600 }}>
          {expanded ? 'HIDE DETAILS' : 'VIEW DETAILS'}
        </div>
      </div>
      
      {expanded && (
        <div style={{ padding: '0px' }}>
          <div style={{ padding: '20px', background: 'rgba(0,0,0,0.2)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#aaa', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Question</h5>
            <div style={{ color: '#e0e0e0', lineHeight: 1.5, fontSize: '0.95rem', whiteSpace: 'pre-wrap', textAlign: 'left' }}>
              {qna.question.replace(/\\n/g, '\n')}
            </div>
          </div>
          
          <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#9b59b6', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Golden Reasoning</h5>
            <div style={{ color: '#b3b3b3', fontSize: '0.9rem', fontStyle: 'italic', whiteSpace: 'pre-wrap', maxHeight: '300px', overflowY: 'auto', paddingRight: '10px', textAlign: 'left' }}>
              {qna.reasoning.replace(/\\n/g, '\n')}
            </div>
          </div>
          
          <div style={{ padding: '20px' }}>
            <h5 style={{ margin: '0 0 10px 0', color: '#61dafb', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'left' }}>Golden Answer</h5>
            <div className="markdown-body" style={{ color: '#fff', fontSize: '0.95rem', overflowX: 'auto', textAlign: 'left' }}>
              <ReactMarkdown>{qna.answer.replace(/\\n/g, '\n')}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
