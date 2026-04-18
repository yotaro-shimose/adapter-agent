import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { API_BASE } from '../constants';
import type { OpenBookQA, OpenBookTrajectory } from '../types';

export const ShadowTuning: React.FC = () => {
  const [experimentNames, setExperimentNames] = useState<string[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>('');
  const [qas, setQas] = useState<OpenBookQA[]>([]);
  const [trajectories, setTrajectories] = useState<OpenBookTrajectory[]>([]);
  const [activeTab, setActiveTab] = useState<'seeds' | 'train' | 'holdout'>('seeds');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const loadExperiments = () => {
    return fetch(`${API_BASE}/api/openbook/experiments`)
      .then(r => r.json())
      .then(data => {
        setExperimentNames(data);
        if (data.length > 0 && !selectedExp) {
          setSelectedExp(data[0]);
        }
        return data;
      })
      .catch(e => console.error("Failed to load OpenBook experiments:", e));
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadData = (expName: string) => {
    setLoading(true);
    return fetch(`${API_BASE}/api/openbook/${expName}/data`)
      .then(r => r.json())
      .then(data => {
        setQas(data.qas);
        setTrajectories(data.trajectories);
        setLoading(false);
      })
      .catch(e => {
        console.error("Failed to load OpenBook data:", e);
        setLoading(false);
      });
  };

  useEffect(() => {
    if (!selectedExp) return;
    loadData(selectedExp);
  }, [selectedExp]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadExperiments();
    if (selectedExp) {
      await loadData(selectedExp);
    }
    setRefreshing(false);
  };

  const trainSet = trajectories.filter(t => t.dataset === 'train');
  const holdoutSet = trajectories.filter(t => t.dataset === 'holdout');
  const untaggedSet = trajectories.filter(t => !t.dataset);

  return (
    <div style={{ display: 'flex', height: '100%', fontFamily: '"Inter", sans-serif', textAlign: 'left', backgroundColor: '#0f111a', color: '#fff' }}>
      {/* Sidebar */}
      <div style={{ width: '350px', borderRight: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', background: 'rgba(15, 17, 26, 0.5)' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#ccc' }}>OpenBook Experiment</h3>
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
            value={selectedExp} 
            onChange={e => setSelectedExp(e.target.value)}
            style={{ width: '100%', padding: '10px', background: '#1a1d27', color: '#fff', border: '1px solid #333', borderRadius: '6px', outline: 'none' }}
          >
            {experimentNames.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
          <h4 style={{ color: '#888', textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '1px', marginBottom: '15px' }}>Summary Info</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
             <StatBox label="SFT Seeds" value={qas.length} color="#9b59b6" />
             <StatBox label="Train Set Samples" value={trainSet.length} color="#61dafb" />
             <StatBox label="Holdout Set Samples" value={holdoutSet.length} color="#2ecc71" />
             {untaggedSet.length > 0 && <StatBox label="Untagged Samples" value={untaggedSet.length} color="#888" />}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '30px', background: 'radial-gradient(circle at top right, rgba(155, 89, 182, 0.05), transparent 40%)' }}>
        <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '5px' }}>Shadow Tuning Dashboard</h2>
          <p style={{ color: '#888', marginBottom: '30px' }}>Inspect the logical bridges and executable code generated during rejection sampling.</p>

          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.1)', marginBottom: '30px', gap: '20px' }}>
            <TabButton active={activeTab === 'seeds'} onClick={() => setActiveTab('seeds')} label="SFT Seeds" color="#9b59b6" count={qas.length} />
            <TabButton active={activeTab === 'train'} onClick={() => setActiveTab('train')} label="Train Set" color="#61dafb" count={trainSet.length} />
            <TabButton active={activeTab === 'holdout'} onClick={() => setActiveTab('holdout')} label="Holdout Set" color="#2ecc71" count={holdoutSet.length} />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', paddingBottom: '100px' }}>
            {loading ? (
              <div style={{ textAlign: 'center', padding: '50px', color: '#555' }}>Loading experiment data...</div>
            ) : activeTab === 'seeds' ? (
              qas.length === 0 ? <EmptyState /> : qas.map(qa => <QACard key={qa.id} qa={qa} />)
            ) : activeTab === 'train' ? (
              trainSet.length === 0 ? <EmptyState /> : trainSet.map(t => <TrajectoryCard key={t.id} t={t} />)
            ) : (
              holdoutSet.length === 0 ? <EmptyState /> : holdoutSet.map(t => <TrajectoryCard key={t.id} t={t} />)
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const StatBox = ({ label, value, color }: { label: string, value: number, color: string }) => (
  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '8px', borderLeft: `4px solid ${color}` }}>
    <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '4px' }}>{label}</div>
    <div style={{ fontSize: '1.2rem', fontWeight: 600, color: '#fff' }}>{value}</div>
  </div>
);

const TabButton = ({ active, onClick, label, color, count }: { active: boolean, onClick: () => void, label: string, color: string, count: number }) => (
  <button 
    onClick={onClick}
    style={{
      padding: '12px 20px',
      background: 'none',
      border: 'none',
      color: active ? color : '#888',
      borderBottom: active ? `2px solid ${color}` : 'none',
      cursor: 'pointer',
      fontSize: '0.95rem',
      fontWeight: active ? 600 : 400,
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    }}
  >
    {label}
    <span style={{ fontSize: '0.75rem', background: active ? color : 'rgba(255,255,255,0.1)', color: active ? '#fff' : '#888', padding: '2px 6px', borderRadius: '10px' }}>{count}</span>
  </button>
);

const QACard = ({ qa }: { qa: OpenBookQA }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ background: 'rgba(25, 28, 41, 0.7)', border: '1px solid rgba(155, 89, 182, 0.2)', borderRadius: '12px', overflow: 'hidden' }}>
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{ padding: '20px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
      >
        <div>
          <div style={{ fontSize: '0.75rem', color: '#9b59b6', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Seed Problem: {qa.knowledge_id}</div>
          <div style={{ fontSize: '1rem', fontWeight: 500 }}>{qa.title}</div>
        </div>
        <div style={{ color: '#888', fontSize: '1.2rem' }}>{expanded ? '−' : '+'}</div>
      </div>
      {expanded && (
        <div style={{ padding: '0 20px 20px 20px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
           <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '0.8rem', color: '#555', marginBottom: '8px', textTransform: 'uppercase' }}>Seed Question</div>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, color: '#ccc' }}>{qa.question}</div>
           </div>
           <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '0.8rem', color: '#555', marginBottom: '8px', textTransform: 'uppercase' }}>Reference Answer (Hint)</div>
              <pre style={{ background: '#0a0c14', padding: '15px', borderRadius: '8px', overflowX: 'auto', fontSize: '0.9rem', color: '#61dafb' }}>{qa.answer}</pre>
           </div>
        </div>
      )}
    </div>
  );
};

const TrajectoryCard = ({ t }: { t: OpenBookTrajectory }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ 
      background: 'rgba(25, 28, 41, 0.7)', 
      border: `1px solid ${t.success ? 'rgba(46, 204, 113, 0.2)' : 'rgba(255, 77, 77, 0.2)'}`, 
      borderRadius: '12px', 
      overflow: 'hidden' 
    }}>
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{ 
          padding: '20px', 
          cursor: 'pointer', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          background: t.success ? 'linear-gradient(90deg, rgba(46, 204, 113, 0.05), transparent)' : 'linear-gradient(90deg, rgba(255, 77, 77, 0.05), transparent)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
           <div style={{ 
              width: '8px', height: '8px', borderRadius: '50%', 
              background: t.success ? '#2ecc71' : '#ff4d4d',
              boxShadow: t.success ? '0 0 10px #2ecc71' : '0 0 10px #ff4d4d'
           }} />
           <div>
              <div style={{ fontSize: '0.75rem', color: '#888', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Trajectory ID: {t.id}</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 500 }}>{t.question.split('\n')[0].substring(0, 80)}...</div>
           </div>
        </div>
        <div style={{ color: '#888', fontSize: '1.2rem' }}>{expanded ? '−' : '+'}</div>
      </div>
      {expanded && (
        <div style={{ padding: '0 20px 20px 20px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
           <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '0.8rem', color: '#9b59b6', marginBottom: '8px', textTransform: 'uppercase' }}>Logical Bridge (Reasoning)</div>
              <div style={{ color: '#999', fontStyle: 'italic', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{t.reasoning}</div>
           </div>
           <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '0.8rem', color: '#61dafb', marginBottom: '8px', textTransform: 'uppercase' }}>Executable Solution (Rust)</div>
              <div className="markdown-body" style={{ background: '#0a0c14', padding: '15px', borderRadius: '8px' }}>
                <ReactMarkdown>{t.answer}</ReactMarkdown>
              </div>
           </div>
           {(t.execution_output || t.verification_output) && (
             <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                {t.execution_output && (
                  <div style={{ marginBottom: t.verification_output ? '15px' : 0 }}>
                    <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px', textTransform: 'uppercase' }}>Execution Logs</div>
                    <pre style={{ margin: 0, fontSize: '0.85rem', color: '#ff4d4d', whiteSpace: 'pre-wrap' }}>{t.execution_output}</pre>
                  </div>
                )}
                {t.verification_output && (
                  <div>
                    <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px', textTransform: 'uppercase' }}>Verification Details</div>
                    <pre style={{ margin: 0, fontSize: '0.85rem', color: t.success ? '#2ecc71' : '#ff4d4d', whiteSpace: 'pre-wrap' }}>{t.verification_output}</pre>
                  </div>
                )}
             </div>
           )}
        </div>
      )}
    </div>
  );
};

const EmptyState = () => (
  <div style={{ textAlign: 'center', padding: '100px', color: '#444' }}>
    <p>No data available for this category.</p>
  </div>
);
