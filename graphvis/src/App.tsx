import { useState } from 'react';
import { GraphCanvasComponent } from './components/GraphCanvas';
import { WikiVis } from './components/WikiVis';
import { RLRolloutExplorer } from './components/RLRolloutExplorer';
import { SftCacheExplorer } from './components/SftCacheExplorer';
import './App.css';

function App() {
  const [view, setView] = useState<'graph' | 'wiki' | 'rl_rollout' | 'sft_cache'>('rl_rollout');

  return (
    <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', backgroundColor: '#0f111a', color: '#fff', boxSizing: 'border-box' }}>
       <div style={{ padding: '15px 30px', background: 'rgba(15, 17, 26, 0.8)', backdropFilter: 'blur(10px)', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: '15px', alignItems: 'center', zIndex: 100 }}>
         <h2 style={{ margin: 0, marginRight: '20px', fontSize: '1.2rem', fontWeight: 600, background: 'linear-gradient(90deg, #61dafb, #9b59b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Adapter Agent Vis</h2>
         <button onClick={() => setView('graph')} style={{ padding: '8px 16px', background: view === 'graph' ? 'rgba(97, 218, 251, 0.15)' : 'transparent', color: view === 'graph' ? '#61dafb' : '#888', border: `1px solid ${view === 'graph' ? '#61dafb' : '#444'}`, borderRadius: '6px', cursor: 'pointer', transition: 'all 0.2s' }}>Hierarchical Graph</button>
         <button onClick={() => setView('rl_rollout')} style={{ padding: '8px 16px', background: view === 'rl_rollout' ? 'rgba(241, 196, 15, 0.15)' : 'transparent', color: view === 'rl_rollout' ? '#f1c40f' : '#888', border: `1px solid ${view === 'rl_rollout' ? '#f1c40f' : '#444'}`, borderRadius: '6px', cursor: 'pointer', transition: 'all 0.2s' }}>RL Rollouts</button>
         <button onClick={() => setView('sft_cache')} style={{ padding: '8px 16px', background: view === 'sft_cache' ? 'rgba(46, 204, 113, 0.15)' : 'transparent', color: view === 'sft_cache' ? '#2ecc71' : '#888', border: `1px solid ${view === 'sft_cache' ? '#2ecc71' : '#444'}`, borderRadius: '6px', cursor: 'pointer', transition: 'all 0.2s' }}>SFT Caches</button>
         <button onClick={() => setView('wiki')} style={{ padding: '8px 16px', background: view === 'wiki' ? 'rgba(46, 204, 113, 0.15)' : 'transparent', color: view === 'wiki' ? '#2ecc71' : '#888', border: `1px solid ${view === 'wiki' ? '#2ecc71' : '#444'}`, borderRadius: '6px', cursor: 'pointer', transition: 'all 0.2s' }}>Wiki Explorer</button>
       </div>
       <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
         {view === 'graph' && <GraphCanvasComponent />}
         {view === 'rl_rollout' && <RLRolloutExplorer />}
         {view === 'sft_cache' && <SftCacheExplorer />}
         {view === 'wiki' && <WikiVis />}
       </div>
    </div>
  );
}

export default App;
