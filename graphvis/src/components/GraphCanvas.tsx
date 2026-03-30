import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods, NodeObject, LinkObject } from 'react-force-graph-2d';
import ReactMarkdown from 'react-markdown';

interface GraphNodeMetadata {
  instruction: string;
  success_count: number;
  total_count: number;
  is_solved: boolean;
  is_executing: boolean;
  slice_count: number;
  gen_count: number;
  citations: { knowledge_id: string, turn_index: number, content?: string | null, title?: string | null }[];
  slices: { question: string, answer: string, reasoning: string }[];
  knowledge_content?: string | null;
  knowledge_title?: string | null;
}

interface CustomNode extends NodeObject {
  id: string;
  metadata: GraphNodeMetadata;
  color: string;
  type: 'task' | 'knowledge';
  label: string;
}

interface CustomLink extends LinkObject {
  source: string | CustomNode;
  target: string | CustomNode;
  type: 'decomposition' | 'citation' | 'generation';
}

interface GraphExportNode {
  id: string;
  label: string;
  type?: string;
  metadata: GraphNodeMetadata;
}

interface GraphExportEdge {
  id: string;
  source: string;
  target: string;
}

interface ContentPart {
  type: string;
  text?: string;
  thinking?: string;
}

interface TrajectoryTurn {
  role: 'assistant' | 'user' | 'tool' | 'system';
  content: string | ContentPart[];
  tool_calls?: any[];
  unparsed_tool_calls?: any[];
  metadata?: {
    thought?: string;
    tool_calls?: any[];
    is_error?: boolean;
    knowledge_id?: string;
  };
}

interface TrajectoryData {
  id: number;
  taskId: string;
  instruction: string;
  conclusion: string;
  reward: number;
  trials: TrajectoryTurn[];
  citations: { knowledge_id: string, content: string, turn_index: number }[];
  final_knowledge?: string | null;
  final_knowledge_title?: string | null;
  created_at: string;
}

interface GraphExportData {
  nodes: GraphExportNode[];
  edges: GraphExportEdge[];
}

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://localhost:8000' 
  : `http://${window.location.hostname}:8000`;

export const GraphCanvasComponent: React.FC = () => {
  const [data, setData] = useState<{ nodes: CustomNode[], links: CustomLink[] } | null>(null);
  const [displayData, setDisplayData] = useState<{ nodes: CustomNode[], links: CustomLink[] } | null>(null);
  const [viewMode, setViewMode] = useState<'global' | 'local'>('global');
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const [focusDepth, setFocusDepth] = useState<number>(2);

  const [showKnowledge, setShowKnowledge] = useState<boolean>(true);
  const [selectedNode, setSelectedNode] = useState<CustomNode | null>(null);
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState<number>(0);
  const [loadingTraj, setLoadingTraj] = useState<boolean>(false);
  const [trajError, setTrajError] = useState<string | null>(null);
  
  const [experiments, setExperiments] = useState<string[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);
  
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);
  
  const handleNodeSelect = useCallback((node: CustomNode) => {
    setSelectedNode(node);
    setTrajectories([]);
    setSelectedAttemptIndex(0);
    if (fgRef.current && node.x !== undefined && node.y !== undefined) {
      fgRef.current.centerAt(node.x, node.y, 600);
      fgRef.current.zoom(2.5, 600);
    }
  }, []);

  const loadData = useCallback(async () => {
    if (!selectedExperiment) {
      setIsInitializing(false);
      return;
    }
    setError(null);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

      const response = await fetch(`${API_BASE}/api/${encodeURIComponent(selectedExperiment)}/graph`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${await response.text()}`);
      }
      
      const json: GraphExportData = await response.json();

      const taskNodes: CustomNode[] = json.nodes.map(n => ({
        id: n.id,
        label: n.id,
        name: n.metadata.instruction,
        val: 5,
        metadata: {
          ...n.metadata,
          citations: n.metadata.citations || []
        },
        color: n.id === 'pseudo_root' ? '#e67e22' : (n.type === 'knowledge' ? '#9b59b6' : (n.metadata.is_solved ? '#28a745' : n.metadata.is_executing ? '#f1c40f' : '#555')),
        type: (n.type as any) || 'task'
      }));

      const knowledge_ids = new Set<string>();
      taskNodes.forEach(tn => {
        tn.metadata.citations.forEach(c => knowledge_ids.add(c.knowledge_id));
      });

      const taskNodeMap = new Map<string, CustomNode>();
      taskNodes.forEach(n => taskNodeMap.set(n.id, n));

      const knowledgeNodes: CustomNode[] = Array.from(knowledge_ids).map(kid => {
        const sourceTask = taskNodeMap.get(kid);
        
        // Find any citation that has the content for this knowledge ID
        let capturedContent: string | null = null;
        let capturedTitle: string | null = null;
        json.nodes.forEach(n => {
          n.metadata.citations.forEach(c => {
            if (c.knowledge_id === kid) {
              if (c.content) capturedContent = c.content;
              if (c.title) capturedTitle = c.title;
            }
          });
        });

        const knowledge_title = sourceTask?.metadata.knowledge_title || capturedTitle || `Knowledge ${kid}`;

        return {
          id: kid,
          label: kid,
          name: knowledge_title,
          val: 10,
          metadata: {
            instruction: `Knowledge ID: ${kid}`,
            success_count: 0,
            total_count: 0,
            is_solved: true,
            is_executing: false,
            slice_count: 0,
            gen_count: 0,
            citations: [],
            slices: [],
            knowledge_content: sourceTask?.metadata.knowledge_content || capturedContent,
            knowledge_title: knowledge_title
          },
          color: '#9b59b6', // Purple
          type: 'knowledge'
        };
      });

      const decompositionLinks: CustomLink[] = json.edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: 'decomposition'
      }));

      const citationLinks: CustomLink[] = [];
      taskNodes.forEach(tn => {
        tn.metadata.citations.forEach(c => {
          const isGeneration = tn.id === c.knowledge_id;
          citationLinks.push({
            id: `cit-${tn.id}-${c.knowledge_id}-${c.turn_index}`,
            source: tn.id,
            target: c.knowledge_id,
            type: isGeneration ? 'generation' : 'citation'
          });
        });
      });

      setData({ 
        nodes: [...taskNodes, ...knowledgeNodes], 
        links: [...decompositionLinks, ...citationLinks] 
      });
      setIsInitializing(false);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. The backend might be overloaded or unresponsive.');
      } else {
        console.error('CRITICAL: Failed to load graph data:', err);
        setError(`Failed to load graph data: ${err.message || String(err)}. Check if backend is running at ${API_BASE}`);
      }
      setData({ nodes: [], links: [] });
      setIsInitializing(false);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    if (fgRef.current) {
      // Increase link distance to spread nodes more
      const linkForce = fgRef.current.d3Force('link');
      if (linkForce) (linkForce as any).distance(60);
      
      // Increase repulsion (charge)
      const chargeForce = fgRef.current.d3Force('charge');
      if (chargeForce) (chargeForce as any).strength(-120);
    }
  }, [fgRef.current, data]);

  useEffect(() => {
    const fetchExps = () => {
      fetch(`${API_BASE}/api/experiments`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          setExperiments(data);
          if (data.length > 0 && !selectedExperiment) {
            setSelectedExperiment(data[0]);
          } else if (data.length === 0) {
            setIsInitializing(false);
          }
        })
        .catch(err => {
          console.error("Failed to fetch experiments", err);
          setError(`Could not connect to backend at ${API_BASE}. Make sure 'just vis' is running and port 8000 is accessible.`);
          setIsInitializing(false);
        });
    };

    fetchExps();
    const interval = setInterval(fetchExps, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData, selectedExperiment]);

  useEffect(() => {
    if (!data) return;

    let neighborhood: Set<string>;
    if (viewMode === 'global' || !focusNodeId) {
      // In global mode, start with all node IDs
      neighborhood = new Set(data.nodes.map(n => n.id));
    } else {
      // In local mode, BFS to find neighborhood
      neighborhood = new Set<string>();
      neighborhood.add(focusNodeId);

      let currentLevel = [focusNodeId];
      for (let i = 0; i < focusDepth; i++) {
        const nextLevel: string[] = [];
        data.links.forEach(link => {
          const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id;
          const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id;

          if (currentLevel.includes(sourceId) && !neighborhood.has(targetId)) {
            neighborhood.add(targetId);
            nextLevel.push(targetId);
          } else if (currentLevel.includes(targetId) && !neighborhood.has(sourceId)) {
            neighborhood.add(sourceId);
            nextLevel.push(sourceId);
          }
        });
        currentLevel = nextLevel;
      }
    }

    let filteredNodes = data.nodes.filter(n => neighborhood.has(n.id));
    
    // Hide knowledge nodes if requested
    if (!showKnowledge) {
      filteredNodes = filteredNodes.filter(n => n.type !== 'knowledge');
    }

    const filteredLinks = data.links.filter(l => {
      const sId = typeof l.source === 'string' ? l.source : (l.source as any).id;
      const tId = typeof l.target === 'string' ? l.target : (l.target as any).id;
      return neighborhood.has(sId) && neighborhood.has(tId) && filteredNodes.find(n => n.id === sId) && filteredNodes.find(n => n.id === tId);
    });

    setDisplayData({ nodes: filteredNodes, links: filteredLinks });
  }, [data, viewMode, focusNodeId, focusDepth, showKnowledge]);

  useEffect(() => {
    if (selectedNode && selectedNode.type === 'task' && selectedExperiment) {
      setLoadingTraj(true);
      setTrajError(null);
      fetch(`${API_BASE}/api/${encodeURIComponent(selectedExperiment)}/trajectory/${encodeURIComponent(selectedNode.id)}`)
        .then(async res => {
          if (!res.ok) {
            throw new Error(`Failed to fetch trajectories: ${res.status} ${await res.text()}`);
          }
          return res.json();
        })
        .then(data => {
          // data is now an array of trajectories
          setTrajectories(Array.isArray(data) ? data : []);
          setSelectedAttemptIndex((data && data.length > 0) ? data.length - 1 : 0); // Default to latest
          setLoadingTraj(false);
        })
        .catch(err => {
          console.error("Failed to fetch trajectory:", err);
          setTrajError(err.message || String(err));
          setLoadingTraj(false);
          setTrajectories([]);
        });
    } else {
      setTrajectories([]);
    }
  }, [selectedNode, selectedExperiment]);

  if (experiments.length === 0) {
    return <div style={{ color: 'white', padding: '20px', fontFamily: 'monospace' }}>
      <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '10px' }}>Waiting for experiments to start...</div>
      <div style={{ color: '#888' }}>Searching for directories in logs/Adapter_Agent/</div>
    </div>;
  }

  if (error) {
    return <div style={{ color: '#ff4d4d', padding: '20px', fontFamily: 'monospace', border: '1px solid #ff4d4d', margin: '20px', borderRadius: '4px', backgroundColor: 'rgba(255, 77, 77, 0.1)' }}>
      <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '10px' }}>⚠️ Error Loading Graph</div>
      <div style={{ wordBreak: 'break-all' }}>{error}</div>
      <button 
        onClick={() => loadData()} 
        style={{ marginTop: '15px', padding: '8px 16px', cursor: 'pointer', background: '#444', color: 'white', border: 'none', borderRadius: '4px' }}
      >
        Retry Fetching
      </button>
    </div>;
  }

  if (!data && isInitializing) {
    return <div style={{ color: 'white', padding: '20px', fontFamily: 'monospace' }}>
      <div className="spinner" style={{ marginBottom: '10px' }}>Loading graph data...</div>
      <div style={{ color: '#888', fontSize: '12px' }}>Requesting {selectedExperiment}...</div>
    </div>;
  }

  const renderTrajectory = () => {
    if (loadingTraj) return (
      <div style={{ padding: '30px', textAlign: 'center' }}>
        <div className="spinner-small" style={{ margin: '0 auto 10px auto' }}></div>
        <div style={{ color: '#888', fontStyle: 'italic', fontSize: '13px' }}>Loading trajectories...</div>
      </div>
    );
    
    if (trajError) return (
      <div style={{ 
        margin: '20px 0', 
        padding: '16px', 
        background: 'rgba(231, 76, 60, 0.1)', 
        border: '1px solid rgba(231, 76, 60, 0.3)', 
        borderRadius: '8px',
        color: '#e74c3c',
        fontSize: '13px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>⚠️ Failed to load trajectories</div>
        <div style={{ opacity: 0.8, marginBottom: '12px' }}>{trajError}</div>
        <button 
          onClick={() => {
            // Trigger a re-fetch by toggling a state if needed, 
            // but here we can just re-run the logic by forcing a re-render or re-calling the effect logic
            // simplest for this UI is letting the user click Refresh or de-select/re-select
          }}
          style={{ 
            background: '#e74c3c', 
            color: 'white', 
            border: 'none', 
            padding: '4px 10px', 
            borderRadius: '4px', 
            cursor: 'not-allowed', // Implementation detail: retry not easily wired here without refactoring
            fontSize: '11px',
            opacity: 0.5
          }}
        >
          Check Backend Logs
        </button>
      </div>
    );

    if (!trajectories || trajectories.length === 0) return (
      <div style={{ 
        padding: '40px 20px', 
        textAlign: 'center', 
        color: '#666', 
        border: '1px dashed #333', 
        borderRadius: '12px',
        marginTop: '20px'
      }}>
        <div style={{ fontSize: '24px', marginBottom: '10px' }}>📁</div>
        <div style={{ fontSize: '14px' }}>No trajectories found for this task</div>
        <div style={{ fontSize: '11px', marginTop: '5px', opacity: 0.7 }}>
          Trajectories are recorded when an agent finishes a trial and the result is verified.
        </div>
      </div>
    );

    const trajectory = trajectories[selectedAttemptIndex];
    if (!trajectory || !trajectory.trials) return null;

    return (
      <div style={{ marginTop: '30px', borderTop: '1px solid #333', paddingTop: '20px' }}>
        {/* Attempt Selector */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '8px', fontWeight: 'bold' }}>
            Attempts ({trajectories.length})
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {trajectories.map((t, idx) => (
              <button
                key={t.id}
                onClick={() => setSelectedAttemptIndex(idx)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  background: selectedAttemptIndex === idx ? 'rgba(97, 218, 251, 0.2)' : 'rgba(255,255,255,0.05)',
                  color: selectedAttemptIndex === idx ? '#61dafb' : '#ccc',
                  border: `1px solid ${selectedAttemptIndex === idx ? '#61dafb' : '#444'}`,
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                #{idx + 1} {t.reward > 0 ? '✅' : '❌'} ({new Date(t.created_at).toLocaleTimeString()})
              </button>
            ))}
          </div>
        </div>

        <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '15px', fontWeight: 'bold' }}>
          Reasoning Trace ({trajectory.trials.length} turns)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {trajectory.trials.map((turn, i) => {
            const contentStr = typeof turn.content === 'string' 
              ? turn.content 
              : Array.isArray(turn.content) 
                ? turn.content.map((p: any) => 
                    p.type === 'thinking' ? `<think>${p.thinking}</think>` : (p.text || '')
                  ).join('')
                : String(turn.content || '');

            const isThought = turn.role === 'assistant' && contentStr.includes('<think>');
            const thoughtMatch = contentStr.match(/<think>([\s\S]*?)<\/think>/);
            const restContent = contentStr.replace(/<think>[\s\S]*?<\/think>/, '').trim();

            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {isThought && thoughtMatch && (
                  <div style={{ 
                    background: 'rgba(97, 218, 251, 0.05)', 
                    borderLeft: '2px solid #61dafb', 
                    padding: '12px', 
                    fontSize: '13px', 
                    color: '#a0e4f1', 
                    fontStyle: 'italic',
                    borderRadius: '0 8px 8px 0',
                    textAlign: 'left'
                  }}>
                    <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase' }}>Thought</div>
                    <ReactMarkdown>{thoughtMatch[1].trim()}</ReactMarkdown>
                  </div>
                )}
                {(turn.role === 'assistant' && (restContent || (turn.tool_calls && turn.tool_calls.length > 0) || (turn.unparsed_tool_calls && turn.unparsed_tool_calls.length > 0))) && (
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', border: '1px solid #333', textAlign: 'left' }}>
                    <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase' }}>Assistant</div>
                    {restContent && <ReactMarkdown>{restContent}</ReactMarkdown>}
                    
                    {turn.tool_calls && turn.tool_calls.length > 0 && (
                      <div style={{ marginTop: restContent ? '12px' : '0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {turn.tool_calls.map((tc, idx) => (
                          <div key={idx} style={{ background: 'rgba(97, 218, 251, 0.1)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(97, 218, 251, 0.2)' }}>
                            <div style={{ fontSize: '10px', color: '#61dafb', fontWeight: 'bold', marginBottom: '4px' }}>🛠 CALL: {tc.function?.name}</div>
                            <pre style={{ fontSize: '11px', margin: 0, whiteSpace: 'pre-wrap', color: '#ddd' }}>{typeof tc.function?.arguments === 'string' ? tc.function.arguments : JSON.stringify(tc.function?.arguments, null, 2)}</pre>
                          </div>
                        ))}
                      </div>
                    )}

                    {turn.unparsed_tool_calls && turn.unparsed_tool_calls.length > 0 && (
                      <div style={{ marginTop: (restContent || (turn.tool_calls && turn.tool_calls.length > 0)) ? '12px' : '0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {turn.unparsed_tool_calls.map((utc, idx) => (
                          <div key={idx} style={{ background: 'rgba(231, 76, 60, 0.1)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(231, 76, 60, 0.2)' }}>
                            <div style={{ fontSize: '10px', color: '#e74c3c', fontWeight: 'bold', marginBottom: '4px' }}>⚠️ UNPARSED CALL</div>
                            <pre style={{ fontSize: '11px', margin: 0, whiteSpace: 'pre-wrap', color: '#ddd' }}>{utc.arguments || JSON.stringify(utc)}</pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {turn.role === 'tool' && (() => {
                  const cit = trajectory.citations.find(c => c.turn_index === i);
                  return (
                    <div style={{ background: 'rgba(40, 167, 69, 0.05)', borderLeft: '2px solid #28a745', padding: '12px', borderRadius: '0 8px 8px 0', textAlign: 'left' }}>
                      <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '4px', textTransform: 'uppercase', color: '#28a745' }}>
                        {cit ? `Knowledge Acquired (#${cit.knowledge_id})` : 'Tool Result'}
                      </div>
                      {cit ? (
                        <div style={{ fontSize: '13px', color: '#eee' }}>
                          <ReactMarkdown>{cit.content}</ReactMarkdown>
                        </div>
                      ) : (
                        <pre style={{ fontSize: '12px', whiteSpace: 'pre-wrap', color: '#ccc', margin: 0, overflowX: 'auto' }}>{contentStr}</pre>
                      )}
                    </div>
                  );
                })()}
                {(turn.role === 'user' || turn.role === 'system') && (
                  <div style={{ background: 'rgba(255,255,255,0.05)', padding: '12px', borderRadius: '8px', border: '1px solid #444', textAlign: 'left' }}>
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

  const renderDetailPanel = () => {
    if (!selectedNode) return null;
    const m = selectedNode.metadata;
    const isKnowledge = selectedNode.type === 'knowledge';

    return (
      <div
        style={{
          position: 'fixed',
          top: '80px',
          right: '25px',
          width: '500px',
          maxHeight: 'calc(100vh - 120px)',
          background: 'rgba(20, 20, 20, 0.95)',
          color: 'white',
          padding: '24px',
          borderRadius: '16px',
          border: '1px solid #444',
          boxShadow: '0 20px 50px rgba(0,0,0,0.7)',
          zIndex: 3000,
          overflowY: 'auto',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          backdropFilter: 'blur(15px)',
          transition: 'all 0.3s ease',
        }}
        className="detail-scroll"
      >
        <style>
          {`
            .detail-scroll::-webkit-scrollbar { width: 6px; }
            .detail-scroll::-webkit-scrollbar-track { background: transparent; }
            .detail-scroll::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
            .detail-scroll::-webkit-scrollbar-thumb:hover { background: #777; }
            .slice-box:hover { background: rgba(255,255,255,0.05); }
            .markdown-content { text-align: left; }
            .markdown-content p { margin: 8px 0; }
            .markdown-content code { background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 4px; font-family: monospace; }
            .markdown-content pre { background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; overflow-x: auto; margin: 12px 0; border: 1px solid #444; }
            .markdown-content pre code { background: transparent; padding: 0; }
            .markdown-content ul, .markdown-content ol { padding-left: 20px; margin: 8px 0; }
          `}
        </style>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '12px', gap: '10px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 'bold', color: '#61dafb', fontSize: '20px', letterSpacing: '0.5px' }}>
              {isKnowledge ? 'Verified Knowledge' : 'Task Analysis'}
            </div>
          </div>
          <button
            onClick={() => {
              setFocusNodeId(selectedNode.id);
              setViewMode('local');
            }}
            style={{
              padding: '6px 12px',
              background: 'rgba(97, 218, 251, 0.2)',
              color: '#61dafb',
              border: '1px solid #61dafb',
              borderRadius: '6px',
              fontSize: '12px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            Focus
          </button>
          <button
            onClick={() => setSelectedNode(null)}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '18px',
            }}
          >
            ×
          </button>
        </div>

        <div className="markdown-content" style={{ fontSize: '15px', lineHeight: '1.6', marginBottom: '24px', color: '#fff', background: 'rgba(97, 218, 251, 0.1)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(97, 218, 251, 0.2)' }}>
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

        {renderTrajectory()}

        {(() => {
          const trajectory = trajectories[selectedAttemptIndex];
          if (!trajectory || !trajectory.citations || trajectory.citations.length === 0) return null;

          // Deduplicate by knowledge_id
          const uniqueCitations = Array.from(
            new Map(trajectory.citations.map(c => [c.knowledge_id, c])).values()
          );

          return (
            <div style={{ marginBottom: '24px' }}>
              <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
                Used Knowledge in this attempt ({uniqueCitations.length})
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {uniqueCitations.map((c, i) => (
                  <div key={i} style={{
                    background: 'rgba(155, 89, 182, 0.2)',
                    border: '1px solid rgba(155, 89, 182, 0.4)',
                    padding: '6px 12px',
                    borderRadius: '16px',
                    fontSize: '12px',
                  }}>
                    <span style={{ color: '#9b59b6', fontWeight: 'bold' }}>#{c.turn_index}</span>: {c.knowledge_id}
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {isKnowledge && data && (
          <div style={{ marginBottom: '24px' }}>
            <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
              Impacted Tasks
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {data.links
                .filter(l => (typeof l.target === 'string' ? l.target : (l.target as any).id) === selectedNode.id)
                .map((l, i) => {
                  const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id;
                  const sourceNode = data.nodes.find(n => n.id === sourceId);
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
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        lineHeight: '1.4',
                        cursor: sourceNode ? 'pointer' : 'default',
                        transition: 'background 0.2s'
                      }}
                      onMouseOver={(e) => sourceNode && (e.currentTarget.style.background = 'rgba(255,255,255,0.1)')}
                      onMouseOut={(e) => sourceNode && (e.currentTarget.style.background = 'rgba(255,255,255,0.05)')}
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
            <div className="detail-scroll" style={{ maxHeight: '400px', overflowY: 'auto', paddingRight: '12px' }}>
              {m.slices.map((s, i) => (
                <div key={i} className="slice-box" style={{ borderBottom: '1px solid #333', padding: '16px 0' }}>
                  <div className="markdown-content" style={{ fontSize: '14px', color: '#fff', marginBottom: '8px', lineHeight: '1.4', fontWeight: '500' }}>
                    <span style={{ color: '#61dafb', fontWeight: 'bold', marginRight: '8px' }}>Question:</span> {s.question}
                  </div>
                  <div className="markdown-content" style={{ fontSize: '13px', color: '#bbb', marginBottom: '8px', lineHeight: '1.4' }}>
                    <span style={{ color: '#f1c40f', fontWeight: '600', marginRight: '8px' }}>Think:</span>
                    <ReactMarkdown>{s.reasoning}</ReactMarkdown>
                  </div>
                  <div className="markdown-content" style={{ fontSize: '13px', color: '#eee', lineHeight: '1.4', background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '6px' }}>
                    <span style={{ color: '#28a745', fontWeight: '600', marginRight: '8px' }}>Answer:</span>
                    <ReactMarkdown>{s.answer}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: '#000' }}>
      <div style={{
          position: 'fixed',
          top: '20px',
          left: '20px',
          zIndex: 2000,
          display: 'flex',
          gap: '12px',
          padding: '8px',
          background: 'rgba(0,0,0,0.5)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          border: '1px solid #333',
          alignItems: 'center'
      }}>
        {experiments.length > 0 && (
          <select 
            value={selectedExperiment || ''} 
            onChange={(e) => setSelectedExperiment(e.target.value)}
            style={{
              padding: '8px 12px',
              background: 'rgba(255,255,255,0.1)',
              color: 'white',
              border: '1px solid #444',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold',
              outline: 'none',
              fontSize: '13px'
            }}
          >
            {experiments.map(exp => (
              <option key={exp} value={exp} style={{ background: '#222' }}>{exp}</option>
            ))}
          </select>
        )}

        <button 
          onClick={loadData}
          style={{
            padding: '8px 16px',
            background: 'rgba(255,255,255,0.05)',
            color: 'white',
            border: '1px solid #444',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            transition: 'all 0.2s',
          }}
          onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
          onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
        >
          Refresh
        </button>

        <button 
          onClick={() => setShowKnowledge(!showKnowledge)}
          style={{
            padding: '8px 16px',
            background: showKnowledge ? 'rgba(155, 89, 182, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showKnowledge ? '#9b59b6' : '#888',
            border: `1px solid ${showKnowledge ? '#9b59b6' : '#444'}`,
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: 'bold',
            transition: 'all 0.2s',
          }}
        >
          {showKnowledge ? 'Hide' : 'Show'} Knowledge
        </button>

        {viewMode === 'local' && (
          <>
            <button 
              onClick={() => {
                setViewMode('global');
                setFocusNodeId(null);
              }}
              style={{
                padding: '8px 16px',
                background: 'rgba(231, 76, 60, 0.2)',
                color: '#e74c3c',
                border: '1px solid #e74c3c',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
              }}
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
                onChange={(e) => setFocusDepth(parseInt(e.target.value))} 
                style={{ width: '80px', cursor: 'pointer' }}
              />
            </div>
          </>
        )}
      </div>

      {renderDetailPanel()}

      <ForceGraph2D
        ref={fgRef}
        graphData={displayData || { nodes: [], links: [] }}
        onNodeClick={(node) => handleNodeSelect(node as CustomNode)}
        nodeLabel="name"
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as CustomNode;
          const m = n.metadata;
          const label = n.type === 'knowledge' ? (m.knowledge_title || n.id) : m.instruction;
          const truncated = label.length > 30 ? label.substring(0, 30) + '...' : label;
          const fontSize = 12 / globalScale;
          const isPseudoRoot = n.id === 'pseudo_root';
          const radius = isPseudoRoot ? 16 : (n.type === 'knowledge' ? 12 : 8);
          
          ctx.font = `${fontSize}px system-ui`;
          
          // Selection highlight
          if (selectedNode && selectedNode.id === n.id) {
            ctx.beginPath();
            ctx.arc(n.x!, n.y!, radius + 4, 0, 2 * Math.PI, false);
            ctx.strokeStyle = n.type === 'knowledge' ? '#9b59b6' : '#61dafb';
            ctx.lineWidth = 3 / globalScale;
            ctx.stroke();
          }

          // Draw node circle
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = n.color;
          ctx.fill();

          if (n.type === 'knowledge') {
            ctx.font = `${fontSize * 1.5}px system-ui`;
            ctx.fillText('📚', n.x! - radius/2, n.y! + radius/2);
            ctx.font = `${fontSize}px system-ui`;
          }

          if (isPseudoRoot) {
            ctx.font = `${fontSize * 1.8}px system-ui`;
            ctx.fillText('🏠', n.x! - radius/2, n.y! + radius/2);
            ctx.font = `${fontSize}px system-ui`;
          }

          if (m.is_executing) {
            ctx.font = `${fontSize * 1.5}px system-ui`;
            ctx.fillText('⚡', n.x! - radius - 2, n.y! - radius - 2);
            ctx.font = `${fontSize}px system-ui`;
          }

          const labelYOffset = radius + 8;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#fff';
          ctx.fillText(truncated, n.x!, n.y! + labelYOffset);
          
          const stats = `${m.success_count}/${m.total_count}` + (m.gen_count > 0 ? ` 📂${m.gen_count}` : '');
          ctx.font = `${fontSize * 0.8}px system-ui`;
          ctx.fillStyle = '#aaa';
          ctx.fillText(stats, n.x!, n.y! + labelYOffset + fontSize * 1.2);
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as CustomNode;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, 14, 0, 2 * Math.PI, false);
          ctx.fill();
        }}
        linkColor={(link) => {
          const l = link as CustomLink;
          if (l.type === 'generation') return 'rgba(155, 89, 182, 0.9)';
          if (l.type === 'citation') return 'rgba(155, 89, 182, 0.3)';
          return '#333';
        }}
        linkWidth={(link) => {
          const l = link as CustomLink;
          if (l.type === 'generation') return 2.5;
          if (l.type === 'citation') return 1.5;
          return 1.5;
        }}
        linkLineDash={(link) => (link as CustomLink).type === 'citation' ? [3, 3] : null}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        linkCurvature={(link) => ['citation', 'generation'].includes((link as CustomLink).type) ? 0.2 : 0}
        backgroundColor="#000"
      />
    </div>
  );
};
