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
  slices: { question: string, answer: string, reasoning: string }[];
}

interface CustomNode extends NodeObject {
  metadata: GraphNodeMetadata;
  color: string;
}

interface CustomLink extends LinkObject {
  source: string | CustomNode;
  target: string | CustomNode;
}

interface GraphExportNode {
  id: string;
  label: string;
  metadata: GraphNodeMetadata;
}

interface GraphExportEdge {
  id: string;
  source: string;
  target: string;
}

interface GraphExportData {
  nodes: GraphExportNode[];
  edges: GraphExportEdge[];
}

export const GraphCanvasComponent: React.FC = () => {
  const [data, setData] = useState<{ nodes: CustomNode[], links: CustomLink[] } | null>(null);
  const [selectedNode, setSelectedNode] = useState<CustomNode | null>(null);
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);

  const loadData = useCallback(async () => {
    try {
      const response = await fetch('/data.json');
      const json: GraphExportData = await response.json();

      const nodes: CustomNode[] = json.nodes.map(n => ({
        id: n.id,
        name: n.metadata.instruction,
        val: 5,
        metadata: n.metadata,
        color: n.metadata.is_solved ? '#28a745' : n.metadata.is_executing ? '#f1c40f' : '#555'
      }));

      const links: CustomLink[] = json.edges.map(e => ({
        source: e.source,
        target: e.target
      }));

      setData({ nodes, links });
    } catch (error) {
      console.error('Failed to load graph data:', error);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Configure forces when engine starts
  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-300);
      fgRef.current.d3Force('link')?.distance(100);
      fgRef.current.d3Force('center')?.strength(0.1);
    }
  }, [data]);

  if (!data) {
    return <div style={{ color: 'white', padding: '20px' }}>Loading graph data...</div>;
  }

  const renderDetailPanel = () => {
    if (!selectedNode) return null;
    const m = selectedNode.metadata;
    
    return (
      <div style={{
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
      }} className="detail-scroll">
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
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '12px' }}>
          <div>
            <div style={{ fontWeight: 'bold', color: '#61dafb', fontSize: '20px', letterSpacing: '0.5px' }}>Task Analysis</div>
          </div>
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
              fontSize: '18px'
            }}
          >
            ×
          </button>
        </div>

        <div className="markdown-content" style={{ fontSize: '15px', lineHeight: '1.6', marginBottom: '24px', color: '#fff', background: 'rgba(97, 218, 251, 0.1)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(97, 218, 251, 0.2)' }}>
          <ReactMarkdown>{m.instruction}</ReactMarkdown>
        </div>

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

        <div style={{ marginBottom: '24px', background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '10px' }}>
          <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '10px', fontWeight: 'bold' }}>Simulation Stats</div>
          <div style={{ display: 'flex', gap: '20px', fontSize: '14px' }}>
            <div><span style={{ color: '#888' }}>Tree:</span> <span style={{ color: '#fff' }}>{m.gen_count} nodes</span></div>
            <div><span style={{ color: '#888' }}>Slices:</span> <span style={{ color: '#fff' }}>{m.slice_count}</span></div>
          </div>
        </div>

        {m.slices.length > 0 && (
          <div>
            <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
              Knowledge Slices (${m.slices.length})
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
      <button 
        onClick={loadData}
        style={{
          position: 'fixed',
          top: '20px',
          left: '20px',
          zIndex: 2000,
          padding: '12px 24px',
          background: 'rgba(255,255,255,0.05)',
          color: 'white',
          border: '1px solid #444',
          borderRadius: '8px',
          cursor: 'pointer',
          fontWeight: 'bold',
          transition: 'all 0.2s',
          backdropFilter: 'blur(5px)',
        }}
        onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
        onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
      >
        Refresh Visualization
      </button>

      {renderDetailPanel()}

      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        onNodeClick={(node) => setSelectedNode(node as CustomNode)}
        onNodeHover={(node) => {
          if (node) setSelectedNode(node as CustomNode);
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as CustomNode;
          const m = n.metadata;
          const label = m.instruction;
          const truncated = label.length > 25 ? label.substring(0, 25) + '...' : label;
          const fontSize = 12 / globalScale;
          const radius = 8;
          
          ctx.font = `${fontSize}px system-ui`;
          
          // Selection highlight
          if (selectedNode && selectedNode.id === n.id) {
            ctx.beginPath();
            ctx.arc(n.x!, n.y!, radius + 4, 0, 2 * Math.PI, false);
            ctx.strokeStyle = '#61dafb';
            ctx.lineWidth = 3 / globalScale;
            ctx.stroke();
          }

          // Draw node circle
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = n.color;
          ctx.fill();

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
        linkColor={() => '#333'}
        linkWidth={1.5}
        backgroundColor="#000"
      />
    </div>
  );
};
