import React, { useEffect, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';

import { useExperiments } from '../hooks/useExperiments';
import { useGraphData } from '../hooks/useGraphData';
import { useTrajectories } from '../hooks/useTrajectories';
import { useFilteredGraph } from '../hooks/useFilteredGraph';

import { ControlPanel } from './ControlPanel';
import { DetailPanel } from './DetailPanel';

import { getNodeLabel, truncateText, getNodeRadius } from '../utils';
import { COLORS, FORCE_GRAPH_SETTINGS } from '../constants';
import type { CustomNode } from '../types';
import './GraphCanvas.css';

export const GraphCanvasComponent: React.FC = () => {
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);
  
  // State for view controls
  const [viewMode, setViewMode] = useState<'global' | 'local'>('global');
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const [focusDepth, setFocusDepth] = useState<number>(1);
  const [showKnowledge, setShowKnowledge] = useState<boolean>(true);
  const [selectedNode, setSelectedNode] = useState<CustomNode | null>(null);

  // Custom Hooks
  const { experiments, selectedExperiment, setSelectedExperiment, error: expError } = useExperiments();
  const { data, error: graphError, isInitializing, loadData: refreshGraph } = useGraphData(selectedExperiment);
  const taskId = selectedNode && selectedNode.type === 'task' ? selectedNode.id : null;
  const { trajectories, selectedAttemptIndex, setSelectedAttemptIndex, loadingTraj, trajError } = useTrajectories(taskId, selectedExperiment);
  
  const displayData = useFilteredGraph({
    data,
    viewMode,
    focusNodeId,
    focusDepth,
    showKnowledge
  });

  const handleNodeSelect = useCallback((node: CustomNode) => {
    setSelectedNode(node);
    setSelectedAttemptIndex(0);
    if (fgRef.current && node.x !== undefined && node.y !== undefined) {
      fgRef.current.centerAt(node.x, node.y, 600);
      fgRef.current.zoom(2.5, 600);
    }
  }, [setSelectedAttemptIndex]);

  useEffect(() => {
    if (fgRef.current && data) {
      const linkForce = fgRef.current.d3Force('link');
      if (linkForce) (linkForce as any).distance(FORCE_GRAPH_SETTINGS.LINK_DISTANCE);
      
      const chargeForce = fgRef.current.d3Force('charge');
      if (chargeForce) (chargeForce as any).strength(FORCE_GRAPH_SETTINGS.REPULSION_STRENGTH);
    }
  }, [data]);

  // Loading States
  if (experiments.length === 0 && !expError) {
    return <div className="detail-panel" style={{ position: 'unset', width: 'auto', margin: '20px' }}>Waiting for experiments...</div>;
  }

  if (graphError || expError) {
    return (
      <div className="detail-panel" style={{ color: COLORS.ERROR, border: `1px solid ${COLORS.ERROR}`, position: 'unset', width: 'auto', margin: '20px' }}>
        <div style={{ fontSize: '20px', fontWeight: 'bold' }}>⚠️ Error Loading Graph</div>
        <div>{graphError || expError}</div>
        <button onClick={refreshGraph} style={{ marginTop: '15px' }}>Retry</button>
      </div>
    );
  }

  if (!data && isInitializing) {
    return <div className="detail-panel" style={{ position: 'unset', width: 'auto', margin: '20px' }}>Loading graph data...</div>;
  }

  return (
    <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: '#000' }}>
      <ControlPanel
        experiments={experiments}
        selectedExperiment={selectedExperiment}
        onSelectExperiment={setSelectedExperiment}
        onRefresh={refreshGraph}
        showKnowledge={showKnowledge}
        onToggleKnowledge={() => setShowKnowledge(!showKnowledge)}
        viewMode={viewMode}
        onSetGlobalView={() => {
          setViewMode('global');
          setFocusNodeId(null);
        }}
        focusDepth={focusDepth}
        onSetFocusDepth={setFocusDepth}
      />

      {selectedNode && (
        <DetailPanel
          selectedNode={selectedNode}
          onClose={() => setSelectedNode(null)}
          onFocus={(nodeId) => {
            setFocusNodeId(nodeId);
            setViewMode('local');
          }}
          handleNodeSelect={handleNodeSelect}
          trajectories={trajectories}
          selectedAttemptIndex={selectedAttemptIndex}
          onSelectAttempt={setSelectedAttemptIndex}
          loadingTraj={loadingTraj}
          trajError={trajError}
          allNodes={data?.nodes || []}
          allLinks={data?.links || []}
        />
      )}

      <ForceGraph2D
        ref={fgRef}
        graphData={displayData || { nodes: [], links: [] }}
        onNodeClick={(node) => handleNodeSelect(node as CustomNode)}
        nodeLabel={(node) => {
          const n = node as CustomNode;
          const label = getNodeLabel(n);
          return `
            <div style="
              background: rgba(15, 15, 20, 0.95);
              border: 1px solid rgba(255, 255, 255, 0.15);
              border-radius: 10px;
              padding: 12px 16px;
              max-width: 480px;
              color: #fff;
              font-family: inherit;
              box-shadow: 0 10px 30px rgba(0,0,0,0.6);
              backdrop-filter: blur(10px);
            ">
              <div style="font-weight: 800; color: ${n.type === 'knowledge' ? COLORS.KNOWLEDGE_NODE : COLORS.PRIMARY_ACCENT}; margin-bottom: 6px; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8;">
                ${n.type === 'knowledge' ? '📚 Verified Knowledge' : '⚡ Task Instruction'}
              </div>
              <div style="font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; font-weight: 400; color: #eee;">
                ${label}
              </div>
              <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 11px; color: #888; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; gap: 10px;">
                  <span>Success: <b style="color: #fff">${n.metadata.success_count}/${n.metadata.total_count}</b></span>
                  ${n.type === 'task' && n.metadata.gen_count > 0 ? `<span>Generated: <b style="color: #fff">${n.metadata.gen_count}</b></span>` : ''}
                </div>
                <span style="font-family: monospace; opacity: 0.6;">${n.id}</span>
              </div>
            </div>
          `;
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as CustomNode;
          const m = n.metadata;
          const label = getNodeLabel(n);
          const truncated = truncateText(label, 30);
          const fontSize = 15 / globalScale;
          const radius = getNodeRadius(n);
          
          ctx.font = `${fontSize}px system-ui`;
          if (selectedNode && selectedNode.id === n.id) {
            ctx.beginPath();
            ctx.arc(n.x!, n.y!, radius + 4, 0, 2 * Math.PI, false);
            ctx.strokeStyle = n.type === 'knowledge' ? COLORS.KNOWLEDGE_NODE : COLORS.PRIMARY_ACCENT;
            ctx.lineWidth = 3 / globalScale;
            ctx.stroke();
          }

          ctx.beginPath();
          ctx.arc(n.x!, n.y!, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = n.color;
          ctx.fill();

          const isPseudoRoot = n.id === 'pseudo_root';
          if (n.type === 'knowledge') {
            ctx.font = `${fontSize * 1.5}px system-ui`;
            ctx.fillText('📚', n.x! - radius/2, n.y! + radius/2);
          } else if (isPseudoRoot) {
            ctx.font = `${fontSize * 1.8}px system-ui`;
            ctx.fillText('🏠', n.x! - radius/2, n.y! + radius/2);
          }

          if (m.is_executing) {
            ctx.font = `${fontSize * 1.5}px system-ui`;
            ctx.fillText('⚡', n.x! - radius - 2, n.y! - radius - 2);
          }

          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#fff';
          ctx.font = `${fontSize}px system-ui`;
          ctx.fillText(truncated, n.x!, n.y! + radius + 10);
          
          const stats = n.type === 'knowledge' 
            ? `Used: ${m.unique_task_success_count}/${m.unique_task_total_count} Tks`
            : `${m.success_count}/${m.total_count}` + (m.gen_count > 0 ? ` 📂${m.gen_count}` : '');
            
          ctx.font = `${fontSize * 0.8}px system-ui`;
          ctx.fillStyle = '#aaa';
          ctx.fillText(stats, n.x!, n.y! + radius + 12 + fontSize);
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as CustomNode;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, getNodeRadius(n) + 4, 0, 2 * Math.PI, false);
          ctx.fill();
        }}
        linkColor={(link) => {
          const l = link as any;
          if (l.type === 'generation') return COLORS.GENERATION_LINK;
          if (l.type === 'citation') return COLORS.CITATION_LINK;
          return COLORS.DECOMPOSITION_LINK;
        }}
        linkWidth={(link) => {
          const l = link as any;
          return l.type === 'generation' ? 2.5 : 1.5;
        }}
        linkLineDash={(link) => (link as any).type === 'citation' ? [3, 3] : null}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        linkCurvature={(link) => ['citation', 'generation'].includes((link as any).type) ? 0.2 : 0}
        backgroundColor="#000"
      />
    </div>
  );
};
