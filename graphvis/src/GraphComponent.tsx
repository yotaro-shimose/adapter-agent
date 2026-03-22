import React from 'react';
import { GraphCanvas, type GraphCanvasRef } from 'reagraph';

const nodes = [
  { id: '1', label: 'Node 1' },
  { id: '2', label: 'Node 2' },
  { id: '3', label: 'Node 3' },
  { id: '4', label: 'Node 4' },
];

const edges = [
  { id: '1-2', source: '1', target: '2', label: 'Edge 1-2' },
  { id: '2-3', source: '2', target: '3', label: 'Edge 2-3' },
  { id: '3-1', source: '3', target: '1', label: 'Edge 3-1' },
  { id: '3-4', source: '3', target: '4', label: 'Edge 3-4' },
];

export const GraphComponent: React.FC = () => {
  const graphRef = React.useRef<GraphCanvasRef>(null);

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
      <GraphCanvas
        ref={graphRef}
        nodes={nodes}
        edges={edges}
      />
    </div>
  );
};
