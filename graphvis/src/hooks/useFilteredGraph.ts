import { useState, useEffect } from 'react';
import type { CustomNode, CustomLink } from '../types';

interface FilterProps {
  data: { nodes: CustomNode[], links: CustomLink[] } | null;
  viewMode: 'global' | 'local';
  focusNodeId: string | null;
  focusDepth: number;
}

export function useFilteredGraph({ data, viewMode, focusNodeId, focusDepth }: FilterProps) {
  const [displayData, setDisplayData] = useState<{ nodes: CustomNode[], links: CustomLink[] } | null>(null);

  useEffect(() => {
    if (!data) return;

    let neighborhood: Set<string>;
    if (viewMode === 'global' || !focusNodeId) {
      neighborhood = new Set(data.nodes.map(n => n.id));
    } else {
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

    const filteredNodes = data.nodes.filter(n => neighborhood.has(n.id));

    const filteredLinks = data.links.filter(l => {
      const sId = typeof l.source === 'string' ? l.source : (l.source as any).id;
      const tId = typeof l.target === 'string' ? l.target : (l.target as any).id;
      return neighborhood.has(sId) && neighborhood.has(tId);
    });

    setDisplayData({ nodes: filteredNodes, links: filteredLinks });
  }, [data, viewMode, focusNodeId, focusDepth]);

  return displayData;
}
