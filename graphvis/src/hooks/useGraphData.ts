import { useState, useCallback, useEffect } from 'react';
import { API_BASE, COLORS } from '../constants';
import type { CustomNode, CustomLink, GraphExportData } from '../types';

export function useGraphData(selectedExperiment: string | null) {
  const [data, setData] = useState<{ nodes: CustomNode[], links: CustomLink[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);

  const loadData = useCallback(async () => {
    if (!selectedExperiment) {
      setIsInitializing(false);
      return;
    }
    setError(null);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`${API_BASE}/api/${encodeURIComponent(selectedExperiment)}/graph`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
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
        color: n.id === 'pseudo_root' ? COLORS.PSEUDO_ROOT : (n.type === 'knowledge' ? COLORS.KNOWLEDGE_NODE : (n.metadata.is_solved ? COLORS.SOLVED_TASK : n.metadata.is_executing ? COLORS.EXECUTING_TASK : COLORS.QUEUED_TASK)),
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
          color: COLORS.KNOWLEDGE_NODE,
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
        setError('Request timed out.');
      } else {
        console.error('Failed to load graph data:', err);
        setError(`Failed to load graph data: ${err.message || String(err)}`);
      }
      setData({ nodes: [], links: [] });
      setIsInitializing(false);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    loadData();
  }, [loadData, selectedExperiment]);

  return { data, error, isInitializing, loadData };
}
