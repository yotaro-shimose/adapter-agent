import { useState, useCallback, useEffect } from 'react';
import { API_BASE, COLORS } from '../constants';
import { getNodeLabel } from '../utils';
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
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      const response = await fetch(`${API_BASE}/api/${encodeURIComponent(selectedExperiment)}/graph`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const json: GraphExportData = await response.json();

      const taskNodes: CustomNode[] = json.nodes.map(n => {
        const node: CustomNode = {
          id: n.id,
          label: n.id,
          metadata: {
            ...n.metadata,
            citations: n.metadata.citations || []
          },
          color: n.id === 'pseudo_root' ? COLORS.PSEUDO_ROOT : (n.type === 'knowledge' ? COLORS.KNOWLEDGE_NODE : (n.metadata.is_solved ? COLORS.SOLVED_TASK : n.metadata.is_executing ? COLORS.EXECUTING_TASK : COLORS.QUEUED_TASK)),
          type: (n.type as any) || 'task',
          val: 5
        };
        node.name = getNodeLabel(node);
        return node;
      });

      const knowledge_ids = new Set<string>();
      const knowledge_stats_map = new Map<string, { 
        total_citations: number, 
        success_citations: number,
        tasks_attempted: Set<string>,
        tasks_solved: Set<string>
      }>();
      
      taskNodes.forEach(tn => {
        tn.metadata.citations.forEach(c => {
          knowledge_ids.add(c.knowledge_id);
          const stats = knowledge_stats_map.get(c.knowledge_id) || { 
            total_citations: 0, 
            success_citations: 0, 
            tasks_attempted: new Set<string>(), 
            tasks_solved: new Set<string>() 
          };
          
          stats.total_citations += 1;
          if (tn.metadata.is_solved) {
            stats.success_citations += 1;
            stats.tasks_solved.add(tn.id);
          }
          stats.tasks_attempted.add(tn.id);
          
          knowledge_stats_map.set(c.knowledge_id, stats);
        });
      });

      const taskNodeMap = new Map<string, CustomNode>();
      taskNodes.forEach(n => taskNodeMap.set(n.id, n));

      const knowledgeIdToTaskId = new Map<string, string>();
      taskNodes.forEach(tn => {
        if (tn.metadata.generated_knowledge_id) {
          knowledgeIdToTaskId.set(tn.metadata.generated_knowledge_id, tn.id);
        }
      });

      const knowledgeNodes: CustomNode[] = Array.from(knowledge_ids).map(kid => {
        const generatorTaskId = knowledgeIdToTaskId.get(kid);
        const sourceTask = generatorTaskId ? taskNodeMap.get(generatorTaskId) : undefined;
        
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
        const stats = knowledge_stats_map.get(kid) || { 
          total_citations: 0, 
          success_citations: 0, 
          tasks_attempted: new Set<string>(), 
          tasks_solved: new Set<string>() 
        };

        return {
          id: kid,
          label: kid,
          name: knowledge_title,
          // Scale by unique successful tasks as requested
          val: 12 + stats.tasks_solved.size * 8, 
          metadata: {
            instruction: `Knowledge ID: ${kid}`,
            success_count: stats.success_citations,
            total_count: stats.total_citations,
            unique_task_success_count: stats.tasks_solved.size,
            unique_task_total_count: stats.tasks_attempted.size,
            is_solved: true,
            is_executing: false,
            slice_count: 0,
            gen_count: 0,
            citations: [],
            slices: [],
            knowledge_content: sourceTask?.metadata.knowledge_content || capturedContent,
            knowledge_title: knowledge_title,
            generator_task_id: generatorTaskId
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
