import type { CustomNode, TrajectoryTurn } from './types';

/**
 * Extracts and formats the content string from a TrajectoryTurn.
 * Handles both plain strings and arrays of content parts (with thinking/thought blocks).
 */
export const getTrajectoryContent = (turn: TrajectoryTurn): string => {
  if (typeof turn.content === 'string') {
    return turn.content;
  }
  
  if (Array.isArray(turn.content)) {
    return turn.content.map((p: any) => 
      p.type === 'thinking' ? `<think>${p.thinking}</think>` : (p.text || '')
    ).join('');
  }
  
  return String(turn.content || '');
};

/**
 * Returns the display label for a node based on its type and metadata.
 */
export const getNodeLabel = (node: CustomNode): string => {
  const m = node.metadata;
  if (node.type === 'knowledge') {
    return m.knowledge_title || node.id;
  }
  return m.instruction || node.id;
};

/**
 * Truncates text to a specified length with an ellipsis.
 */
export const truncateText = (text: string, length: number): string => {
  if (text.length <= length) return text;
  return text.substring(0, length) + '...';
};

/**
 * Calculates the radius for a node based on its type and role.
 */
export const getNodeRadius = (node: CustomNode): number => {
  const m = node.metadata;
  if (node.id === 'pseudo_root') return 24;
  if (node.type === 'knowledge') {
    // Knowledge nodes scale with broad utility (unique tasks successfully solved). 
    // This highlights knowledge that is useful across different problem sets.
    const taskScale = (m.unique_task_success_count || 0) * 6;
    return Math.min(16 + taskScale, 48); // Min 16, Cap at 48
  }
  return 14; // Default task node radius
};
