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
  if (node.id === 'pseudo_root') return 16;
  if (node.type === 'knowledge') return 12;
  return 8;
};
