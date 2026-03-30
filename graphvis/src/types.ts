import type { NodeObject, LinkObject } from 'react-force-graph-2d';

export interface GraphNodeMetadata {
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

export interface CustomNode extends NodeObject {
  id: string;
  metadata: GraphNodeMetadata;
  color: string;
  type: 'task' | 'knowledge';
  label: string;
  name?: string;
  val?: number;
}

export interface CustomLink extends LinkObject {
  id: string;
  source: string | CustomNode;
  target: string | CustomNode;
  type: 'decomposition' | 'citation' | 'generation';
}

export interface GraphExportNode {
  id: string;
  label: string;
  type?: string;
  metadata: GraphNodeMetadata;
}

export interface GraphExportEdge {
  id: string;
  source: string;
  target: string;
}

export interface ContentPart {
  type: string;
  text?: string;
  thinking?: string;
}

export interface TrajectoryTurn {
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

export interface TrajectoryData {
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

export interface GraphExportData {
  nodes: GraphExportNode[];
  edges: GraphExportEdge[];
}
