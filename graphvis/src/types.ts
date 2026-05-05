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
  generator_task_id?: string;
  generated_knowledge_id?: string;
  unique_task_success_count?: number;
  unique_task_total_count?: number;
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

export interface TaskSummary {
  task_id: string;
  instruction: string;
  total_count: number;
  success_count: number;
  max_reward: number | null;
  latest_created_at: string | null;
}

export interface SimpleRun {
  simple_train_id: string;
  created_at: string | null;
  latest_rollout_at: string | null;
  total_rollouts: number;
  success_count: number;
  max_rl_step: number | null;
}

export interface SimpleRunStepSummary {
  rl_step: number;
  suite_name: string;
  total_count: number;
  success_count: number;
  avg_reward: number | null;
  unique_tasks: number;
}

export interface SimpleRolloutListItem {
  id: number;
  rl_step: number;
  suite_name: string;
  task_id: string;
  group_idx: number;
  sample_idx: number;
  num_samples: number;
  instruction: string;
  parsed: boolean;
  success: boolean;
  reward: number;
  sampling_client_version: number;
  created_at: string | null;
}

export interface SimpleRolloutDetail extends SimpleRolloutListItem {
  simple_train_id: string;
  answer: string;
  reasoning: string;
  execution_output: string;
  verification_output: string;
}

export interface SftCacheSummary {
  id: string;
  granular_id: string | null;
  library_name: string | null;
  description: string | null;
  created_at: string | null;
  latest_item_at: string | null;
  total_items: number;
  verified_items: number;
  unique_knowledges: number;
}

export interface SftCacheItemListItem {
  id: number;
  cache_id: string;
  knowledge_id: string;
  knowledge_title: string;
  question: string;
  verified: boolean;
  conclusion: string;
  created_at: string | null;
}

export interface SftTrialMessage {
  role: 'system' | 'user' | 'assistant' | 'tool' | string;
  content: string | ContentPart[];
  // tinker messages may carry tool_calls / tool_call_id but the synthesis
  // solver doesn't use them (XML tag-based protocol instead).
  tool_calls?: unknown;
  tool_call_id?: unknown;
  [k: string]: unknown;
}

export interface SftCacheItemDetail extends SftCacheItemListItem {
  reasoning: string;
  answer: string;
  verifier_reasoning: string;
  reward: number | null;
  trials: SftTrialMessage[] | null;
}

