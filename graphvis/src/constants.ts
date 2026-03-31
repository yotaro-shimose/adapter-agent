export const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://localhost:8000' 
  : `http://${window.location.hostname}:8000`;

export const COLORS = {
  PSEUDO_ROOT: '#e67e22',
  SOLVED_TASK: '#28a745',
  EXECUTING_TASK: '#f1c40f',
  QUEUED_TASK: '#555',
  KNOWLEDGE_NODE: '#9b59b6',
  GENERATION_LINK: 'rgba(155, 89, 182, 0.9)',
  CITATION_LINK: 'rgba(155, 89, 182, 0.3)',
  DECOMPOSITION_LINK: '#333',
  PRIMARY_ACCENT: '#61dafb',
  ERROR: '#ff4d4d',
};

export const FORCE_GRAPH_SETTINGS = {
  LINK_DISTANCE: 150,
  REPULSION_STRENGTH: -1000,
};
