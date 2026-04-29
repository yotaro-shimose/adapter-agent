import React, { useMemo, useState } from 'react';
import { useExperiments } from '../hooks/useExperiments';
import { useTasks } from '../hooks/useTasks';
import { useTrajectories } from '../hooks/useTrajectories';
import { TrajectoryView } from './TrajectoryView';
import type { TaskSummary } from '../types';

type SortKey = 'recent' | 'hardest' | 'attempts';
type FilterKey = 'all' | 'solved' | 'failed' | 'mixed';

export const TrajectoryExplorer: React.FC = () => {
  const { experiments, selectedExperiment, setSelectedExperiment, error: expError, isLoading: expLoading } = useExperiments();
  const { tasks, loading: tasksLoading, error: tasksError, reload: reloadTasks } = useTasks(selectedExperiment);

  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('recent');
  const [filter, setFilter] = useState<FilterKey>('all');

  const { trajectories, selectedAttemptIndex, setSelectedAttemptIndex, loadingTraj, trajError } =
    useTrajectories(selectedTaskId, selectedExperiment);

  const filteredTasks = useMemo(() => {
    const q = search.trim().toLowerCase();
    let xs = tasks.filter(t => {
      if (q && !(t.instruction.toLowerCase().includes(q) || t.task_id.toLowerCase().includes(q))) return false;
      const successRate = t.total_count > 0 ? t.success_count / t.total_count : 0;
      if (filter === 'solved' && t.success_count === 0) return false;
      if (filter === 'failed' && t.success_count > 0) return false;
      if (filter === 'mixed' && (successRate === 0 || successRate === 1)) return false;
      return true;
    });
    xs = [...xs];
    if (sortKey === 'recent') {
      xs.sort((a, b) => (b.latest_created_at ?? '').localeCompare(a.latest_created_at ?? ''));
    } else if (sortKey === 'hardest') {
      xs.sort((a, b) => {
        const ra = a.total_count > 0 ? a.success_count / a.total_count : 1;
        const rb = b.total_count > 0 ? b.success_count / b.total_count : 1;
        return ra - rb;
      });
    } else if (sortKey === 'attempts') {
      xs.sort((a, b) => b.total_count - a.total_count);
    }
    return xs;
  }, [tasks, search, sortKey, filter]);

  const summary = useMemo(() => {
    const totalTasks = tasks.length;
    const solvedTasks = tasks.filter(t => t.success_count > 0).length;
    const totalTraj = tasks.reduce((s, t) => s + t.total_count, 0);
    const successTraj = tasks.reduce((s, t) => s + t.success_count, 0);
    return { totalTasks, solvedTasks, totalTraj, successTraj };
  }, [tasks]);

  return (
    <div style={{ display: 'flex', height: '100%', fontFamily: '"Inter", sans-serif', textAlign: 'left', background: '#0f111a', color: '#fff' }}>
      {/* Sidebar */}
      <div style={{ width: '380px', borderRight: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', background: 'rgba(15, 17, 26, 0.5)' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#ccc' }}>Experiment</h3>
            <button
              onClick={reloadTasks}
              disabled={tasksLoading || !selectedExperiment}
              style={{ background: 'transparent', border: '1px solid #444', color: tasksLoading ? '#555' : '#888', borderRadius: '4px', padding: '4px 8px', fontSize: '0.75rem', cursor: tasksLoading ? 'default' : 'pointer' }}
            >
              {tasksLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          <select
            value={selectedExperiment ?? ''}
            onChange={e => { setSelectedExperiment(e.target.value); setSelectedTaskId(null); }}
            disabled={expLoading || experiments.length === 0}
            style={{ width: '100%', padding: '10px', background: '#1a1d27', color: '#fff', border: '1px solid #333', borderRadius: '6px', outline: 'none' }}
          >
            {experiments.length === 0 && <option value="">{expLoading ? 'Loading...' : 'No experiments'}</option>}
            {experiments.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
          {expError && <div style={{ color: '#e74c3c', fontSize: '11px', marginTop: '8px' }}>{expError}</div>}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '15px' }}>
            <Stat label="Tasks" value={`${summary.solvedTasks}/${summary.totalTasks}`} color="#61dafb" />
            <Stat label="Trajectories" value={`${summary.successTraj}/${summary.totalTraj}`} color="#2ecc71" />
          </div>
        </div>

        <div style={{ padding: '12px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <input
            type="text"
            placeholder="Search instruction..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ padding: '8px 10px', background: '#1a1d27', color: '#fff', border: '1px solid #333', borderRadius: '6px', outline: 'none', fontSize: '12px' }}
          />
          <div style={{ display: 'flex', gap: '6px' }}>
            <select value={sortKey} onChange={e => setSortKey(e.target.value as SortKey)} style={selectStyle}>
              <option value="recent">Recent</option>
              <option value="hardest">Hardest</option>
              <option value="attempts">Most attempts</option>
            </select>
            <select value={filter} onChange={e => setFilter(e.target.value as FilterKey)} style={selectStyle}>
              <option value="all">All</option>
              <option value="solved">Solved</option>
              <option value="failed">Failed</option>
              <option value="mixed">Mixed</option>
            </select>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {tasksError && <div style={{ padding: '20px', color: '#e74c3c', fontSize: '12px' }}>Failed to load tasks: {tasksError}</div>}
          {!tasksError && filteredTasks.length === 0 && !tasksLoading && (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: '#555', fontSize: '13px' }}>No tasks match.</div>
          )}
          {filteredTasks.map(t => (
            <TaskRow
              key={t.task_id}
              task={t}
              selected={t.task_id === selectedTaskId}
              onClick={() => setSelectedTaskId(t.task_id)}
            />
          ))}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '30px 40px', background: 'radial-gradient(circle at top right, rgba(97, 218, 251, 0.04), transparent 40%)' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          {!selectedTaskId ? (
            <div style={{ textAlign: 'center', padding: '120px 20px', color: '#555' }}>
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>📂</div>
              <div style={{ fontSize: '14px' }}>Select a task on the left to inspect its trajectories.</div>
            </div>
          ) : (
            <>
              <div style={{ marginBottom: '10px', fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>Task</div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 600, margin: 0, marginBottom: '20px', color: '#eee', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
                {tasks.find(t => t.task_id === selectedTaskId)?.instruction ?? selectedTaskId}
              </h2>
              <TrajectoryView
                trajectories={trajectories}
                selectedAttemptIndex={selectedAttemptIndex}
                onSelectAttempt={setSelectedAttemptIndex}
                loading={loadingTraj}
                error={trajError}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const selectStyle: React.CSSProperties = {
  flex: 1,
  padding: '6px 8px',
  background: '#1a1d27',
  color: '#fff',
  border: '1px solid #333',
  borderRadius: '6px',
  outline: 'none',
  fontSize: '12px',
};

const Stat: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => (
  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 12px', borderRadius: '8px', borderLeft: `3px solid ${color}` }}>
    <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: '2px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
    <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff' }}>{value}</div>
  </div>
);

const TaskRow: React.FC<{ task: TaskSummary; selected: boolean; onClick: () => void }> = ({ task, selected, onClick }) => {
  const rate = task.total_count > 0 ? task.success_count / task.total_count : 0;
  const color = task.success_count === 0 ? '#ff4d4d' : rate === 1 ? '#2ecc71' : '#f1c40f';
  return (
    <div
      onClick={onClick}
      style={{
        padding: '12px 20px',
        cursor: 'pointer',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: selected ? 'rgba(97, 218, 251, 0.08)' : 'transparent',
        borderLeft: selected ? '3px solid #61dafb' : '3px solid transparent',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => { if (!selected) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
      onMouseLeave={e => { if (!selected) e.currentTarget.style.background = 'transparent'; }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: color, flexShrink: 0 }} />
        <span style={{ fontSize: '11px', color, fontWeight: 600 }}>
          {task.success_count}/{task.total_count}
        </span>
        {task.latest_created_at && (
          <span style={{ fontSize: '10px', color: '#666', marginLeft: 'auto' }}>
            {new Date(task.latest_created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
          </span>
        )}
      </div>
      <div style={{ fontSize: '12px', color: selected ? '#fff' : '#bbb', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
        {task.instruction || task.task_id}
      </div>
    </div>
  );
};
