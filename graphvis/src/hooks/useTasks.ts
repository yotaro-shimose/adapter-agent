import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../constants';
import type { TaskSummary } from '../types';

export function useTasks(experimentName: string | null) {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    if (!experimentName) {
      setTasks([]);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/api/${encodeURIComponent(experimentName)}/tasks`)
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: TaskSummary[]) => {
        setTasks(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || String(err));
        setLoading(false);
        setTasks([]);
      });
  }, [experimentName]);

  useEffect(() => {
    load();
  }, [load]);

  return { tasks, loading, error, reload: load };
}
