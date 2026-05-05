import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../constants';
import type { SimpleRun, SimpleRunStepSummary, SimpleRolloutListItem, SimpleRolloutDetail } from '../types';

export function useSimpleRuns() {
  const [runs, setRuns] = useState<SimpleRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/simple_runs`)
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SimpleRun[]) => {
        setRuns(Array.isArray(data) ? data : []);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return { runs, loading, error, reload };
}

export function useSimpleRunSummary(simpleTrainId: string | null) {
  const [summary, setSummary] = useState<SimpleRunStepSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    if (!simpleTrainId) {
      setSummary([]);
      return;
    }
    setLoading(true);
    fetch(`${API_BASE}/api/simple_runs/${encodeURIComponent(simpleTrainId)}/summary`)
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SimpleRunStepSummary[]) => {
        setSummary(Array.isArray(data) ? data : []);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || String(err));
        setLoading(false);
        setSummary([]);
      });
  }, [simpleTrainId]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { summary, loading, error, reload };
}

export interface RolloutFilters {
  rl_step?: number;
  suite_name?: string;
  task_id?: string;
  success?: 'true' | 'false';
  limit?: number;
}

export function useSimpleRunRollouts(simpleTrainId: string | null, filters: RolloutFilters) {
  const [rollouts, setRollouts] = useState<SimpleRolloutListItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fkey = JSON.stringify(filters);

  useEffect(() => {
    if (!simpleTrainId) {
      setRollouts([]);
      return;
    }
    setLoading(true);
    const params = new URLSearchParams();
    if (filters.rl_step !== undefined) params.set('rl_step', String(filters.rl_step));
    if (filters.suite_name) params.set('suite_name', filters.suite_name);
    if (filters.task_id) params.set('task_id', filters.task_id);
    if (filters.success) params.set('success', filters.success);
    if (filters.limit) params.set('limit', String(filters.limit));
    const q = params.toString();
    const url = `${API_BASE}/api/simple_runs/${encodeURIComponent(simpleTrainId)}/rollouts${q ? `?${q}` : ''}`;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SimpleRolloutListItem[]) => {
        setRollouts(Array.isArray(data) ? data : []);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        setError(err.message || String(err));
        setLoading(false);
        setRollouts([]);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simpleTrainId, fkey]);

  return { rollouts, loading, error };
}

export function useSimpleRolloutDetail(simpleTrainId: string | null, rolloutId: number | null) {
  const [detail, setDetail] = useState<SimpleRolloutDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!simpleTrainId || rolloutId == null) {
      setDetail(null);
      return;
    }
    setLoading(true);
    const controller = new AbortController();
    fetch(`${API_BASE}/api/simple_runs/${encodeURIComponent(simpleTrainId)}/rollout/${rolloutId}`, {
      signal: controller.signal,
    })
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SimpleRolloutDetail) => {
        setDetail(data);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        setError(err.message || String(err));
        setLoading(false);
        setDetail(null);
      });
    return () => controller.abort();
  }, [simpleTrainId, rolloutId]);

  return { detail, loading, error };
}
