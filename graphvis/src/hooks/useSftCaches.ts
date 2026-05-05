import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../constants';
import type { SftCacheSummary, SftCacheItemListItem, SftCacheItemDetail } from '../types';

export function useSftCaches() {
  const [caches, setCaches] = useState<SftCacheSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/sft_caches`)
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SftCacheSummary[]) => {
        setCaches(Array.isArray(data) ? data : []);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => { reload(); }, [reload]);

  return { caches, loading, error, reload };
}

export interface SftItemFilters {
  knowledge_id?: string;
  verified?: 'true' | 'false';
  limit?: number;
}

export function useSftCacheItems(
  cacheId: string | null,
  filters: SftItemFilters,
  refreshToken: number = 0,
) {
  const [items, setItems] = useState<SftCacheItemListItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fkey = JSON.stringify(filters);

  useEffect(() => {
    if (!cacheId) {
      setItems([]);
      return;
    }
    setLoading(true);
    const params = new URLSearchParams();
    if (filters.knowledge_id) params.set('knowledge_id', filters.knowledge_id);
    if (filters.verified) params.set('verified', filters.verified);
    if (filters.limit) params.set('limit', String(filters.limit));
    const q = params.toString();
    const url = `${API_BASE}/api/sft_caches/${encodeURIComponent(cacheId)}/items${q ? `?${q}` : ''}`;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SftCacheItemListItem[]) => {
        setItems(Array.isArray(data) ? data : []);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        setError(err.message || String(err));
        setLoading(false);
        setItems([]);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheId, fkey, refreshToken]);

  return { items, loading, error };
}

export function useSftCacheItemDetail(
  cacheId: string | null,
  itemId: number | null,
  refreshToken: number = 0,
) {
  const [detail, setDetail] = useState<SftCacheItemDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!cacheId || itemId == null) {
      setDetail(null);
      return;
    }
    setLoading(true);
    const controller = new AbortController();
    fetch(`${API_BASE}/api/sft_caches/${encodeURIComponent(cacheId)}/item/${itemId}`, {
      signal: controller.signal,
    })
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SftCacheItemDetail) => {
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
  }, [cacheId, itemId, refreshToken]);

  return { detail, loading, error };
}
