import { useState, useEffect } from 'react';
import { API_BASE } from '../constants';

export function useExperiments() {
  const [experiments, setExperiments] = useState<string[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchExps = () => {
      fetch(`${API_BASE}/api/experiments`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          setExperiments(data);
          if (data.length > 0 && !selectedExperiment) {
            setSelectedExperiment(data[0]);
          }
          setIsLoading(false);
        })
        .catch(err => {
          console.error("Failed to fetch experiments", err);
          setError(`Could not connect to backend at ${API_BASE}. Make sure 'just vis' is running.`);
          setIsLoading(false);
        });
    };

    fetchExps();
    const interval = setInterval(fetchExps, 5000);
    return () => clearInterval(interval);
  }, [selectedExperiment]);

  return { experiments, selectedExperiment, setSelectedExperiment, error, isLoading };
}
