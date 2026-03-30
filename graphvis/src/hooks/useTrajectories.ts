import { useState, useEffect } from 'react';
import { API_BASE } from '../constants';
import type { TrajectoryData, CustomNode } from '../types';

export function useTrajectories(selectedNode: CustomNode | null, selectedExperiment: string | null) {
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState<number>(0);
  const [loadingTraj, setLoadingTraj] = useState<boolean>(false);
  const [trajError, setTrajError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedNode && selectedNode.type === 'task' && selectedExperiment) {
      setLoadingTraj(true);
      setTrajError(null);
      fetch(`${API_BASE}/api/${encodeURIComponent(selectedExperiment)}/trajectory/${encodeURIComponent(selectedNode.id)}`)
        .then(async res => {
          if (!res.ok) {
            throw new Error(`Failed to fetch trajectories: ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          setTrajectories(Array.isArray(data) ? data : []);
          setSelectedAttemptIndex((data && data.length > 0) ? data.length - 1 : 0);
          setLoadingTraj(false);
        })
        .catch(err => {
          console.error("Failed to fetch trajectory:", err);
          setTrajError(err.message || String(err));
          setLoadingTraj(false);
          setTrajectories([]);
        });
    } else {
      setTrajectories([]);
    }
  }, [selectedNode, selectedExperiment]);

  return { trajectories, selectedAttemptIndex, setSelectedAttemptIndex, loadingTraj, trajError };
}
