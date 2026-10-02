import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { EmotionDashboard } from './components/EmotionDashboard';
import { HealthCheckResponse } from './types';
import { checkBackendHealth } from './services/api';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;

    const pollHealth = async () => {
      try {
        const data = await checkBackendHealth();
        if (isMounted) {
          setHealth(data);
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setHealth(null);
          setLoading(false);
        }
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 10000); // 10s poll

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-graphite-950 text-offwhite-100 flex flex-col selection:bg-tealaccent-500/30 selection:text-tealaccent-400 font-sans">
      <Navbar health={health} loading={loading} />
      <div className="flex-1">
        <EmotionDashboard />
      </div>
      <footer className="border-t border-graphite-800/80 bg-graphite-950 py-4 text-center text-xs text-offwhite-400 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Real-Time Facial Emotion Recognition — Deep Learning Portfolio</span>
          <span>MIT License &bull; Venkat Reddy</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
