import React from 'react';
import { Github, Cpu } from 'lucide-react';
import { HealthCheckResponse } from '../types';

interface NavbarProps {
  health: HealthCheckResponse | null;
  loading: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ health, loading }) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="border-b border-graphite-800 bg-graphite-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-tealaccent-900/50 border border-tealaccent-600/30 flex items-center justify-center text-tealaccent-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-offwhite-50 text-base tracking-tight">EmotionDetect</span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-graphite-800 text-tealaccent-400 border border-graphite-700">
                CNN v0.1
              </span>
            </div>
            <p className="text-xs text-offwhite-400 font-normal">Real-Time Facial Emotion Recognition</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Backend Status Indicator */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-graphite-850 border border-graphite-800 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                loading
                  ? 'bg-amber-400 animate-pulse'
                  : isHealthy
                  ? 'bg-tealaccent-400'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-offwhite-300 font-mono">
              {loading ? 'Connecting...' : isHealthy ? 'Engine Online' : 'Engine Offline'}
            </span>
          </div>

          <a
            href="https://github.com/venkatreddy3/EmotionDetect"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1.5 text-xs text-offwhite-300 hover:text-offwhite-50 px-3 py-1.5 rounded-lg bg-graphite-800 hover:bg-graphite-700 transition border border-graphite-700"
          >
            <Github className="w-4 h-4" />
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
};
