import React from 'react';
import { Clock, Users, Zap } from 'lucide-react';

interface MetricsOverlayProps {
  fps: number;
  latencyMs: number;
  facesCount: number;
}

export const MetricsOverlay: React.FC<MetricsOverlayProps> = ({ fps, latencyMs, facesCount }) => {
  return (
    <div className="grid grid-cols-3 gap-3">
      <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-3.5 flex items-center space-x-3">
        <div className="p-2 rounded-lg bg-graphite-800 text-tealaccent-400">
          <Zap className="w-4 h-4" />
        </div>
        <div>
          <div className="text-[11px] text-offwhite-400 font-mono">FRAME RATE</div>
          <div className="text-base font-semibold text-offwhite-100 font-mono">
            {fps > 0 ? `${fps.toFixed(0)} FPS` : '--'}
          </div>
        </div>
      </div>

      <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-3.5 flex items-center space-x-3">
        <div className="p-2 rounded-lg bg-graphite-800 text-tealaccent-400">
          <Clock className="w-4 h-4" />
        </div>
        <div>
          <div className="text-[11px] text-offwhite-400 font-mono">LATENCY</div>
          <div className="text-base font-semibold text-offwhite-100 font-mono">
            {latencyMs > 0 ? `${latencyMs.toFixed(1)} ms` : '--'}
          </div>
        </div>
      </div>

      <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-3.5 flex items-center space-x-3">
        <div className="p-2 rounded-lg bg-graphite-800 text-tealaccent-400">
          <Users className="w-4 h-4" />
        </div>
        <div>
          <div className="text-[11px] text-offwhite-400 font-mono">FACES</div>
          <div className="text-base font-semibold text-offwhite-100 font-mono">{facesCount}</div>
        </div>
      </div>
    </div>
  );
};
