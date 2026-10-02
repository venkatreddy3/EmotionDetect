import React, { useState, useEffect } from 'react';
import { WebcamStream } from './WebcamStream';
import { ConfidenceBars } from './ConfidenceBars';
import { MetricsOverlay } from './MetricsOverlay';
import { FaceEmotionPrediction } from '../types';
import { Layers, Terminal } from 'lucide-react';

export const EmotionDashboard: React.FC = () => {
  const [activePrediction, setActivePrediction] = useState<FaceEmotionPrediction | null>(null);
  const [latency, setLatency] = useState<number>(0);
  const [facesCount, setFacesCount] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);

  // Approximate client-side FPS counter
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();

    const interval = setInterval(() => {
      const now = performance.now();
      const delta = (now - lastTime) / 1000;
      if (delta > 0 && activePrediction) {
        setFps(frameCount / delta);
      } else if (!activePrediction) {
        setFps(0);
      }
      frameCount = 0;
      lastTime = now;
    }, 1000);

    return () => clearInterval(interval);
  }, [activePrediction]);

  const handlePredictionUpdate = (
    pred: FaceEmotionPrediction | null,
    latencyMs: number,
    faces: number
  ) => {
    setActivePrediction(pred);
    setLatency(latencyMs);
    setFacesCount(faces);
  };

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Hero Overview */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between pb-4 border-b border-graphite-800 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-offwhite-50 tracking-tight flex items-center space-x-2.5">
            <span>Inference Dashboard</span>
            <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-tealaccent-900/30 text-tealaccent-400 border border-tealaccent-700/30">
              Live Stream
            </span>
          </h1>
          <p className="text-xs text-offwhite-400 mt-1 max-w-2xl">
            Real-time multi-stage pipeline: OpenCV face detection, ROI normalization (48x48), and Deep CNN 7-class emotion classification.
          </p>
        </div>

        <MetricsOverlay fps={fps} latencyMs={latency} facesCount={facesCount} />
      </div>

      {/* Main Grid: Stream & Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-7 space-y-4">
          <WebcamStream onPredictionUpdate={handlePredictionUpdate} />

          {/* Model Information Card */}
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-4 text-xs font-mono text-offwhite-300 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-offwhite-400">
              <Layers className="w-4 h-4 text-tealaccent-400" />
              <span>Target Architecture:</span>
              <span className="text-offwhite-200">Mini-Xception / Depthwise Separable Conv2D</span>
            </div>
            <div className="text-offwhite-400">
              Classes: <span className="text-tealaccent-400">7 (Ekman Taxonomy)</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <ConfidenceBars prediction={activePrediction} />

          {/* Pipeline Details */}
          <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center space-x-2 text-offwhite-100 text-sm font-medium font-mono uppercase tracking-wider">
              <Terminal className="w-4 h-4 text-tealaccent-400" />
              <span>Pipeline Stages</span>
            </div>
            <ul className="space-y-2.5 text-xs text-offwhite-300">
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-tealaccent-400 mt-1.5 shrink-0" />
                <span><strong className="text-offwhite-100">Stage 1:</strong> Frame ingestion & aspect-ratio normalization</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-tealaccent-400 mt-1.5 shrink-0" />
                <span><strong className="text-offwhite-100">Stage 2:</strong> Frontal face detection via Haar Cascade / DNN</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-tealaccent-400 mt-1.5 shrink-0" />
                <span><strong className="text-offwhite-100">Stage 3:</strong> Grayscale crop (48x48) & pixel rescaling [0, 1]</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-tealaccent-400 mt-1.5 shrink-0" />
                <span><strong className="text-offwhite-100">Stage 4:</strong> CNN forward pass & Softmax probability extraction</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
};
