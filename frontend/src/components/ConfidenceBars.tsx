import React from 'react';
import { FaceEmotionPrediction } from '../types';

interface ConfidenceBarsProps {
  prediction: FaceEmotionPrediction | null;
}

const ALL_EMOTIONS = [
  'Angry',
  'Disgust',
  'Fear',
  'Happy',
  'Sad',
  'Surprise',
  'Neutral',
];

export const ConfidenceBars: React.FC<ConfidenceBarsProps> = ({ prediction }) => {
  const probs = prediction?.probabilities || {};

  return (
    <div className="bg-graphite-900 border border-graphite-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-offwhite-100 uppercase tracking-wider font-mono">
          Emotion Probabilities
        </h3>
        {prediction && (
          <span className="text-xs font-mono text-tealaccent-400 bg-tealaccent-900/30 px-2 py-0.5 rounded border border-tealaccent-700/40">
            {prediction.dominant_emotion} ({(prediction.confidence * 100).toFixed(1)}%)
          </span>
        )}
      </div>

      <div className="space-y-3">
        {ALL_EMOTIONS.map((emotion) => {
          const score = probs[emotion] !== undefined ? probs[emotion] : 0;
          const percentage = (score * 100).toFixed(1);
          const isDominant = prediction?.dominant_emotion === emotion;

          return (
            <div key={emotion} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className={isDominant ? 'font-semibold text-tealaccent-400' : 'text-offwhite-400'}>
                  {emotion}
                </span>
                <span className="font-mono text-offwhite-300">{percentage}%</span>
              </div>
              <div className="w-full h-2 bg-graphite-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 rounded-full ${
                    isDominant
                      ? 'bg-gradient-to-r from-tealaccent-600 to-tealaccent-400 shadow-[0_0_8px_rgba(20,184,166,0.5)]'
                      : 'bg-graphite-600'
                  }`}
                  style={{ width: `${Math.max(score * 100, 2)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
