import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, CameraOff, AlertCircle } from 'lucide-react';
import { FaceEmotionPrediction } from '../types';
import { predictEmotionFromBlob } from '../services/api';

interface WebcamStreamProps {
  onPredictionUpdate: (prediction: FaceEmotionPrediction | null, latency: number, faces: number) => void;
}

export const WebcamStream: React.FC<WebcamStreamProps> = ({ onPredictionUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const [isActive, setIsActive] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // Start Webcam
  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsActive(true);
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Unable to access camera';
      setError(errMsg);
    }
  };

  // Stop Webcam
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    onPredictionUpdate(null, 0, 0);

    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  };

  // Frame Capture & Inference Loop
  const captureAndPredict = useCallback(async () => {
    if (!isActive || !videoRef.current || !canvasRef.current || isProcessing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayCanvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      setIsProcessing(true);
      try {
        const result = await predictEmotionFromBlob(blob);
        const primary = result.predictions[0] || null;
        onPredictionUpdate(primary, result.inference_time_ms, result.num_faces_detected);

        // Draw bounding box on overlay
        if (overlay) {
          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;
          const oCtx = overlay.getContext('2d');
          if (oCtx) {
            oCtx.clearRect(0, 0, overlay.width, overlay.height);

            result.predictions.forEach((p) => {
              const { x, y, width, height } = p.bounding_box;
              oCtx.strokeStyle = '#14B8A6';
              oCtx.lineWidth = 2.5;
              oCtx.strokeRect(x, y, width, height);

              // Label badge
              const label = `${p.dominant_emotion} (${Math.round(p.confidence * 100)}%)`;
              oCtx.font = '600 13px Inter, sans-serif';
              const textWidth = oCtx.measureText(label).width;

              oCtx.fillStyle = '#0B0C0E';
              oCtx.fillRect(x, Math.max(0, y - 24), textWidth + 12, 22);

              oCtx.fillStyle = '#2DD4BF';
              oCtx.fillText(label, x + 6, Math.max(16, y - 8));
            });
          }
        }
      } catch (e) {
        // Silently skip frame error to allow smooth loop recovery
      } finally {
        setIsProcessing(false);
      }
    }, 'image/jpeg', 0.85);
  }, [isActive, isProcessing, onPredictionUpdate]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isActive) {
      interval = setInterval(captureAndPredict, 150); // ~6.6 FPS inference request rate
    }
    return () => clearInterval(interval);
  }, [isActive, captureAndPredict]);

  return (
    <div className="bg-graphite-900 border border-graphite-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
      {/* Video Container */}
      <div className="relative aspect-video bg-graphite-950 flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full h-full object-cover ${isActive ? 'block' : 'hidden'}`}
        />
        <canvas ref={canvasRef} className="hidden" />
        <canvas
          ref={overlayCanvasRef}
          className={`absolute inset-0 w-full h-full pointer-events-none ${isActive ? 'block' : 'hidden'}`}
        />

        {!isActive && !error && (
          <div className="text-center p-6 space-y-3">
            <div className="w-12 h-12 rounded-full bg-graphite-850 border border-graphite-700 mx-auto flex items-center justify-center text-offwhite-400">
              <Camera className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-offwhite-200">Camera is offline</p>
              <p className="text-xs text-offwhite-400">Start the stream to run real-time facial emotion recognition</p>
            </div>
          </div>
        )}

        {error && (
          <div className="p-6 text-center space-y-2">
            <AlertCircle className="w-8 h-8 text-rose-400 mx-auto" />
            <p className="text-sm text-rose-300 font-medium">{error}</p>
            <p className="text-xs text-offwhite-400">Ensure camera permissions are granted in your browser.</p>
          </div>
        )}
      </div>

      {/* Control Toolbar */}
      <div className="p-4 bg-graphite-850 border-t border-graphite-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {!isActive ? (
            <button
              onClick={startCamera}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-tealaccent-600 hover:bg-tealaccent-500 text-graphite-950 font-medium text-xs transition shadow-sm"
            >
              <Camera className="w-4 h-4" />
              <span>Start Stream</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-graphite-800 hover:bg-graphite-700 text-offwhite-200 font-medium text-xs transition border border-graphite-700"
            >
              <CameraOff className="w-4 h-4" />
              <span>Stop Stream</span>
            </button>
          )}
        </div>

        <div className="flex items-center space-x-3 text-xs text-offwhite-400 font-mono">
          <span className="flex items-center space-x-1.5">
            <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-tealaccent-400 animate-pulse' : 'bg-graphite-600'}`} />
            <span>{isActive ? 'LIVE INFERENCE' : 'STANDBY'}</span>
          </span>
        </div>
      </div>
    </div>
  );
};
