export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceEmotionPrediction {
  face_id: number;
  bounding_box: BoundingBox;
  dominant_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface PredictionResponse {
  success: boolean;
  num_faces_detected: number;
  predictions: FaceEmotionPrediction[];
  inference_time_ms: number;
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  model_path?: string;
}

export type EmotionType =
  | 'Angry'
  | 'Disgust'
  | 'Fear'
  | 'Happy'
  | 'Sad'
  | 'Surprise'
  | 'Neutral';
