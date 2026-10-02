import { HealthCheckResponse, PredictionResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkBackendHealth(): Promise<HealthCheckResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function predictEmotionFromBlob(blob: Blob): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Prediction request failed: ${response.statusText}`);
  }

  return response.json();
}
