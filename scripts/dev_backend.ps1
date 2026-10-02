# Launch FastAPI Backend in Development Mode with Live Reload
Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Cyan
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
