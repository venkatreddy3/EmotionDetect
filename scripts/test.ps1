# Run All Unit and Integration Tests
Write-Host "Running Python Test Suite..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "`nRunning Frontend Lint & Type Checks..." -ForegroundColor Cyan
Set-Location frontend
npm run lint
npm run build
Set-Location ..
