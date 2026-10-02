# PowerShell environment setup script
Write-Host "Setting up Python virtual environment and dependencies..." -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Created .venv" -ForegroundColor Green
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

Write-Host "`nSetting up frontend dependencies..." -ForegroundColor Cyan
Set-Location frontend
npm install
Set-Location ..

Write-Host "`nEnvironment setup completed successfully!" -ForegroundColor Green
