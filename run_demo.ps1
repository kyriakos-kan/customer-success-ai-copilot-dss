# Customer Success AI Copilot - laptop demo launcher (Windows PowerShell)
# Usage, from the project folder:   .\run_demo.ps1
# If scripts are blocked:            powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv-demo")) {
    Write-Host "First run: creating Python 3.12 environment (a few minutes)..." -ForegroundColor Cyan
    py -3.12 -m venv .venv-demo
    .\.venv-demo\Scripts\python.exe -m pip install --upgrade pip
    .\.venv-demo\Scripts\python.exe -m pip install -r requirements\demo.txt
}

Write-Host "Starting the demo at http://localhost:8501  (Ctrl+C to stop)" -ForegroundColor Green
.\.venv-demo\Scripts\python.exe -m streamlit run demo\app.py --server.headless false
