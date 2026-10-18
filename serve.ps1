# Run the service from a plain PowerShell window (no uv needed once .venv-local exists).
#   .\serve.ps1              -> http://127.0.0.1:8000 (Ctrl-C to stop)
#   .\serve.ps1 -Port 8001
param([int]$Port = 8000)

Set-Location $PSScriptRoot
$env:HF_HUB_OFFLINE = "1"   # weights are cached; never touch the internet while processing video
$py = Join-Path $PSScriptRoot ".venv-local\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }

if (-not (Test-Path .env)) {
    $key = -join ((48..57) + (97..122) | Get-Random -Count 40 | ForEach-Object { [char]$_ })
    (Get-Content .env.example) -replace '^API_KEY=.*', "API_KEY=$key" | Out-File .env -Encoding ascii
    Write-Host "Created .env with a new API key"
}

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) { $ollama = Get-Item "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ErrorAction SilentlyContinue }
if (-not (Get-Process ollama -ErrorAction SilentlyContinue) -and $ollama) {
    Start-Process $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

& $py -m uvicorn app.main:app --host 0.0.0.0 --port $Port --workers 1
