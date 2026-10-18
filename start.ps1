# Start the service locally and (optionally) expose it with a Cloudflare quick tunnel.
#   .\start.ps1            -> http://127.0.0.1:8000
#   .\start.ps1 -Tunnel    -> also prints a public https://*.trycloudflare.com URL
param([switch]$Tunnel, [int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")

if (-not (Test-Path .env)) {
    $key = -join ((48..57) + (97..122) | Get-Random -Count 40 | ForEach-Object { [char]$_ })
    (Get-Content .env.example) -replace '^API_KEY=.*', "API_KEY=$key" | Set-Content .env -Encoding utf8
    Write-Host "Created .env with a new API key"
}

if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

if ($Tunnel) {
    Start-Process cloudflared -ArgumentList "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:$Port" `
        -RedirectStandardError "$PSScriptRoot\runs\tunnel.log" -WindowStyle Hidden
    Write-Host "Tunnel starting; public URL will appear in runs\tunnel.log (trycloudflare.com)"
}

uv run uvicorn app.main:app --host 0.0.0.0 --port $Port --workers 1
