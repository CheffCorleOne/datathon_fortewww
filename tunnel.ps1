# Public HTTPS URL for the local service without opening ports.
#   .\tunnel.ps1            -> localhost.run (no account; URL changes on every reconnect)
#   .\tunnel.ps1 -Ngrok     -> ngrok with your static domain (needs `ngrok config add-authtoken` once
#                              and NGROK_DOMAIN in .env, e.g. NGROK_DOMAIN=my-team.ngrok-free.app)
# Cloudflare quick tunnels need outbound TCP/UDP 7844, which is blocked on some networks.
param([switch]$Ngrok, [int]$Port = 8000)

Set-Location $PSScriptRoot
if ($Ngrok) {
    $domain = ((Get-Content .env | Select-String "^NGROK_DOMAIN=").Line -split "=", 2)[1]
    while ($true) {
        ngrok http --url=$domain "127.0.0.1:$Port"
        Write-Host "ngrok exited, restarting in 3 s..."
        Start-Sleep -Seconds 3
    }
}
# localhost.run drops idle tunnels ("tunnel activity timeout"): a background job pings the public URL.
$log = Join-Path $PSScriptRoot "runs\tunnel_url.txt"
Start-Job -ArgumentList $log -ScriptBlock {
    param($log)
    while ($true) {
        Start-Sleep -Seconds 45
        $u = Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($u) { try { Invoke-WebRequest -UseBasicParsing "$u/healthz" -TimeoutSec 10 | Out-Null } catch {} }
    }
} | Out-Null
while ($true) {
    ssh -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R "80:127.0.0.1:$Port" nokey@localhost.run 2>&1 |
        ForEach-Object {
            $_
            if ($_ -match "(https://[a-z0-9]+\.lhr\.life)") { $Matches[1] | Out-File $log -Append -Encoding ascii }
        }
    Write-Host "tunnel dropped, reconnecting in 3 s (the URL will change)..."
    Start-Sleep -Seconds 3
}
