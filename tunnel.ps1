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
while ($true) {
    ssh -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R "80:127.0.0.1:$Port" nokey@localhost.run
    Write-Host "tunnel dropped, reconnecting in 3 s (the URL will change)..."
    Start-Sleep -Seconds 3
}
