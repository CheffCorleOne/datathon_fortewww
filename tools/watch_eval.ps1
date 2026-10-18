# Live terminal dashboard for an evaluation run started with tools/eval.py (reads runs/eval_<tag>.log).
#   .\tools\watch_eval.ps1 -Tag v6
param([string]$Tag = "v6", [int]$Total = 40, [int]$Refresh = 2)

$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $root "runs\eval_$Tag.log"
$start = Get-Date

function Bar([double]$frac, [int]$width = 40) {
    $n = [Math]::Round($frac * $width)
    return ("#" * $n) + ("-" * ($width - $n))
}

while ($true) {
    $lines = @()
    if (Test-Path $log) { $lines = @(Get-Content $log -ErrorAction SilentlyContinue | Where-Object { $_ -match '^(OK|ERR) ' }) }
    $tp = 0; $fn = 0; $fp = 0; $tn = 0; $secs = @(); $types = @{}
    foreach ($l in $lines) {
        $gt = $l -match 'gt=V'
        $pred = $l -match 'pred=V'
        if ($gt -and $pred) { $tp++ } elseif ($gt) { $fn++ } elseif ($pred) { $fp++ } else { $tn++ }
        if ($l -match '\[\s*([\d\.]+)s\]') { $secs += [double]$Matches[1] }
        foreach ($m in [regex]::Matches($l, '\d+\.\d+s ([a-z_]+) \(')) { $t = $m.Groups[1].Value; $types[$t] = 1 + [int]$types[$t] }
    }
    $done = $lines.Count
    $acc = if ($done) { ($tp + $tn) / $done } else { 0 }
    $recall = if ($tp + $fn) { $tp / ($tp + $fn) } else { 0 }
    $avg = if ($secs.Count) { ($secs | Measure-Object -Average).Average } else { 0 }
    $eta = [TimeSpan]::FromSeconds([Math]::Max(0, ($Total - $done) * $avg))
    $finished = (Test-Path $log) -and (Select-String -Path $log -Pattern "videos=" -Quiet)

    Clear-Host
    Write-Host ""
    Write-Host "  EVAL $Tag" -ForegroundColor Cyan -NoNewline
    Write-Host ("   elapsed {0:mm\:ss}" -f ((Get-Date) - $start)) -ForegroundColor DarkGray
    Write-Host ("  [{0}] {1}/{2}" -f (Bar ($done / $Total)), $done, $Total)
    Write-Host ""
    Write-Host ("  accuracy  {0,6:P1}" -f $acc) -ForegroundColor $(if ($acc -ge 0.75) { "Green" } elseif ($acc -ge 0.6) { "Yellow" } else { "Red" })
    Write-Host ("  recall    {0,6:P1}   (violations found)" -f $recall)
    Write-Host ("  TP {0,-3} FN {1,-3} FP {2,-3} TN {3,-3}" -f $tp, $fn, $fp, $tn)
    Write-Host ("  avg {0:N0} s/video   ETA {1:mm\:ss}" -f $avg, $eta) -ForegroundColor DarkGray
    if ($types.Count) {
        Write-Host ("  types: " + (($types.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "  ")) -ForegroundColor DarkGray
    }
    Write-Host ""
    foreach ($l in ($lines | Select-Object -Last 14)) {
        $short = if ($l.Length -gt 118) { $l.Substring(0, 118) + "..." } else { $l }
        Write-Host ("  " + $short) -ForegroundColor $(if ($l.StartsWith("OK")) { "Green" } else { "Red" })
    }
    if ($finished) {
        Write-Host ""
        Get-Content $log | Where-Object { $_ -match 'videos=|timecode' } | ForEach-Object { Write-Host ("  " + $_) -ForegroundColor Cyan }
        break
    }
    Start-Sleep -Seconds $Refresh
}
