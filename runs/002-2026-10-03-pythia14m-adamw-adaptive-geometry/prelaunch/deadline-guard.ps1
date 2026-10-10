param([string]$PodId, [string]$DeadlineUtc, [string]$CliPath, [string]$ReceiptPath)
$ErrorActionPreference = 'Stop'
$taskDeadline = [DateTimeOffset]::Parse($DeadlineUtc)
while ([DateTimeOffset]::UtcNow -lt $taskDeadline) {
    $remainingSeconds = ($taskDeadline - [DateTimeOffset]::UtcNow).TotalSeconds
    Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min(60, [Math]::Ceiling($remainingSeconds))))
}
# Stop rather than delete at the deadline so recovery artifacts survive a lost controller.
$guardOutput = & $CliPath pod stop $PodId 2>&1 | Out-String
[ordered]@{pod_id=$PodId; deadline=$DeadlineUtc; action='stop'; exit_code=$LASTEXITCODE; output=$guardOutput; checked_at=[DateTimeOffset]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
