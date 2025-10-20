param(
  [Parameter(Mandatory=$true,Position=0)][ValidateSet("start","stop","restart","status","remove")][string]$cmd,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$rest
)
$TaskName = 'PaperTradingBot'
$WorkDir  = 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'
$ScriptPS1= Join-Path $WorkDir 'start_bot.ps1'
$Exe      = 'C:\Program Files\PowerShell\7\pwsh.exe'
function New-StartScript([string]$line){
  Set-Content -Path $ScriptPS1 -Value '$ErrorActionPreference = "Continue"'
  Add-Content -Path $ScriptPS1 -Value "Set-Location '$WorkDir'"
  # Do not delete log at startup; Python runner writes a process header and trims on market close
  Add-Content -Path $ScriptPS1 -Value 'while ($true) {'
  Add-Content -Path $ScriptPS1 -Value "  $line 2>&1 | Tee-Object -FilePath '.\bot.log' -Append"
  Add-Content -Path $ScriptPS1 -Value '  Start-Sleep -Seconds 10'
  Add-Content -Path $ScriptPS1 -Value '}'
}
switch ($cmd) {
  'start' {
    if (-not $rest) { Write-Error 'Usage: start python runner.py -t .0065'; exit 1 }
    $line = ($rest -join ' ')
    if ($line -notmatch '^\s*python\b') { Write-Error 'First arg must be python'; exit 1 }
    New-StartScript $line
    try {
      Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
      $act = New-ScheduledTaskAction -Execute $Exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPS1`""
      $dailyTime = (Get-Date).Date.AddHours(9).AddMinutes(25)
      $wakeDaily = New-ScheduledTaskTrigger -Daily -At $dailyTime
      $trg = @($wakeDaily; New-ScheduledTaskTrigger -AtStartup; New-ScheduledTaskTrigger -AtLogOn)
      $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
      $set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew -WakeToRun
      Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg -Principal $principal -Settings $set -Description 'Paper trading bot' | Out-Null
    } catch {
      schtasks /Delete /TN $TaskName /F >$null 2>&1
      $tr="`"$Exe`" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPS1`""
      schtasks /Create /TN $TaskName /SC ONLOGON /TR $tr /RL LIMITED >$null
    }
    schtasks /Run /TN $TaskName >$null
    Write-Host "Started. Log: $WorkDir\bot.log"
  }
  'stop'    { schtasks /End /TN $TaskName >$null 2>&1; Write-Host 'Stopped (will start at next logon/boot unless removed).' }
  'restart' { schtasks /End /TN $TaskName >$null 2>&1; schtasks /Run /TN $TaskName >$null; Write-Host 'Restarted.' }
  'status'  { Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo }
  'remove'  { schtasks /End /TN $TaskName >$null 2>&1; schtasks /Delete /TN $TaskName /F >$null; Write-Host 'Removed task.' }
}
