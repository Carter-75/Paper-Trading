param(
  [Parameter(Mandatory=$true,Position=0)][ValidateSet("start","pause","resume","stop","restart","status","remove","stop-forever")][string]$cmd,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$rest
)
$TaskName = 'PaperTradingBot'
$WorkDir  = 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'
$ScriptPS1= Join-Path $WorkDir 'start_bot.ps1'
$Exe      = 'C:\Program Files\PowerShell\7\pwsh.exe'
$CmdFile  = Join-Path $WorkDir 'last_start_cmd.txt'
function New-StartScript([string]$line){
  Set-Content -Path $ScriptPS1 -Value '$ErrorActionPreference = "Continue"'
  Add-Content -Path $ScriptPS1 -Value "Set-Location '$WorkDir'"
  # Hint the Python runner we're in a scheduled-task context so it can exit during long market-closed windows
  Add-Content -Path $ScriptPS1 -Value '$env:SCHEDULED_TASK_MODE=''1'''
  # Avoid double-writes: tell Python to disable file logging when PowerShell Tee is active
  Add-Content -Path $ScriptPS1 -Value '$env:BOT_TEE_LOG=''1'''
  # Diagnostic: write an INIT line so we can confirm the task invoked the script
  Add-Content -Path $ScriptPS1 -Value "Add-Content -Path '.\bot.log' -Value ('INIT ' + (Get-Date).ToString('s') + ' user=' + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)"
  # Do not delete log at startup; Python runner writes a process header and trims on market close
  Add-Content -Path $ScriptPS1 -Value 'while ($true) {'
  Add-Content -Path $ScriptPS1 -Value "  $line 2>&1 | Tee-Object -FilePath '.\bot.log' -Append"
  Add-Content -Path $ScriptPS1 -Value '  Start-Sleep -Seconds 10'
  Add-Content -Path $ScriptPS1 -Value '}'
  try { Set-Content -Path $CmdFile -Value $line } catch {}
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
  'pause'  {
    schtasks /End /TN $TaskName >$null 2>&1
    schtasks /Change /TN $TaskName /DISABLE >$null 2>&1
    Write-Host 'Paused task (disabled triggers and stopped current run).'
  }
  'resume' {
    schtasks /Change /TN $TaskName /ENABLE >$null 2>&1
    schtasks /Run /TN $TaskName >$null 2>&1
    Write-Host 'Resumed task (enabled and started).'
  }
  'stop'    { schtasks /End /TN $TaskName >$null 2>&1; Write-Host 'Stopped (will start at next logon/boot unless removed).' }
  'restart' {
    $line = $null
    if ($rest -and $rest.Count -gt 0) { $line = ($rest -join ' ') }
    elseif (Test-Path $CmdFile) { try { $line = Get-Content -Path $CmdFile -Raw } catch {} }
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
      if (-not $line) { Write-Error "Task missing and no command provided. Usage: restart python -u runner.py ..."; break }
      if ($line -notmatch '^\s*python\b') { Write-Error 'First arg must be python'; break }
      New-StartScript $line
      try {
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
      schtasks /Run /TN $TaskName >$null 2>&1
      Write-Host 'Created and started.'
      break
    }
    if ($line) {
      if ($line -notmatch '^\s*python\b') { Write-Error 'First arg must be python'; break }
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
    }
    schtasks /End /TN $TaskName >$null 2>&1
    schtasks /Change /TN $TaskName /ENABLE >$null 2>&1
    schtasks /Run /TN $TaskName >$null 2>&1
    Write-Host 'Restarted.'
  }
  'status'  { Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Get-ScheduledTaskInfo }
  'remove'  { schtasks /End /TN $TaskName >$null 2>&1; schtasks /Delete /TN $TaskName /F >$null; Write-Host 'Removed task.' }
  'stop-forever' { schtasks /End /TN $TaskName >$null 2>&1; schtasks /Delete /TN $TaskName /F >$null; Write-Host 'Stopped and removed task (forever).' }
}
