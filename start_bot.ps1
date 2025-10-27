$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'
$env:SCHEDULED_TASK_MODE='1'
$env:BOT_TEE_LOG='1'
Add-Content -Path '.\bot.log' -Value ('INIT ' + (Get-Date).ToString('s') + ' user=' + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
while ($true) {
  python -u runner.py -t 0.25 -m 100 2>&1 | Tee-Object -FilePath '.\bot.log' -Append
  $exitCode = $LASTEXITCODE
  if ($exitCode -eq 0) {
    # Exit code 0 = market closed cleanly, stop and let scheduled task handle next run
    Add-Content -Path '.\bot.log' -Value "Market closed - task will restart automatically when market opens"
    exit 0
  } else {
    # Non-zero exit = crash, restart quickly
    Add-Content -Path '.\bot.log' -Value "Bot crashed (exit $exitCode) - restarting in 10s"
    Start-Sleep -Seconds 10
  }
}
