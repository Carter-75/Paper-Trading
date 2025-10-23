$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'
$env:SCHEDULED_TASK_MODE='1'
$env:BOT_TEE_LOG='1'
Add-Content -Path '.\bot.log' -Value ('INIT ' + (Get-Date).ToString('s') + ' user=' + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
while ($true) {
  python -u runner.py -t 1 -m 100 2>&1 | Tee-Object -FilePath '.\bot.log' -Append
  Start-Sleep -Seconds 10
}
