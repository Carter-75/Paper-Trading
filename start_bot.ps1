$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Portfolio-Websites (Mostly)\Paper-Trading'
$env:SCHEDULED_TASK_MODE="1"
$env:BOT_TEE_LOG="1"

# --- Sleep lock helpers (prevents sleep only while trading loop is active) ---
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
"@
function Acquire-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000001) }
function Release-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000000) }

Add-Content -Path .\\bot.log -Value ("BOT_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
Acquire-AwakeLock
try {
  while ($true) {
    $exitCode = 1
    try {
      & python -u runner.py 2>&1 | Tee-Object -FilePath .\\bot.log -Append
      $exitCode = $LASTEXITCODE
    } catch { $exitCode = 1 }
    if ($exitCode -eq 0) { Add-Content -Path .\\bot.log -Value "Market closed - releasing sleep lock and exiting"; break }
    Add-Content -Path .\\bot.log -Value ("Bot crashed (exit " + $exitCode + ") - restarting in 10s")
    Start-Sleep -Seconds 10
  }
} finally { Release-AwakeLock }
