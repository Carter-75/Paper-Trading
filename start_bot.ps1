$ErrorActionPreference = "Continue"
Set-Location "$PSScriptRoot"
$env:SCHEDULED_TASK_MODE="1"

# --- Singleton guard (prevents multiple scheduled-task triggers from spawning duplicate bot loops) ---
$mutexName = "Global\\PaperTradingBotLauncher"
$createdNew = $false
try {
  $mutex = New-Object System.Threading.Mutex($true, $mutexName, [ref]$createdNew)
  if (-not $createdNew) {
    Add-Content -Path .\bot.log -Value ("BOT_LAUNCHER_ALREADY_RUNNING " + (Get-Date).ToString('s'))
    exit 0
  }
} catch {
  # continue if mutex fails
}

# --- Sleep lock helpers (prevents sleep only while trading loop is active) ---
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
"@
function Acquire-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000001) }
function Release-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000000) }

Add-Content -Path .\bot.log -Value ("BOT_INIT " + (Get-Date).ToString('s') + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
Acquire-AwakeLock
try {
  while ($true) {
    $exitCode = 1
    try {
      Add-Content -Path .\bot.log -Value ("BOT_RUN " + (Get-Date).ToString('s') + " launching python -u runner.py")
      # Use cmd redirection so exit codes are reliable (PowerShell pipelines can mask them)
      cmd /c "python -u runner.py >> bot.log 2>&1"
      $exitCode = $LASTEXITCODE
    } catch {
      $exitCode = 1
      Add-Content -Path .\bot.log -Value ("BOT_RUN_EXCEPTION " + (Get-Date).ToString('s') + " " + $_.Exception.Message)
    }

    if ($exitCode -eq 0) {
      Add-Content -Path .\bot.log -Value ("BOT_EXIT_0 " + (Get-Date).ToString('s') + " Market closed - releasing sleep lock and exiting")
      break
    }

    Add-Content -Path .\bot.log -Value ("BOT_CRASH " + (Get-Date).ToString('s') + " exit=" + $exitCode + " - restarting in 10s")
    Start-Sleep -Seconds 10
  }
} finally {
  Release-AwakeLock
}
