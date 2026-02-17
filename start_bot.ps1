$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Portfolio-Websites (Mostly)\Paper-Trading'
$env:SCHEDULED_TASK_MODE="1"
$env:BOT_TEE_LOG="1"

# --- Hardcoded Python Path from Generator ---
$py = "C:\Python311\python.exe"
Add-Content -Path .\bot.log -Value ("BOT_INIT_SCRIPT " + (Get-Date).ToString("s") + " using python: $py")

# --- Aggressive Takeover (Supervisor Killer) ---
$mutexName = "Global\PaperTradingBotLauncher"
$createdNew = $false
try {
  $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew)
  if (-not $createdNew) {
    $msg = "BOT_LAUNCHER_BLOCKED: Old instance detected. Initiating SUPERVISOR KILL. " + (Get-Date).ToString("s")
    Write-Host $msg -ForegroundColor Yellow
    Add-Content -Path .\bot.log -Value $msg

    # 1. Kill the PARENT PowerShell supervisor (start_bot.ps1)
    Get-CimInstance Win32_Process -Filter "Name LIKE '%pwsh%' OR Name LIKE '%powershell%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "start_bot.ps1" -and $_.ProcessId -ne $PID } |
    ForEach-Object {
       Add-Content -Path .\bot.log -Value ("TAKING OVER: Killing Old Supervisor PID " + $_.ProcessId)
       taskkill /F /PID $_.ProcessId | Out-Null
    }

    # 2. Kill orphaned runner.py instances
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "runner.py" -and $_.ProcessId -ne $PID } |
    ForEach-Object {
       Add-Content -Path .\bot.log -Value ("TAKING OVER: Killing Old Runner PID " + $_.ProcessId)
       taskkill /F /PID $_.ProcessId | Out-Null
    }
    Start-Sleep -Seconds 3

    # Re-acquire mutex
    try { $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew) } catch {}
  }
} catch {
  $err = $_.Exception.Message
  Add-Content -Path .\bot.log -Value ("MUTEX_ERROR (Ignored for Takeover): " + $err)
}

# --- Sleep lock helpers ---
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
"@
function Acquire-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000001) }
function Release-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000000) }

Add-Content -Path .\bot.log -Value ("BOT_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
Acquire-AwakeLock
try {
  while ($true) {
    $exitCode = 1
    try {
      & "C:\Python311\python.exe" -u runner.py 2>&1 | Tee-Object -FilePath .\bot.log -Append
      $exitCode = $LASTEXITCODE
    } catch { $exitCode = 1 }
    if ($exitCode -eq 0) { Add-Content -Path .\bot.log -Value "Market closed - releasing sleep lock and exiting"; break }
    Add-Content -Path .\bot.log -Value ("Bot crashed (exit " + $exitCode + ") - restarting in 10s")
    Start-Sleep -Seconds 10
  }
} finally { Release-AwakeLock }
