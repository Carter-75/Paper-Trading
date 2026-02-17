# start_dashboard.ps1
# Runs dashboard.py 24/7 with restart-on-crash.
# Aggressive Takeover logic included.

$ErrorActionPreference = "Continue"
Set-Location 'C:\Users\carte\OneDrive\Desktop\Code\Portfolio-Websites (Mostly)\Paper-Trading'

$logPath = ".\dashboard.log"

# --- Hardcoded Python Path from Generator ---
$py = "C:\Python311\python.exe"
Add-Content -Path $logPath -Value ("DASH_INIT_SCRIPT " + (Get-Date).ToString("s") + " using python: $py")

# --- Aggressive Takeover (Supervisor Killer) ---
$mutexName = "Global\PaperTradingDashboardLauncher"
$createdNew = $false
try {
  $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew)
  if (-not $createdNew) {
    Add-Content -Path $logPath -Value ("DASH_LAUNCHER BLOCKED: Old instance detected. Initiating SUPERVISOR KILL. " + (Get-Date).ToString("s"))

    # 1. Kill the PARENT PowerShell supervisor (start_dashboard.ps1)
    Get-CimInstance Win32_Process -Filter "Name LIKE '%pwsh%' OR Name LIKE '%powershell%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "start_dashboard.ps1" -and $_.ProcessId -ne $PID } |
    ForEach-Object {
       Add-Content -Path $logPath -Value ("TAKING OVER: Killing Old Dashboard Supervisor PID " + $_.ProcessId)
       taskkill /F /PID $_.ProcessId | Out-Null
    }

    # 2. Kill invalid occupants of port 5000
    function Get-ListeningPids5000 {
      try {
        $lines = (netstat -ano | Select-String ":5000" | Select-String "LISTENING").Line
        if (-not $lines) { return @() }
        return $lines | ForEach-Object { [int](($_ -split "\s+")[-1]) } | Sort-Object -Unique
      } catch { return @() }
    }
    $pids = Get-ListeningPids5000
    foreach ($p in $pids) {
       Add-Content -Path $logPath -Value ("TAKING OVER: Killing Port 5000 occupant PID " + $p)
       taskkill /F /PID $p | Out-Null
    }

    # 3. Kill other dashboard.py instances
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "dashboard\.py" -and $_.ProcessId -ne $PID } |
    ForEach-Object {
       Add-Content -Path $logPath -Value ("TAKING OVER: Killing old dashboard PID " + $_.ProcessId)
       taskkill /F /PID $_.ProcessId | Out-Null
    }
    Start-Sleep -Seconds 3

    # Re-acquire mutex
    try { $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew) } catch {}
  }
}
catch {
  Add-Content -Path $logPath -Value ("DASH_MUTEX_ERROR " + $_.Exception.Message)
}

Add-Content -Path $logPath -Value ("DASH_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)

while ($true) {
  # --- LOG TRUNCATION (Safety) ---
  try {
    if (Test-Path $logPath) {
      $logInfo = Get-Item $logPath
      if ($logInfo.Length -gt 1MB) {
        $tail = Get-Content $logPath -Tail 1000
        $tail | Set-Content $logPath
        Add-Content -Path $logPath -Value ("DASH_LOG_ROTATED " + (Get-Date).ToString("s"))
      }
    }
  } catch { }

  try {
    & "C:\Python311\python.exe" -u dashboard.py 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
  }
  catch {
    $exitCode = 1
    Add-Content -Path $logPath -Value ("Dashboard exception: " + $_.Exception.Message)
  }

  if ($exitCode -eq 0) {
      Add-Content -Path $logPath -Value ("Dashboard exited cleanly (0).")
  } else {
      Add-Content -Path $logPath -Value ("Dashboard exited (" + $exitCode + ") - restarting in 5s")
  }
  Start-Sleep -Seconds 5
}
