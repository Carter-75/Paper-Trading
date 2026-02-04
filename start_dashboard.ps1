# start_dashboard.ps1
# Runs dashboard.py 24/7 with restart-on-crash.
# Singleton launcher (mutex) to prevent duplicates.
# If port 5000 is occupied, FORCE-KILL the occupant and take over.

$ErrorActionPreference = "Continue"
Set-Location "$PSScriptRoot"

$logPath = ".\dashboard.log"

# --- Singleton guard ---
$mutexName = "Global\\PaperTradingDashboardLauncher"
$createdNew = $false
try {
  $mutex = New-Object System.Threading.Mutex($true, $mutexName, [ref]$createdNew)
  if (-not $createdNew) {
    Add-Content -Path $logPath -Value ("DASH_LAUNCHER already running; exiting. " + (Get-Date).ToString('s'))
    exit 0
  }
} catch {
  # If mutex fails, continue (better to run than to be down)
}

Add-Content -Path $logPath -Value ("DASH_INIT " + (Get-Date).ToString('s') + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)

function Get-ListeningPids5000 {
  try {
    $lines = (netstat -ano | Select-String ":5000" | Select-String "LISTENING").Line
    if (-not $lines) { return @() }
    return $lines | ForEach-Object { [int](($_ -split "\s+")[-1]) } | Sort-Object -Unique
  } catch {
    return @()
  }
}

function Kill-DashboardPy {
  try {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'dashboard\.py' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  } catch { }
}

while ($true) {
  # Take over port 5000 no matter what is holding it
  $pids = Get-ListeningPids5000
  if ($pids.Count -gt 0) {
    Add-Content -Path $logPath -Value ("DASH_TAKEOVER killing port 5000 occupant pids=" + ($pids -join ',') + " " + (Get-Date).ToString('s'))
    $pids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
  }

  # Clean up any stray dashboard.py python processes
  Kill-DashboardPy

  try {
    python -u dashboard.py 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
  } catch {
    $exitCode = 1
    Add-Content -Path $logPath -Value ("Dashboard exception: " + $_.Exception.Message)
  }

  Add-Content -Path $logPath -Value ("Dashboard exited (" + $exitCode + ") - restarting in 5s")
  Start-Sleep -Seconds 5
}
