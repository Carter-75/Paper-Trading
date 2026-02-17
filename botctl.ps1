
# botctl.ps1 (patched v7 - Variable Interpolation Fix)

param(
  [Parameter(Mandatory = $true, Position = 0)]
  [ValidateSet('start', 'restart', 'stop', 'status', 'remove')]
  [string]$cmd,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$rest
)

$ErrorActionPreference = 'Stop'

$TaskName = 'PaperTradingBot'
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Ensure WorkDir is absolute
$WorkDir = (Resolve-Path $WorkDir).Path

$BotScript = Join-Path $WorkDir 'start_bot.ps1'
$DashScript = Join-Path $WorkDir 'start_dashboard.ps1'
$CmdFile = Join-Path $WorkDir 'last_start_cmd.txt'

$Pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $Pwsh) { $Pwsh = 'C:\Program Files\PowerShell\7\pwsh.exe' }

# --- Resolve Python ONCE at generation time ---
$GlobalPy = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $GlobalPy) { 
    # Fallback/Guess if not in PATH
    if (Test-Path "C:\Python311\python.exe") { $GlobalPy = "C:\Python311\python.exe" }
    else { $GlobalPy = "python" } 
}

function Test-Administrator {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p = New-Object Security.Principal.WindowsPrincipal($id)
  return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Require-Admin {
  if (-not (Test-Administrator)) {
    throw "Run PowerShell as Administrator (UAC) for task management."
  }
}

function Task-Exists([string]$Name) {
  schtasks /Query /TN $Name *> $null
  return ($LASTEXITCODE -eq 0)
}

function Task-Stop([string]$Name) {
  schtasks /End /TN $Name *> $null
}

function Task-Delete([string]$Name) {
  schtasks /Delete /TN $Name /F *> $null
}

function Task-Run([string]$Name) {
  schtasks /Run /TN $Name *> $null
}

function Get-LocalTimeStringForEastern925 {
  $et = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
  $local = [System.TimeZoneInfo]::Local

  # Build an "Unspecified" DateTime for 9:25 AM ET so timezone conversion works reliably
  $todayEt = [DateTime]::SpecifyKind([DateTime]::Today.AddHours(9).AddMinutes(25), [DateTimeKind]::Unspecified)
  $todayUtc = [System.TimeZoneInfo]::ConvertTimeToUtc($todayEt, $et)
  $todayLocal = [System.TimeZoneInfo]::ConvertTimeFromUtc($todayUtc, $local)
  return $todayLocal.ToString('HH:mm')
}

function Write-BotStartScript([string]$BotCommand) {
  $content = @()
  $content += '$ErrorActionPreference = "Continue"'
  # Hardcode Working Directory
  $content += "Set-Location '$WorkDir'"
  $content += '$env:SCHEDULED_TASK_MODE="1"'
  $content += '$env:BOT_TEE_LOG="1"'
  $content += ''
  $content += '# --- Hardcoded Python Path from Generator ---'
  $content += '$py = "' + $GlobalPy + '"'
  $content += 'Add-Content -Path .\bot.log -Value ("BOT_INIT_SCRIPT " + (Get-Date).ToString("s") + " using python: $py")'
  $content += ''
  $content += '# --- Aggressive Takeover (Supervisor Killer) ---'
  $content += '$mutexName = "Global\PaperTradingBotLauncher"'
  $content += '$createdNew = $false'
  $content += 'try {'
  $content += '  $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew)'
  $content += '  if (-not $createdNew) {'
  $content += '    $msg = "BOT_LAUNCHER_BLOCKED: Old instance detected. Initiating SUPERVISOR KILL. " + (Get-Date).ToString("s")'
  $content += '    Write-Host $msg -ForegroundColor Yellow'
  $content += '    Add-Content -Path .\bot.log -Value $msg'
  $content += ''
  $content += '    # 1. Kill the PARENT PowerShell supervisor (start_bot.ps1)'
  $content += '    Get-CimInstance Win32_Process -Filter "Name LIKE ''%pwsh%'' OR Name LIKE ''%powershell%''" -ErrorAction SilentlyContinue |'
  $content += '    Where-Object { $_.CommandLine -match "start_bot.ps1" -and $_.ProcessId -ne $PID } |'
  $content += '    ForEach-Object {'
  $content += '       Add-Content -Path .\bot.log -Value ("TAKING OVER: Killing Old Supervisor PID " + $_.ProcessId)'
  $content += '       taskkill /F /PID $_.ProcessId | Out-Null'
  $content += '    }'
  $content += ''
  $content += '    # 2. Kill orphaned runner.py instances'
  $content += '    Get-CimInstance Win32_Process -Filter "Name=''python.exe''" -ErrorAction SilentlyContinue |'
  $content += '    Where-Object { $_.CommandLine -match "runner.py" -and $_.ProcessId -ne $PID } |'
  $content += '    ForEach-Object {'
  $content += '       Add-Content -Path .\bot.log -Value ("TAKING OVER: Killing Old Runner PID " + $_.ProcessId)'
  $content += '       taskkill /F /PID $_.ProcessId | Out-Null'
  $content += '    }'
  $content += '    Start-Sleep -Seconds 3'
  $content += ''
  $content += '    # Re-acquire mutex'
  $content += '    try { $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew) } catch {}'
  $content += '  }'
  $content += '} catch {'
  $content += '  $err = $_.Exception.Message'
  $content += '  Add-Content -Path .\bot.log -Value ("MUTEX_ERROR (Ignored for Takeover): " + $err)'
  $content += '}'
  $content += ''
  $content += '# --- Sleep lock helpers ---'
  $content += 'Add-Type -Namespace Win32 -Name Power -MemberDefinition @"'
  $content += '  [DllImport("kernel32.dll", SetLastError=true)]'
  $content += '  public static extern uint SetThreadExecutionState(uint esFlags);'
  $content += '"@'
  $content += 'function Acquire-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000001) }'
  $content += 'function Release-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000000) }'
  $content += ''
  $content += 'Add-Content -Path .\bot.log -Value ("BOT_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)'
  $content += 'Acquire-AwakeLock'
  $content += 'try {'
  $content += '  while ($true) {'
  $content += '    $exitCode = 1'
  $content += '    try {'
  
  # FIX: Interpolate $GlobalPy during generation
  $content += '      & "' + $GlobalPy + '" -u runner.py 2>&1 | Tee-Object -FilePath .\bot.log -Append'
  
  $content += '      $exitCode = $LASTEXITCODE'
  $content += '    } catch { $exitCode = 1 }'
  $content += '    if ($exitCode -eq 0) { Add-Content -Path .\bot.log -Value "Market closed - releasing sleep lock and exiting"; break }'
  $content += '    Add-Content -Path .\bot.log -Value ("Bot crashed (exit " + $exitCode + ") - restarting in 10s")'
  $content += '    Start-Sleep -Seconds 10'
  $content += '  }'
  $content += '} finally { Release-AwakeLock }'

  Set-Content -Path $BotScript -Value ($content -join "`r`n") -Encoding UTF8
  Set-Content -Path $CmdFile -Value $BotCommand -Encoding UTF8
}

function Write-DashStartScript {
  $content = @()
  $content += '# start_dashboard.ps1'
  $content += '# Runs dashboard.py 24/7 with restart-on-crash.'
  $content += '# Aggressive Takeover logic included.'
  $content += ''
  $content += '$ErrorActionPreference = "Continue"'
  # Hardcode Working Directory
  $content += "Set-Location '$WorkDir'"
  $content += ''
  $content += '$logPath = ".\dashboard.log"'
  $content += ''
  $content += '# --- Hardcoded Python Path from Generator ---'
  $content += '$py = "' + $GlobalPy + '"'
  $content += 'Add-Content -Path $logPath -Value ("DASH_INIT_SCRIPT " + (Get-Date).ToString("s") + " using python: $py")'
  $content += ''
  $content += '# --- Aggressive Takeover (Supervisor Killer) ---'
  $content += '$mutexName = "Global\PaperTradingDashboardLauncher"'
  $content += '$createdNew = $false'
  $content += 'try {'
  $content += '  $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew)'
  $content += '  if (-not $createdNew) {'
  $content += '    Add-Content -Path $logPath -Value ("DASH_LAUNCHER BLOCKED: Old instance detected. Initiating SUPERVISOR KILL. " + (Get-Date).ToString("s"))'
  $content += ''
  $content += '    # 1. Kill the PARENT PowerShell supervisor (start_dashboard.ps1)'
  $content += '    Get-CimInstance Win32_Process -Filter "Name LIKE ''%pwsh%'' OR Name LIKE ''%powershell%''" -ErrorAction SilentlyContinue |'
  $content += '    Where-Object { $_.CommandLine -match "start_dashboard.ps1" -and $_.ProcessId -ne $PID } |'
  $content += '    ForEach-Object {'
  $content += '       Add-Content -Path $logPath -Value ("TAKING OVER: Killing Old Dashboard Supervisor PID " + $_.ProcessId)'
  $content += '       taskkill /F /PID $_.ProcessId | Out-Null'
  $content += '    }'
  $content += ''
  $content += '    # 2. Kill invalid occupants of port 5000'
  $content += '    function Get-ListeningPids5000 {'
  $content += '      try {'
  $content += '        $lines = (netstat -ano | Select-String ":5000" | Select-String "LISTENING").Line'
  $content += '        if (-not $lines) { return @() }'
  $content += '        return $lines | ForEach-Object { [int](($_ -split "\s+")[-1]) } | Sort-Object -Unique'
  $content += '      } catch { return @() }'
  $content += '    }'
  $content += '    $pids = Get-ListeningPids5000'
  $content += '    foreach ($p in $pids) {'
  $content += '       Add-Content -Path $logPath -Value ("TAKING OVER: Killing Port 5000 occupant PID " + $p)'
  $content += '       taskkill /F /PID $p | Out-Null'
  $content += '    }'
  $content += ''
  $content += '    # 3. Kill other dashboard.py instances'
  $content += '    Get-CimInstance Win32_Process -Filter "Name=''python.exe''" -ErrorAction SilentlyContinue |'
  $content += '    Where-Object { $_.CommandLine -match "dashboard\.py" -and $_.ProcessId -ne $PID } |'
  $content += '    ForEach-Object {'
  $content += '       Add-Content -Path $logPath -Value ("TAKING OVER: Killing old dashboard PID " + $_.ProcessId)'
  $content += '       taskkill /F /PID $_.ProcessId | Out-Null'
  $content += '    }'
  $content += '    Start-Sleep -Seconds 3'
  $content += ''
  $content += '    # Re-acquire mutex'
  $content += '    try { $mutex = New-Object -TypeName System.Threading.Mutex -ArgumentList $true, $mutexName, ([ref]$createdNew) } catch {}'
  $content += '  }'
  $content += '}'
  $content += 'catch {'
  $content += '  Add-Content -Path $logPath -Value ("DASH_MUTEX_ERROR " + $_.Exception.Message)'
  $content += '}'
  $content += ''
  $content += 'Add-Content -Path $logPath -Value ("DASH_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)'
  $content += ''
  $content += 'while ($true) {'
  $content += '  # --- LOG TRUNCATION (Safety) ---'
  $content += '  try {'
  $content += '    if (Test-Path $logPath) {'
  $content += '      $logInfo = Get-Item $logPath'
  $content += '      if ($logInfo.Length -gt 1MB) {'
  $content += '        $tail = Get-Content $logPath -Tail 1000'
  $content += '        $tail | Set-Content $logPath'
  $content += '        Add-Content -Path $logPath -Value ("DASH_LOG_ROTATED " + (Get-Date).ToString("s"))'
  $content += '      }'
  $content += '    }'
  $content += '  } catch { }'
  $content += ''
  $content += '  try {'
  
  # FIX: Interpolate $GlobalPy during generation
  $content += '    & "' + $GlobalPy + '" -u dashboard.py 2>&1 | Tee-Object -FilePath $logPath -Append'
  
  $content += '    $exitCode = $LASTEXITCODE'
  $content += '  }'
  $content += '  catch {'
  $content += '    $exitCode = 1'
  $content += '    Add-Content -Path $logPath -Value ("Dashboard exception: " + $_.Exception.Message)'
  $content += '  }'
  $content += ''
  $content += '  if ($exitCode -eq 0) {'
  $content += '      Add-Content -Path $logPath -Value ("Dashboard exited cleanly (0).")'
  $content += '  } else {'
  $content += '      Add-Content -Path $logPath -Value ("Dashboard exited (" + $exitCode + ") - restarting in 5s")'
  $content += '  }'
  $content += '  Start-Sleep -Seconds 5'
  $content += '}'
  
  Set-Content -Path $DashScript -Value ($content -join "`r`n") -Encoding UTF8
}

function CreateOrUpdateTask([string]$Name, [string]$Trigger, [string]$TimeOpt, [string]$ScriptPath) {
  $tr = '"' + $Pwsh + '" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $ScriptPath + '"'
  if (Task-Exists $Name) { Task-Delete $Name }

  if ($Trigger -eq 'DAILY') {
    schtasks /Create /TN $Name /SC DAILY /ST $TimeOpt /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  }
  elseif ($Trigger -eq 'ONLOGON') {
    schtasks /Create /TN $Name /SC ONLOGON /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  }
  elseif ($Trigger -eq 'ONSTART') {
    schtasks /Create /TN $Name /SC ONSTART /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  }
}

function Ensure-Tasks([string]$BotCommand) {
  Require-Admin

  Write-BotStartScript $BotCommand
  Write-DashStartScript

  # Dashboard task 24/7
  CreateOrUpdateTask 'PaperTradingDashboard' 'ONSTART' '' $DashScript

  # Bot tasks
  $dailyTime = Get-LocalTimeStringForEastern925
  CreateOrUpdateTask $TaskName 'ONLOGON' '' $BotScript
  CreateOrUpdateTask ($TaskName + '-Startup') 'ONSTART' '' $BotScript
  CreateOrUpdateTask ($TaskName + '-Daily925ET') 'DAILY' $dailyTime $BotScript

  return $dailyTime
}

function Status {
  Require-Admin
  Write-Host "--- Paper-Trading Status ---" -ForegroundColor Cyan
  
  $botProc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'runner.py' }
  if ($botProc) {
    Write-Host "Bot: RUNNING (PID: $($botProc.ProcessId))" -ForegroundColor Green
  }
  else {
    Write-Host "Bot: NOT RUNNING" -ForegroundColor Red
  }

  $dashProc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'dashboard.py' }
  if ($dashProc) {
    Write-Host "Dashboard: RUNNING (PID: $($dashProc.ProcessId))" -ForegroundColor Green
  }
  else {
    Write-Host "Dashboard: NOT RUNNING" -ForegroundColor Red
  }

  Write-Host "`n--- Scheduled Tasks ---" -ForegroundColor Cyan
  foreach ($n in @($TaskName, $TaskName + '-Startup', $TaskName + '-Daily925ET', 'PaperTradingDashboard')) {
    if (Task-Exists $n) {
      $info = schtasks /Query /TN $n /FO LIST /V | Select-String "Status:" | ForEach-Object { $_.ToString().Trim() }
      Write-Host "Task $($n): $info"
    }
    else {
      Write-Host "Task $($n): NOT FOUND" -ForegroundColor Gray
    }
  }
}

switch ($cmd) {
  'start' {
    if (-not $rest -or $rest.Count -lt 1) { throw 'Usage: .\botctl.ps1 start python -u runner.py' }
    $line = ($rest -join ' ')
    if ($line -notmatch '^\s*python\b') { throw 'Command must start with python' }

    $dailyTime = Ensure-Tasks $line

    Task-Run 'PaperTradingDashboard'
    Task-Run $TaskName

    "OK: started. Daily bot wake (local time) = $dailyTime"
  }

  'restart' {
    $line = $null
    if ($rest -and $rest.Count -gt 0) { $line = ($rest -join ' ') }
    elseif (Test-Path $CmdFile) { $line = (Get-Content -Path $CmdFile -Raw).Trim() }
    if (-not $line) { throw 'No saved command found. Run start once with the command.' }

    $dailyTime = Ensure-Tasks $line

    Task-Stop $TaskName
    Task-Stop ($TaskName + '-Startup')
    Task-Stop ($TaskName + '-Daily925ET')
    
    Write-Host "NUCLEAR: Force killing bot-related python processes..." -ForegroundColor Yellow
    
    # Target specific scripts to avoid collateral damage
    $targets = @('runner.py', 'dashboard.py', 'start_dashboard.ps1', 'start_bot.ps1')
    
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | 
    ForEach-Object {
      $p = $_
      $procCmdLimit = $p.CommandLine
      if ($null -eq $procCmdLimit) { return }
        
      $matchFound = $false
      foreach ($t in $targets) {
        if ($procCmdLimit -match $t) { 
          $matchFound = $true 
          break 
        }
      }
        
      if ($matchFound) {
        Write-Host "Killing PID $($p.ProcessId): $procCmdLimit" -ForegroundColor Gray
        # Use taskkill for better force-kill capability (especially against SYSTEM processes)
        taskkill /F /PID $p.ProcessId | Out-Null
      }
    }
    
    Start-Sleep -Seconds 2

    Task-Run 'PaperTradingDashboard'
    Task-Run $TaskName

    "OK: restarted. Daily bot wake (local time) = $dailyTime"
  }

  'stop' {
    Require-Admin
    Task-Stop $TaskName
    "OK: stopped bot (tasks remain). Dashboard stays up."
  }

  'remove' {
    Require-Admin
    foreach ($n in @($TaskName, $TaskName + '-Startup', $TaskName + '-Daily925ET', 'PaperTradingDashboard')) {
      if (Task-Exists $n) { Task-Delete $n }
    }
    "OK: removed bot+dashboard tasks"
  }

  'status' {
    Status
  }
}
