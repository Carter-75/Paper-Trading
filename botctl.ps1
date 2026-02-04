# botctl.ps1 (patched v2 - bot + 24/7 dashboard)

param(
  [Parameter(Mandatory=$true,Position=0)]
  [ValidateSet('start','restart','stop','status','remove')]
  [string]$cmd,

  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$rest
)

$ErrorActionPreference = 'Stop'

$TaskName = 'PaperTradingBot'
$WorkDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotScript = Join-Path $WorkDir 'start_bot.ps1'
$DashScript = Join-Path $WorkDir 'start_dashboard.ps1'
$CmdFile  = Join-Path $WorkDir 'last_start_cmd.txt'

$Pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $Pwsh) { $Pwsh = 'C:\\Program Files\\PowerShell\\7\\pwsh.exe' }

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
  $content += "Set-Location '$WorkDir'"
  $content += '$env:SCHEDULED_TASK_MODE="1"'
  $content += '$env:BOT_TEE_LOG="1"'
  $content += ''
  $content += '# --- Sleep lock helpers (prevents sleep only while trading loop is active) ---'
  $content += 'Add-Type -Namespace Win32 -Name Power -MemberDefinition @"'
  $content += '  [DllImport("kernel32.dll", SetLastError=true)]'
  $content += '  public static extern uint SetThreadExecutionState(uint esFlags);'
  $content += '"@'
  $content += 'function Acquire-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000001) }'
  $content += 'function Release-AwakeLock { [void][Win32.Power]::SetThreadExecutionState(0x80000000) }'
  $content += ''
  $content += 'Add-Content -Path .\\bot.log -Value ("BOT_INIT " + (Get-Date).ToString("s") + " user=" + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)'
  $content += 'Acquire-AwakeLock'
  $content += 'try {'
  $content += '  while ($true) {'
  $content += '    $exitCode = 1'
  $content += '    try {'
  $content += "      & $BotCommand 2>&1 | Tee-Object -FilePath .\\bot.log -Append"
  $content += '      $exitCode = $LASTEXITCODE'
  $content += '    } catch { $exitCode = 1 }'
  $content += '    if ($exitCode -eq 0) { Add-Content -Path .\\bot.log -Value "Market closed - releasing sleep lock and exiting"; break }'
  $content += '    Add-Content -Path .\\bot.log -Value ("Bot crashed (exit " + $exitCode + ") - restarting in 10s")'
  $content += '    Start-Sleep -Seconds 10'
  $content += '  }'
  $content += '} finally { Release-AwakeLock }'

  Set-Content -Path $BotScript -Value ($content -join "`r`n") -Encoding UTF8
  Set-Content -Path $CmdFile -Value $BotCommand -Encoding UTF8
}

function CreateOrUpdateTask([string]$Name,[string]$Trigger,[string]$TimeOpt,[string]$ScriptPath) {
  $tr = '"' + $Pwsh + '" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $ScriptPath + '"'
  if (Task-Exists $Name) { Task-Delete $Name }

  if ($Trigger -eq 'DAILY') {
    schtasks /Create /TN $Name /SC DAILY /ST $TimeOpt /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  } elseif ($Trigger -eq 'ONLOGON') {
    schtasks /Create /TN $Name /SC ONLOGON /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  } elseif ($Trigger -eq 'ONSTART') {
    schtasks /Create /TN $Name /SC ONSTART /TR $tr /RL HIGHEST /RU SYSTEM /F | Out-Null
  }
}

function Ensure-Tasks([string]$BotCommand) {
  Require-Admin

  Write-BotStartScript $BotCommand

  # Dashboard task 24/7
  CreateOrUpdateTask 'PaperTradingDashboard' 'ONSTART' '' $DashScript

  # Bot tasks
  $dailyTime = Get-LocalTimeStringForEastern925
  CreateOrUpdateTask $TaskName 'ONLOGON' '' $BotScript
  CreateOrUpdateTask ($TaskName + '-Startup') 'ONSTART' '' $BotScript
  CreateOrUpdateTask ($TaskName + '-Daily925ET') 'DAILY' $dailyTime $BotScript

  return $dailyTime
}

switch ($cmd) {
  'start' {
    if (-not $rest -or $rest.Count -lt 1) { throw 'Usage: .\\botctl.ps1 start python -u runner.py' }
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
    Start-Sleep -Seconds 1

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
    foreach($n in @($TaskName, $TaskName+'-Startup', $TaskName+'-Daily925ET', 'PaperTradingDashboard')) {
      if (Task-Exists $n) { Task-Delete $n }
    }
    "OK: removed bot+dashboard tasks"
  }

  'status' {
    Require-Admin
    foreach($n in @($TaskName, $TaskName+'-Startup', $TaskName+'-Daily925ET', 'PaperTradingDashboard')) {
      "--- $n ---"
      if (Task-Exists $n) { schtasks /Query /TN $n /FO LIST /V } else { "(not found)" }
    }
  }
}

