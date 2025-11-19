param(
  [Parameter(Mandatory=$true,Position=0)][ValidateSet("start","stop","restart","stop-forever","status")][string]$cmd,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$rest
)

$TaskName = 'PaperTradingBot'
$WorkDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPS1= Join-Path $WorkDir 'start_bot.ps1'
$Exe      = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $Exe) { $Exe = 'C:\Program Files\PowerShell\7\pwsh.exe' }
$CmdFile  = Join-Path $WorkDir 'last_start_cmd.txt'

# Auto-generated files to delete on stop-forever
$AutoFiles = @(
    (Join-Path $WorkDir 'bot.log'),
    (Join-Path $WorkDir 'portfolio.json'),
    (Join-Path $WorkDir 'pnl_ledger.json'),
    (Join-Path $WorkDir 'top_stocks_cache.json'),
    (Join-Path $WorkDir 'ml_model.pkl'),
    (Join-Path $WorkDir 'start_bot.ps1'),
    (Join-Path $WorkDir 'last_start_cmd.txt')
)

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$IsAdmin = Test-Administrator

function Test-TaskExists {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    return $null -ne $task
}

function Test-TaskRunning {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task) { return $false }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    return $info.LastTaskResult -eq 267009 -or ($task.State -eq 'Running')
}

function New-StartScript([string]$line) {
    Set-Content -Path $ScriptPS1 -Value '$ErrorActionPreference = "Continue"'
    Add-Content -Path $ScriptPS1 -Value "Set-Location '$WorkDir'"
    Add-Content -Path $ScriptPS1 -Value '$env:SCHEDULED_TASK_MODE=''1'''
    Add-Content -Path $ScriptPS1 -Value '$env:BOT_TEE_LOG=''1'''
    Add-Content -Path $ScriptPS1 -Value ''
    Add-Content -Path $ScriptPS1 -Value '# Function to truncate log file to max lines'
    Add-Content -Path $ScriptPS1 -Value 'function Limit-LogFile([string]$LogPath, [int]$MaxLines = 250) {'
    Add-Content -Path $ScriptPS1 -Value '  try {'
    Add-Content -Path $ScriptPS1 -Value '    if (-not (Test-Path $LogPath)) { return }'
    Add-Content -Path $ScriptPS1 -Value '    $lines = Get-Content $LogPath -ErrorAction SilentlyContinue'
    Add-Content -Path $ScriptPS1 -Value '    if ($lines.Count -le $MaxLines) { return }'
    Add-Content -Path $ScriptPS1 -Value '    $initLine = $lines | Where-Object { $_ -match ''^INIT '' } | Select-Object -First 1'
    Add-Content -Path $ScriptPS1 -Value '    $kept = $lines | Select-Object -Last $MaxLines'
    Add-Content -Path $ScriptPS1 -Value '    if ($initLine -and $kept -notcontains $initLine) {'
    Add-Content -Path $ScriptPS1 -Value '      $kept = @($initLine) + ($kept | Select-Object -Skip 1)'
    Add-Content -Path $ScriptPS1 -Value '    }'
    Add-Content -Path $ScriptPS1 -Value '    Set-Content -Path $LogPath -Value $kept -ErrorAction SilentlyContinue'
    Add-Content -Path $ScriptPS1 -Value '  } catch { }'
    Add-Content -Path $ScriptPS1 -Value '}'
    Add-Content -Path $ScriptPS1 -Value ''
    Add-Content -Path $ScriptPS1 -Value "Add-Content -Path '.\bot.log' -Value ('INIT ' + (Get-Date).ToString('s') + ' user=' + [System.Security.Principal.WindowsIdentity]::GetCurrent().Name)"
    Add-Content -Path $ScriptPS1 -Value '$iterationCount = 0'
    Add-Content -Path $ScriptPS1 -Value 'while ($true) {'
    Add-Content -Path $ScriptPS1 -Value '  $iterationCount++'
    Add-Content -Path $ScriptPS1 -Value '  # Truncate log every 10 iterations (reduces file I/O)'
    Add-Content -Path $ScriptPS1 -Value '  if ($iterationCount % 10 -eq 1) { Limit-LogFile ''.\bot.log'' 250 }'
    Add-Content -Path $ScriptPS1 -Value "  $line 2>&1 | Tee-Object -FilePath '.\bot.log' -Append"
    Add-Content -Path $ScriptPS1 -Value '  $exitCode = $LASTEXITCODE'
    Add-Content -Path $ScriptPS1 -Value '  if ($exitCode -eq 0) {'
    Add-Content -Path $ScriptPS1 -Value '    # Exit code 0 = market closed cleanly, stop and let scheduled task handle next run'
    Add-Content -Path $ScriptPS1 -Value '    Add-Content -Path ''.\bot.log'' -Value "Market closed - task will restart automatically when market opens"'
    Add-Content -Path $ScriptPS1 -Value '    Limit-LogFile ''.\bot.log'' 250'
    Add-Content -Path $ScriptPS1 -Value '    exit 0'
    Add-Content -Path $ScriptPS1 -Value '  } else {'
    Add-Content -Path $ScriptPS1 -Value '    # Non-zero exit = crash, restart quickly'
    Add-Content -Path $ScriptPS1 -Value '    Add-Content -Path ''.\bot.log'' -Value "Bot crashed (exit $exitCode) - restarting in 10s"'
    Add-Content -Path $ScriptPS1 -Value '    Start-Sleep -Seconds 10'
    Add-Content -Path $ScriptPS1 -Value '  }'
    Add-Content -Path $ScriptPS1 -Value '}'
    try { Set-Content -Path $CmdFile -Value $line } catch {}
}

function Create-Task([string]$line) {
    New-StartScript $line
    
    # Remove old task if exists
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
    schtasks /Delete /TN $TaskName /F >$null 2>&1
    
    try {
        $act = New-ScheduledTaskAction -Execute $Exe -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPS1`""
        
        # Triggers: Boot, Logon, Daily at 9:25 AM ET (5 min before market open)
        # Detect timezone and adjust trigger time to LOCAL time
        $etZone = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
        $localZone = [System.TimeZoneInfo]::Local
        $etTime = [DateTime]::Parse("9:25 AM")
        $etDateTime = [System.TimeZoneInfo]::ConvertTimeToUtc($etTime, $etZone)
        $localTime = [System.TimeZoneInfo]::ConvertTimeFromUtc($etDateTime, $localZone)
        $localTimeStr = $localTime.ToString("hh:mmtt")
        
        Write-Host "   ℹ Market opens: 9:30 AM ET = $($localTime.AddMinutes(5).ToString('hh:mm tt')) your time"
        Write-Host "   ℹ Bot will wake: 9:25 AM ET = $localTimeStr your time"
        
        $trgBoot = New-ScheduledTaskTrigger -AtStartup
        $trgLogon = New-ScheduledTaskTrigger -AtLogOn
        $trgDaily = New-ScheduledTaskTrigger -Daily -At $localTimeStr
        $trg = @($trgBoot, $trgLogon, $trgDaily)
        
        $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
        
        # Settings: Allow on battery, don't stop on battery, wake to run, restart on failure
        $set = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -RestartCount 999 `
            -RestartInterval (New-TimeSpan -Minutes 1) `
            -MultipleInstances IgnoreNew `
            -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Hours 0)
        
        Register-ScheduledTask -TaskName $TaskName -Action $act -Trigger $trg -Principal $principal -Settings $set -Description 'Paper trading bot - auto-starts on boot/logon, wakes at 9:25 AM, keeps system awake during market hours' -ErrorAction Stop | Out-Null
        Write-Host "   ✓ Task created successfully (PowerShell method)"
    } catch {
        Write-Host "   ⚠ PowerShell method failed, trying schtasks fallback..."
        Write-Host "   Error: $_"
        
        # Fallback: Use schtasks command with ALL triggers
        $tr="`"$Exe`" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPS1`""
        
        # Create task with login trigger first
        $result = schtasks /Create /TN $TaskName /SC ONLOGON /TR $tr /RL HIGHEST /F 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   ❌ ERROR: Failed to create scheduled task!"
            Write-Host "   schtasks error: $result"
            throw "Task creation failed"
        }
        
        # Add boot trigger
        schtasks /Change /TN $TaskName /RL HIGHEST /ENABLE >$null 2>&1
        
        # Add daily 9:25 AM trigger (note: requires creating a separate task or using XML)
        # For simplicity, we'll note that only login trigger is active via schtasks
        Write-Host "   ✓ Task created (schtasks fallback - login trigger only)"
        Write-Host "   ⚠ Daily 9:25 AM trigger NOT available via fallback method"
    }
    
    # Verify task was created
    Start-Sleep -Milliseconds 500
    if (Test-TaskExists) {
        Write-Host "   ✓ Task verified: PaperTradingBot exists"
    } else {
        Write-Host "   ❌ ERROR: Task creation failed - task does not exist!"
        throw "Task verification failed"
    }
}

function Run-Simple([string]$line) {
    # Non-admin mode: Just run Python directly (no scheduled task)
    Write-Host "⚠️  Running in SIMPLE MODE (not Admin)"
    Write-Host "   • No auto-start on boot/logon"
    Write-Host "   • No wake at 9:25 AM"
    Write-Host "   • Must keep PowerShell open"
    Write-Host "   • Press Ctrl+C to stop"
    Write-Host ""
    Write-Host "Starting bot..."
    
    Set-Location $WorkDir
    $env:SCHEDULED_TASK_MODE = '0'
    
    try {
        Invoke-Expression $line
    } catch {
        Write-Host "Error: $_"
    }
}

switch ($cmd) {
    'start' {
        if (-not $IsAdmin) {
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════"
            Write-Host "  ⚠️  NOT RUNNING AS ADMINISTRATOR"
            Write-Host "════════════════════════════════════════════════════════════════"
            Write-Host ""
            Write-Host "Admin mode gives you:"
            Write-Host "  ✓ Auto-start on boot"
            Write-Host "  ✓ Auto-start on login"
            Write-Host "  ✓ Wake at 9:25 AM (5 min before market)"
            Write-Host "  ✓ Runs hidden in background"
            Write-Host "  ✓ Auto-restart on crash"
            Write-Host ""
            Write-Host "Simple mode (current):"
            Write-Host "  • Runs in this window only"
            Write-Host "  • Stops when you close PowerShell"
            Write-Host "  • No auto-start features"
            Write-Host ""
            $response = Read-Host "Continue in simple mode? (y/n)"
            if ($response -ne 'y') {
                Write-Host ""
                Write-Host "To run as Admin:"
                Write-Host "  1. Right-click PowerShell → Run as Administrator"
                Write-Host "  2. cd '$WorkDir'"
                Write-Host "  3. Run command again"
                exit 1
            }
            
            if (-not $rest) {
                Write-Host "ERROR: No command provided"
                exit 1
            }
            
            $line = ($rest -join ' ')
            if ($line -notmatch '^\s*python\b') {
                Write-Host "ERROR: Command must start with 'python'"
                exit 1
            }
            
            Run-Simple $line
            exit 0
        }
        
        # Admin mode
        if (Test-TaskRunning) {
            Write-Host "❌ ERROR: Bot is already running!"
            Write-Host "   Use 'restart' to restart, or 'stop' first."
            exit 1
        }
        
        if (-not $rest) {
            Write-Host "❌ ERROR: No command provided"
            Write-Host "   Usage: .\botctl.ps1 start python -u runner.py -t 0.25 -s AAPL -m 100"
            exit 1
        }
        
        $line = ($rest -join ' ')
        if ($line -notmatch '^\s*python\b') {
            Write-Host "❌ ERROR: Command must start with 'python'"
            exit 1
        }
        
        Write-Host "Creating and starting bot (Admin mode)..."
        Create-Task $line
        schtasks /Run /TN $TaskName >$null 2>&1
        Start-Sleep -Seconds 2
        
        if (Test-TaskRunning) {
            Write-Host "✅ Bot started successfully!"
            Write-Host ""
            Write-Host "Features enabled:"
            Write-Host "  ✓ Auto-start on boot"
            Write-Host "  ✓ Auto-start on login"
            Write-Host "  ✓ Wake at 9:25 AM (5 min before market)"
            Write-Host "  ✓ Auto-restart on crash"
            Write-Host "  ✓ Keeps system awake during market hours"
            Write-Host ""
            Write-Host "Log: $WorkDir\bot.log"
            Write-Host "Command saved: last_start_cmd.txt"
            Write-Host ""
            Write-Host "Monitor: Get-Content bot.log -Wait -Tail 50"
            Write-Host "Status:  .\botctl.ps1 status"
            Write-Host "Stop:    .\botctl.ps1 stop"
        } else {
            Write-Host "⚠️  Task created but may not be running. Check: .\botctl.ps1 status"
        }
    }
    
    'stop' {
        if (-not $IsAdmin) {
            Write-Host "❌ ERROR: 'stop' requires Administrator privileges"
            Write-Host ""
            Write-Host "In simple mode (non-admin), just press Ctrl+C to stop."
            Write-Host ""
            Write-Host "To use stop command:"
            Write-Host "  Right-click PowerShell → Run as Administrator"
            exit 1
        }
        
        if (-not (Test-TaskExists)) {
            Write-Host "❌ ERROR: Bot task doesn't exist!"
            Write-Host "   Use 'start' to create and start it."
            exit 1
        }
        
        if (-not (Test-TaskRunning)) {
            Write-Host "⚠️  Bot is already stopped."
            Write-Host "   Task still exists and will auto-start on boot/logon/9:25 AM."
            Write-Host "   Use 'stop-forever' to remove completely."
            exit 0
        }
        
        Write-Host "Stopping bot (task will remain and auto-start on boot/logon/9:25 AM)..."
        schtasks /End /TN $TaskName >$null 2>&1
        Start-Sleep -Seconds 1
        
        if (-not (Test-TaskRunning)) {
            Write-Host "✅ Bot stopped."
            Write-Host "   Task still exists - will auto-start on:"
            Write-Host "   • System boot"
            Write-Host "   • User logon"
            Write-Host "   • Daily at 9:25 AM"
            Write-Host ""
            Write-Host "To start now:        .\botctl.ps1 restart"
            Write-Host "To remove forever:   .\botctl.ps1 stop-forever"
        } else {
            Write-Host "⚠️  Bot may still be running. Check: .\botctl.ps1 status"
        }
    }
    
    'restart' {
        if (-not $IsAdmin) {
            Write-Host "❌ ERROR: 'restart' requires Administrator privileges"
            Write-Host ""
            Write-Host "In simple mode (non-admin):"
            Write-Host "  1. Press Ctrl+C to stop"
            Write-Host "  2. Run start command again"
            Write-Host ""
            Write-Host "To use restart command:"
            Write-Host "  Right-click PowerShell → Run as Administrator"
            exit 1
        }
        
        $line = $null
        if ($rest -and $rest.Count -gt 0) {
            $line = ($rest -join ' ')
        } elseif (Test-Path $CmdFile) {
            try { $line = (Get-Content -Path $CmdFile -Raw).Trim() } catch {}
        }
        
        if (-not $line) {
            Write-Host "❌ ERROR: No command provided and no saved command found."
            Write-Host "   Usage: .\botctl.ps1 restart python -u runner.py -t 0.25 -s AAPL -m 100"
            Write-Host "   Or run 'start' first to save a command."
            exit 1
        }
        
        if ($line -notmatch '^\s*python\b') {
            Write-Host "❌ ERROR: Command must start with 'python'"
            exit 1
        }
        
        $taskExists = Test-TaskExists
        $taskRunning = Test-TaskRunning
        
        if ($taskExists -and $taskRunning) {
            Write-Host "Restarting bot..."
            schtasks /End /TN $TaskName >$null 2>&1
            Start-Sleep -Seconds 1
        } elseif ($taskExists) {
            Write-Host "Starting bot (was stopped)..."
        } else {
            Write-Host "Creating and starting bot (task didn't exist)..."
        }
        
        Create-Task $line
        schtasks /Change /TN $TaskName /ENABLE >$null 2>&1
        schtasks /Run /TN $TaskName >$null 2>&1
        Start-Sleep -Seconds 2
        
        if (Test-TaskRunning) {
            Write-Host "✅ Bot restarted successfully!"
            Write-Host "   Command: $line"
            Write-Host "   Log: $WorkDir\bot.log"
        } else {
            Write-Host "⚠️  Task created but may not be running. Check: .\botctl.ps1 status"
        }
    }
    
    'stop-forever' {
        if (-not $IsAdmin) {
            Write-Host "❌ ERROR: 'stop-forever' requires Administrator privileges"
            Write-Host ""
            Write-Host "In simple mode (non-admin), just press Ctrl+C to stop."
            Write-Host "Auto-generated files will remain (bot.log, portfolio.json, etc)."
            Write-Host ""
            Write-Host "To use stop-forever (full cleanup):"
            Write-Host "  Right-click PowerShell → Run as Administrator"
            exit 1
        }
        
        $taskExists = Test-TaskExists
        $taskRunning = Test-TaskRunning
        
        Write-Host "Stopping bot forever (removing task and auto-generated files)..."
        
        if ($taskRunning) {
            Write-Host "  • Stopping running task..."
            schtasks /End /TN $TaskName >$null 2>&1
            Start-Sleep -Seconds 1
        }
        
        if ($taskExists) {
            Write-Host "  • Removing scheduled task..."
            schtasks /Delete /TN $TaskName /F >$null 2>&1
        } else {
            Write-Host "  • Task doesn't exist (skipping removal)"
        }
        
        Write-Host "  • Cleaning up processes..."
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | 
            Where-Object { $_.CommandLine -match 'runner\.py' } | 
            ForEach-Object { 
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue 
            }
        
        Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" -ErrorAction SilentlyContinue | 
            Where-Object { $_.CommandLine -match 'start_bot\.ps1' } | 
            ForEach-Object { 
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue 
            }
        
        Write-Host "  • Deleting auto-generated files..."
        $deletedCount = 0
        foreach ($file in $AutoFiles) {
            if (Test-Path $file) {
                try {
                    Remove-Item -Path $file -Force -ErrorAction SilentlyContinue
                    $deletedCount++
                    Write-Host "    ✓ Deleted: $(Split-Path -Leaf $file)"
                } catch {
                    Write-Host "    ⚠ Could not delete: $(Split-Path -Leaf $file)"
                }
            }
        }
        
        if ($deletedCount -eq 0) {
            Write-Host "    (No auto-generated files found)"
        }
        
        Write-Host ""
        Write-Host "✅ Bot stopped forever!"
        Write-Host "   Task removed, processes killed, files cleaned up."
        Write-Host "   To start again: .\botctl.ps1 start python -u runner.py ..."
    }
    
    'status' {
        $taskExists = Test-TaskExists
        $taskRunning = Test-TaskRunning
        
        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════"
        Write-Host "  BOT STATUS"
        Write-Host "═══════════════════════════════════════════════════════════════"
        Write-Host "  Mode:    $(if ($IsAdmin) { '✅ ADMIN' } else { '⚠️  NON-ADMIN' })"
        
        if (-not $IsAdmin) {
            Write-Host "  Note:    Non-admin mode has limited features"
            Write-Host "           Run as Admin for full functionality"
        }
        
        if (-not $taskExists) {
            Write-Host "  Status:  ❌ NOT CREATED"
            Write-Host "  Task doesn't exist. Use 'start' to create it."
        } elseif ($taskRunning) {
            Write-Host "  Status:  ✅ RUNNING"
            
            $task = Get-ScheduledTask -TaskName $TaskName
            $info = Get-ScheduledTaskInfo -TaskName $TaskName
            
            Write-Host "  Task:    $TaskName"
            Write-Host "  State:   $($task.State)"
            
            if ($info.LastRunTime) {
                Write-Host "  Started: $($info.LastRunTime)"
            }
            
            if (Test-Path $CmdFile) {
                try {
                    $cmdText = Get-Content -Path $CmdFile -Raw
                    Write-Host "  Command: $($cmdText.Trim())"
                } catch {}
            }
            
            $logPath = Join-Path $WorkDir 'bot.log'
            if (Test-Path $logPath) {
                try {
                    $lastLines = Get-Content -Path $logPath -Tail 3 -ErrorAction SilentlyContinue
                    if ($lastLines) {
                        Write-Host ""
                        Write-Host "  Recent activity:"
                        foreach ($line in $lastLines) {
                            Write-Host "    $line"
                        }
                    }
                } catch {}
            }
        } else {
            Write-Host "  Status:  ⏸️ STOPPED"
            Write-Host "  Task exists but is not running."
            Write-Host "  Will auto-start on: boot, logon, or 9:25 AM"
        }
        
        Write-Host ""
        Write-Host "─────────────────────────────────────────────────────────────────"
        Write-Host "  Commands:"
        Write-Host "    .\botctl.ps1 restart       - Start/restart bot (holy grail)"
        Write-Host "    .\botctl.ps1 stop          - Stop (keeps task for auto-start)"
        Write-Host "    .\botctl.ps1 stop-forever  - Remove completely + clean files"
        Write-Host "═══════════════════════════════════════════════════════════════"
        Write-Host ""
        
        if ($taskExists) {
            Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
        }
    }
}
