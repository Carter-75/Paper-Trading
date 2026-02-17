
function Get-ProcessTree {
    $procs = Get-CimInstance Win32_Process
    $pythons = $procs | Where-Object { $_.Name -eq 'python.exe' }
    
    foreach ($p in $pythons) {
        $parent = $procs | Where-Object { $_.ProcessId -eq $p.ParentProcessId }
        [PSCustomObject]@{
            PID         = $p.ProcessId
            Name        = $p.Name
            CommandLine = $p.CommandLine
            ParentPID   = $p.ParentProcessId
            ParentName  = $parent.Name
            ParentCmd   = $parent.CommandLine
        }
    }
}

Get-ProcessTree | Format-List
