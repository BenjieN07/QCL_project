$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
foreach ($module in @('qcl.audit', 'qcl.verify_evaluation', 'qcl.depth_report', 'qcl.boundary_report')) {
    if ($module -eq 'qcl.audit') {
        & $pythonExe -m $module --compare results/smoke/repeat
    } elseif ($module -eq 'qcl.verify_evaluation') {
        & $pythonExe -m $module
    } else {
        & $pythonExe -m $module --verify
    }
    if ($LASTEXITCODE -ne 0) { throw "$module audit failed" }
}
& $pythonExe -m qcl.mirror_report
if ($LASTEXITCODE -ne 0) { throw 'Mirrored-training audit failed' }
& $pythonExe -m qcl.hardware --audit-only
if ($LASTEXITCODE -ne 0) { throw 'Shot/circuit audit failed; install requirements-hardware.txt' }
& $pythonExe -m qcl.hardware_report
if ($LASTEXITCODE -ne 0) { throw 'Saved hardware-result audit failed' }
& $pythonExe -m qcl.checkpoints --verify
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint/progress integrity audit failed' }
Write-Host 'Review passed. No training or hardware submission was performed.'
