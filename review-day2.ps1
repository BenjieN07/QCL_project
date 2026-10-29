param([switch]$Repeat)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Training tests failed.' }
if ($Repeat) {
    $repeatPath = 'results/day2/repeat-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    & $pythonExe -m qcl.train --output $repeatPath
    if ($LASTEXITCODE -ne 0) { throw 'Repeated training failed.' }
    & $pythonExe -m qcl.audit --compare $repeatPath
} else {
    & $pythonExe -m qcl.audit --compare results/day2/repeat
}
if ($LASTEXITCODE -ne 0) { throw 'Saved-run audit failed.' }
