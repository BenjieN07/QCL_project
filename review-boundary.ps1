$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $pythonExe -m qcl.boundary_report --verify
if ($LASTEXITCODE -ne 0) { throw 'Boundary-study audit failed.' }
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\boundary\evaluation\REVIEW.md') -Encoding utf8
