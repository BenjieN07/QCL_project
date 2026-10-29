$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $pythonExe -m qcl.verify_evaluation
if ($LASTEXITCODE -ne 0) { throw 'Saved evaluation audit failed.' }
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\day3\evaluation\REVIEW.md')
