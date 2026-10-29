$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $pythonExe -m qcl.pilot_report --verify
if ($LASTEXITCODE -ne 0) { throw 'Pilot verification failed.' }
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\paper_pilot\evaluation\REVIEW.md') -Encoding utf8
