$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $pythonExe -m qcl.depth_report --verify
if ($LASTEXITCODE -ne 0) { throw 'Depth-study audit failed.' }
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\day4\evaluation\REVIEW.md') -Encoding utf8
