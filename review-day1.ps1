$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Create the virtual environment and install requirements first; see README.md.'
}
& $pythonExe -m qcl.prepare
if ($LASTEXITCODE -ne 0) { throw 'Dataset checks failed.' }
& $pythonExe -m qcl.benchmark
if ($LASTEXITCODE -ne 0) { throw 'Circuit checks failed.' }
Get-Content -LiteralPath (Join-Path $PSScriptRoot 'results\day1\REVIEW.md')
