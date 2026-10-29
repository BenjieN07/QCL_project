$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mplconfig'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& .\.venv\Scripts\python.exe -m qcl.mirror_report
if ($LASTEXITCODE -ne 0) { throw 'Training/evaluation audit failed' }
& .\.venv\Scripts\python.exe -m qcl.hardware --audit-only
if ($LASTEXITCODE -ne 0) { throw 'Hardware artifact audit failed' }
Write-Host 'Review passed. No model training or hardware jobs were performed.'
