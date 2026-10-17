param([int]$Port = 7861)
$ErrorActionPreference = 'Stop'
$MunicipalRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $MunicipalRoot
& "$MunicipalRoot\.venv\Scripts\python.exe" -m surveillance.municipal --port $Port
exit $LASTEXITCODE
