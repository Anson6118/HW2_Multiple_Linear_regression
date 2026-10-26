$ErrorActionPreference = 'Stop'
Invoke-Item -LiteralPath (Join-Path $PSScriptRoot 'web\dist\index.html')
