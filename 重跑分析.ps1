$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw '請依 README.md 先建立 .venv 並安裝 requirements.txt。'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython scripts/analyze.py
    if ($LASTEXITCODE -ne 0) { throw '分析失敗。' }
    & $taskPython scripts/build_report.py
    if ($LASTEXITCODE -ne 0) { throw '報告產製失敗。' }
    & $taskPython scripts/export_pdf.py
    if ($LASTEXITCODE -ne 0) { throw 'PDF 產製失敗；HTML 報告已保存。' }
} finally {
    Pop-Location
}
