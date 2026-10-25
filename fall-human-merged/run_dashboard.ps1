$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$ProjectRoot = $PSScriptRoot
$VenvPython  = Join-Path $ProjectRoot '.venv-yolo\Scripts\python.exe'
$Frontend    = Join-Path $ProjectRoot 'frontend'
$Dist        = Join-Path $Frontend 'dist'

if (!(Test-Path -LiteralPath $VenvPython)) {
    $LocalPython = Join-Path $ProjectRoot 'local_python.txt'
    if (Test-Path -LiteralPath $LocalPython) {
        $ConfiguredPython = (Get-Content -LiteralPath $LocalPython -Raw).Trim()
        if (Test-Path -LiteralPath $ConfiguredPython -PathType Leaf) {
            $VenvPython = $ConfiguredPython
        }
    }
}
if (!(Test-Path -LiteralPath $VenvPython)) {
    throw 'Runtime Python belum tersedia. Jalankan setup_dashboard.cmd terlebih dahulu.'
}

if (!(Test-Path -LiteralPath $Dist)) {
    Write-Host 'Build frontend...'
    if (!(Test-Path -LiteralPath (Join-Path $Frontend 'node_modules'))) {
        & npm --prefix $Frontend install
        if ($LASTEXITCODE -ne 0) { throw 'npm install gagal.' }
    }
    & npm --prefix $Frontend run build
    if ($LASTEXITCODE -ne 0) { throw 'Build frontend gagal.' }
}

Write-Host ''
Write-Host 'Dashboard berjalan di http://localhost:8000'
Write-Host 'Tekan Ctrl+C untuk berhenti.'
Write-Host ''

& $VenvPython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
