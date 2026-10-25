$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_HUB_CACHE = Join-Path $PSScriptRoot '.cache\huggingface\hub'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.cache\ultralytics'
$env:YOLO_AUTOINSTALL = 'false'
$venvPython = Join-Path $PSScriptRoot '.venv-yolo\Scripts\python.exe'

if (!(Test-Path -LiteralPath $venvPython)) {
    $candidates = @()
    if ($env:FALL_POSE_PYTHON) { $candidates += $env:FALL_POSE_PYTHON }
    $localHint = Join-Path $PSScriptRoot 'local_python.txt'
    if (Test-Path -LiteralPath $localHint) {
        $candidates += (Get-Content -LiteralPath $localHint -Raw).Trim()
    }
    foreach ($version in @('3.12', '3.11')) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            try {
                $candidate = & py "-$version" -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0) { $candidates += $candidate }
            } catch { }
        }
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += $pythonCommand.Source }
    $pythonBase = $null
    foreach ($candidate in $candidates) {
        if (!(Test-Path -LiteralPath $candidate)) { continue }
        try {
            & $candidate -c 'import sys; import tkinter; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonBase = $candidate; break }
        } catch { }
    }
    if (!$pythonBase) {
        throw 'Python yang sesuai belum ditemukan. Pasang Python 3.12 dari python.org (sertakan Tcl/Tk), lalu jalankan ulang.'
    }
    Write-Host 'Menyiapkan lingkungan Python proyek...'
    & $pythonBase -m venv (Join-Path $PSScriptRoot '.venv-yolo')
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan lingkungan Python gagal.' }
}

$dependenciesReady = $false
try {
    & $venvPython -c 'import torch, torchvision, cv2, PIL, huggingface_hub, safetensors, ultralytics, lap; from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification; assert tuple(int(p) for p in torch.__version__.split(chr(43))[0].split(chr(46))[:2]) >= (2, 6)' 2>$null
    $dependenciesReady = $LASTEXITCODE -eq 0
} catch { }
if (!$dependenciesReady) {
    Write-Host 'Menyiapkan paket aplikasi: PyTorch CPU/GPU dan VideoMAE...'
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw 'Pemasangan PyTorch gagal. Periksa koneksi internet.' }
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Pemasangan paket gagal. Periksa koneksi internet.' }
}
# Model download is an explicit action (download_model.py or the GUI checkbox).
# Starting the GUI alone does not download weights or activate any camera.
& $venvPython (Join-Path $PSScriptRoot 'app.py')
if ($LASTEXITCODE -ne 0) { throw 'Aplikasi berhenti dengan kesalahan.' }
