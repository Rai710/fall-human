$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Lingkungan Python proyek. Folder lama '.venv-yolo' tetap dipakai jika sudah ada
# (agar lingkungan yang sudah terpasang tidak perlu dibuat ulang); selain itu '.venv'.
$venvDir = Join-Path $PSScriptRoot '.venv'
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv-yolo\Scripts\python.exe')) {
    $venvDir = Join-Path $PSScriptRoot '.venv-yolo'
}
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

function Test-Py([string]$code) {
    try {
        $null = & $venvPython -c $code 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# Daftar versi CUDA PyTorch yang cocok dengan driver NVIDIA (kosong jika tidak ada GPU NVIDIA).
function Get-CudaTags {
    if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return @() }
    try {
        $text = (& nvidia-smi 2>$null | Out-String)
    } catch {
        return @()
    }
    $tags = @()
    if ($text -match 'CUDA Version:\s*(\d+)\.(\d+)') {
        $code = [int]$Matches[1] * 100 + [int]$Matches[2]
        if ($code -ge 1300) { $tags += 'cu130' }
        if ($code -ge 1208) { $tags += 'cu128' }
        if ($code -ge 1206) { $tags += 'cu126' }
    }
    return $tags
}

function Install-Torch([string]$tag, [bool]$force) {
    $pipArgs = @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', "https://download.pytorch.org/whl/$tag")
    if ($force) { $pipArgs += '--force-reinstall' }
    & $venvPython @pipArgs
    if ($LASTEXITCODE -ne 0) { return $false }
    if ($tag -eq 'cpu') { return $true }
    return (Test-Py 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)')
}

if (!(Test-Path -LiteralPath $venvPython)) {
    $candidates = @()
    if ($env:FALL_POSE_PYTHON) { $candidates += $env:FALL_POSE_PYTHON }
    $localHint = Join-Path $PSScriptRoot 'local_python.txt'
    if (Test-Path -LiteralPath $localHint) {
        $candidates += (Get-Content -LiteralPath $localHint -Raw).Trim()
    }
    foreach ($version in @('3.12', '3.11', '3.13', '3.10')) {
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
    & $pythonBase -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan lingkungan Python gagal.' }
}

# 1. Paket. Hanya dijalankan jika ada yang belum terpasang.
if (!(Test-Py 'import torch, torchvision, transformers, mediapipe, cv2, PIL')) {
    Write-Host 'Memasang paket untuk penggunaan pertama (perlu internet, ukuran besar)...'
    # mediapipe memakai opencv-contrib-python; opencv-python biasa bentrok dengannya.
    try { $null = & $venvPython -m pip uninstall -y opencv-python 2>$null } catch { }
    $torchReady = $false
    foreach ($tag in @(Get-CudaTags)) {
        Write-Host "GPU NVIDIA terdeteksi. Memasang PyTorch CUDA ($tag)..."
        if (Install-Torch $tag $false) { $torchReady = $true; break }
    }
    if (!$torchReady) {
        Write-Host 'Memasang PyTorch untuk CPU...'
        if (!(Install-Torch 'cpu' $false)) { throw 'Pemasangan PyTorch gagal. Periksa koneksi internet.' }
    }
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Pemasangan paket gagal. Periksa koneksi internet.' }
}

# 2. Lingkungan lama yang sudah terpasang PyTorch CPU tetapi komputernya punya GPU NVIDIA:
#    coba pindah ke PyTorch CUDA satu kali saja (dicatat di .cuda_checked agar tidak diulang).
$cudaMarker = Join-Path $PSScriptRoot '.cuda_checked'
if (!(Test-Path -LiteralPath $cudaMarker)) {
    $tags = @(Get-CudaTags)
    if ($tags.Count -gt 0 -and !(Test-Py 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)')) {
        Write-Host 'GPU NVIDIA terdeteksi, tetapi PyTorch belum memakai CUDA. Memasang versi GPU (sekali saja)...'
        $switched = $false
        foreach ($tag in $tags) {
            if (Install-Torch $tag $true) { $switched = $true; break }
        }
        if (!$switched) {
            Write-Host 'PyTorch CUDA tidak berhasil dipasang; aplikasi tetap berjalan di CPU.'
            if (!(Test-Py 'import torch, torchvision')) {
                if (!(Install-Torch 'cpu' $true)) { throw 'PyTorch rusak dan gagal dipulihkan. Hapus folder lingkungan Python lalu jalankan ulang.' }
            }
        }
        Set-Content -LiteralPath $cudaMarker -Value (Get-Date -Format 's')
    }
}

# 3. Model (VideoMAE dan model kerangka). Dilewati cepat jika sudah ada.
& $venvPython (Join-Path $PSScriptRoot 'download_model.py')
if ($LASTEXITCODE -ne 0) { throw 'Model VideoMAE tidak berhasil disiapkan. Periksa koneksi internet.' }

& $venvPython (Join-Path $PSScriptRoot 'app.py')
if ($LASTEXITCODE -ne 0) { throw 'Aplikasi berhenti dengan kesalahan.' }
