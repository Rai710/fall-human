$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$ProjectRoot   = $PSScriptRoot
$VenvDir       = Join-Path $ProjectRoot '.venv-yolo'
$VenvPython    = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements  = Join-Path $ProjectRoot 'requirements.txt'

function Invoke-Pip {
    param([string[]]$PipCommand)
    Write-Host "pip $($PipCommand -join ' ')"
    & $VenvPython -m pip --require-virtualenv --disable-pip-version-check --no-input `
        --timeout 120 --retries 5 @PipCommand
    if ($LASTEXITCODE -ne 0) {
        throw "pip gagal: $($PipCommand -join ' ')"
    }
}

if (!(Test-Path -LiteralPath $VenvPython)) {
    Write-Host 'Mencari Python 3.10-3.13...'
    $pythonBase = $null

    foreach ($version in @('3.12', '3.13')) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            try {
                $candidate = & py "-$version" -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $candidate) {
                    $pythonBase = $candidate
                    break
                }
            } catch { }
        }
    }

    if (!$pythonBase) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $pythonBase = $pythonCommand.Source }
    }

    if (!$pythonBase) {
        throw 'Python tidak ditemukan. Pasang Python 3.12 dari python.org (sertakan Tcl/Tk).'
    }

    & $pythonBase -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'
    if ($LASTEXITCODE -ne 0) {
        throw 'Versi Python harus 3.10 sampai 3.13.'
    }

    Write-Host "Membuat .venv-yolo dengan: $pythonBase"
    & $pythonBase -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw 'Gagal membuat .venv-yolo.' }
}

Write-Host '1/5 Upgrade pip...'
Invoke-Pip -PipCommand @('install', '--upgrade', 'pip')

Write-Host '2/5 Install PyTorch CPU...'
Invoke-Pip -PipCommand @('install', 'torch', 'torchvision',
    '--index-url', 'https://download.pytorch.org/whl/cpu')

Write-Host '3/5 Install dependency inti (VideoMAE)...'
Invoke-Pip -PipCommand @('install', '-r', $Requirements)

Write-Host '4/5 Install dependency dashboard...'
Invoke-Pip -PipCommand @('install', 'fastapi', 'uvicorn[standard]',
    'python-multipart', 'pytest', 'httpx')

Write-Host '5/5 Verifikasi import...'
& $VenvPython -c "import torch, torchvision, cv2, transformers, fastapi, uvicorn, numpy; print('Semua dependency OK')"
if ($LASTEXITCODE -ne 0) { throw 'Verifikasi import gagal.' }

Write-Host 'Memeriksa konsistensi dependency...'
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check menemukan konflik dependency.' }

Write-Host ''
Write-Host 'SELESAI'
Write-Host "Python : $VenvPython"
Write-Host 'Model VideoMAE akan diunduh otomatis saat sesi pertama dijalankan.'
Write-Host 'Jalankan run_dashboard.cmd untuk dashboard web.'
Write-Host 'Jalankan MULAI.cmd untuk aplikasi desktop.'
