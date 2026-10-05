$ErrorActionPreference = "Stop"

Write-Host "=== Thesis environment setup: Windows + NVIDIA GPU ==="
if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

$torchOk = $false
python -c "import torch; print('torch=', torch.__version__, 'cuda_build=', torch.version.cuda, 'cuda_available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -eq 0) { $torchOk = $true }

if (-not $torchOk) {
    Write-Host "Installing/upgrading the official CUDA 12.6 PyTorch wheel for Windows..."
    python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed." }
}

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt installation failed." }

python -c "import torch; print('torch=', torch.__version__); print('cuda_build=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed after environment setup." }
Write-Host "Environment verification complete."
