$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }

Write-Host "=== V3.1 FULL AUTHORITATIVE FINAL RUN ==="
Write-Host "Run this only after V3.1 validation metrics are accepted/frozen."
Write-Host "The run retrains all learned models from scratch and evaluates the held-out test once."
& .\scripts\run_final_experiment.ps1
if ($LASTEXITCODE -ne 0) { throw "V3.1 full final experiment failed." }
