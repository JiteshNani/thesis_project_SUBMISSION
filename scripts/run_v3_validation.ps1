param(
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE): python $($Args -join ' ')" }
}

Write-Host "=== V3.1 VALIDATION-ONLY IMPROVEMENT RUN ==="
Write-Host "This stage uses TRAIN + VALIDATION only. It does NOT evaluate the held-out test set."

$required = @("models\slot_model_best.pth", "data\manifest\manifest.csv", "data\manifest\slot_metadata.json")
foreach ($p in $required) {
    if (-not (Test-Path $p)) { throw "Missing $p. Run the existing V2 preparation/slot-CNN stage first." }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "experiments\before_v31_validation_$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($p in @("models\hybrid3dvlm_trainable.pth", "results\vlm_training_history.json", "results\vlm_run_metadata.json", "data\rag_index")) {
    if (Test-Path $p) { Copy-Item -Recurse -Force $p $backup }
}

Write-Host "Compiling + tests..."
Invoke-Python -m compileall -q .
Invoke-Python -m pytest -q

Write-Host "Building corrected train-only RAG index..."
if (Test-Path "data\rag_index") { Remove-Item -Recurse -Force "data\rag_index" }
Invoke-Python build_rag_index.py

Write-Host "Training corrected VLM on train split with validation-loss checkpoint selection..."
if ($Quick) {
    Invoke-Python train_vlm.py --epochs 1 --max-train-batches 8 --max-val-cases 8
    Write-Host "Quick mode: validation metrics below are execution-only; do not freeze the thesis from them."
    Invoke-Python validate_v3_report_models.py --max-cases 8
} else {
    Invoke-Python train_vlm.py
    Invoke-Python validate_v3_report_models.py
}

Write-Host "=== V3.1 VALIDATION RUN COMPLETE ==="
Write-Host "Send results\v3_validation_report_metrics.json before running the authoritative final test."
