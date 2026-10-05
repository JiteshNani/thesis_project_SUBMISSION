$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE): python $($Args -join ' ')" }
}

Write-Host "=== V3.2 RAG ENHANCEMENT - DEVELOPMENT HALF ONLY ==="
Write-Host "This does NOT touch the held-out test split."
Write-Host "It preserves the frozen V3.1 checkpoint and trains a separate RAG-specific LoRA checkpoint."

$required = @(
    "models\slot_model_best.pth",
    "models\hybrid3dvlm_trainable.pth",
    "data\manifest\manifest.csv",
    "data\manifest\slot_metadata.json"
)
foreach ($p in $required) {
    if (-not (Test-Path $p)) { throw "Missing required frozen artifact: $p" }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "experiments\before_v32_rag_dev_$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($p in @("models\hybrid3dvlm_rag_v32.pth", "data\rag_index_v32", "results\v32_rag_training_history.json", "results\v32_rag_run_metadata.json")) {
    if (Test-Path $p) { Copy-Item -Recurse -Force $p $backup }
}

Write-Host "Compiling + complete project tests..."
Invoke-Python -m compileall -q .
Invoke-Python -m pytest -q

Write-Host "Removing only stale V3.2 RAG artifacts (V3.1 remains untouched)..."
if (Test-Path "data\rag_index_v32") { Remove-Item -Recurse -Force "data\rag_index_v32" }
if (Test-Path "models\hybrid3dvlm_rag_v32.pth") { Remove-Item -Force "models\hybrid3dvlm_rag_v32.pth" }

Write-Host "Training one pre-declared TRAIN-only RAG-specific LoRA adaptation epoch..."
Invoke-Python train_rag_v32.py

Write-Host "Evaluating only the deterministic DEVELOPMENT half of the validation split..."
Invoke-Python validate_v32_rag.py --subset development

Write-Host "=== V3.2 RAG DEVELOPMENT COMPLETE ==="
Write-Host "If the gate says PASS, send results\v32_rag_development_metrics.json before running confirmation."
Write-Host "DO NOT edit RAG settings after seeing the confirmation subset."
