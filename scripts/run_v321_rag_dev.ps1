$ErrorActionPreference = "Stop"

function Run-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): python $($Args -join ' ')"
    }
}

Write-Host "=== V3.2.1 RAG REFINEMENT - DEVELOPMENT HALF ONLY ==="
Write-Host "No held-out TEST case is used."
Write-Host "Frozen V3.1, slot CNN and U-Net remain untouched."

Write-Host "Compiling + complete project tests..."
Run-Python -m compileall -q src tests train_rag_v321.py validate_v321_rag.py
Run-Python -m pytest -q

Write-Host "Removing only stale V3.2.1 artifacts..."
$items = @(
    "models\hybrid3dvlm_rag_v321.pth",
    "results\v321_rag_training_history.json",
    "results\v321_rag_run_metadata.json",
    "results\v321_rag_development_metrics.json",
    "results\v321_rag_development_outputs.json"
)
foreach ($item in $items) {
    if (Test-Path $item) { Remove-Item -Force $item }
}
if (Test-Path "data\rag_index_v321") {
    Remove-Item -Recurse -Force "data\rag_index_v321"
}

Write-Host "Training V3.2.1 TRAIN-only retrieval-conditioned LoRA branch..."
Run-Python train_rag_v321.py

Write-Host "Evaluating deterministic DEVELOPMENT half of validation only..."
Run-Python validate_v321_rag.py --subset development

Write-Host "=== V3.2.1 DEVELOPMENT PASS ==="
Write-Host "Do NOT run confirmation more than once. Send the development metrics/outputs to ChatGPT first."
