$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE): python $($Args -join ' ')" }
}

Write-Host "=== V3.2 RAG ENHANCEMENT - ONE-TIME CONFIRMATION HALF ==="
Write-Host "Run only after the development gate is accepted and the V3.2 design is frozen."
Write-Host "This does NOT retrain and does NOT touch the held-out test split."

if (-not (Test-Path "models\hybrid3dvlm_rag_v32.pth")) { throw "Missing frozen V3.2 checkpoint. Run development first." }
if (-not (Test-Path "data\rag_index_v32\faiss_index.bin")) { throw "Missing frozen V3.2 RAG index. Run development first." }

Invoke-Python -m compileall -q .
Invoke-Python -m pytest -q
Invoke-Python validate_v32_rag.py --subset confirmation

Write-Host "=== V3.2 RAG CONFIRMATION COMPLETE ==="
Write-Host "Send results\v32_rag_confirmation_metrics.json and do not tune this configuration further."
