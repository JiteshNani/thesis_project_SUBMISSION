$ErrorActionPreference = "Stop"

Write-Host "=== V3.2.1 RAG - ONE-TIME CONFIRMATION HALF ==="
Write-Host "Run this only after the development gate has passed and the V3.2.1 design is frozen."
Write-Host "This still does NOT use the held-out TEST split."

python validate_v321_rag.py --subset confirmation
if ($LASTEXITCODE -ne 0) {
    throw "V3.2.1 confirmation evaluation failed with exit $LASTEXITCODE"
}

Write-Host "=== V3.2.1 CONFIRMATION COMPLETE ==="
Write-Host "Do not tune V3.2.1 after reading confirmation results."
