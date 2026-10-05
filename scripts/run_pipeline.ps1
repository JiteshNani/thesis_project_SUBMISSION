param(
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): python $($Args -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path "models", "outputs", "results", "experiments", "data\manifest" | Out-Null

Write-Host "=== 1/16 Prepare data + deterministic split + laterality audit ==="
Invoke-Python prepare_data.py
Invoke-Python audit_laterality_targets.py
if ($SmokeTest) { Invoke-Python validate_project.py } else {
    Invoke-Python -c "import torch; import sys; print(torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 20)"
    Invoke-Python validate_project.py --final
}

if ($SmokeTest) {
    Write-Host "SMOKE TEST MODE: outputs are execution checks only and MUST NOT be used in the thesis."
    Write-Host "=== 2/16 Train slot CNN (smoke) ==="
    Invoke-Python train_slot_model.py --epochs 1 --max-train-batches 2
    Write-Host "=== 3/16 Evaluate slot CNN ==="
    Invoke-Python evaluate_slot_model.py
    Write-Host "=== 4/16 Train segmentation U-Net (smoke) ==="
    Invoke-Python train_segmentation_model.py --epochs 1 --max-train-batches 1 --max-val-cases 2
    Write-Host "=== 5/16 Evaluate segmentation ==="
    Invoke-Python evaluate_segmentation_model.py --save-predictions 1 --max-cases 1
    Write-Host "=== 6/16 Train hybrid VLM with frozen CNN slots (smoke) ==="
    Invoke-Python train_vlm.py --epochs 1 --max-train-batches 1 --max-val-cases 1
    Write-Host "=== 7/16 Evaluate base VLM ==="
    Invoke-Python evaluate_vlm.py --max-cases 1
    Write-Host "=== 8/16 Build train-only RAG ==="
    Invoke-Python build_rag_index.py
    Write-Host "=== 9/16 Evaluate VLM+RAG ==="
    Invoke-Python evaluate_vlm_rag.py --max-cases 1
    Write-Host "SMOKE PIPELINE PASS. Stopping before authoritative full-test comparison/metrics because smoke outputs intentionally cover only a subset."
    Write-Host "Do NOT use smoke outputs in the thesis."
    return
} else {
    Write-Host "=== 2/16 Train slot CNN ==="
    Invoke-Python train_slot_model.py
    Write-Host "=== 3/16 Evaluate slot CNN + training-majority baseline ==="
    Invoke-Python evaluate_slot_model.py
    Write-Host "=== 4/16 Train genuine MONAI 3D U-Net ==="
    Invoke-Python train_segmentation_model.py
    Write-Host "=== 5/16 Evaluate segmentation on held-out test set ==="
    Invoke-Python evaluate_segmentation_model.py
    Write-Host "=== 6/16 Train hybrid structured-and-visual VLM using frozen CNN-predicted slots ==="
    Invoke-Python train_vlm.py
    Write-Host "=== 7/16 Evaluate base VLM on held-out test set ==="
    Invoke-Python evaluate_vlm.py
    Write-Host "=== 8/16 Build verified train-only TextBraTS RAG index ==="
    Invoke-Python build_rag_index.py
    Write-Host "=== 9/16 Evaluate VLM+RAG on held-out test set ==="
    Invoke-Python evaluate_vlm_rag.py
}

Write-Host "=== 10/16 Assemble final four-method comparison ==="
Invoke-Python final_comparison.py
Write-Host "=== 11/16 Compute report metrics ==="
if ($SmokeTest) { Invoke-Python compute_report_metrics.py } else { Invoke-Python compute_report_metrics.py --strict }
Write-Host "=== 12/16 Verify held-out result integrity ==="
Invoke-Python verify_final_results.py
Write-Host "=== 13/16 Export thesis tables ==="
Invoke-Python export_thesis_tables.py
Write-Host "=== 14/16 Generate thesis figures ==="
Invoke-Python generate_thesis_figures.py
Write-Host "=== 15/16 Generate authoritative thesis results bundle ==="
Invoke-Python generate_thesis_results_summary.py
Write-Host "=== 16/16 Generate chapter update material ==="
Invoke-Python generate_thesis_update_material.py
Write-Host "DONE. Authoritative outputs:"
Write-Host "  results\thesis_results_bundle.json"
Write-Host "  results\thesis_results_summary.txt"
Write-Host "  results\thesis_chapter4_update_material.md"
Write-Host "  results\thesis_chapter5_update_material.md"
Write-Host "  results\thesis_chapter6_update_material.md"
