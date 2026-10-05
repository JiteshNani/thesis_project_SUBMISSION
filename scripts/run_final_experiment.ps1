param(
    [switch]$KeepExisting
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

Write-Host "=== AUTHORITATIVE FINAL THESIS EXPERIMENT ==="
Write-Host "This run creates the single results package that should be used for thesis updating."

if (-not $KeepExisting) {
    Write-Host "Cleaning generated checkpoints/results/indexes from earlier smoke or stale runs..."
    $targets = @("models", "outputs", "results", "experiments", "data\manifest", "data\rag_index")
    foreach ($target in $targets) {
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    }
}
New-Item -ItemType Directory -Force -Path "models", "outputs", "results", "experiments", "data\manifest" | Out-Null

Write-Host "Compiling Python source..."
Invoke-Python -m compileall -q .
Write-Host "Running complete unit/contract test suite before expensive training..."
Invoke-Python -m pytest -q
Write-Host "Verifying mandatory medical-imaging dependencies and executing the real imaging tests..."
Invoke-Python -c "import nibabel, monai; print('nibabel=', nibabel.__version__, 'monai=', monai.__version__)"
Invoke-Python -m pytest -q tests/test_imaging_core.py

Write-Host "Recording exact installed Python packages..."
& python -m pip freeze | Out-File -Encoding utf8 results\pip_freeze.txt
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed." }

& .\scripts\run_pipeline.ps1
if ($LASTEXITCODE -ne 0) { throw "Final pipeline failed." }

Write-Host "=== FINAL RUN COMPLETE ==="
Write-Host "The result bundle and chapter-update files are authoritative for the current run. Do not carry numbers forward from older runs."
Write-Host "Authoritative outputs:"
Write-Host "  results\thesis_results_bundle.json"
Write-Host "  results\thesis_results_summary.txt"
Write-Host "  results\pip_freeze.txt"
