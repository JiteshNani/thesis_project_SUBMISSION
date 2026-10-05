# Final Run Instructions

From Windows PowerShell in the extracted project directory:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m compileall -q .
python -m pytest -q
.\scripts\run_final_experiment.ps1
```

If `.venv` has not been created yet:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
.\scripts\run_final_experiment.ps1
```

Do not use smoke-test outputs in the thesis. After the full run, use:

```text
results\thesis_results_bundle.json
results\thesis_results_summary.txt
```

The first VLM/RAG/report-metric run needs internet access to download the configured Hugging Face models/resources. The raw BraTS/TextBraTS dataset must be provided manually under `data\raw`.
