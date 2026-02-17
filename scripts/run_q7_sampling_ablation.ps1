# Commands for Q7: Sampling Steps Ablation
# Run these commands in PowerShell

# Q7 tests how DDPM performs with fewer sampling steps (faster but potentially lower quality)
# We compare KID scores and visual quality across different step counts

# Use your main epsilon parametrization model (the fully trained one)
$CHECKPOINT = "logs/ddpm_modal/ddpm_20260123_090933/checkpoints/ddpm_final.pt"  # Use final checkpoint or your best one
$METHOD = "ddpm"

Write-Host "Q7: Sampling Steps Ablation Study" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Testing how model performs with fewer sampling steps (faster sampling)" -ForegroundColor Yellow
Write-Host "Comparing: 100, 300, 500, 700, 900, 1000 steps" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# KID Evaluations for each step count
# ============================================================================
Write-Host "KID Evaluations (1000 samples each)" -ForegroundColor Green
Write-Host "-" * 60 -ForegroundColor Green

$step_counts = @(100, 300, 500, 700, 900, 1000)

foreach ($steps in $step_counts) {
    Write-Host "`nEvaluating KID with $steps steps..." -ForegroundColor Yellow
    modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT --metrics kid --num-samples 1000 --num-steps $steps
}

# ============================================================================
# Qualitative Samples (1 sample per step count)
# ============================================================================
Write-Host "`n`nQualitative Samples (1 sample per step count)" -ForegroundColor Green
Write-Host "-" * 60 -ForegroundColor Green

foreach ($steps in $step_counts) {
    Write-Host "`nGenerating 1 sample with $steps steps..." -ForegroundColor Yellow
    modal run --detach modal_app.py::main --action sample --method $METHOD --checkpoint $CHECKPOINT --num-samples 1 --num-steps $steps
}

Write-Host "`n`nAll commands queued! Check Modal dashboard for progress." -ForegroundColor Cyan
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  - KID evaluations: 100, 300, 500, 700, 900, 1000 steps (1000 samples each)" -ForegroundColor White
Write-Host "  - Qualitative samples: 1 sample per step count (for visual comparison)" -ForegroundColor White
Write-Host "`nNote: You'll compare KID scores to see how quality degrades with fewer steps." -ForegroundColor Yellow
