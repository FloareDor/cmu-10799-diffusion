# Commands for Q6: Alternative Parametrization Comparison
# Run these commands in PowerShell

# X0 parametrization model (alternative)
$CHECKPOINT_X0 = "logs/ddpm_modal_x0/ddpm_20260123_091726/checkpoints/ddpm_final.pt"

# Original epsilon parametrization model
$CHECKPOINT_EPSILON_50K = "logs/ddpm_modal/ddpm_20260123_090933/checkpoints/ddpm_0050000.pt"
$CHECKPOINT_EPSILON_110K = "logs/ddpm_modal/ddpm_20260123_090933/checkpoints/ddpm_0110000.pt"

$METHOD = "ddpm"

Write-Host "Q6: Alternative Parametrization Comparison" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# ============================================================================
# X0 Parametrization Model (Alternative)
# ============================================================================
Write-Host "`nX0 Parametrization Model (Alternative)" -ForegroundColor Green
Write-Host "-" * 60 -ForegroundColor Green

# Q6(b): Generate grid of 16 samples
Write-Host "`nQ6(b): Generate grid of 16 samples (x0 model)" -ForegroundColor Yellow
modal run --detach modal_app.py::main --action sample --method $METHOD --checkpoint $CHECKPOINT_X0 --num-samples 16

# Q6(b): Evaluate KID with 1000 samples
Write-Host "`nQ6(b): Evaluate KID with 1000 samples (x0 model)" -ForegroundColor Yellow
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT_X0 --metrics kid --num-samples 1000

# ============================================================================
# Original Epsilon Parametrization Model (for comparison)
# ============================================================================
Write-Host "`nOriginal Epsilon Parametrization Model (for Q6(c) comparison)" -ForegroundColor Green
Write-Host "-" * 60 -ForegroundColor Green

# Evaluate KID with 1000 samples for comparison (50k checkpoint)
Write-Host "`nQ6(c): Evaluate KID with 1000 samples (epsilon model @ 50k - for comparison)" -ForegroundColor Yellow
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT_EPSILON_50K --metrics kid --num-samples 1000

# Evaluate KID with 1000 samples for comparison (110k checkpoint)
Write-Host "`nQ6(c): Evaluate KID with 1000 samples (epsilon model @ 110k - for comparison)" -ForegroundColor Yellow
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT_EPSILON_110K --metrics kid --num-samples 1000

Write-Host "`nAll commands queued! Check Modal dashboard for progress." -ForegroundColor Cyan
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  - X0 model: Grid + KID (for Q6b)" -ForegroundColor White
Write-Host "  - Epsilon model @ 50k: KID (for Q6c comparison)" -ForegroundColor White
Write-Host "  - Epsilon model @ 110k: KID (for Q6c comparison)" -ForegroundColor White
