# Final commands for Q6: Alternative Parametrization (x0 prediction) Evaluations
# Run these commands in PowerShell

$CHECKPOINT = "logs/ddpm_modal_x0/ddpm_20260123_091726/checkpoints/ddpm_final.pt"
$METHOD = "ddpm"

Write-Host "Q6: Alternative Parametrization (x0 prediction) Evaluations" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Q6(b): Generate grid of 16 samples
Write-Host "`nQ6(b): Generate grid of 16 samples" -ForegroundColor Green
modal run --detach modal_app.py::main --action sample --method $METHOD --checkpoint $CHECKPOINT --num-samples 16

# Q6(b): Evaluate KID with 1000 samples (1000 steps)
Write-Host "`nQ6(b): Evaluate KID with 1000 samples (1000 steps)" -ForegroundColor Green
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT --metrics kid --num-samples 1000

Write-Host "`nAll commands queued! Check Modal dashboard for progress." -ForegroundColor Cyan
Write-Host "`nFor Q6(c), compare these results with your original parametrization (epsilon) model." -ForegroundColor Yellow
