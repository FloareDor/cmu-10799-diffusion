# Commands to run evaluations for Q6: Alternative Parametrization (x0 prediction)
# Run these commands in PowerShell

$CHECKPOINT = "logs/ddpm_modal_x0/ddpm_20260123_091726/checkpoints/ddpm_final.pt"
$METHOD = "ddpm"

Write-Host "Q6: Alternative Parametrization (x0 prediction) Evaluations" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host "`nQ6(b): Generate grid of 16 samples" -ForegroundColor Green
modal run --detach modal_app.py::main --action sample --method $METHOD --checkpoint $CHECKPOINT --num-samples 16

Write-Host "`nQ6(b): Evaluate KID with 1000 samples (1000 steps)" -ForegroundColor Green
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT --metrics kid --num-samples 1000

Write-Host "`nAll commands queued! Check Modal dashboard for progress." -ForegroundColor Cyan
Write-Host "`nNote: For Q6(c), you'll need to compare these results with your original parametrization (epsilon prediction) model." -ForegroundColor Yellow
Write-Host "Make sure you have the KID score from the original model to complete the comparison." -ForegroundColor Yellow
