#!/bin/bash
# Commands to run evaluations for Q6: Alternative Parametrization (x0 prediction)

CHECKPOINT="logs/ddpm_modal_x0/ddpm_20260123_091726/checkpoints/ddpm_final.pt"
METHOD="ddpm"

echo "============================================================"
echo "Q6: Alternative Parametrization (x0 prediction) Evaluations"
echo "============================================================"

# Q6(b): Generate grid of 16 samples
echo ""
echo "Running: Generate grid of 16 samples (Q6b)"
modal run --detach modal_app.py::main --action sample --method $METHOD --checkpoint $CHECKPOINT --num-samples 16

# Q6(b): Evaluate KID with 1000 samples (1000 steps - default)
echo ""
echo "Running: Evaluate KID with 1000 samples, 1000 steps (Q6b)"
modal run --detach modal_app.py::main --action evaluate_torch_fidelity --method $METHOD --checkpoint $CHECKPOINT --metrics kid --num-samples 1000

echo ""
echo "All commands queued! Check Modal dashboard for progress."
echo ""
echo "Note: For Q6(c), you'll need to compare these results with your original parametrization (epsilon prediction) model."
echo "Make sure you have the KID score from the original model to complete the comparison."
