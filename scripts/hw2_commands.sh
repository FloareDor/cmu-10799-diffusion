#!/bin/bash

# =============================================================================
# CMU 10799 HW2: Flow Matching & DDIM Command List
# =============================================================================

# --- 1. TRIAL RUNS (Smoke Tests) ---
# Use these to ensure code doesn't crash before leaving it overnight
# modal run --detach modal_app.py::main --action train --method flow_matching --config configs/flow_matching_modal.yaml --iterations 50 --overfit-single-batch
# modal run --detach modal_app.py::train_1gpu --action train --method ddpm --config configs/ddpm_modal.yaml --iterations 50 --overfit-single-batch

# --- 2. MAIN TRAINING (OVERNIGHT) ---
# Launch and detach. Check status via Modal dashboard.

# Train Flow Matching
modal run --detach modal_app.py::main --action train --method flow_matching --config configs/flow_matching_modal.yaml

# Train DDPM (if needed)
# modal run --detach modal_app.py::main --action train --method ddpm --config configs/ddpm_modal.yaml


# --- 3. Q7 ABLATION STUDY ---
# Run this after DDPM training is complete.
# Replace PATH_TO_CHECKPOINT with the actual path in your volume (e.g. checkpoints/ddpm_modal/ddpm_final.pt)
# modal run --detach modal_app.py::run_q7_ablation --checkpoint PATH_TO_CHECKPOINT --steps-list "10,50,100,250,500,1000"


# --- 4. QUALITATIVE COMPARISONS ---

# Flow Matching Samples (Euler 50 steps)
# modal run --detach modal_app.py::main --action sample --method flow_matching --checkpoint checkpoints/flow_matching_modal/flow_matching_final.pt --num_steps 50

# DDIM Deterministic (100 steps)
# modal run --detach modal_app.py::main --action sample --method ddpm --checkpoint checkpoints/ddpm_modal/ddpm_final.pt --sampler ddim --num_steps 100

# DDPM Stochastic (100 steps)
# modal run --detach modal_app.py::main --action sample --method ddpm --checkpoint checkpoints/ddpm_modal/ddpm_final.pt --sampler ddpm --num_steps 100
