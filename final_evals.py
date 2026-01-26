"""
Q7: Sampling Steps Ablation Study (Native Modal Version)

Run with: python final_evals.py
(Requires: modal deploy modal_app.py)
"""

import modal

# Configurations
steps_list = [100, 300, 500, 700, 900, 1000]
checkpoint = "logs/ddpm_modal/ddpm_20260123_090933/checkpoints/ddpm_final.pt"

# Connect to deployed functions
try:
    evaluate_torch_fidelity = modal.Function.from_name("cmu-10799-diffusion", "evaluate_torch_fidelity")
    sample_fn = modal.Function.from_name("cmu-10799-diffusion", "sample")
except Exception:
    print("Error: App not deployed. Run 'modal deploy modal_app.py' first.")
    exit(1)

def run_batch_evaluations():
    print(f"--- Q7: Sampling Steps Ablation Study ---")
    
    # ---------------------------------------------------------
    # Part 1: KID Evaluations
    # ---------------------------------------------------------
    print("🚀 Launching KID evaluation jobs...")
    
    kid_jobs = []
    for steps in steps_list:
        # Create the arguments dictionary
        job_args = {
            "method": "ddpm",
            "checkpoint": checkpoint,
            "metrics": "kid",
            "num_samples": 1000,
            "num_steps": steps,
        }
        # .spawn returns immediately (non-blocking)
        # **job_args unpacks the dict into keyword arguments
        evaluate_torch_fidelity.spawn(**job_args)
        kid_jobs.append(steps)
        
    print(f"✅ Spawned {len(kid_jobs)} KID jobs")

    # ---------------------------------------------------------
    # Part 2: Qualitative Samples
    # ---------------------------------------------------------
    print("🖼️  Launching sample generation jobs...")
    
    sample_jobs = []
    for steps in steps_list:
        job_args = {
            "method": "ddpm",
            "checkpoint": checkpoint,
            "num_samples": 1,
            "num_steps": steps,
        }
        sample_fn.spawn(**job_args)
        sample_jobs.append(steps)

    print(f"✅ Spawned {len(sample_jobs)} sampling jobs")
    print("\nCheck the Modal dashboard to monitor progress.")

if __name__ == "__main__":
    run_batch_evaluations()