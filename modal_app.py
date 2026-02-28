"""
Modal Configuration for CMU 10799 Diffusion Homework

Defines Modal environment and training functions for cloud GPU training.

See docs/QUICKSTART-MODAL.md for setup and usage instructions.

All parameters are read from config YAML files first, then overridden by command-line arguments.
"""

import time
import modal

# =============================================================================
# Modal App Definition
# =============================================================================

# Create the Modal app
app = modal.App("cmu-10799-diffusion")

# Define the container image with all dependencies
# This mirrors the CPU-only environment (environments/environment-cpu.yml)
# but installs GPU-enabled PyTorch automatically on Modal's GPU machines
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"CACHE_BUST4": str(time.time())})  # Forces rebuild every time - remove when done developing
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "opencv-python-headless>=4.0.0",
        "rembg>=2.0.0",
        "onnxruntime>=1.16.0",
        "wandb>=0.15.0",
        "datasets>=2.0.0",  # For HuggingFace Hub dataset loading
        "torch-fidelity>=0.3.0",  # Comprehensive evaluation metrics
        "lpips>=0.1.4",           # Perceptual loss (PixelGen, arXiv 2602.02493)
    )
    # Copy the local project directory into the image
    .add_local_dir(".", "/root", ignore=[".git", ".venv*", "venv", "__pycache__", "logs", "checkpoints", "*.md", "docs", "environments", "notebooks"])
)

# Create a persistent volume for checkpoints and data
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

# =============================================================================
# Training Function
# =============================================================================

def _train_impl(
    method: str,
    config_path: str,
    resume_from: str,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    """
    Internal training implementation.

    Reads config from YAML file, applies command-line overrides.
    """
    import os
    import sys
    import yaml
    import tempfile
    import subprocess
    import time
    import json
    from datetime import datetime

    sys.path.insert(0, "/root")

    # Load config
    config_tag = method
    if config_path is None:
        config_path = f"/root/configs/{method}.yaml"
    else:
        config_path = f"/root/{config_path}"
        config_tag = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get num_gpus from config
    config_device = config['infrastructure'].get('device', 'cuda')
    num_gpus = config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
    if num_gpus is None:
        num_gpus = 1 if config_device == 'cuda' else 0

    # Read from_hub from config
    from_hub = config['data'].get('from_hub', False)

    # Apply command-line overrides if provided
    if num_iterations is not None:
        config['training']['num_iterations'] = num_iterations
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if learning_rate is not None:
        config['training']['learning_rate'] = learning_rate

    # Set Modal-specific paths
    config['data']['repo_name'] = "electronickale/cmu-10799-celeba64-subset"
    # Respect config-provided root. For relative paths, pin under /data volume.
    configured_root = str(config['data'].get('root', 'celeba'))
    if configured_root.startswith("/"):
        modal_root = configured_root
    else:
        normalized = configured_root.lstrip("./")
        if normalized.startswith("data/"):
            normalized = normalized[5:]
        modal_root = f"/data/{normalized}"
    config['data']['root'] = modal_root
    config['checkpoint']['dir'] = f"/data/checkpoints/{config_tag}"
    config['logging']['dir'] = f"/data/logs/{config_tag}"

    # Create directories
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)

    resume_path = f"/data/{resume_from}" if resume_from else None

    # Track wall-clock time for training summary (answers Q7b)
    train_start_time = time.time()
    train_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use torchrun for multi-GPU, direct import for single GPU
    if num_gpus > 1:
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                yaml.safe_dump(config, temp_file)
                temp_config_path = temp_file.name

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "/root/train.py",
                "--method", method,
                "--config", temp_config_path,
            ]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            if overfit_single_batch:
                cmd.append("--overfit-single-batch")

            subprocess.run(cmd, check=True)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        from train import train as run_training
        run_training(method_name=method, config=config, resume_path=resume_path, overfit_single_batch=overfit_single_batch)

    # Save training summary JSON for homework answers (Q7b, Q8)
    train_end_time = time.time()
    elapsed_seconds = train_end_time - train_start_time
    elapsed_h = int(elapsed_seconds // 3600)
    elapsed_m = int((elapsed_seconds % 3600) // 60)
    elapsed_s = int(elapsed_seconds % 60)

    import subprocess as _sp
    try:
        gpu_info = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        gpu_info = f"{num_gpus}x GPU (nvidia-smi unavailable)"

    summary = {
        "method": method,
        "config": config_tag,
        "num_gpus": num_gpus,
        "gpu_info": gpu_info,
        "num_iterations": config['training']['num_iterations'],
        "batch_size_per_gpu": config['training']['batch_size'],
        "effective_batch_size": config['training']['batch_size'] * num_gpus,
        "learning_rate": config['training']['learning_rate'],
        "mixed_precision": config['infrastructure'].get('mixed_precision', False),
        "sample_every": config['training'].get('sample_every'),
        "save_every": config['training'].get('save_every'),
        "train_start": train_start_str,
        "train_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_seconds": round(elapsed_seconds, 1),
        "wall_time_human": f"{elapsed_h}h {elapsed_m}m {elapsed_s}s",
        "checkpoint_dir": config['checkpoint']['dir'],
        "log_dir": config['logging']['dir'],
        "resume_from": resume_from,
    }

    summary_dir = f"/data/hw3_answers/{config_tag}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[HW3] Training summary saved to {summary_path}")
    print(f"[HW3] Wall-clock time: {summary['wall_time_human']}")
    print(f"[HW3] GPU: {gpu_info}")
    print(f"[HW3] Effective batch size: {summary['effective_batch_size']} ({num_gpus} GPUs x {config['training']['batch_size']})")

    volume.commit()
    return f"Training complete! Checkpoints saved to /data/checkpoints/{method}"


# Create training functions for different GPU counts
@app.function(image=image, gpu="L40S:1", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_1gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:2", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_2gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:3", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_3gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:4", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_4gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:5", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_5gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:6", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_6gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:7", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_7gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:8", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_8gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

# Map GPU counts to functions
TRAIN_FUNCTIONS = {
    1: train_1gpu,
    2: train_2gpu,
    3: train_3gpu,
    4: train_4gpu,
    5: train_5gpu,
    6: train_6gpu,
    7: train_7gpu,
    8: train_8gpu,
}


# =============================================================================
# Sampling Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,  # 3 hours
    volumes={"/data": volume},
)
def sample(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = None,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = None,
    save_path: str = None,
):
    """
    Generate samples from a trained model.

    Uses sample.py via subprocess, similar to how training uses train.py.
    """
    import os
    import subprocess
    from datetime import datetime

    # Set up paths
    checkpoint_path = f"/data/{checkpoint}"
    if save_path is not None:
        output_path = f"/data/{save_path}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/data/samples/{method}_{timestamp}.png"
        # Include num_steps in filename to avoid collisions when running multiple jobs in parallel
        if num_steps is not None:
            output_path = f"/data/samples/{method}_{num_steps}steps_{sampler}_{timestamp}.png"
        else:
            output_path = f"/data/samples/{method}_{sampler}_{timestamp}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build command to run sample.py
    cmd = [
        "python", "/root/sample.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--grid",
        "--output", output_path,
        "--sampler", sampler,
        "--eta", str(eta),
    ]

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if edge_source is not None:
        cmd.extend(["--edge_source", edge_source])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Samples saved to {output_path}"


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,  # 4 hours
    volumes={"/data": volume},
)
def generate_paired_triplet_strip(
    baseline_checkpoint: str,
    hw4_checkpoint: str,
    output_path: str = "hw4_eval/poster/paired_triplet_strip_edge_base_hw4.png",
    method: str = "flow_matching",
    num_pairs: int = 3,
    num_steps: int = 50,
    sampler: str = "ddpm",
    eta: float = 0.0,
    dataset_split: str = "train",
    batch_size: int = 8,
    seed: int = 42,
    condition_source: str = "baseline",
    layout: str = "triplet_rows",
):
    """
    Generate paired triplets [edge | baseline | hw4] using identical edge inputs.
    """
    import os
    import subprocess

    base_ckpt = f"/data/{baseline_checkpoint}"
    hw_ckpt = f"/data/{hw4_checkpoint}"
    out_path = f"/data/{output_path}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "python",
        "/root/scripts/generate_paired_triplet_strip.py",
        "--baseline_checkpoint", base_ckpt,
        "--hw4_checkpoint", hw_ckpt,
        "--method", method,
        "--num_pairs", str(num_pairs),
        "--batch_size", str(batch_size),
        "--num_steps", str(num_steps),
        "--sampler", sampler,
        "--eta", str(eta),
        "--dataset_split", dataset_split,
        "--seed", str(seed),
        "--condition_source", condition_source,
        "--layout", layout,
        "--output_path", out_path,
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    volume.commit()
    return result.stdout.strip() if result.stdout else f"Saved to {out_path}"


# =============================================================================
# Dataset Download Function
# =============================================================================

@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def download_dataset():
    """
    Download the dataset from HuggingFace Hub to Modal volume.

    Caches the dataset in Arrow format at /data/celeba. After downloading,
    training with from_hub=true will automatically use this cached version
    instead of redownloading.
    """
    import sys
    sys.path.insert(0, "/root")

    from datasets import load_dataset
    import os

    print("Downloading dataset from HuggingFace Hub...")
    dataset = load_dataset("electronickale/cmu-10799-celeba64-subset")

    # Save to volume in Arrow format
    os.makedirs("/data/celeba", exist_ok=True)
    dataset.save_to_disk("/data/celeba")

    volume.commit()

    print(f"Dataset cached to /data/celeba")
    print(f"Train set size: {len(dataset['train'])}")
    return "Dataset download complete! Training with from_hub=true with root = '/data/celeba' will now use this cached version."


# =============================================================================
# Dataset Background-Removal Precompute (Parallel)
# =============================================================================

def _to_data_path(path: str) -> str:
    if path.startswith("/data/"):
        return path
    return f"/data/{path.lstrip('/')}"


@app.function(
    image=image,
    timeout=60 * 60 * 6,  # 6 hours per shard
    volumes={"/data": volume},
)
def precompute_bg_removed_shard(
    source_root: str = "celeba",
    output_root: str = "celeba_bg_removed_shards",
    split: str = "train",
    shard_idx: int = 0,
    num_shards: int = 6,
    model_name: str = "u2netp",
):
    import os
    import shutil
    import numpy as np
    from PIL import Image
    from datasets import load_from_disk
    from rembg import remove, new_session

    source_path = _to_data_path(source_root)
    output_path = _to_data_path(output_root)
    session = new_session(model_name)

    if not os.path.exists(os.path.join(source_path, "dataset_dict.json")):
        raise FileNotFoundError(
            f"Source dataset not found at {source_path}. Run --action download first."
        )

    ds_dict = load_from_disk(source_path)
    if split not in ds_dict:
        raise ValueError(f"Split '{split}' not found in source dataset. Available: {list(ds_dict.keys())}")

    split_ds = ds_dict[split]
    shard = split_ds.shard(num_shards=num_shards, index=shard_idx, contiguous=False)

    print(f"[bg-shard] split={split} shard={shard_idx+1}/{num_shards} count={len(shard)}")

    def _remove_bg(example):
        image = example["image"].convert("RGB")
        rgba = remove(image, session=session).convert("RGBA")
        rgba_arr = np.array(rgba, dtype=np.uint8)
        rgb = rgba_arr[..., :3].astype(np.float32)
        alpha = (rgba_arr[..., 3:4].astype(np.float32) / 255.0)
        out = np.clip(rgb * alpha, 0.0, 255.0).astype(np.uint8)
        example["image"] = Image.fromarray(out, mode="RGB")
        return example

    processed = shard.map(_remove_bg, desc=f"BG remove {split} shard {shard_idx}")

    shard_path = os.path.join(output_path, f"{split}_shard_{shard_idx:02d}")
    if os.path.exists(shard_path):
        shutil.rmtree(shard_path)
    processed.save_to_disk(shard_path)
    volume.commit()

    return {"split": split, "shard_idx": shard_idx, "count": len(processed), "path": shard_path}


@app.function(
    image=image,
    timeout=60 * 60 * 24,  # 24 hours orchestration
    volumes={"/data": volume},
)
def precompute_bg_removed_dataset_parallel(
    source_root: str = "celeba",
    output_root: str = "celeba_bg_removed",
    model_name: str = "u2netp",
    num_shards: int = 6,
    splits: str = "train,validation",
):
    import os
    import shutil
    from datasets import load_from_disk, DatasetDict, concatenate_datasets

    source_path = _to_data_path(source_root)
    output_path = _to_data_path(output_root)
    shard_root = f"{output_path}_shards"

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not os.path.exists(os.path.join(source_path, "dataset_dict.json")):
        raise FileNotFoundError(
            f"Source dataset not found at {source_path}. Run --action download first."
        )

    source_ds = load_from_disk(source_path)
    requested_splits = [s.strip() for s in splits.split(",") if s.strip()]
    split_names = [s for s in requested_splits if s in source_ds]
    if not split_names:
        raise ValueError(
            f"No valid splits requested. Requested={requested_splits}, available={list(source_ds.keys())}"
        )

    if os.path.exists(shard_root):
        shutil.rmtree(shard_root)
    os.makedirs(shard_root, exist_ok=True)

    # Launch shards in parallel.
    handles = []
    for split in split_names:
        for shard_idx in range(num_shards):
            handle = precompute_bg_removed_shard.spawn(
                source_root=source_path,
                output_root=shard_root,
                split=split,
                shard_idx=shard_idx,
                num_shards=num_shards,
                model_name=model_name,
            )
            handles.append((split, shard_idx, handle))

    for split, shard_idx, handle in handles:
        result = handle.get()
        print(f"[bg-done] split={split} shard={shard_idx} count={result['count']}")

    # Merge split shards into one DatasetDict and save.
    merged = {}
    for split in split_names:
        split_shards = []
        for shard_idx in range(num_shards):
            shard_path = os.path.join(shard_root, f"{split}_shard_{shard_idx:02d}")
            split_shards.append(load_from_disk(shard_path))
        merged[split] = concatenate_datasets(split_shards)

    final_ds = DatasetDict(merged)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    final_ds.save_to_disk(output_path)

    # Keep only the merged dataset.
    shutil.rmtree(shard_root, ignore_errors=True)
    volume.commit()

    counts = {split: len(final_ds[split]) for split in split_names}
    return f"Background-removed dataset saved to {output_path} with splits: {counts}"


@app.function(
    image=image,
    timeout=60 * 60 * 2,  # merge only
    volumes={"/data": volume},
)
def merge_bg_removed_from_shards(
    shard_root: str = "celeba_bg_removed_shards",
    output_root: str = "celeba_bg_removed",
    cleanup_shards: bool = False,
):
    import os
    import re
    import shutil
    from collections import defaultdict
    from datasets import load_from_disk, DatasetDict, concatenate_datasets

    shard_path_root = _to_data_path(shard_root)
    output_path = _to_data_path(output_root)

    if not os.path.isdir(shard_path_root):
        raise FileNotFoundError(f"Shard root not found: {shard_path_root}")

    # Discover shard directories like: train_shard_00
    split_shards = defaultdict(list)
    pat = re.compile(r"^(.+)_shard_(\d+)$")
    for entry in os.listdir(shard_path_root):
        full = os.path.join(shard_path_root, entry)
        if not os.path.isdir(full):
            continue
        m = pat.match(entry)
        if not m:
            continue
        split_name, shard_idx = m.group(1), int(m.group(2))
        split_shards[split_name].append((shard_idx, full))

    if not split_shards:
        raise RuntimeError(f"No shard directories found under {shard_path_root}")

    merged = {}
    for split_name, shard_infos in split_shards.items():
        shard_infos.sort(key=lambda x: x[0])
        parts = [load_from_disk(p) for _, p in shard_infos]
        merged[split_name] = concatenate_datasets(parts)
        print(f"[merge] split={split_name} parts={len(parts)} rows={len(merged[split_name])}")

    final_ds = DatasetDict(merged)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    final_ds.save_to_disk(output_path)

    if cleanup_shards:
        shutil.rmtree(shard_path_root, ignore_errors=True)

    volume.commit()
    counts = {split: len(final_ds[split]) for split in final_ds.keys()}
    return f"Merged background-removed dataset at {output_path} with splits: {counts}"


# =============================================================================
# Evaluation Function (using torch-fidelity)
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_torch_fidelity(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    metrics: str = "fid,kid",
    num_samples: int = 5000,
    batch_size: int = 128,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = None,
    override: bool = False,
    save_log_path: str = None,
):
    """
    Evaluate using torch-fidelity CLI.

    Uses the fidelity command to compute metrics directly.

    Args:
        method: 'ddpm'
        checkpoint: Path to checkpoint (relative to /data)
        metrics: Comma-separated: 'fid', 'kid', 'is' (default: 'fid,kid')
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_steps: Sampling steps (optional)
        sampler: Sampling method for DDPM ('ddpm' or 'ddim')
        eta: Eta parameter for DDIM sampling
        edge_source: Optional conditioning source for conditional models.
                     Use "dataset" to draw conditions from the training split.
        override: Force regenerate samples even if they exist
        save_log_path: Optional path to save results log file (relative to /data)
    """
    import sys
    import subprocess
    from pathlib import Path
    sys.path.insert(0, "/root")

    checkpoint_path = f"/data/{checkpoint}"

    # Put samples in same parent dir as checkpoint under samples/
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Include sampling config in folder names so different samplers/steps do not collide
    folder_parts = []
    if num_steps:
        folder_parts.append(f"{num_steps}steps")
    if sampler:
        folder_parts.append(sampler)
    folder_suffix = f"_{'_'.join(folder_parts)}" if folder_parts else ""
    
    generated_dir = str(checkpoint_dir / "samples" / f"generated{folder_suffix}")
    cache_dir = str(checkpoint_dir / "samples" / f"cache{folder_suffix}")
    # -----------------------------------------------------------

    # Prepare dataset path for torch-fidelity
    # torch-fidelity needs actual image files, not Arrow format
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    # Extract images from Arrow format if not already done
    import os
    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset['train']

        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item['image']
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)

            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    dataset_path = dataset_images_path

    # Step 1: Generate samples
    print("=" * 60)
    print("Step 1/2: Generating samples...")
    print("=" * 60)

    import os
    import shutil
    import glob

    # Check if samples already exist
    need_generation = True
    if os.path.exists(generated_dir) and not override:
        # Check for both png and jpg files
        existing_samples = (
            glob.glob(os.path.join(generated_dir, "*.png")) +
            glob.glob(os.path.join(generated_dir, "*.jpg")) + 
            glob.glob(os.path.join(generated_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)

        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(generated_dir)
    elif os.path.exists(generated_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(generated_dir)

    if need_generation:
        sample_cmd = [
            "python", "/root/sample.py",
            "--checkpoint", checkpoint_path,
            "--method", method,
            "--output_dir", generated_dir,
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
            "--sampler", sampler,
            "--eta", str(eta),
        ]

        if num_steps:
            sample_cmd.extend(["--num_steps", str(num_steps)])
        if edge_source is not None:
            sample_cmd.extend(["--edge_source", edge_source])

        subprocess.run(sample_cmd, check=True)
        print(f"Generated {num_samples} samples to {generated_dir}")
    else:
        print(f"Using existing samples from {generated_dir}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)

    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--cache-root", cache_dir,
        "--input1", generated_dir,
        "--input2", dataset_path,
    ]

    if "fid" in metrics:
        fidelity_cmd.append("--fid")
    if "kid" in metrics:
        fidelity_cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        fidelity_cmd.append("--isc")

    print(f"\nRunning command: {' '.join(fidelity_cmd)}\n")

    try:
        result = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)

        # Save results to log file if path provided
        if save_log_path:
            log_path = f"/data/{save_log_path}"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w') as f:
                f.write(f"# Evaluation Results\n")
                f.write(f"# Method: {method}\n")
                f.write(f"# Checkpoint: {checkpoint}\n")
                f.write(f"# Num Steps: {num_steps}\n")
                f.write(f"# Sampler: {sampler}\n")
                f.write(f"# Eta: {eta}\n")
                f.write(f"# Num Samples: {num_samples}\n")
                f.write(f"# Metrics: {metrics}\n\n")
                f.write(result.stdout)
            print(f"Results saved to {log_path}")

        volume.commit()
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the error output to help debug
        print(f"\nError running fidelity command:")
        print(f"Command: {' '.join(fidelity_cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        raise


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_controllability(
    method: str = "flow_matching",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = 1000,
    batch_size: int = 64,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = "dataset",
    edge_split: str = "train",
    eval_edge_method: str = "canny",
    mask_threshold: float = 0.5,
    tolerance_px: int = 1,
    canny_sigma: float = 1.2,
    canny_low: int = 80,
    canny_high: int = 200,
    xdog_sigma: float = 0.29,
    seed: int = 42,
    save_log_path: str = None,
):
    """
    Evaluate conditional controllability via edge adherence metrics.

    Produces JSON with:
    - edge_adherence_iou_mean/std
    - edge_precision_mean/std
    - edge_recall_mean/std
    - edge_f1_mean/std
    """
    import json
    import os
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"

    cmd = [
        "python",
        "/root/scripts/evaluate_controllability.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--sampler", sampler,
        "--eta", str(eta),
        "--edge_source", edge_source,
        "--edge_split", edge_split,
        "--eval_edge_method", eval_edge_method,
        "--mask_threshold", str(mask_threshold),
        "--tolerance_px", str(tolerance_px),
        "--canny_sigma", str(canny_sigma),
        "--canny_low", str(canny_low),
        "--canny_high", str(canny_high),
        "--xdog_sigma", str(xdog_sigma),
        "--seed", str(seed),
    ]
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if save_log_path is not None:
        log_path = f"/data/{save_log_path}"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd.extend(["--save_log_path", log_path])

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()

    parsed = None
    try:
        parsed = json.loads(stdout)
    except Exception:
        pass

    if save_log_path is not None and parsed is None:
        # Fallback if script stdout includes non-JSON lines.
        log_path = f"/data/{save_log_path}"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout)

    volume.commit()
    return stdout


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_csg(
    method: str = "flow_matching",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = 1000,
    batch_size: int = 64,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = "dataset",
    edge_split: str = "train",
    eval_edge_method: str = "canny",
    mask_threshold: float = 0.5,
    tolerance_px: int = 1,
    canny_sigma: float = 1.2,
    canny_low: int = 80,
    canny_high: int = 200,
    xdog_sigma: float = 0.29,
    seed: int = 42,
    save_log_path: str = None,
):
    """
    Evaluate Condition Sensitivity Gap (CSG):
    CSG = EAS_IoU(matched condition) - EAS_IoU(mismatched condition)
    """
    import json
    import os
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"

    cmd = [
        "python",
        "/root/scripts/evaluate_csg.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--sampler", sampler,
        "--eta", str(eta),
        "--edge_source", edge_source,
        "--edge_split", edge_split,
        "--eval_edge_method", eval_edge_method,
        "--mask_threshold", str(mask_threshold),
        "--tolerance_px", str(tolerance_px),
        "--canny_sigma", str(canny_sigma),
        "--canny_low", str(canny_low),
        "--canny_high", str(canny_high),
        "--xdog_sigma", str(xdog_sigma),
        "--seed", str(seed),
    ]
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if save_log_path is not None:
        log_path = f"/data/{save_log_path}"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd.extend(["--save_log_path", log_path])

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()

    parsed = None
    try:
        parsed = json.loads(stdout)
    except Exception:
        pass

    if save_log_path is not None and parsed is None:
        log_path = f"/data/{save_log_path}"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout)

    volume.commit()
    return stdout


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_lpips_reference(
    method: str = "flow_matching",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = 1000,
    batch_size: int = 64,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = "dataset",
    edge_split: str = "train",
    seed: int = 42,
    save_log_path: str = None,
):
    """
    Evaluate LPIPS to reference source images for the same conditioning edges.
    Lower is better.
    """
    import json
    import os
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    cmd = [
        "python",
        "/root/scripts/evaluate_lpips_reference.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--sampler", sampler,
        "--eta", str(eta),
        "--edge_source", edge_source,
        "--edge_split", edge_split,
        "--seed", str(seed),
    ]
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if save_log_path is not None:
        log_path = f"/data/{save_log_path}"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cmd.extend(["--save_log_path", log_path])

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()

    parsed = None
    try:
        parsed = json.loads(stdout)
    except Exception:
        pass

    if save_log_path is not None and parsed is None:
        log_path = f"/data/{save_log_path}"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout)

    volume.commit()
    return stdout


# =============================================================================
# Helper Functions for Logging
# =============================================================================

@app.function(
    image=image,
    timeout=60 * 5,
    volumes={"/data": volume},
)
def save_summary_log(log_path: str, content: str):
    """
    Save a summary log file to the Modal volume.

    Args:
        log_path: Path relative to /data (e.g., "q7_ablation/summary.txt")
        content: Content to write to the file
    """
    import os

    full_path = f"/data/{log_path}"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as f:
        f.write(content)

    volume.commit()
    print(f"Summary saved to {full_path}")
    return full_path


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main(
    action: str = "train",
    method: str = "ddpm",
    config: str = None,
    checkpoint: str = None,
    resume_from: str = None,
    iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_samples: int = None,
    num_steps: int = None,
    sampler: str = "ddpm",
    eta: float = 0.0,
    edge_source: str = None,
    edge_split: str = "train",
    eval_edge_method: str = "canny",
    mask_threshold: float = 0.5,
    tolerance_px: int = 1,
    canny_sigma: float = 1.2,
    canny_low: int = 80,
    canny_high: int = 200,
    xdog_sigma: float = 0.29,
    seed: int = 42,
    save_path: str = None,
    metrics: str = None,
    save_log_path: str = None,
    overfit_single_batch: bool = False,
    override: bool = False,
    bg_source_root: str = "celeba",
    bg_output_root: str = "celeba_bg_removed",
    bg_shard_root: str = "celeba_bg_removed_shards",
    bg_model: str = "u2netp",
    bg_num_shards: int = 6,
    bg_splits: str = "train,validation",
):
    """
    Main entry point for Modal CLI.

    See docs/QUICKSTART-MODAL.md for usage instructions.

    All parameters are read from config YAML files first, then overridden by command-line arguments.
    """
    if action == "download":
        result = download_dataset.remote()
        print(result)
    elif action == "precompute_bg_removed":
        result = precompute_bg_removed_dataset_parallel.remote(
            source_root=bg_source_root,
            output_root=bg_output_root,
            model_name=bg_model,
            num_shards=bg_num_shards,
            splits=bg_splits,
        )
        print(result)
    elif action == "merge_bg_removed":
        result = merge_bg_removed_from_shards.remote(
            shard_root=bg_shard_root,
            output_root=bg_output_root,
            cleanup_shards=False,
        )
        print(result)
    elif action == "train":
        # Read config to determine GPU count
        import yaml

        local_config_path = config or f"configs/{method}.yaml"
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)

        # Get num_gpus from config
        config_device = local_config['infrastructure'].get('device', 'cuda')
        num_gpus = local_config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
        if num_gpus is None:
            num_gpus = 1 if config_device == 'cuda' else 0

        # Get the appropriate training function
        train_fn = TRAIN_FUNCTIONS.get(num_gpus)
        if train_fn is None:
            raise ValueError(
                f"Unsupported num_gpus={num_gpus} in config. "
                f"Supported: 1-8"
            )

        result = train_fn.remote(
            method=method,
            config_path=config,
            resume_from=resume_from,
            num_iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_single_batch=overfit_single_batch,
        )
        print(result)
    elif action == "sample":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"
        
        sample_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'num_samples': num_samples,
            'num_steps': num_steps,
            'edge_source': edge_source,
        }
        
        # Add optional sampler and eta if they exist in locals or are passed
        # Note: the main function signature needs to include them
        if 'sampler' in locals() and sampler is not None:
            sample_kwargs['sampler'] = sampler
        if 'eta' in locals() and eta is not None:
            sample_kwargs['eta'] = eta
        if save_path is not None:
            sample_kwargs['save_path'] = save_path

        result = sample.remote(**sample_kwargs)
        print(result)
    elif action == "evaluate" or action == "evaluate_torch_fidelity":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        eval_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'override': override,
        }
        if metrics is not None:
            eval_kwargs['metrics'] = metrics
        if num_samples is not None:
            eval_kwargs['num_samples'] = num_samples
        if batch_size is not None:
            eval_kwargs['batch_size'] = batch_size
        if num_steps is not None:
            eval_kwargs['num_steps'] = num_steps
        if 'sampler' in locals() and sampler is not None:
            eval_kwargs['sampler'] = sampler
        if 'eta' in locals() and eta is not None:
            eval_kwargs['eta'] = eta
        if save_log_path is not None:
            eval_kwargs['save_log_path'] = save_log_path
        if edge_source is not None:
            eval_kwargs['edge_source'] = edge_source

        result = evaluate_torch_fidelity.remote(**eval_kwargs)
        print(result)
    elif action == "evaluate_controllability":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        controllability_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'num_samples': num_samples if num_samples is not None else 1000,
            'batch_size': batch_size if batch_size is not None else 64,
            'sampler': sampler,
            'eta': eta,
            'edge_source': edge_source if edge_source is not None else "dataset",
            'edge_split': edge_split,
            'eval_edge_method': eval_edge_method,
            'mask_threshold': mask_threshold,
            'tolerance_px': tolerance_px,
            'canny_sigma': canny_sigma,
            'canny_low': canny_low,
            'canny_high': canny_high,
            'xdog_sigma': xdog_sigma,
            'seed': seed,
        }
        if num_steps is not None:
            controllability_kwargs['num_steps'] = num_steps
        if save_log_path is not None:
            controllability_kwargs['save_log_path'] = save_log_path

        result = evaluate_controllability.remote(**controllability_kwargs)
        print(result)
    elif action == "evaluate_csg":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        csg_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'num_samples': num_samples if num_samples is not None else 1000,
            'batch_size': batch_size if batch_size is not None else 64,
            'sampler': sampler,
            'eta': eta,
            'edge_source': edge_source if edge_source is not None else "dataset",
            'edge_split': edge_split,
            'eval_edge_method': eval_edge_method,
            'mask_threshold': mask_threshold,
            'tolerance_px': tolerance_px,
            'canny_sigma': canny_sigma,
            'canny_low': canny_low,
            'canny_high': canny_high,
            'xdog_sigma': xdog_sigma,
            'seed': seed,
        }
        if num_steps is not None:
            csg_kwargs['num_steps'] = num_steps
        if save_log_path is not None:
            csg_kwargs['save_log_path'] = save_log_path

        result = evaluate_csg.remote(**csg_kwargs)
        print(result)
    elif action == "evaluate_lpips_reference":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        lpips_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'num_samples': num_samples if num_samples is not None else 1000,
            'batch_size': batch_size if batch_size is not None else 64,
            'sampler': sampler,
            'eta': eta,
            'edge_source': edge_source if edge_source is not None else "dataset",
            'edge_split': edge_split,
            'seed': seed,
        }
        if num_steps is not None:
            lpips_kwargs['num_steps'] = num_steps
        if save_log_path is not None:
            lpips_kwargs['save_log_path'] = save_log_path

        result = evaluate_lpips_reference.remote(**lpips_kwargs)
        print(result)
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: download, precompute_bg_removed, merge_bg_removed, train, sample, evaluate, evaluate_controllability, evaluate_csg, evaluate_lpips_reference")


@app.local_entrypoint()
def run_q7_ablation(
    checkpoint: str = "logs/ddpm_modal/ddpm_20260123_090933/checkpoints/ddpm_final.pt",
    steps_list: str = "100,300,500,700,900,1000",
):
    """.
    Q7: Sampling Steps Ablation Study using Modal Batch Processing.
    """
    # Parse steps list
    steps_list_parsed = [int(s.strip()) for s in steps_list.split(",")]
    
    print(f"\n--- Q7: Sampling Steps Ablation Study ---")
    print(f"Testing: {steps_list_parsed} steps")
    print(f"Checkpoint: {checkpoint}\n")

    # ============================================================================
    # Part 1: KID Evaluations (1000 samples each)
    # ============================================================================
    print(f"--- Part 1: KID Evaluations (1000 samples each) ---")
    print("[run] Launching KID evaluation jobs...")

    kid_handles = []
    for steps in steps_list_parsed:
        # We use .spawn() instead of .spawn_map() so we can pass keyword arguments
        # .spawn() returns a handle that we can wait on
        handle = evaluate_torch_fidelity.spawn(
            method="ddpm",
            checkpoint=checkpoint,
            metrics="kid",
            num_samples=1000,
            num_steps=steps,
            save_log_path=f"q7_ablation/kid_{steps}steps.txt",
        )
        kid_handles.append((steps, handle))

    print(f"[ok] Submitted {len(steps_list_parsed)} KID evaluation jobs!")
    print("[wait] Waiting for results...\n")

    # Wait for all jobs and collect results
    kid_results = {}
    for steps, handle in kid_handles:
        print(f"\n{'='*60}")
        print(f"Results for {steps} steps:")
        print(f"{'='*60}")
        try:
            result = handle.get()
            print(result)
            kid_results[steps] = result
        except Exception as e:
            print(f"[err] Error for {steps} steps: {e}")
            kid_results[steps] = f"ERROR: {e}"

    # ============================================================================
    # Part 2: Qualitative Samples (1 sample per step count)
    # ============================================================================
    print(f"\n--- Part 2: Qualitative Samples (1 sample per step count) ---")
    print("[run] Launching sample generation jobs...")

    sample_handles = []
    for steps in steps_list_parsed:
        handle = sample.spawn(
            method="ddpm",
            checkpoint=checkpoint,
            num_samples=1,
            num_steps=steps,
        )
        sample_handles.append((steps, handle))

    print(f"[ok] Submitted {len(steps_list_parsed)} sample generation jobs!")
    print("[wait] Waiting for sample generation to complete...\n")

    # Wait for all sample jobs
    for steps, handle in sample_handles:
        try:
            result = handle.get()
            print(f"[ok] {steps} steps: {result}")
        except Exception as e:
            print(f"[err] Error generating sample for {steps} steps: {e}")

    # ============================================================================
    # Summary - Save consolidated log file
    # ============================================================================
    print("\n[summary]")
    print(f"  - KID evaluations: {len(steps_list_parsed)} jobs (1000 samples each)")
    print(f"  - Qualitative samples: {len(steps_list_parsed)} jobs (1 sample each)")

    # Build consolidated summary content
    from datetime import datetime
    summary_lines = [
        "=" * 60,
        "Q7 Sampling Steps Ablation Study - KID Results Summary",
        "=" * 60,
        f"Checkpoint: {checkpoint}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Steps tested: {steps_list_parsed}",
        "",
        "-" * 60,
        "KID Scores by Step Count:",
        "-" * 60,
        "",
    ]

    for steps in steps_list_parsed:
        summary_lines.append(f"--- {steps} steps ---")
        if steps in kid_results:
            summary_lines.append(kid_results[steps])
        else:
            summary_lines.append("No result recorded")
        summary_lines.append("")

    summary_content = "\n".join(summary_lines)

    # Save summary to volume
    print("\n[run] Saving consolidated summary log...")
    save_summary_log.remote("q7_ablation/kid_summary.txt", summary_content)

    print(f"\n[ok] All jobs completed.")
    print(f"[path] Logs saved to /data/q7_ablation/ on Modal volume")
    print(f"   - Individual logs: kid_<steps>steps.txt")
    print(f"   - Summary: kid_summary.txt")
