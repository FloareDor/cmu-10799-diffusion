"""
Download VoxCeleb Dataset from Hugging Face

This script downloads the VoxCeleb dataset (video/audio).
Default repo: 101arrowz/vox_celeb (VoxCeleb1)

Usage:
    python download_voxceleb.py --output_dir ./data/voxceleb --split test
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

def download_voxceleb(
    repo_name: str = "ProgramComputer/voxceleb",
    output_dir: str = "./data/voxceleb",
    split: str = "test",
    max_samples: int = None,
    streaming: bool = True
):
    """
    Download VoxCeleb dataset.
    
    Args:
        repo_name: HuggingFace repo name
        output_dir: Output directory
        split: Split to download
        max_samples: Optional limit on number of samples
        streaming: Whether to stream dataset (useful for large datasets)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return

    print("=" * 60)
    print(f"Downloading VoxCeleb from {repo_name}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    # We use streaming=True by default for large datasets like VoxCeleb
    print(f"Loading dataset (streaming={streaming})...")
    try:
        dataset = load_dataset(repo_name, split=split, streaming=streaming)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Try 'google/voxceleb1' or check your HF token.")
        return

    print("Download started...")
    
    count = 0
    # Create subdirectories
    (output_dir / "video").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    
    metadata = []
    
    for i, item in enumerate(tqdm(dataset)):
        if max_samples and i >= max_samples:
            break
            
        # Inspect item structure on first run
        if i == 0:
            print(f"\nSample data keys: {item.keys()}")
            
        # Try to find video/audio paths or bytes
        # Structure varies by repo. 
        # ProgramComputer/voxceleb often has 'file' or 'path' pointing to .mp4
        
        # We need to save the actual file.
        # If it's streaming, 'audio' or 'video' might be a dict with 'path' and 'bytes'
        
        file_id = item.get('id', item.get('client_id', f"{i:06d}"))
        
        # Handle different potential keys
        # This is a generic handler, might need adjustment for specific repo
        if 'video' in item:
            # If it's bytes
            video_data = item['video']
            if isinstance(video_data, dict) and 'bytes' in video_data:
                mode = 'wb'
                data = video_data['bytes']
            else:
                # If it's a path, we might need to download it if it's a URL, 
                # but load_dataset usually handles downloading.
                # If streaming, we might just get metadata.
                pass
                
        # For now, let's just inspect what we get and save it if possible.
        # Assuming ProgramComputer/voxceleb structure or 101arrowz
        
        # If we can't easily save video, we might just stop and report structure.
        if i == 0:
           print(f"First item: {str(item)[:500]}...")

        count += 1
        
    print(f"\nProcessed {count} samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default='ProgramComputer/voxceleb')
    parser.add_argument('--output_dir', type=str, default='./data/voxceleb')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_samples', type=int, default=10)
    
    args = parser.parse_args()
    
    download_voxceleb(
        repo_name=args.repo,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples
    )

if __name__ == '__main__':
    main()
