"""
Download VoxCeleb Videos from YouTube

This script:
1. Downloads vox1_meta.csv from Hugging Face
2. Parses it to get YouTube IDs
3. Downloads videos using yt-dlp
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
import yt_dlp

def download_voxceleb_yt(
    output_dir: str = "./data/voxceleb_yt",
    max_samples: int = 10,
    split: str = "test"
):
    print(f"Downloading metadata...")
    try:
        meta_path = hf_hub_download(
            repo_id="ProgramComputer/voxceleb",
            filename="vox1/vox1_meta.csv",
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return

    print(f"Metadata saved to {meta_path}")
    
    # Read metadata
    # The CSV format usually has:
    # VoxCeleb1:
    # VGGFace1 ID, Name, Gender, Nationality, Set (dev/test)
    # But ProgramComputer might strictly follow original:
    # VoxCeleb1 meta: VoxCeleb1 ID, VGGFace1 ID, Gender, Nationality, Set
    
    # Wait, the meta csv usually maps ID to Name. 
    # The video mapping is in a different file for VoxCeleb1?
    # Original VoxCeleb1 has list of files.
    # The file paths in VoxCeleb1 are like `id10270/5r0y8H4yM4k/00001.mp4`
    # checking the content of meta.csv will help.
    
    df = pd.read_csv(meta_path, sep='\t')
    print("Metadata columns:", df.columns)
    print(df.head())
    
    # We need a file that maps IDs to YouTube URLs.
    # For VoxCeleb, the directory structure IS the mapping: id/video_id/clip_id
    # But where is the list of all video IDs?
    # It might be in `vox1_test_txt.zip` or similar, containing checks.
    
    # Let's see if we can get a file list.
    try:
        txt_path = hf_hub_download(
            repo_id="ProgramComputer/voxceleb",
            filename="vox1/vox1_test_txt.zip",
            repo_type="dataset"
        )
        import zipfile
        with zipfile.ZipFile(txt_path, 'r') as z:
            # List files in zip
            files = z.namelist()
            print(f"Found {len(files)} files in text zip")
            # Example: txt/id10270/5r0y8H4yM4k/00001.txt
            # The middle part '5r0y8H4yM4k' is the YouTube ID!
            
            # Let's extract unique YouTube IDs from this list
            youtube_ids = set()
            for f in files:
                parts = f.split('/')
                if len(parts) >= 3:
                    yt_id = parts[2]
                    youtube_ids.add(yt_id)
            
            print(f"Found {len(youtube_ids)} unique YouTube IDs in test set")
            
    except Exception as e:
        print(f"Error reading text zip: {e}")
        return
        
    # Download videos
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
    }

    count = 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for yt_id in list(youtube_ids)[:max_samples]:
            url = f"https://www.youtube.com/watch?v={yt_id}"
            print(f"Downloading {url} ...")
            try:
                ydl.download([url])
                count += 1
            except Exception as e:
                print(f"Failed to download {yt_id}: {e}")
                
    print(f"Downloaded {count} videos to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data/voxceleb_video')
    parser.add_argument('--max_samples', type=int, default=5)
    args = parser.parse_args()
    
    download_voxceleb_yt(
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
