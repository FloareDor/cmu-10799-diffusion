import argparse
from huggingface_hub import list_repo_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', default='ProgramComputer/voxceleb')
    args = parser.parse_args()
    
    print(f"Listing files in {args.repo_id}...")
    try:
        files = list_repo_files(args.repo_id, repo_type="dataset")
        for f in files[:20]:
            print(f)
        if len(files) > 20:
            print(f"... and {len(files)-20} more.")
            
        # Check for video extensions
        video_files = [f for f in files if f.endswith('.mp4') or f.endswith('.avi')]
        print(f"\nFound {len(video_files)} video files.")
        if video_files:
            print(f"Example: {video_files[0]}")
            
        # Check for zips that might contain video
        zips = [f for f in files if f.endswith('.zip') or f.endswith('.tar.gz')]
        print(f"\nFound {len(zips)} archives.")
        if zips:
            print(f"Example: {zips[0]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
