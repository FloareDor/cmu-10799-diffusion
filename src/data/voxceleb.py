
import os
import random
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video, read_video_timestamps

class VoxCelebDataset(Dataset):
    """
    Dataset for Audio-Conditioned Face Inpainting using VoxCeleb videos.
    """
    def __init__(
        self,
        root: str = "./data/voxceleb_video",
        image_size: int = 64,
        audio_window_sec: float = 0.2, # Window size for audio (0.2s is common for lip sync)
        sample_rate: int = 16000,
        fps: int = 25, # VoxCeleb is usually 25fps
    ):
        self.root = root
        self.image_size = image_size
        self.audio_window_sec = audio_window_sec
        self.sample_rate = sample_rate
        self.fps = fps
        
        self.video_files = [
            os.path.join(root, f) for f in os.listdir(root) 
            if f.endswith('.mp4') or f.endswith('.avi')
        ]
        
        if len(self.video_files) == 0:
            print(f"Warning: No videos found in {root}")
            
        print(f"Found {len(self.video_files)} videos in {root}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Audio transform: Resample to target rate
        self.audio_transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)


    def __len__(self):
        # We process videos dynamically. 
        # Ideally we'd know total frames, but that's slow to count.
        # So we'll just treat "length" as number of videos * 100 samples per video 
        # or just return number of videos and random sample inside __getitem__?
        # Standard approach for video datasets is usually pre-processing frames.
        # But for this baseline, let's just say length = num_videos * 50
        return len(self.video_files) * 50

    def __getitem__(self, idx):
        # Map linear index to video index
        vid_idx = idx // 50
        video_path = self.video_files[vid_idx % len(self.video_files)]
        
        try:
            # Get metadata (duration/timestamps)
            # This can be slow, might want to cache?
            pts, video_fps = read_video_timestamps(video_path, pts_unit='sec')
            
            # If failed to get timestamps or empty
            if len(pts) < 10:
                # Pick another video randomly
                return self.__getitem__(random.randint(0, len(self) - 1))
                
            # Pick a random frame index, avoiding edges
            frame_idx = random.randint(5, len(pts) - 5)
            timestamp = pts[frame_idx]
            
            # Read video frame (just 1 frame)
            # read_video allows start/end pts.
            # We assume constant FPS for simplicity to find end_pts
            # but read_video is a bit tricky with exact frames.
            # simpler: read a small chunk around timestamp
            
            v_frames, a_frames, info = read_video(
                video_path, 
                start_pts=timestamp, 
                end_pts=timestamp + (1.0/self.fps), 
                pts_unit='sec'
            )
            
            if v_frames.shape[0] == 0:
                 return self.__getitem__(random.randint(0, len(self) - 1))
                 
            # Take first frame found
            image = v_frames[0].permute(2, 0, 1) # HWC -> CHW
            
            # Process Audio
            # We need audio centered around this frame (or causal)
            # Window: timestamp +/- window/2 ? 
            # Or just timestamp to timestamp + window?
            # Let's do centered: [t - window/2, t + window/2]
            
            # If a_frames is returned by read_video, it corresponds to the video chunk.
            # But we only read 1 frame (0.04s). We need 0.2s of audio.
            # So we should have requested a larger chunk?
            # Or read audio separately? 
            # read_video returns ALL audio if you don't specify stream? 
            # Actually read_video crops audio to video timestamps usually.
            
            # Better approach: Read audio separately or read 0.2s of video and pick middle frame.
            start_sec = max(0, timestamp - self.audio_window_sec / 2)
            end_sec = start_sec + self.audio_window_sec
            
            v_chunk, a_chunk, info = read_video(
                video_path,
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit='sec'
            )
            
            # if audio is empty
            if a_chunk.numel() == 0:
                 return self.__getitem__(random.randint(0, len(self) - 1))

            # Pick the middle video frame of this chunk
            mid_v = v_chunk.shape[0] // 2
            image = v_chunk[mid_v].permute(2, 0, 1) # HWC -> CHW
            
            # Preprocess Image
            # Center Crop to face (heuristic: VoxCeleb is mostly centered faces)
            # We crop the center square.
            C, H, W = image.shape
            min_dim = min(H, W)
            cropper = transforms.CenterCrop(min_dim)
            image = cropper(image)
            image = self.transform(image) # Resize to 64x64, Normalize
            
            # Create Masked Image
            # Mask lower half (mouth region)
            # 64x64 image. Mask rows 32 to 64?
            masked_image = image.clone()
            masked_image[:, 32:, :] = 0  # Zero out lower half
            
            # Preprocess Audio
            # Mix down to mono
            audio = a_chunk 
            if audio.shape[0] > 1: # channels dim is 0 or 1? read_video gives (samples, channels) usually? 
                # wait, torchvision read_video audio is (C, N) or (N, C)?
                # Doc says: (samples, channels) usually, wait...
                # Let's check info['audio_fps'] if needed.
                # Actually, standard is (1, N) for mono.
                # If shape is (N, C), permute.
                pass
                
            # Ensure shape (C, N)
            if audio.shape[-1] < 5: # likely channels is last
                audio = audio.permute(1, 0)
                
            # Mono
            audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample
            # We need to know original rate. info['audio_fps']
            orig_freq = info['audio_fps']
            if orig_freq != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=self.sample_rate)
                audio = resampler(audio)
            
            # Pad/Truncate to fixed length
            # 0.2s * 16000 = 3200 samples
            target_len = int(self.audio_window_sec * self.sample_rate)
            if audio.shape[1] < target_len:
                padding = target_len - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            else:
                audio = audio[:, :target_len]
                
            # Return dict matching FlowMatching expectations
            # Clean image is the target x_1
            # We return: x_1, condition
            
            return image, {'image': masked_image, 'audio': audio}

        except Exception as e:
            # print(f"Error loading {video_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

def unnormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

