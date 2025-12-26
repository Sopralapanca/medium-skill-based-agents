import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torch import Tensor
from skills.skill_interface import Skill, model_forward

class Autoencoder(nn.Module):
    def __init__(self, channels=1):
        super(Autoencoder, self).__init__()
        # Encoder - takes 4 stacked frames as input
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Changed to accept 4 input channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # changed stride from 2 to 1 wrt original
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Decoder - outputs single frame reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4),  # Output 1 channel
            nn.Sigmoid()  # Sigmoid to ensure pixel values are between 0 and 1
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def autoencoder_input_trans(self, x: Tensor):
        # x is of shape 32x4x84x84, because there are 4 frame stacked, pick only the last frame and return a tensor of shape 32x1x84x84
        x = x[:, -1:, :, :]
        return x.float()

    def get_skill(self, device):
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"

        self.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        self.to(device)
        self.eval()
        
        input_transformation_function = self.autoencoder_input_trans
        return Skill("autoencoder", input_transformation_function, self.encoder, model_forward, None)
    
    
class AutoencoderDataset(Dataset):
    def __init__(self, episode_paths, idxs, frame_size):
        """
        Args:
            episode_paths: Dictionary mapping episode index to full path (e.g., {0: 'env_1/episode_1', 1: 'env_1/episode_2', ...})
            idxs: List of episode indices to use (for train/val split)
            frame_size: Size to resize images to
        """
        super().__init__()
        self.episode_paths = episode_paths
        self.idxs = idxs
        self.frame_size = frame_size
        
        # Pre-cache episode metadata to avoid repeated directory listings
        print("Caching episode metadata...")
        self.episode_lengths = {}
        self.valid_episodes = []  # Only episodes with at least 4 frames
        for idx in idxs:
            episode_path = episode_paths[idx]
            num_images = len([f for f in os.listdir(episode_path) if f.endswith('.png')])
            if num_images >= 4:
                self.episode_lengths[idx] = num_images
                self.valid_episodes.append(idx)
        print(f"Cached {len(self.valid_episodes)} valid episodes")

    def __len__(self):
        return len(self.valid_episodes)  # One sample per episode per epoch

    def __getitem__(self, idx):
        # Choose a random episode from the valid episodes (with pre-cached lengths)
        episode_idx = np.random.choice(self.valid_episodes)
        episode_path = self.episode_paths[episode_idx]
        num_images = self.episode_lengths[episode_idx]
        
        # Choose a random timestep (ensuring we can get 4 consecutive frames)
        t = np.random.randint(3, num_images)
        
        # Pre-allocate numpy array for better performance
        frames = np.empty((4, self.frame_size, self.frame_size), dtype=np.float32)
        
        # Load 4 consecutive frames and stack them
        for j, i in enumerate(range(t-3, t+1)):  # Load frames at t-3, t-2, t-1, t
            pil = Image.open(f"{episode_path}/{i}.png").convert('L').resize((self.frame_size, self.frame_size), Image.BILINEAR)
            frames[j] = np.asarray(pil, dtype=np.float32) / 255.0
        
        # Convert to tensors
        input_tensor = torch.from_numpy(frames).contiguous()
        target_tensor = torch.from_numpy(frames[-1:]).contiguous()  # Keep as (1, H, W)
        
        return input_tensor, target_tensor