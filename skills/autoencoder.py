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

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # Choose a random episode from the available indices
        num_images = 0
        while num_images < 4:
            episode_idx = np.random.choice(self.idxs)
            episode_path = self.episode_paths[episode_idx]
            
            # Get number of images in this episode
            num_images = len([f for f in os.listdir(episode_path) if f.endswith('.png')])
        
        # Choose a random timestep (ensuring we can get 4 consecutive frames)
        t = np.random.randint(3, num_images)
        
        # Load 4 consecutive frames and stack them
        frames = []
        for i in range(t-3, t+1):  # Load frames at t-3, t-2, t-1, t
            pil = Image.open(f"{episode_path}/{i}.png").convert('L').resize((self.frame_size, self.frame_size))
            img = np.array(pil, dtype=np.float32) / 255.0
            frames.append(img)
        
        # Stack frames to create input (4, H, W)
        stacked_frames = np.stack(frames, axis=0)
        input_tensor = torch.from_numpy(stacked_frames).contiguous().float()
        
        # Target is just the last frame (1, H, W)
        target_tensor = torch.from_numpy(frames[-1]).unsqueeze(0).contiguous().float()
        
        return input_tensor, target_tensor