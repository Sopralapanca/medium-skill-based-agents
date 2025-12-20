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
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # changed stride from 2 to 1 wrt original
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=8, stride=4),
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
        model_path = "skills/torch_models/nature-encoder.pt"

        model = self.load_state_dict(torch.load(model_path, map_location=device), strict=True).to(device)
        model.eval()
        input_transformation_function = self.autoencoder_input_trans
        return Skill("autoencoder", input_transformation_function, model.encoder, model_forward, None)
    
    
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
        episode_idx = np.random.choice(self.idxs)
        episode_path = self.episode_paths[episode_idx]
        
        # Get number of images in this episode
        num_images = len([f for f in os.listdir(episode_path) if f.endswith('.png')])
        
        # Choose a random timestep
        t = np.random.randint(0, num_images)
        # Load and preprocess the image as grayscale (single channel)
        pil = Image.open(f"{episode_path}/{t}.png").convert('L').resize((self.frame_size, self.frame_size))
        img = np.array(pil, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).contiguous().float()
        # add channel dimension -> (1, H, W)
        img = img.unsqueeze(0)
        return img