import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
import os
import random
from .skill_interface import Skill, model_forward
from torch import Tensor


def obj_key_input_trans(x: Tensor):
    x = x.float()
    x = x[:, -1, ...]
    return x.unsqueeze(1)

class Encoder(nn.Module):
    def __init__(self, inp_ch=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(inp_ch, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.encoder(inp)
        return out

class KeyNet(nn.Module):
    def __init__(self, inp_ch=3, K=1):
        super().__init__()

        self.keynet = nn.Sequential(
            nn.Conv2d(inp_ch, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.reg = nn.Conv2d(128, K, 1)

    def forward(self, inp):
        x = self.keynet(inp)
        out = self.reg(x)
        return out

class RefineNet(nn.Module):
    def __init__(self, num_ch):
        super().__init__()

        self.refine_net = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_ch, 7, 1, 3),
            nn.BatchNorm2d(num_ch),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.refine_net(inp)
        return out

# https://github.com/ethanluoyc/transporter-pytorch/blob/master/transporter.py
def spatial_softmax(features):
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output

def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1) # N, K, 2

def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)

    return g_yx

def transport(source_keypoints, target_keypoints, source_features, target_features):
    out = source_features
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
    return out

class Transporter(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.encoder = Encoder(inp_ch=1)
        self.key_net = KeyNet(inp_ch=1, K=4)
        self.refine_net = RefineNet(num_ch=1)
        self.std = std
        
    def forward(self, source_img, target_img):
        # source img
        source_features = self.encoder(source_img)
        source_kn = self.key_net(source_img)
        source_kn = spatial_softmax(source_kn)
        source_keypoints = gaussian_map(source_kn, self.std)

        # target img
        target_features = self.encoder(target_img)
        target_kn = self.key_net(target_img)
        target_kn = spatial_softmax(target_kn)
        target_keypoints = gaussian_map(target_kn, self.std)

        # transport
        transport_features = transport(source_keypoints.detach(),
                                    target_keypoints,
                                    source_features.detach(),
                                    target_features)

        # RefineNet
        out = self.refine_net(transport_features)
        return out
    
    def get_skill(self, device, keynet_or_encoder='encoder'):
        input_transformation_function = obj_key_input_trans
        # here I should load the .pt model
        model_path = "skills/torch_models/vid-obj-key.pt"
        
        model = Transporter(std=self.std)
        state = torch.load(model_path, map_location=device)
        # load_state_dict returns an _IncompatibleKeys namedtuple; call it but keep the model
        _ = model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        
        if keynet_or_encoder == 'encoder':
            return Skill("obj_key_key", input_transformation_function, model.encoder, model_forward, None)
        elif keynet_or_encoder == 'keynet':
            return Skill("obj_key_enc", input_transformation_function, model.key_net, model_forward, None)
        else:
            raise ValueError("keynet_or_encoder must be either 'encoder' or 'keynet'")


class ObjectKeypointsDataset(Dataset):
    def __init__(self, path, ep=10, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.ep = ep

        # PRE-CACHE directory structure to avoid repeated os.listdir() calls
        self.envs = os.listdir(self.path)
        self.episode_lengths = {}

        print("Caching episode lengths...")
        for env in self.envs:
            env_path = f"{self.path}/{env}"
            for n in range(1, ep + 1):
                ep_path = f"{env_path}/episode_{n}"
                if os.path.exists(ep_path):
                    length = len(os.listdir(ep_path))
                    self.episode_lengths[(env, n)] = length
        print(f"Cached {len(self.episode_lengths)} episodes")

    def __len__(self):
        return len(self.episode_lengths) * 100  # Approximate

    def __getitem__(self, index):
        n, env, t, tp1 = index

        imgt = Image.open(f"{self.path}/{env}/episode_{n}/{t}.png")
        imgtp1 = Image.open(f"{self.path}/{env}/episode_{n}/{tp1}.png")

        if self.transform is not None:
            imgt = self.transform(imgt)
            imgtp1 = self.transform(imgtp1)

        return imgt, imgtp1

    def get_trajectory(self, env, idx):
        max_ep_len = self.episode_lengths.get((env, idx), 0)
        images = [Image.open(f'{self.path}/{env}/episode_{idx}/{t}.png')
                  for t in range(max_ep_len)]
        return [self.transform(im) for im in images]


class ObjectKeypointsSampler(Sampler):
    def __init__(self, dataset, cache_size=10000):
        self.dataset = dataset
        self.envs = os.listdir(self.dataset.path)
        self.cache_size = cache_size

        # PRE-CACHE: Build episode metadata once
        print("Building episode metadata cache...")
        self.episode_metadata = []
        for env in self.envs:
            env_path = os.path.join(self.dataset.path, env)
            for n in range(1, self.dataset.ep + 1):
                episode_path = os.path.join(env_path, f"episode_{n}")
                if os.path.exists(episode_path):
                    num_images = len(os.listdir(episode_path))
                    if num_images >= 21:
                        self.episode_metadata.append({
                            'env': env,
                            'episode': n,
                            'length': num_images
                        })
        if len(self.episode_metadata) == 0:
            raise ValueError("No valid episodes found in the dataset path. All episodes must have at least 21 frames.")
        print(f"Cached {len(self.episode_metadata)} valid episodes")
        self._generate_samples()

    def _generate_samples(self):
        """Generate a batch of samples to avoid per-iteration overhead"""
        self.samples = []
        for _ in range(self.cache_size):
            metadata = random.choice(self.episode_metadata)
            num_images = metadata['length']

            t_ind = np.random.randint(0, num_images - 20)
            tp1_ind = t_ind + np.random.randint(1, min(20, num_images - t_ind) + 1)

            self.samples.append((
                metadata['episode'],
                metadata['env'],
                t_ind,
                tp1_ind
            ))

    def __iter__(self):
        idx = 0
        while True:
            if idx >= len(self.samples):
                self._generate_samples()
                idx = 0

            yield self.samples[idx]
            idx += 1

    def __len__(self):
        return self.cache_size