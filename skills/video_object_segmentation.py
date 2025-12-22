import torch
import torch.nn as nn
import numpy as np
from kornia.losses import ssim_loss
import glob
import os
import random
from PIL import Image
import cv2
from .skill_interface import Skill
from torch import Tensor


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class VideoObjectSegmentationModel(nn.Module):
    def __init__(self, device="cpu", emb_size=512, K=20, depth=24, H=84, W=84, curriculum_steps=100000):
        super().__init__()
        self.device = device

        # Regularization curriculum: linearly increase from 0 to 1 over curriculum_steps
        # Paper uses 100k steps
        self.of_reg_cur = 0.0
        self.of_reg_inc = 1.0 / curriculum_steps
        self.emb_size = emb_size
        self.K = K
        self.depth = depth
        self.H = H
        self.W = W
        self.final_conv_size = 7 * 7 * 64
        self.flow_c = 0.01

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten(),
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_conv = nn.Linear(self.final_conv_size, emb_size)

        self.obj_trans = nn.Linear(emb_size, self.K * 2)
        self.cam_trans = nn.Linear(emb_size, 2)

        self.fc_m1 = nn.Linear(emb_size, 21 * 21 * self.depth)
        self.conv_m1 = nn.Conv2d(self.depth, self.depth, 3, 1, 1)
        self.conv_m2 = nn.Conv2d(self.depth, self.depth, 3, 1, 1)
        self.conv_m3 = nn.Conv2d(self.depth, self.K, 1, 1)
        self.upsample1 = nn.UpsamplingBilinear2d(size=(42, 42))
        self.upsample2 = nn.UpsamplingBilinear2d(size=(84, 84))

        # Cache mesh grid - create once and register as buffer
        self.register_buffer('mesh_grid', self._create_mesh_grid())
        
        # Cache img_size_f - create once
        self.register_buffer('img_size_f', torch.from_numpy(
            self.flow_c * np.array([[self.H], [self.W]], dtype=np.float32)
        ))

    def compute_masks(self, input):
        # input shape = [ BS x C x H x W ]

        # Basic CNN
        x = self.cnn(input)
        x = self.relu(self.fc_conv(x))

        # Object Masks
        m = self.fc_m1(x)
        m = m.view((-1, self.depth, 21, 21))
        m = self.upsample1(m)

        # conv 5
        y = self.conv_m1(m)  # padding=1 => padding='same'
        m = self.relu(m + y)
        m = self.upsample2(m)

        # conv 6
        z = self.conv_m2(m)
        m = self.relu(m + z)

        # [ BS x K x H x W ]
        m = self.conv_m3(m)
        m = self.sigmoid(m)

        return m

    def forward(self, input):
        # input shape = [ BS x C x H x W ]

        # Basic CNN
        x = self.cnn(input)
        x = self.relu(self.fc_conv(x))

        # Object Masks
        m = self.fc_m1(x)
        m = m.view((-1, self.depth, 21, 21))
        m = self.upsample1(m)

        # conv 5
        y = self.conv_m1(m)  # padding=1 => padding='same'
        m = self.relu(m + y)
        m = self.upsample2(m)

        # conv 6
        z = self.conv_m2(m)
        m = self.relu(m + z)

        # [ BS x K x H x W ]
        m = self.conv_m3(m)
        m = self.sigmoid(m)
        self.object_masks = m

        # Object Translation
        ot = self.obj_trans(x)
        # [ BS x K x 2 ]
        ot = torch.reshape(ot, (-1, self.K, 2))

        # Mesh Grid
        #mesh_grid = self._create_mesh_grid().to(self.device)
        mesh_grid = self.mesh_grid

        # Optical Flow
        # [ BS x K x 1 x H x W ]
        m_reshape = torch.unsqueeze(m, 2)

        # [ BS x K x 2 x 1 x 1 ]
        ot_reshape = torch.unsqueeze(torch.unsqueeze(ot, -1), -1)

        translation_masks = m_reshape * ot_reshape
        self.translation_masks = translation_masks

        # [ BS x 2 x H x W ]
        flow = torch.sum(translation_masks, 1)

        # [ BS x 2 x H*W ]
        flat_flow = torch.reshape(flow, (-1, 2, self.W * self.H))

        # Camera Translation
        # [ BS x 2 ]
        ct = self.cam_trans(x)
        ct = torch.unsqueeze(ct, -1)

        # add camera translation to flow
        # [ BS x 2 x H*W ]
        flat_flow = flat_flow + ct

        # Add in the default coordinates
        # img_size_f = torch.from_numpy(
        #     self.flow_c * np.array([[self.H], [self.W]], dtype=np.float32)
        # ).to(self.device)
        img_size_f = self.img_size_f
        img_size_flat_flow = img_size_f * flat_flow

        # [ BS x 2 x H*W]
        sampling_coords = torch.add(img_size_flat_flow, mesh_grid)

        # Computer transformed image
        y_s = sampling_coords[:, 0, :]
        ys_flat = torch.reshape(y_s, (-1,))

        x_s = sampling_coords[:, 1, :]
        xs_flat = torch.reshape(x_s, (-1,))

        # x1 -> x0
        x1 = input[:, 1, :, :]
        source_frames = torch.unsqueeze(x1, 1)

        # Interpolate
        out = self._interpolate(source_frames, xs_flat, ys_flat, (1, self.H, self.W))
        # [ BS x 1 x H x W ]
        out = torch.reshape(out, (input.size(0), 1, self.H, self.W))

        return out

    def compute_loss(self, x, x_):
        # DSSIM
        out_loss = ssim_loss(x, x_, 11)

        # L1 reg for translations masks
        of_loss_reg = (
            torch.abs(self.translation_masks).mean(-1).mean(-1).mean(-1).mean(-1)
        )

        loss = out_loss + self.of_reg_cur * of_loss_reg

        return loss.mean()

    def update_reg(self):
        # increase regularization
        self.of_reg_cur = min(self.of_reg_cur + self.of_reg_inc, 1)

    def _create_mesh_grid(self):
        x_lin = torch.linspace(0.0, self.W - 1.0, self.W)
        y_lin = torch.linspace(0.0, self.H - 1.0, self.H)

        grid_x, grid_y = torch.meshgrid(x_lin, y_lin)
        # meshgrid -> pytorch != tf :)
        grid_x = grid_x.t()
        grid_y = grid_y.t()

        grid_x = torch.reshape(grid_x, (1, -1))
        grid_y = torch.reshape(grid_y, (1, -1))

        grid = torch.cat([grid_y, grid_x], 0)

        return grid

    def _repeat(self, x, n_rep):
        a = torch.unsqueeze(torch.ones(n_rep, device=self.device), 1)
        rep = a.permute(1, 0)
        x = torch.reshape(x, (-1, 1)).to(torch.float32)
        y = torch.matmul(x, rep)
        y = torch.reshape(y, (-1,))
        return y
    
    def _interpolate(self, im, x, y, out_size):
        bs, c, h, w = im.shape
        
        # Normalize coordinates to [-1, 1]
        x_normalized = 2.0 * x / (w - 1) - 1.0
        y_normalized = 2.0 * y / (h - 1) - 1.0
        
        # Reshape to grid format
        out_h, out_w = out_size[1], out_size[2]
        grid_x = x_normalized.view(bs, out_h, out_w)
        grid_y = y_normalized.view(bs, out_h, out_w)
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        output = torch.nn.functional.grid_sample(
            im, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return output

    def vos_output_masks(self, model, x):
        return model.compute_masks(x)

    def vid_obj_seg_input_trans(self, x: Tensor):
        x = x.float()
        first_frames = torch.mean(x[:, :2, ...], 1)
        second_frames = torch.mean(x[:, 2:, ...], 1)
        s = torch.stack([first_frames, second_frames])
        norm_s = s / 255.0
        return norm_s.permute(1, 0, 2, 3)

    def get_skill(self, device):
        model_path = "skills/torch_models/vid-obj-seg.pt"

        model = VideoObjectSegmentationModel(device=device)
        state = torch.load(model_path, map_location=device)
        _ = model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)

        return Skill(
            "vid_obj_seg",
            self.vid_obj_seg_input_trans,
            model,
            self.vos_output_masks,
            None,
        )

class VOSDataset():
    def __init__(self, batch_size, num_frames, data_path, H=84, W=84, game=None, frame_skip=3):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_skip = frame_skip  # Sample every N frames for more motion
        self.H = H
        self.W = W
        self.data_path = data_path
        self.envs = os.listdir(self.data_path)
        self.episodes = {}

        # NEW: cache for preloaded episodes
        self.cache = {}

        for env in self.envs:
            if game is not None and game.lower() not in env.lower():
                continue

            episode_paths = sorted(os.listdir(os.path.join(self.data_path, env)))

            for episode_path in episode_paths:
                key = env + "/" + episode_path
                frame_paths = sorted(
                    glob.glob(os.path.join(self.data_path, env, episode_path, "*.png")),
                    key=lambda x: int(os.path.basename(x).split('.')[0]),
                )

                if len(frame_paths) == 0:
                    continue

                self.episodes[key] = frame_paths

                # -------- PRELOAD EPISODE INTO RAM --------
                frames = []
                for path in frame_paths:
                    img = Image.open(path).convert("RGB")
                    img_np = np.array(img)

                    # RGB -> Grayscale
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                    # Resize
                    resized = cv2.resize(
                        gray, (self.W, self.H), interpolation=cv2.INTER_AREA
                    )

                    # Normalize to [0, 1]
                    frames.append(resized.astype(np.float32) / 255.0)

                # [T, H, W]
                self.cache[key] = np.stack(frames, axis=0)

        # Train / validation split (unchanged)
        all_episodes = sorted(
            [key for key in self.episodes.keys() if "episode_" in key],
            key=lambda s: int(s.split("/")[-1].split("_")[-1])
        )
        split = int(0.8 * len(all_episodes))
        self.train_data_keys = all_episodes[:split]
        self.valid_data_keys = all_episodes[split:]

        print(f"Loaded {len(self.cache)} episodes into RAM")

    def get_batch(self, data_type):
        if data_type == "train":
            episode_keys = self.train_data_keys
        else:
            episode_keys = self.valid_data_keys

        # Only episodes long enough
        valid_keys = [
            k for k in episode_keys
            if self.cache[k].shape[0] >= (self.num_frames - 1) * self.frame_skip + 1
        ]

        frames = np.zeros(
            (self.batch_size, self.num_frames, self.H, self.W),
            dtype=np.float32
        )

        for bs in range(self.batch_size):
            k = np.random.choice(valid_keys)
            episode = self.cache[k]

            max_start_idx = episode.shape[0] - (self.num_frames - 1) * self.frame_skip - 1
            idx = random.randint(0, max_start_idx)

            # Sample frames with frame_skip spacing
            for i in range(self.num_frames):
                frames[bs, i] = episode[idx + i * self.frame_skip]
        
        if torch.cuda.is_available():
            return torch.from_numpy(frames).pin_memory()
        else:
          return torch.from_numpy(frames)
