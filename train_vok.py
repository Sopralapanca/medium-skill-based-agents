import torch
from pathlib import Path
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader
from skills.video_object_keypoints import (
    ObjectKeypointsDataset,
    ObjectKeypointsSampler,
    Encoder,
    KeyNet,
    RefineNet,
    Transporter
)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

IMG_SZ = 84
# Load config for max iterations
_config_path = Path(__file__).resolve().parent / "configs.yaml"
_config = {}
if _config_path.exists():
    with open(_config_path, "r") as f:
        _config = yaml.safe_load(f) or {}
MAX_ITER = _config.get("max_training_steps", 1000000)
EPISODES = _config.get("EPISODES", 10)
batch_size = _config.get("batch_size", 32)
image_channels = 1
K = 4
lr = 1e-3
lr_decay = 0.95
lr_decay_len = int(1e5)
IMG_SZ = _config.get("IMG_SZ", 84)
data_path = _config.get("data_path", "./data")

def main():
    
    os.makedirs('./skills/torch_models/', exist_ok=True)
    # Ensure GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    encoder = Encoder(image_channels)
    key_net = KeyNet(image_channels, K)
    refine_net = RefineNet(image_channels)
    transporter = Transporter(encoder, key_net, refine_net)
    transporter = transporter.to(device)  # Move to GPU
    transporter.train()

    # Data augmentation & preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SZ, IMG_SZ)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    t_dataset = ObjectKeypointsDataset(data_path, EPISODES, transform=transform)
    t_sampler = ObjectKeypointsSampler(t_dataset)

    # CRITICAL: Use num_workers > 0 for parallel data loading
    # pin_memory=True speeds up CPU->GPU transfer
    t_data_loader = DataLoader(
        t_dataset,
        batch_size=batch_size,
        sampler=t_sampler,
        num_workers=2,  # Adjust based on your CPU cores
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )

    v_dataset = ObjectKeypointsDataset(data_path, EPISODES, transform=transform)
    v_sampler = ObjectKeypointsSampler(v_dataset)
    v_data_loader = DataLoader(
        v_dataset,
        batch_size=2*batch_size,
        sampler=v_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(transporter.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_len, lr_decay)

    pbar = tqdm(enumerate(t_data_loader), total=MAX_ITER, desc="Training")
    VAL_INTERVAL = 500
    best_loss = 100
    best_step = 0
    eval_loss = 0.

    for i, (xt, xtp1) in pbar:
        if i > MAX_ITER:
            break

        # Training step
        transporter.train()
        xt = xt.to(device, non_blocking=True)  # non_blocking for async transfer
        xtp1 = xtp1.to(device, non_blocking=True)

        optimizer.zero_grad()
        reconstruction = transporter(xt, xtp1)
        loss = F.mse_loss(reconstruction, xtp1)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation - only every VAL_INTERVAL steps
        if i % VAL_INTERVAL == 0:
            with torch.no_grad():
                transporter.eval()
                eval_losses = []

                for j, (xv, xv1) in enumerate(v_data_loader):
                    if j >= 5:  # Average over 5 batches
                        break
                    xv = xv.to(device, non_blocking=True)
                    xv1 = xv1.to(device, non_blocking=True)
                    r = transporter(xv, xv1)
                    eval_losses.append(F.mse_loss(r, xv1).item())

                eval_loss = np.mean(eval_losses)

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_step = i
                    torch.save(transporter.state_dict(), './skills/torch_models/vid-obj-key.pt')

        last_lr = scheduler.get_last_lr()[0]

        pbar.set_postfix({
            'train_loss': f'{loss.item():.4f}',
            'val_loss': f'{eval_loss:.4f}',
            'best_val': f'{best_loss:.4f}',
            'best_step': f'{best_step}',
            'lr': f'{last_lr:.2e}'
        })


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
