import torch
from pathlib import Path
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader
from skills.video_object_keypoints import (
    ObjectKeypointsDataset,
    ObjectKeypointsSampler,
    spatial_softmax,
    gaussian_map,
    compute_keypoint_location_mean,
    transport,
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
MAX_ITER = 1000 #_config.get("max_training_steps", 1000000)
batch_size = _config.get("batch_size", 32)
image_channels = 1
K = 4
lr = 1e-3
lr_decay = 0.95
lr_decay_len = int(1e5)
IMG_SZ = _config.get("IMG_SZ", 84)
data_path = "./data_pt" #_config.get("data_path", "./data")

def main():
    
    os.makedirs('./skills/torch_models/', exist_ok=True)
    
    model_path = './skills/torch_models/vid-obj-key.pt'
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Ensure GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    transporter = Transporter(inp_ch=image_channels, K=K, std=0.2)
    transporter = transporter.to(device)  # Move to GPU


    # Data augmentation & preprocessing
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((IMG_SZ, IMG_SZ)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    transform = None
    
    t_dataset = ObjectKeypointsDataset(data_path, transform=transform, game="pong")
    t_sampler = ObjectKeypointsSampler(t_dataset)

    # CRITICAL: Use num_workers > 0 for parallel data loading
    # pin_memory=True speeds up CPU->GPU transfer
    
    
    t_data_loader = DataLoader(
        t_dataset,
        batch_size=batch_size,
        sampler=t_sampler,
        pin_memory=True,  # Faster CPU->GPU transfer
        
    )

    v_dataset = ObjectKeypointsDataset(data_path, transform=transform, game="pong")
    v_sampler = ObjectKeypointsSampler(v_dataset)
    v_data_loader = DataLoader(
        v_dataset,
        batch_size=2*batch_size,
        sampler=v_sampler,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(transporter.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_len, lr_decay)

    pbar = tqdm(enumerate(t_data_loader), total=MAX_ITER, desc="Training")
    VAL_INTERVAL = 2000
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
        
        # forward
        # source img
        source_features = transporter.encoder(xt)
        source_kn = transporter.key_net(xt)
        source_kn = spatial_softmax(source_kn)
        source_keypoints = gaussian_map(source_kn, transporter.std)
        source_kp_coords = compute_keypoint_location_mean(source_kn)

        # target img
        target_features = transporter.encoder(xtp1)
        target_kn = transporter.key_net(xtp1)
        target_kn = spatial_softmax(target_kn)
        target_keypoints = gaussian_map(target_kn, transporter.std)
        target_kp_coords = compute_keypoint_location_mean(target_kn)

        # transport - only detach source_features, NOT source_keypoints
        transport_features = transport(source_keypoints,
                                    target_keypoints,
                                    source_features.detach(),
                                    target_features)
        
        reconstruction = transporter.refine_net(transport_features)
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

        if i % 100 == 0:
            print(f"\nDebug at step {i}:")
            print(f"  target_kp_coords sample: {target_kp_coords[0]}")
            print(f"  target_kn max: {target_kn.max().item():.4f}")
        
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
