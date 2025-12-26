import os
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skills.autoencoder import Autoencoder, AutoencoderDataset
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config_path = Path(__file__).resolve().parent / "configs.yaml"
config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

data_path = config.get("data_path", "./data")
img_sz = config.get("IMG_SZ", 84)
batch_size = 64
max_training_steps = 250000

os.makedirs('./skills/torch_models', exist_ok=True)

def main():
    # Build episode path mapping
    # Map each episode to its full path: {0: 'data/env_1/episode_1', 1: 'data/env_1/episode_2', ...}
    episode_paths = {}
    episode_idx = 0

    # Iterate through all environments in the data folder
    for env_name in sorted(os.listdir(data_path)):
        env_path = os.path.join(data_path, env_name)
        if not os.path.isdir(env_path):
            continue
        
        # Iterate through all episodes in this environment
        for episode_name in sorted(os.listdir(env_path)):
            episode_path = os.path.join(env_path, episode_name)
            if not os.path.isdir(episode_path):
                continue
            
            # Check if episode has images
            images = [f for f in os.listdir(episode_path) if f.endswith('.png')]
            if len(images) > 0:
                episode_paths[episode_idx] = episode_path
                episode_idx += 1

    NUM_EPS = len(episode_paths)
    print(f"Total episodes found: {NUM_EPS}")
    print(f"Environments processed: {len([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])}")

    # Create train/validation split
    eps = np.arange(start=0, stop=NUM_EPS)
    np.random.shuffle(eps)
    split_idx = int(NUM_EPS * 0.8)
    train_idxs = eps[:split_idx]
    val_idxs = eps[split_idx:NUM_EPS]

    print(f"Train episodes: {len(train_idxs)}")
    print(f"Validation episodes: {len(val_idxs)}")

    
    # Create datasets and dataloaders
    dataset_ts = AutoencoderDataset(episode_paths, train_idxs, img_sz)
    train_load = DataLoader(dataset_ts, batch_size, shuffle=True, pin_memory=True)

    dataset_vs = AutoencoderDataset(episode_paths, val_idxs, img_sz)
    val_load = DataLoader(dataset_vs, batch_size, shuffle=False, pin_memory=True)

    # Initialize model
    channels = 1  # Set to 3 if using RGB images
    autoencoder = Autoencoder(channels=channels).to(device)
    autoencoder = torch.compile(autoencoder, mode='default')
    
    criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    print(f"\nStarting training with {max_training_steps} steps...")

    # Training loop with tqdm
    best_loss = float('inf')
    eval_interval = 2000  # run validation every N outer steps (epochs)
    last_val_loss = float('nan')
    
    pbar = tqdm(range(max_training_steps), desc="AE Training", unit="epochs")
    at_step = 0
    for epoch in pbar:
        autoencoder.train()
        train_losses = []

        # training batches
        for i, (inputs, targets) in enumerate(train_load):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        avg_train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0.0

        # validation only every `eval_interval` steps
        avg_val_loss = last_val_loss
        if (epoch + 1) % eval_interval == 0:
            val_losses = []
            with torch.no_grad():
                autoencoder.eval()
                for i, (inputs, targets) in enumerate(val_load):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    out = autoencoder(inputs)
                    vloss = criterion(out, targets)
                    val_losses.append(vloss.detach().cpu().item())

            avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else float('nan')
            last_val_loss = avg_val_loss

            # Save best model when validation improves
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                at_step = epoch + 1
                torch.save(autoencoder.state_dict(), './skills/torch_models/nature-encoder.pt')

        # Convert to tensors so .item() is available in postfix formatting as requested
        tr_loss = torch.tensor(avg_train_loss)
        val_loss = torch.tensor(avg_val_loss)
        best_val_loss = best_loss

        pbar.set_postfix({
            'tr_loss': f'{tr_loss.item():.5f}',
            'val_loss': f'{val_loss.item():.5f}',
            'best_val': f'{best_val_loss:.5f}',
            'at_step': f'{at_step}'
        })


if __name__ == '__main__':
    main()
