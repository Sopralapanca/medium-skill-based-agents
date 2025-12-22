import torch
from tqdm import tqdm
from pathlib import Path
import yaml
import os
from skills.video_object_segmentation import VOSDataset, VideoObjectSegmentationModel

num_frames = 2
# Load shared config
_config_path = Path(__file__).resolve().parent / "configs.yaml"
_config = {}
if _config_path.exists():
    with open(_config_path, "r") as f:
        _config = yaml.safe_load(f) or {}

batch_size = 32 # _config.get("batch_size", 16)  # Paper uses batch_size=16
steps = 250000  # Paper uses 250k steps
lr = 1e-4
max_grad_norm = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./data"
val_frequency = 5000  # Run validation every N steps to save time

game = "pong"
# Use frame_skip=3 to sample frames 3 timesteps apart for more visible motion
# Atari games run at ~60fps, so skip=3 gives ~20fps effective sampling rate
data = VOSDataset(batch_size, num_frames, data_path, game=game, frame_skip=3)
inp = data.get_batch("train").to(device)

# Paper uses curriculum over 100k steps to linearly increase lambda_reg from 0 to 1
curriculum_steps = 100000
model = VideoObjectSegmentationModel(device=device, curriculum_steps=curriculum_steps)
model.to(device)
model = torch.compile(model, mode='default')

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def lamb(epoch):
    return 1 - epoch/steps
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamb)

best_val_loss = 1.
at_step = 0
pbar = tqdm(range(steps), desc="Training")

for i in pbar:
    model.train()
    optimizer.zero_grad()
    inp = data.get_batch("train").to(device)
    
    # inp shape: [batch_size, num_frames, H, W]
    # Extract x0 (first frame) and create input with x0 and x1 (two consecutive frames)
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)  # [batch_size, 1, H, W]
    x1 = torch.unsqueeze(inp[:, 1, :, :], 1)  # [batch_size, 1, H, W]
    model_input = torch.cat([x0, x1], dim=1)  # [batch_size, 2, H, W]
    
    # Model reconstructs x0 from x1 using optical flow
    x0_ = model(model_input)
    tr_loss = model.compute_loss(x0, x0_)
    
    tr_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    # Validation (run less frequently)
    if i % val_frequency == 0:
        with torch.no_grad():
            model.eval()
            inp_val = data.get_batch("val").to(device)
            x0_val = torch.unsqueeze(inp_val[:, 0, :, :], 1)
            x1_val = torch.unsqueeze(inp_val[:, 1, :, :], 1)
            model_input_val = torch.cat([x0_val, x1_val], dim=1)
            x0_val_ = model(model_input_val)
            val_loss = model.compute_loss(x0_val, x0_val_)
    
    model.update_reg()

    # Only update best model when validation is computed
    if i % val_frequency == 0 and val_loss <= best_val_loss:
        best_val_loss = val_loss
        at_step = i
        #print(f"Step: {i} new best val_loss : {val_loss}")
        torch.save(model.state_dict(), './skills/torch_models/vid-obj-seg.pt')


    last_lr = scheduler.get_last_lr()[0]

    pbar.set_postfix({
        'tr_loss': f'{tr_loss.item():.4f}',
        'val_loss': f'{val_loss.item():.4f}',
        'best_val': f'{best_val_loss:.4f}',
        'at_step': f'{at_step}'
    })
