import torch
from tqdm import tqdm
from skills.video_object_segmentation import VOSDataset, VideoObjectSegmentationModel

num_frames = 2
batch_size = 32
steps = 1000000
lr = 1e-4
max_grad_norm = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./data"

data = VOSDataset(batch_size, num_frames, data_path)
inp = data.get_batch("train").to(device)


model = VideoObjectSegmentationModel(device=device)
model.to(device)

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
    x0_ = model(inp)
    x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
    tr_loss = model.compute_loss(x0, x0_)
    tr_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        model.eval()
        inp = data.get_batch("val").to(device)
        x0_ = model(inp)
        x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
        val_loss = model.compute_loss(x0, x0_)

    model.update_reg()

    if val_loss <= best_val_loss:
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
