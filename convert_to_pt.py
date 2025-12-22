import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

IMG_SZ = 84
SRC_ROOT = "./data"       # your current PNG dataset
DST_ROOT = "./data_pt"    # new tensor dataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

os.makedirs(DST_ROOT, exist_ok=True)

envs = os.listdir(SRC_ROOT)

for env in envs:
    src_env = os.path.join(SRC_ROOT, env)
    dst_env = os.path.join(DST_ROOT, env)
    os.makedirs(dst_env, exist_ok=True)

    episodes = os.listdir(src_env)
    for ep in tqdm(episodes, desc=f"Processing {env}"):
        src_ep = os.path.join(src_env, ep)
        dst_ep = os.path.join(dst_env, ep)
        os.makedirs(dst_ep, exist_ok=True)

        frames = sorted(os.listdir(src_ep), key=lambda x: int(x.split(".")[0]))
        for f in frames:
            img = Image.open(os.path.join(src_ep, f))
            tensor = transform(img)
            out_path = os.path.join(dst_ep, f.replace(".png", ".pt"))
            torch.save(tensor, out_path)
