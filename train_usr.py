import random
import os
from atariari.benchmark.episodes import get_episodes
from atariari.methods.encoders import NatureCNN
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_tr_eps = []
all_val_eps = []
ENVS = [
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
]


for env_name in ENVS:
    print(f"Collecting episodes from {env_name}...")
    tr_eps, val_eps = get_episodes(
        steps=1000,
        env_name=env_name,
        collect_mode="random_agent",
        train_mode="train_encoder",
    )

    all_tr_eps.extend(tr_eps)
    all_val_eps.extend(val_eps)

print(f"\nTotal train episodes: {len(all_tr_eps)}")
print(f"Total val episodes: {len(all_val_eps)}")

# Shuffle the episodes
random.shuffle(all_tr_eps)
random.shuffle(all_val_eps)

print("Episodes shuffled!")


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyWandB:
    class Run:
        def __init__(self):
            self.dir = "./skills"

    def __init__(self):
        self.run = self.Run()

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def finish(self):
        pass


dummy_wandb = DummyWandB()
args = {
    # "env_name": "MsPacmanNoFrameskip-v4",
    "pretraining_steps": 10000,
    "no_downsample": True,
    "probe_collect_mode": "random_agent",
    "feature_size": 512,
    "end_with_relu": True,
    "encoder_type": "Nature",
    "patience": 15,
    "epochs": 100,
    "batch_size": 64,
    "lr": 3e-4,
    "method": "vae",
    "end_with_relu": False,
    "wandb": dummy_wandb,
}

args = Args(**args)

observation_shape = all_tr_eps[0][0].shape
encoder = NatureCNN(observation_shape[0], args)

config = {}
config.update(vars(args))
config["obs_space"] = observation_shape
trainer = InfoNCESpatioTemporalTrainer(
    encoder, config, device=device, wandb=dummy_wandb
)
trainer.train(all_tr_eps, all_val_eps)

# save model in /skills/torch_models/state-rep.pt
os.makedirs('./skills/torch_models/', exist_ok=True)
torch.save(encoder.state_dict(), './skills/torch_models/state-rep.pt')
print("Model saved to ./skills/torch_models/state-rep.pt")
