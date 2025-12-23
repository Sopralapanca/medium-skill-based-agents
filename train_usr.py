import random
from pathlib import Path
import yaml
import torch
from skills.unsupervised_state_representation import UnsupervisedStateRepresentationModel, _get_episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
_config_path = Path(__file__).resolve().parent / "configs.yaml"
_config = {}
if _config_path.exists():
    with open(_config_path, "r") as f:
        _config = yaml.safe_load(f) or {}
max_training_steps = 100000
#_config.get("max_training_steps", 10000)

all_tr_eps = []
all_val_eps = []
ENVS = _config.get("ENVS", [])
steps_in_get_episodes = 35000

for env_name in ENVS:
    print(f"Collecting episodes from {env_name}...")
    tr_eps, val_eps = _get_episodes(env_name, steps_in_get_episodes, collect_mode="random_agent")

    all_tr_eps.extend(tr_eps)
    all_val_eps.extend(val_eps)

print(f"\nTotal train episodes: {len(all_tr_eps)}")
print(f"Total val episodes: {len(all_val_eps)}")

# Shuffle the episodes
random.shuffle(all_tr_eps)
random.shuffle(all_val_eps)

state_rep_model = UnsupervisedStateRepresentationModel(
    observation=all_tr_eps[0][0], max_training_steps=max_training_steps, device=device, batch_size=128
)

state_rep_model.train(all_tr_eps, all_val_eps, val_interval=5)
