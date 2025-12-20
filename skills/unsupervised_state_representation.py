from atariari.benchmark.episodes import get_episodes
from atariari.methods.encoders import NatureCNN
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
import os
import torch
from .skill_interface import Skill, model_forward
from torch import Tensor
import torch.nn.functional as F


def _get_episodes(env_name, steps, collect_mode):
    tr_eps, val_eps = get_episodes(
        steps=steps,
        env_name=env_name,
        collect_mode=collect_mode,
        train_mode="train_encoder",
    )
    return tr_eps, val_eps
    
class UnsupervisedStateRepresentationModel:
    def __init__(self, observation, max_training_steps=10000, device=None):
                
        dummy_wandb = DummyWandB()
        args = {
            # "env_name": "MsPacmanNoFrameskip-v4",
            "pretraining_steps": max_training_steps,
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

        self.args = Args(**args)
        
        self.observation_shape = observation.shape
        self.encoder = NatureCNN(input_channels=self.observation_shape[0], args=self.args).to(device)

        config = {}
        config.update(vars(self.args))
        config["obs_space"] = self.observation_shape
        
        self.trainer = InfoNCESpatioTemporalTrainer(
            self.encoder, config, device=device, wandb=dummy_wandb
        )
        
    def train(self, train_episodes, val_episodes):
        self.trainer.train(train_episodes, val_episodes)
        
        os.makedirs('./skills/torch_models/', exist_ok=True)
        torch.save(self.encoder.state_dict(), './skills/torch_models/state-rep.pt')
        print("Model saved to ./skills/torch_models/state-rep.pt")

        if os.path.exists("encoder.pt"):
            os.remove("encoder.pt")
    
    def state_rep_input_trans(self, x: Tensor):
        x = x.float()
        return F.interpolate(x, (160, 210), mode='bilinear', align_corners=True)
    
    def get_skill(self, device):
        input_transformation_function = self.state_rep_input_trans
        model_path = "skills/torch_models/state-rep.pt"

        model = NatureCNN(self.observation_shape[0], args=self.args).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        adapter = None

        return Skill("state_rep_uns", input_transformation_function, model, model_forward, adapter)

    
    


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