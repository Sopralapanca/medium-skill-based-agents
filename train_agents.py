# general imports
import torch
import yaml
import numpy as np
import random
import os
import sys
import argparse

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations

# training imports
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement

from stable_baselines3 import PPO
from rl_zoo3.utils import linear_schedule

from skills.autoencoder import Autoencoder
from skills.unsupervised_state_representation import UnsupervisedStateRepresentationModel
from skills.video_object_keypoints import Transporter
from skills.video_object_segmentation import VideoObjectSegmentationModel

from utils.feature_extractors import WeightSharingAttentionExtractor, SoftHardMOE
from utils.custom_ppo import CustomPPO
from utils.monitor_weights import WeightMonitorCallback, plot_gating_distribution

# IMPORTANT - REGISTER THE ENVIRONMENTS
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback

from dotenv import load_dotenv

import ale_py
gym.register_envs(ale_py)

load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents with skill-based feature extractors')
    parser.add_argument('--entropy_coef', type=float, default=0.0001,
                        help='Entropy coefficient for auxiliary loss (default: 0.0001)')
    parser.add_argument('--load_balance_coef', type=float, default=0.000015,
                        help='Load balance coefficient for auxiliary loss (default: 0.000015)')
    parser.add_argument('--run_id', type=str,
                        help='Run ID for logging and saving models (default: load_balancing_loss_2)')
    return parser.parse_args()


key = os.getenv("WANDB_API_KEY")
if key is None:
    raise ValueError("WANDB_API_KEY not set")

# Parse command line arguments
args = parse_args()


def create_env(env_id, configs, seed=None):
    env = make_atari_env(env_id, n_envs=configs["n_envs"], seed=seed)
    env = VecFrameStack(env, n_stack=configs["n_stacks"])
    env = VecTransposeImage(env)
    return env


def init_wandb(environment_configuration, run_id=None):
  wandb.login(key=key)

  tags = [
      f"fe:{environment_configuration['f_ext_name']}",
      f"game:{environment_configuration['game']}",
  ]

  run = wandb.init(
      project="medium-skill-based-agents",
      name=run_id,  # Set custom run name
      config=environment_configuration,
      sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
      monitor_gym=False,  # auto-upload the videos of agents playing the game
      group=f"{environment_configuration['game']}",
      tags=tags
      # save_code = True,  # optional
  )

  return run


def train_agent(env_id, configs, policy_kwargs, seed, run_id="", train_steps=5000, wandb=False, weight_monitor=False):
    if wandb:
        run = init_wandb(configs, run_id=run_id)
        #run_id = str(run.id)
    else:
        run = None
    
    logdir = "./tensorboard_logs"
            
    vec_envs = create_env(env_id=env_id, configs=configs, seed=seed)
    _ = vec_envs.reset()
    
    #eval_envs = create_env(env_id=env_id, configs=configs, seed=None)

    model = CustomPPO(
        "CnnPolicy",
        vec_envs,
        learning_rate=linear_schedule(environment_configuration["learning_rate"]),
        n_steps=environment_configuration["n_steps"],
        n_epochs=environment_configuration["n_epochs"],
        batch_size=environment_configuration["batch_size"],
        clip_range=linear_schedule(environment_configuration["clip_range"]),
        normalize_advantage=environment_configuration["normalize"],
        ent_coef=environment_configuration["ent_coef"],
        vf_coef=environment_configuration["vf_coef"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
        tensorboard_log=logdir,
    )


    eval_logs = f"eval_logs/{env}/{run_id}"
    os.makedirs(eval_logs, exist_ok=True)

    # eval_callback = EvalCallback(
    #     eval_envs,
    #     n_eval_episodes=100,
    #     best_model_save_path=f"./agents/{run_id}",
    #     log_path=eval_logs,
    #     eval_freq=5000 * environment_configuration["n_envs"],
    #     verbose=0,
    # )
    
    callbacks = [
        # eval_callback
    ]
    
    if wandb:
        callbacks.append(WandbCallback(verbose=0))

    if weight_monitor:
        # Get the feature extractor from the model
        feature_extractor = model.policy.features_extractor

        # Create monitoring callback
        weight_monitor = WeightMonitorCallback(
            feature_extractor=feature_extractor,
            env=env_id,
            run_id=run_id,
            save_freq=1000,  # Save every 1000 steps
            verbose=0,
            use_wandb=True  # Enable wandb logging if wandb is enabled
        )
        
        callbacks.append(weight_monitor)
    try:    
        model.learn(train_steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        weight_monitor._on_training_end()
        sys.exit(0)
    
    if run is not None:
        run.finish()
        
    # #Plot the results
    # if weight_monitor:
    #     weights_file = weight_monitor.weights_file
    #     if os.path.exists(weights_file):
    #         plot_gating_distribution(weights_file, output_dir=weight_monitor.save_path)
    #     else:
    #         raise FileNotFoundError(f"Warning: Weights file not found at {weights_file}")


        

# Load config
_config_path = "./configs.yaml"

_config = {}
with open(_config_path, "r") as f:
    _config = yaml.safe_load(f) or {}


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed for reproducibility
seed = 42

# Set seeds for all libraries
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU

# Enable deterministic operations (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For PyTorch >= 1.8
torch.use_deterministic_algorithms(True, warn_only=True)


#envs = _config.get("ENVS", ["PongNoFrameskip-v4"])[0]
env = "PongNoFrameskip-v4"
with open(f'environment_configs/{env}.yaml', 'r') as file:
        environment_configuration = yaml.safe_load(file)["config"]


environment_configuration["f_ext_kwargs"]["device"] = device  #do not comment this, it is the parameter passed to the feature extractor
environment_configuration["game"] = env


policy_kwargs = dict(
    net_arch={
        "pi": environment_configuration["net_arch_pi"],
        "vf": environment_configuration["net_arch_vf"],
    },
    # activation_fn=torch.nn.ReLU,  # use ReLU in case of multiple layers for the policy learning network
)

test_envs = create_env(env_id=env, configs=environment_configuration, seed=seed)
obs = test_envs.reset()

# init skills
autoencoder = Autoencoder(channels=1).to(device)
usr = UnsupervisedStateRepresentationModel(observation=obs[0], device=device)
vok = Transporter().to(device)
vos = VideoObjectSegmentationModel(device=device)


skills = [
    usr.get_skill(device=device),
    vok.get_skill(device=device, keynet_or_encoder="encoder"),
    vok.get_skill(device=device, keynet_or_encoder="keynet"),
    vos.get_skill(device=device)
]
   

f_ext_kwargs = environment_configuration["f_ext_kwargs"]
environment_configuration["f_ext_name"] = "moe_ext"
environment_configuration["f_ext_class"] = WeightSharingAttentionExtractor
f_ext_kwargs["skills"] = skills
f_ext_kwargs["features_dim"] = 256
f_ext_kwargs["entropy_coef"] = args.entropy_coef
f_ext_kwargs["load_balance_coef"] = args.load_balance_coef

policy_kwargs["features_extractor_class"] = environment_configuration["f_ext_class"]
policy_kwargs["features_extractor_kwargs"] = f_ext_kwargs


train_agent(
    env, 
    environment_configuration, 
    policy_kwargs, 
    seed, 
    run_id=args.run_id, 
    train_steps=1000000, 
    wandb=True,
    weight_monitor=True
)