# general imports
import torch
import yaml
import numpy as np
import random
import os

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
from utils.monitor_moe_weights import GatingMonitorCallback

# IMPORTANT - REGISTER THE ENVIRONMENTS
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback

from dotenv import load_dotenv

import ale_py
gym.register_envs(ale_py)

load_dotenv()


key = os.getenv("WANDB_API_KEY")
if key is None:
    raise ValueError("WANDB_API_KEY not set")


def create_env(env_id, configs, seed=None):
    env = make_atari_env(env_id, n_envs=configs["n_envs"], seed=seed)
    env = VecFrameStack(env, n_stack=configs["n_stacks"])
    env = VecTransposeImage(env)
    return env


def init_wandb(environment_configuration):
  wandb.login(key=key)

  tags = [
      f"fe:{environment_configuration['f_ext_name']}",
      f"game:{environment_configuration['game']}",
  ]

  run = wandb.init(
      project="medium-skill-based-agents",
      config=environment_configuration,
      sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
      monitor_gym=False,  # auto-upload the videos of agents playing the game
      group=f"{environment_configuration['game']}",
      tags=tags
      # save_code = True,  # optional
  )

  return run


def train_agent(env_id, configs, policy_kwargs, seed, train_steps=5000, wandb=False):
    if wandb:
        run = init_wandb(configs)
        monitor_dir = str(run.id)
    else:
        run = None
        monitor_dir = "ppo"
    
    logdir = "./tensorboard_logs"
            
    vec_envs = create_env(env_id=env_id, configs=configs, seed=seed)
    _ = vec_envs.reset()
    
    eval_envs = create_env(env_id=env_id, configs=configs, seed=None)

    model = CustomPPO(  # Changed from PPO to CustomPPO
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


    eval_logs = f"eval_logs/{env}/{monitor_dir}"
    os.makedirs(eval_logs, exist_ok=True)

    # eval_callback = EvalCallback(
    #     eval_envs,
    #     n_eval_episodes=100,
    #     best_model_save_path=f"./agents/{monitor_dir}",
    #     log_path=eval_logs,
    #     eval_freq=5000 * environment_configuration["n_envs"],
    #     verbose=0,
    # )
    
    callbacks = [
        # eval_callback
    ]
    
    if wandb:
        callbacks.append(WandbCallback(verbose=0))

    if configs["f_ext_name"] == "moe_ext" or configs["f_ext_name"] == "wsharing_attention_ext":
        # Get the feature extractor from the model
        feature_extractor = model.policy.features_extractor

        # Create monitoring callback
        gating_monitor = GatingMonitorCallback(
            feature_extractor=feature_extractor,
            env=env_id,
            save_freq=500,  # Save every 500 steps
            save_path="./gating_weights",
            verbose=0
        )
        
        callbacks.append(gating_monitor)
        
    model.learn(train_steps, callback=callbacks, progress_bar=True) #tb_log_name=run.id)
    
    if run is not None:
        run.finish()

# Load config
_config_path = "./configs.yaml"

_config = {}
with open(_config_path, "r") as f:
    _config = yaml.safe_load(f) or {}


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # ignore tensorflow warnings about CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

seed = None
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
environment_configuration["f_ext_class"] = SoftHardMOE
f_ext_kwargs["skills"] = skills
f_ext_kwargs["features_dim"] = 256

# Exploration and load balancing parameters
f_ext_kwargs["min_temperature"] = 0.1  # Try 0.01, 0.05, 0.1, 0.2
f_ext_kwargs["temperature_decay"] = 0.99998    # Try 0.001, 0.01, 0.05


policy_kwargs["features_extractor_class"] = environment_configuration["f_ext_class"]
policy_kwargs["features_extractor_kwargs"] = f_ext_kwargs



train_agent(env, environment_configuration, policy_kwargs, seed, train_steps=5000, wandb=False)