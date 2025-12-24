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

from utils.feature_extractors import WeightSharingAttentionExtractor, MixtureOfExpertsExtractor

# IMPORTANT - REGISTER THE ENVIRONMENTS
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback

from dotenv import load_dotenv

import ale_py
gym.register_envs(ale_py)


load_dotenv()


# key = os.getenv("WANDB_API_KEY")
# if key is None:
#     raise ValueError("WANDB_API_KEY not set")

# wandb.login(key=key)


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


# monitor_dir = f"monitor/{run.id}"
monitor_dir = "ppo"

vec_envs = make_atari_env(env, n_envs=environment_configuration["n_envs"], seed=seed)
vec_envs = VecFrameStack(vec_envs, n_stack=environment_configuration["n_stacks"])
vec_envs = VecTransposeImage(vec_envs)

# execute some steps with random moves
obs = vec_envs.reset()

for i in range(10):
    action = [vec_envs.action_space.sample() for _ in range(environment_configuration["n_envs"])]
    obs, rewards, dones, info = vec_envs.step(action)

# obs[0] has shape (4, 84, 84) because there are 4 stacked environments, take the first
observation = obs[0][-1]


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

for feature_extractor_class in ["ppo", "wsa", "moe"]:

    if feature_extractor_class == "wsa":
        environment_configuration["f_ext_name"] = "wsharing_attention_ext"
        environment_configuration["f_ext_class"] = WeightSharingAttentionExtractor
    elif feature_extractor_class == "moe":
        environment_configuration["f_ext_name"] = "moe_ext"
        environment_configuration["f_ext_class"] = MixtureOfExpertsExtractor
    
    # else ppo default CNN policy
    
    tags = [
        f"fe:{environment_configuration['f_ext_name']}",
        f"game:{environment_configuration['game']}",
    ]

    # run = wandb.init(
    #     project="medium-skill-based-agents",
    #     config=environment_configuration,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     monitor_gym=False,  # auto-upload the videos of agents playing the game
    #     group=f"{environment_configuration['game']}",
    #     tags=tags
    #     # save_code = True,  # optional
    # )


    if feature_extractor_class != "ppo":
        f_ext_kwargs["skills"] = skills
    
    f_ext_kwargs["features_dim"] = 256

    policy_kwargs = {}
    if "f_ext_class" in environment_configuration:
        policy_kwargs["features_extractor_class"] = environment_configuration["f_ext_class"]

    # Only pass extractor kwargs when a custom extractor is used
    if "f_ext_kwargs" in environment_configuration and feature_extractor_class != "ppo":
        policy_kwargs["features_extractor_kwargs"] = f_ext_kwargs

    policy_kwargs["net_arch"] = {
        "pi": environment_configuration["net_arch_pi"],
        "vf": environment_configuration["net_arch_vf"],
    }

    logdir = "./tensorboard_logs"

    model = PPO(
        "CnnPolicy",
        vec_envs,
        learning_rate=linear_schedule(environment_configuration["learning_rate"]),
        n_steps=128,
        n_epochs=4,
        batch_size=environment_configuration["batch_size"],
        clip_range=linear_schedule(environment_configuration["clip_range"]),
        normalize_advantage=environment_configuration["normalize"],
        ent_coef=environment_configuration["ent_coef"],
        vf_coef=environment_configuration["vf_coef"],
        policy_kwargs=policy_kwargs if len(policy_kwargs) > 0 else None,
        verbose=1,
        device=device,
        tensorboard_log=logdir,
    )



    eval_env = make_atari_env(env, n_envs=environment_configuration["n_envs"])
    eval_env = VecFrameStack(eval_env, n_stack=environment_configuration["n_stacks"])
    eval_env = VecTransposeImage(eval_env)

    eval_logs = f"eval_logs/{env}/{monitor_dir}"
    os.makedirs(eval_logs, exist_ok=True)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=21, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=100,
        best_model_save_path=f"./agents/{monitor_dir}",
        log_path=eval_logs,
        eval_freq=5000 * environment_configuration["n_envs"],
        verbose=0,
        callback_on_new_best=callback_on_best
    )

    callbacks = [
        #WandbCallback(verbose=0),
        eval_callback
    ]

    model.learn(2000, callback=callbacks) #tb_log_name=run.id)
    #run.finish()
