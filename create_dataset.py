import os
import numpy as np
from tqdm import tqdm 
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt  # Only needed if you want to plot somewhere else
import yaml
import random


HF_CACHE_DIR = ".hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["SB3_CACHE"] = HF_CACHE_DIR

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gymnasium as gym # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_atari_env # noqa: E402
from stable_baselines3.common.vec_env import VecFrameStack # noqa: E402

from huggingface_sb3 import load_from_hub # noqa: E402
import ale_py # noqa: E402

gym.register_envs(ale_py)

# Load shared configuration
_config_path = Path(__file__).resolve().parent / "configs.yaml"
_config = {}
if _config_path.exists():
    with open(_config_path, "r") as f:
        _config = yaml.safe_load(f) or {}

ENVS = _config.get("ENVS", [])


EPISODES = _config.get("EPISODES", 10)
MAX_STEPS_PER_EPISODE_RANDOM_AGENT = _config.get("MAX_STEPS_PER_EPISODE_RANDOM_AGENT", 100)
MAX_STEPS_PER_EPISODE_EXPERT_AGENT = _config.get("MAX_STEPS_PER_EPISODE_EXPERT_AGENT", 100)
BASE_DATA_PATH = _config.get("data_path", "./data")


class FrameCollector:
    """Class to handle Atari gameplay and save frames directly to disk."""

    def __init__(self, env_name, n_stack=4):
        self.env_name = env_name
        self.n_stack = n_stack
        self.model = None
        self.env = None
        self.base_path = BASE_DATA_PATH
        self.create_environment()
        
    def create_environment(self):
        try:
            self.env = make_atari_env(self.env_name, n_envs=1, seed=42)
            self.env = VecFrameStack(self.env, n_stack=self.n_stack)
            return True
        except Exception as e:
            print(f"Error creating environment {self.env_name}: {e}")
            raise e

    def load_model(self):
        if self.use_random_agent:
            print("Using random agent (no model provided)")
            self.model = None
            return True

        try:
            checkpoint = load_from_hub(
                repo_id=f"sb3/ppo-{self.env_name}",
                filename=f"ppo-{self.env_name}.zip",
            )

            # Some models saved with older SB3 / Python versions serialize
            # callables (schedules / lambdas) that cannot be deserialized
            # in newer runtimes (code object signature mismatch). Provide
            # safe replacements via `custom_objects` to avoid those errors.
            custom_objects = {
                "learning_rate": 2.5e-4,
                "lr_schedule": (lambda _: 2.5e-4),
                "clip_range": (lambda _: 0.2),
            }

            try:
                self.model = PPO.load(checkpoint, custom_objects=custom_objects)
            except Exception as inner_e:
                # If the full replacement fails, try a minimal replacement
                # and if that still fails fall back to using a random agent.
                print(f"Warning: full custom_objects load failed: {inner_e}")
                try:
                    self.model = PPO.load(checkpoint, custom_objects={"learning_rate": 2.5e-4})
                except Exception as inner_e2:
                    print(f"Error loading model after fallback: {inner_e2}")
                    print("Falling back to standard loading")
                    self.model = PPO.load(checkpoint)
            return True
        except Exception as e:
            raise RuntimeError(f"Error loading model for {self.env_name}: {e}")

    def collect_frames_gameplay(self, n_episodes=3, max_steps_per_episode=1000, use_random_agent=False):
        print(f"Starting gameplay for {n_episodes} episodes...")
        self.use_random_agent = use_random_agent
        suffix = "_random" if self.use_random_agent else "_expert"
        self.output_dir = os.path.join(self.base_path, self.env_name + suffix)
        os.makedirs(self.output_dir, exist_ok=True)

        
        self.load_model()

        for episode in tqdm(range(n_episodes)):
            episode_dir = Path(self.output_dir) / f"episode_{episode + 1}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            for step in range(max_steps_per_episode):
                if self.use_random_agent:
                    action = np.array([self.env.action_space.sample()])
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                step_result = self.env.step(action)

                if len(step_result) == 5:
                    obs, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, _, done, _ = step_result

                try:
                    if hasattr(self.env, "render"):
                        frame = self.env.render()
                    else:
                        frame = self.env.get_images()[0]
                except:
                    try:
                        frame = self.env.envs[0].render()
                    except:
                        frame = None

                if frame is not None:
                    frame_path = episode_dir / f"{step}.png"
                    img = Image.fromarray(frame)
                    img.save(frame_path)

                if hasattr(done, "__len__"):
                    if done[0]:
                        break
                else:
                    if done:
                        break

        print(f"Frames saved to {self.output_dir}")
        return True

    def display_sample_frames(self, n_frames=6):
        """Display n random frames from a random episode stored on disk"""

        if not os.path.exists(self.output_dir):
            print("Output directory does not exist!")
            return

        episode_dirs = [
            d
            for d in Path(self.output_dir).iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]

        if not episode_dirs:
            print("No episode directories found!")
            return

        # Pick one random episode
        chosen_episode = random.choice(episode_dirs)
        frame_files = sorted(chosen_episode.glob("*.png"))

        if not frame_files:
            print(f"No frames found in episode {chosen_episode.name}!")
            return

        sample_files = random.sample(frame_files, min(n_frames, len(frame_files)))
        sample_images = [Image.open(f) for f in sample_files]

        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, img in enumerate(sample_images):
            axes[i].imshow(img)
            axes[i].set_title(f"{sample_files[i].stem}")
            axes[i].axis("off")

        for i in range(len(sample_images), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    for env in ENVS:
        print(f"Collecting frames for: {env}")
        collector = FrameCollector(env_name=env)

        collector.collect_frames_gameplay(
            n_episodes=EPISODES,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE_RANDOM_AGENT,
            use_random_agent=True
        )
        
        # collector.collect_frames_gameplay(
        #     n_episodes=EPISODES,
        #     max_steps_per_episode=MAX_STEPS_PER_EPISODE_EXPERT_AGENT,
        #     use_random_agent=False
        # )
