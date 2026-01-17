import torch
import yaml
import numpy as np
import random
import os
from typing import List

# PufferLib imports
import pufferlib
import pufferlib.vector
import pufferlib.emulation
from pufferlib import pufferl

# Your existing skill imports
from skills.autoencoder import Autoencoder
from skills.unsupervised_state_representation import UnsupervisedStateRepresentationModel
from skills.video_object_keypoints import Transporter
from skills.video_object_segmentation import VideoObjectSegmentationModel
from skills.skill_interface import Skill

from utils.puffer_wsa import AttentionPolicy

# WandB for logging
import wandb
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_env_creator(env_id):
    """Create environment creator function for PufferLib"""
    def env_creator(buf=None, seed=None):
        # Import here to avoid issues with multiprocessing
        import gymnasium as gym
        import ale_py
        from stable_baselines3.common.atari_wrappers import (
            NoopResetEnv,
            MaxAndSkipEnv,
            EpisodicLifeEnv,
            FireResetEnv,
            ClipRewardEnv,
        )
        from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
        
        gym.register_envs(ale_py)
        
        # Create the base environment
        env = gym.make(env_id)
        
        # Apply SB3 Atari wrappers (same as make_atari_env does)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        
        # Apply preprocessing: grayscale, resize, framestack
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=(84, 84))
        env = FrameStackObservation(env, stack_size=4)
        
        # Wrap with PufferLib emulation for compatibility
        # PufferLib handles vectorization, so this is a single env
        env = pufferlib.emulation.GymnasiumPufferEnv(
            env=env,
            buf=buf,    # Shared memory buffer
            seed=seed   # Environment seed
        )
        
        return env
    
    return env_creator


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_pufferlib(
    env_id: str,
    skills: List[Skill],
    config: dict,
    device: str = "cpu",
    use_wandb: bool = False,
):
    """
    Train agent using PufferLib with custom attention-based feature extractor.
    
    Args:
        env_id: Environment name (e.g., "PongNoFrameskip-v4")
        skills: List of pretrained skill models
        config: Training configuration dictionary
        device: Device for training ('cuda' or 'cpu')
        use_wandb: Whether to use Weights & Biases logging
    """
    
    # Create vectorized environment
    env_creator = make_env_creator(env_id)
    
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=config.get("n_envs", 8),
        backend=pufferlib.vector.Multiprocessing,  # or Serial for debugging
        overwork=True,  # Allow more workers than hardware cores
    )
    
    # Create policy
    policy = AttentionPolicy(
        env=vecenv.driver_env,
        skills=skills,
        features_dim=config.get("features_dim", 256),
        hidden_size=config.get("hidden_size", 512),
        device=device,
    ).to(device)
    
    # Setup WandB logging
    if use_wandb:
        wandb.init(
            project="pufferlib-skill-based-agents",
            config=config,
            sync_tensorboard=True,
            group=env_id,
            tags=["attention", "skills", env_id],
        )
    
    batch_size = config.get("batch_size", 256)
    
    # Standard PPO default: 4 minibatches per epoch
    minibatch_size = batch_size // 4 
    
    class DictConfig:
        """Config class that supports both dict and attribute access"""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getitem__(self, key):
            return self.__dict__[key]

        def __setitem__(self, key, value):
            self.__dict__[key] = value


        def get(self, key, default=None):
            return self.__dict__.get(key, default) 

    train_config = DictConfig(
        # --- Environment ---
        env=env_id,
        
        # --- Device & Backend ---
        device=device,
        cpu_offload=False,   # Keep replay buffer on GPU (set True if OOM)
        
        # --- Network Architecture ---
        use_rnn=False,       # Set True only if using LSTM/GRU
        compile=False,       # Set True for torch.compile speedup (optional)
        compile_mode="reduce-overhead",
        
        # --- Data & Batching (The Missing Keys) ---
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        max_minibatch_size=minibatch_size, # Cap for GPU memory safety
        bptt_horizon=16,     # Sequence length for RNNs (ignored if use_rnn=False but required)
        
        # --- Training Hyperparameters ---
        total_timesteps=config.get("total_timesteps", 1000000),
        learning_rate=config.get("learning_rate", 2.5e-4),
        num_steps=config.get("n_steps", 128),
        num_envs=config.get("n_envs", 8),
        update_epochs=config.get("n_epochs", 4),
        
        # --- Optimizer ---
        optimizer='adam',    # 'adam' or 'muon'
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        
        # --- PPO Specifics ---
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=config.get("clip_range", 0.1),
        vf_clip_coef=0.1,    # Value function clipping coefficient
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=0.5,
        normalize_advantage=config.get("normalize", True),
        target_kl=None,      # Target KL divergence (optional, usually None)
        
        # --- V-trace & Prioritization ---
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.0,      # Prioritization exponent (0 = no prioritization)
        prio_beta0=0.4,      # Initial importance sampling weight
        
        # --- Learning Rate Scheduling ---
        anneal_lr=True,      # Cosine annealing of learning rate
        min_lr_ratio=0.1,    # Minimum LR as fraction of initial LR
        
        # --- Precision & AMP ---
        precision='float32', # 'float32' or 'bfloat16'
        amp=True,            # Automatic mixed precision
        
        # --- Reproducibility ---
        seed=config.get("seed", 1),
        torch_deterministic=False,
        
        # --- Logging & Checkpointing ---
        data_dir="experiments",
        checkpoint_interval=10000,
        save_overlay=True,
        verbose=1,
    )
    
    # Create PuffeRL trainer
    trainer = pufferl.PuffeRL(
        config=train_config,
        vecenv=vecenv,
        policy=policy,
    )
    
    # Training loop
    try:
        while trainer.global_step < train_config.total_timesteps:
            # Collect experience
            trainer.evaluate()
            
            # Train on collected data
            trainer.train()
            
            # Print progress every epoch with reward stats
            if trainer.epoch % 10 == 0:
                # Get episode statistics
                if 'episode_return' in trainer.stats:
                    returns = trainer.stats['episode_return']
                    if len(returns) > 0:
                        mean_return = sum(returns) / len(returns)
                        print(f"\nEpoch {trainer.epoch} - Mean Episode Return: {mean_return:.2f} "
                              f"(from {len(returns)} episodes)")
            
            # Log metrics
            if use_wandb:
                trainer.wandb_log()
            
            # Print dashboard less frequently
            if trainer.global_step % 10000 == 0:
                trainer.print_dashboard()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Cleanup
        trainer.close()
        if use_wandb:
            wandb.finish()
    
    return trainer


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load configuration
    env_id = "PongNoFrameskip-v4"
    
    with open(f'environment_configs/{env_id}.yaml', 'r') as file:
        config = yaml.safe_load(file)["config"]
    
    # Set seed
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize skills (same as your original code)
    print("Loading pretrained skills...")
    
    # Create a temporary environment to initialize skills (same preprocessing as training)
    import gymnasium as gym
    import ale_py
    from stable_baselines3.common.atari_wrappers import (
        NoopResetEnv,
        MaxAndSkipEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        ClipRewardEnv,
    )
    from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
    
    gym.register_envs(ale_py)
    
    temp_env = gym.make(env_id)
    temp_env = NoopResetEnv(temp_env, noop_max=30)
    temp_env = MaxAndSkipEnv(temp_env, skip=4)
    temp_env = EpisodicLifeEnv(temp_env)
    if "FIRE" in temp_env.unwrapped.get_action_meanings():
        temp_env = FireResetEnv(temp_env)
    temp_env = ClipRewardEnv(temp_env)
    temp_env = GrayscaleObservation(temp_env, keep_dim=False)
    temp_env = ResizeObservation(temp_env, shape=(84, 84))
    temp_env = FrameStackObservation(temp_env, stack_size=4)
    temp_env = pufferlib.emulation.GymnasiumPufferEnv(env=temp_env)
    
    obs, _ = temp_env.reset()
    
    # Convert to torch tensor with correct shape for skills initialization
    # obs from FrameStackObservation should be (4, 84, 84)
    obs = torch.from_numpy(obs).float().to(device) / 255.0
    
    print(f"Observation shape for skill initialization: {obs.shape}")  # Should be (1, 4, 84, 84)
  
    
    # Initialize skills
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
    
    temp_env.close()
    
    print(f"Loaded {len(skills)} skills")
    
    # Update config for PufferLib
    config.update({
        "features_dim": 256,
        "hidden_size": 512,
        "total_timesteps": 1000000,
    })
    
    # Check for WandB key
    wandb_key = os.getenv("WANDB_API_KEY", None)
    use_wandb = False #wandb_key is not None
    
    # Train
    print(f"Starting training on {env_id}")
    print(f"Device: {device}")
    print(f"WandB logging: {use_wandb}")
    
    trainer = train_with_pufferlib(
        env_id=env_id,
        skills=skills,
        config=config,
        device=device,
        use_wandb=use_wandb,
    )
    
    print("Training complete!")
