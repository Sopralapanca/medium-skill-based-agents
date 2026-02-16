from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
import numpy as np
import envpool

class EnvPoolWrapper(VecEnv):
    """
    Wrapper to make EnvPool compatible with Stable Baselines3.
    """
    def __init__(self, env):
        self.env = env
        self.num_envs = env.spec.config.num_envs
        
        # Get the original observation space from EnvPool
        obs_shape = env.observation_space.shape  # Should be (4, 84, 84) with EnvPool
        
        # Create a proper Box observation space for images
        observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
        
        # Convert EnvPool action space to Gymnasium action space
        # EnvPool uses its own Discrete class, we need Gymnasium's
        if hasattr(env.action_space, 'n'):
            # Discrete action space
            action_space = spaces.Discrete(env.action_space.n)
        elif hasattr(env.action_space, 'shape'):
            # Box action space (continuous)
            action_space = spaces.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype
            )
        else:
            # Fallback: use the original action space
            action_space = env.action_space
        
        super().__init__(self.num_envs, observation_space, action_space)
        
    def reset(self):
        obs, _ = self.env.reset()  # EnvPool returns (obs, info)
        return obs
    
    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self.actions)
        # Combine terminated and truncated into done for SB3
        done = np.logical_or(terminated, truncated)
        
        # Convert info from EnvPool format to SB3 format
        # EnvPool returns info as a dict, but SB3 expects a list of dicts
        if isinstance(info, dict):
            # Create a list of empty dicts for each environment
            info_list = [{} for _ in range(self.num_envs)]
            
            # If there are any episode statistics, distribute them
            for key, value in info.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == self.num_envs:
                    for i in range(self.num_envs):
                        info_list[i][key] = value[i]
                # Handle scalar values or keys that don't match num_envs
                elif key != 'env_id':  # Skip env_id as it's not needed per-env
                    for i in range(self.num_envs):
                        info_list[i][key] = value
            
            info = info_list
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
    
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    
    def seed(self, seed=None):
        # EnvPool handles seeding at initialization
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    

def create_envpool_env(env_id, configs, seed=None):
    """
    Create vectorized Atari environment using EnvPool for faster training.
    
    Args:
        env_id: Environment ID (e.g., "PongNoFrameskip-v4")
        configs: Configuration dict with n_envs and n_stacks
        seed: Random seed
    
    Returns:
        Vectorized environment with correct observation shape
    """
    # Convert Gym environment ID to EnvPool format
    # "PongNoFrameskip-v4" -> "Pong-v5"
    env_name = env_id.replace("NoFrameskip-v4", "-v5")
    
    env = envpool.make(
        env_name,
        env_type="gym",
        num_envs=configs["n_envs"],
        seed=seed if seed is not None else 0,
        episodic_life=True,      # Reset on life loss
        reward_clip=True,        # Clip rewards to [-1, 1]
        stack_num=configs["n_stacks"],  # Frame stacking (usually 4)
        gray_scale=True,         # Grayscale observations
        img_height=84,           # Standard Atari preprocessing
        img_width=84,
        frame_skip=4,            # Skip 4 frames between actions
    )
    
    # Wrap with our custom wrapper to make it compatible with SB3
    env = EnvPoolWrapper(env)
    
    # EnvPool already returns observations in (C, H, W) format, 
    # which is what PyTorch expects, so no need for VecTransposeImage!
    
    return env