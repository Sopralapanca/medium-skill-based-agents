import pickle
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

class WeightMonitorCallback(BaseCallback):
    """
    Callback to monitor and save gating distribution during training
    """
    def __init__(self, feature_extractor, env, run_id, save_freq=1000, verbose=0, use_wandb=False):
        super().__init__(verbose)
        self.feature_extractor = feature_extractor
        self.save_freq = save_freq
        self.save_path = "./weights"
        self.weights_file = os.path.join(self.save_path, f"{env}_{run_id}.pkl")
        self.timesteps = []
        self.all_weights = []
        self.env = env
        self.run_id = run_id
        self.use_wandb = use_wandb

        
        os.makedirs(self.save_path, exist_ok=True)
        
    def save_weights(self):
        # Save the complete history
        save_data = {
            'timesteps': self.timesteps,
            'weights': self.all_weights,
            'skill_names': [skill.name for skill in self.feature_extractor.skills]
        }
        
        with open(self.weights_file, 'wb') as f:
            pickle.dump(save_data, f)
        
    def _on_step(self) -> bool:
        # Save weights periodically
        if self.n_calls % self.save_freq == 0:
            if hasattr(self.feature_extractor, 'training_weights') and len(self.feature_extractor.training_weights) > 0:
                # Get the accumulated weights (keep as list to handle variable batch sizes)
                weights = self.feature_extractor.training_weights.copy()
                self.all_weights.append(weights)
                self.timesteps.append(self.num_timesteps)
                
                # Log to wandb if enabled
                if self.use_wandb:
                    self._log_to_wandb(weights)
                
                # Clear the buffer to avoid memory issues
                self.feature_extractor.training_weights = []
                
                self.save_weights()
                
                if self.verbose > 0:
                    total_samples = sum(w.shape[0] for w in weights)
                    print(f"Step {self.num_timesteps}: Saved {len(weights)} weight arrays ({total_samples} total samples)")
        
        return True
    
    def _log_to_wandb(self, weights):
        """Log weight statistics to wandb"""
        # Weights are now pre-averaged over batch: each entry is [num_experts]
        all_weights_list = []
        for w in weights:
            if len(w.shape) == 1:  # 1D: [num_experts]
                all_weights_list.append(w.cpu().numpy())
            elif len(w.shape) == 2:  # 2D: [batch, num_experts] (legacy support)
                all_weights_list.append(w.cpu().numpy())
        
        if not all_weights_list:
            return
        
        # Stack to get [num_timesteps, num_experts] and average over time
        all_weights_stacked = np.stack(all_weights_list, axis=0)
        
        # Calculate mean weight per expert across all timesteps in this checkpoint
        mean_weights = np.mean(all_weights_stacked, axis=0)  # [num_experts]
        
        # Get skill names
        skill_names = [skill.name for skill in self.feature_extractor.skills]
        
        # Log individual expert weights
        log_dict = {}
        for i, name in enumerate(skill_names):
            log_dict[f"attention_weights/{name}"] = mean_weights[i]
        
        # Calculate and log entropy (measure of diversity)
        weights_safe = mean_weights + 1e-10
        weights_safe = weights_safe / weights_safe.sum()
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        log_dict["attention_weights/entropy"] = entropy
        log_dict["attention_weights/max_weight"] = np.max(mean_weights)
        
        # Log the dominant expert
        dominant_idx = np.argmax(mean_weights)
        log_dict["attention_weights/dominant_expert_idx"] = dominant_idx
        
        wandb.log(log_dict, step=self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Save all collected weights at the end of training"""
        # Save any remaining weights
        if hasattr(self.feature_extractor, 'training_weights') and len(self.feature_extractor.training_weights) > 0:
            weights = self.feature_extractor.training_weights.copy()
            self.all_weights.append(weights)
            self.timesteps.append(self.num_timesteps)
            
            # Log final weights to wandb
            if self.use_wandb:
                self._log_to_wandb(weights)
        
        self.save_weights()
    


def plot_gating_distribution(weights_file, output_dir="./gating_plots"):
    """
    Plot the gating distribution over training
    
    Args:
        weights_file: Path to the pickle file containing gating weights
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    with open(weights_file, 'rb') as f:
        data = pickle.load(f)
    
    timesteps = data['timesteps']
    weights = data['weights']
    skill_names = data['skill_names']
    num_experts = len(skill_names)
    
    print(f"Loaded gating weights from {len(timesteps)} checkpoints")
    print(f"Number of experts: {num_experts}")
    print(f"Expert names: {skill_names}")
    
    # Concatenate all weight arrays
    all_weights_concat = []
    for checkpoint_weights in weights:
        # checkpoint_weights is a list of tensors
        for w in checkpoint_weights:
            # w can be either [num_experts] (new format) or [batch, num_experts] (legacy)
            if hasattr(w, 'cpu'):
                w_np = w.cpu().numpy()
            else:
                w_np = w
            
            if len(w_np.shape) == 1:  # [num_experts] - new format (already averaged)
                all_weights_concat.append(w_np[np.newaxis, :])  # Add batch dim: [1, num_experts]
            elif len(w_np.shape) == 2:  # [batch, num_experts] - legacy format
                all_weights_concat.append(w_np)
            else:
                print(f"Warning: Unexpected weight shape {w_np.shape}, skipping")
    
    if not all_weights_concat:
        print("Error: No valid weights found!")
        return None
    
    all_weights_concat = np.concatenate(all_weights_concat, axis=0)  # (total_steps, num_experts)
    
    # Calculate statistics
    print(f"\nTotal steps recorded: {all_weights_concat.shape[0]}")
    print("\nMean weights per expert:")
    for i, name in enumerate(skill_names):
        print(f"  {name}: {np.mean(all_weights_concat[:, i]):.4f}")
    
    # 1. Plot mean weights over time (smoothed)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate moving average
    window_size = min(100, all_weights_concat.shape[0] // 10)
    
    # Plot 1: Individual expert weights over time
    ax = axes[0, 0]
    for i, name in enumerate(skill_names):
        weights_expert = all_weights_concat[:, i]
        # Moving average
        if window_size > 1:
            weights_smooth = np.convolve(weights_expert, np.ones(window_size)/window_size, mode='valid')
            steps = np.arange(len(weights_smooth))
        else:
            weights_smooth = weights_expert
            steps = np.arange(len(weights_smooth))
        ax.plot(steps, weights_smooth, label=name, linewidth=2)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Weight', fontsize=12)
    ax.set_title('Expert Weights Over Time (Smoothed)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stacked area plot
    ax = axes[0, 1]
    weights_smooth_all = []
    for i in range(num_experts):
        weights_expert = all_weights_concat[:, i]
        if window_size > 1:
            weights_smooth = np.convolve(weights_expert, np.ones(window_size)/window_size, mode='valid')
        else:
            weights_smooth = weights_expert
        weights_smooth_all.append(weights_smooth)
    
    weights_smooth_all = np.array(weights_smooth_all).T
    steps = np.arange(weights_smooth_all.shape[0])
    
    ax.stackplot(steps, *[weights_smooth_all[:, i] for i in range(num_experts)], 
                 labels=skill_names, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Weight Distribution', fontsize=12)
    ax.set_title('Expert Weight Distribution (Stacked)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Entropy over time (measure of diversity)
    ax = axes[1, 0]
    entropy = []
    for step_weights in all_weights_concat:
        # Add small epsilon to avoid log(0)
        step_weights_safe = step_weights + 1e-10
        step_weights_safe = step_weights_safe / step_weights_safe.sum()
        ent = -np.sum(step_weights_safe * np.log(step_weights_safe))
        entropy.append(ent)
    
    if window_size > 1:
        entropy_smooth = np.convolve(entropy, np.ones(window_size)/window_size, mode='valid')
        steps = np.arange(len(entropy_smooth))
    else:
        entropy_smooth = entropy
        steps = np.arange(len(entropy_smooth))
    
    ax.plot(steps, entropy_smooth, linewidth=2, color='purple')
    ax.axhline(y=np.log(num_experts), color='r', linestyle='--', label=f'Max Entropy (log {num_experts})')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Entropy (nats)', fontsize=12)
    ax.set_title('Gating Entropy Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final distribution (box plot or histogram)
    ax = axes[1, 1]
    final_weights = all_weights_concat[-1000:, :]  # Last 1000 steps
    
    bp = ax.boxplot([final_weights[:, i] for i in range(num_experts)], 
                     labels=skill_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, num_experts))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Final Weight Distribution (Last 1000 Steps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"gating_distribution_{os.path.basename(weights_file).replace('.pkl', '')}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_file}")
    plt.show()
    
    # Check for collapse
    print("\n" + "="*60)
    print("COLLAPSE ANALYSIS")
    print("="*60)
    
    final_mean_weights = np.mean(final_weights, axis=0)
    max_weight = np.max(final_mean_weights)
    max_expert = skill_names[np.argmax(final_mean_weights)]
    
    print(f"\nFinal mean weights:")
    for i, name in enumerate(skill_names):
        print(f"  {name}: {final_mean_weights[i]:.4f}")
    
    print(f"\nDominant expert: {max_expert} ({max_weight:.4f})")
    
    if max_weight > 0.7:
        print(f"\n⚠️  WARNING: Possible collapse detected! Expert '{max_expert}' dominates with {max_weight:.1%}")
    elif max_weight > 0.5:
        print(f"\n⚡ Expert '{max_expert}' is preferred but not collapsed ({max_weight:.1%})")
    else:
        print(f"\n✓ Good diversity! No single expert dominates (max: {max_weight:.1%})")
    
    return data