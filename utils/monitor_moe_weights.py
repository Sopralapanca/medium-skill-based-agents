import pickle
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt

class GatingMonitorCallback(BaseCallback):
    """
    Callback to monitor and save gating distribution during training
    """
    def __init__(self, feature_extractor, env, save_freq=1000, save_path="./gating_weights", verbose=0):
        super().__init__(verbose)
        self.feature_extractor = feature_extractor
        self.save_freq = save_freq
        self.save_path = save_path
        self.timesteps = []
        self.all_weights = []
        self.env = env
        
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Save weights periodically
        if self.n_calls % self.save_freq == 0:
            if hasattr(self.feature_extractor, 'training_weights') and len(self.feature_extractor.training_weights) > 0:
                # Get the accumulated weights
                weights = np.array(self.feature_extractor.training_weights)
                self.all_weights.append(weights)
                self.timesteps.append(self.num_timesteps)
                
                # Clear the buffer to avoid memory issues
                self.feature_extractor.training_weights = []
                
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Saved gating weights (shape: {weights.shape})")
        
        return True
    
    def _on_training_end(self) -> None:
        """Save all collected weights at the end of training"""
        # Save any remaining weights
        if hasattr(self.feature_extractor, 'training_weights') and len(self.feature_extractor.training_weights) > 0:
            weights = np.array(self.feature_extractor.training_weights)
            self.all_weights.append(weights)
            self.timesteps.append(self.num_timesteps)
        
        # Save the complete history
        save_data = {
            'timesteps': self.timesteps,
            'weights': self.all_weights,
            'skill_names': [skill.name for skill in self.feature_extractor.skills]
        }
        
        save_file = os.path.join(self.save_path, f"gating_weights_{self.env}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nGating weights saved to {save_file}")


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
    for w in weights:
        # w has shape (num_steps_in_chunk, batch_size, num_experts)
        # Average over batch dimension
        w_mean = np.mean(w, axis=1)  # (num_steps_in_chunk, num_experts)
        all_weights_concat.append(w_mean)
    
    all_weights_concat = np.concatenate(all_weights_concat, axis=0)  # (total_steps, num_experts)
    
    # Calculate statistics
    print(f"\nTotal steps recorded: {all_weights_concat.shape[0]}")
    print(f"\nMean weights per expert:")
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