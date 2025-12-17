import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import sys
import os
import glob

def plot_training(logdir):
    # Find the actual log directory
    if not os.path.exists(logdir):
        print(f"Directory {logdir} does not exist")
        return
    
    # Find subdirectories
    subdirs = glob.glob(os.path.join(logdir, "PPO_*"))
    if not subdirs:
        print(f"No PPO logs found in {logdir}")
        return
    
    # Use the most recent one
    logdir = sorted(subdirs)[-1]
    print(f"Loading logs from: {logdir}")
    
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    # Get available scalars
    tags = ea.Tags()['scalars']
    print(f"Available metrics: {tags}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Plot episode reward
    if 'rollout/ep_rew_mean' in tags:
        data = ea.Scalars('rollout/ep_rew_mean')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 0].plot(steps, values, linewidth=2)
        axes[0, 0].set_title('Episode Reward Mean', fontsize=12)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode length
    if 'rollout/ep_len_mean' in tags:
        data = ea.Scalars('rollout/ep_len_mean')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 1].plot(steps, values, linewidth=2, color='green')
        axes[0, 1].set_title('Episode Length Mean', fontsize=12)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot value loss
    if 'train/value_loss' in tags:
        data = ea.Scalars('train/value_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 0].plot(steps, values, linewidth=2, color='red')
        axes[1, 0].set_title('Value Loss', fontsize=12)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot policy loss
    if 'train/policy_gradient_loss' in tags:
        data = ea.Scalars('train/policy_gradient_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 1].plot(steps, values, linewidth=2, color='orange')
        axes[1, 1].set_title('Policy Gradient Loss', fontsize=12)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'training_progress.png'
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "../experiments/logs/hover/"
    plot_training(logdir)

