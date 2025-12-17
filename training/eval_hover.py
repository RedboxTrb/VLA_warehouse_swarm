import sys
sys.path.append('..')
from envs.hover_env import SimpleHoverEnv
from stable_baselines3 import PPO
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
args = parser.parse_args()

print("Loading trained model...")
model = PPO.load("../experiments/checkpoints/hover_100k")

print(f"Testing policy for {args.episodes} episodes (render={args.render})...")
env = SimpleHoverEnv(render_mode=args.render)

success_count = 0
crash_count = 0
timeout_count = 0
total_rewards = []
episode_lengths = []

for episode in range(args.episodes):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    total_rewards.append(episode_reward)
    episode_lengths.append(step + 1)
    
    pos = obs[0:3]
    distance = np.linalg.norm(pos - env.target_pos)
    
    if terminated:
        if distance < 0.15:
            success_count += 1
            status = "SUCCESS"
        else:
            crash_count += 1
            status = "CRASH"
    else:
        timeout_count += 1
        status = "TIMEOUT"
    
    print(f"Episode {episode+1}: {status} (reward={episode_reward:.2f}, steps={step+1}, final_dist={distance:.3f}m)")

env.close()

print(f"\nResults Summary:")
print(f"Success rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
print(f"Crash rate: {crash_count}/{args.episodes}")
print(f"Timeout rate: {timeout_count}/{args.episodes}")
print(f"Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
