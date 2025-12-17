import sys
sys.path.append('..')
from envs.hover_sensors_env import HoverSensorsEnv
from stable_baselines3 import PPO
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Enable rendering')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
parser.add_argument('--model', type=str, default='hover_sensors_200k', help='Model name')
args = parser.parse_args()

print("Loading trained model...")
model = PPO.load(f"../experiments/checkpoints/{args.model}")

print(f"Testing policy for {args.episodes} episodes (render={args.render})...")
env = HoverSensorsEnv(render_mode=args.render, use_ekf=True)

success_count = 0
crash_count = 0
timeout_count = 0
total_rewards = []
episode_lengths = []
estimation_errors = []

for episode in range(args.episodes):
    obs, info = env.reset()
    episode_reward = 0
    episode_errors = []
    
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Track estimation error
        est_state = env.ekf.get_state()
        error = np.linalg.norm(est_state['position'] - env.true_position)
        episode_errors.append(error)
        
        if terminated or truncated:
            break
    
    total_rewards.append(episode_reward)
    episode_lengths.append(step + 1)
    estimation_errors.append(np.mean(episode_errors))
    
    pos = env.true_position
    distance = np.linalg.norm(pos - env.target_pos)
    
    if terminated:
        if distance < 0.2:
            success_count += 1
            status = "SUCCESS"
        else:
            crash_count += 1
            status = "CRASH"
    else:
        timeout_count += 1
        status = "TIMEOUT"
    
    print(f"Episode {episode+1}: {status} (reward={episode_reward:.2f}, steps={step+1}, "
          f"final_dist={distance:.3f}m, avg_est_error={np.mean(episode_errors):.3f}m)")

env.close()

print(f"\nResults Summary:")
print(f"Success rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
print(f"Crash rate: {crash_count}/{args.episodes}")
print(f"Timeout rate: {timeout_count}/{args.episodes}")
print(f"Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
print(f"Average estimation error: {np.mean(estimation_errors):.3f} m")

