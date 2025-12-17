import sys
sys.path.append('.')
from envs.hover_sensors_env import HoverSensorsEnv
import numpy as np

print("Testing Hover Environment with Sensors")
print("=" * 60)

env = HoverSensorsEnv(render_mode=False, use_ekf=True)

obs, info = env.reset()

print(f"Observation space: {env.observation_space}")
print(f"Observation dimension: {obs.shape}")
print(f"First observation: {obs}")
print(f"\nTarget position: {env.target_pos}")

print("\n" + "=" * 60)
print("Running 50 random steps:")
print("=" * 60)

for i in range(50):
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    
    if (i + 1) % 10 == 0:
        est_state = env.ekf.get_state()
        print(f"\nStep {i+1}:")
        print(f"  Estimated position: {est_state['position']}")
        print(f"  True position: {env.true_position}")
        print(f"  Estimation error: {np.linalg.norm(est_state['position'] - env.true_position):.4f} m")
        print(f"  Reward: {reward:.3f}")
    
    if terminated or truncated:
        print(f"\nEpisode ended at step {i+1}")
        break

env.close()
print("\n" + "=" * 60)
print("Test complete")

