import os
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

import sys
sys.path.append('.')
from envs.hover_env import SimpleHoverEnv

print("Creating hover environment with rendering...")
env = SimpleHoverEnv(render_mode=True)

obs, info = env.reset()
print(f"Starting position: {obs[0:3]}")
print(f"Target position: {env.target_pos}")
print("Watch the GUI window - drone should hover near target")

for i in range(300):
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 50 == 0:
        pos = obs[0:3]
        print(f"Step {i}: position={pos}, reward={reward:.3f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {i}")
        if terminated:
            print("Reason: Success or crash")
        else:
            print("Reason: Timeout")
        break

env.close()
print("Test complete")
