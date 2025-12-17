import numpy as np
from PyFlyt.core import Aviary

print("Initializing PyFlyt Aviary (headless mode)...")
aviary = Aviary(
    start_pos=np.array([[0.0, 0.0, 1.0]]),
    start_orn=np.array([[0.0, 0.0, 0.0]]),
    drone_type="quadx",
    render=False
)

print("Aviary created")
print("Running 200 simulation steps...")

for i in range(200):
    aviary.step()
    
    if i % 50 == 0:
        state = aviary.state(0)
        pos = state[0]
        print(f"Step {i}: position={pos}")

# Remove close, not needed in headless mode
print("\nTest complete - PyFlyt working")
