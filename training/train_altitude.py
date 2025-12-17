import sys
sys.path.append('..')
from envs.altitude_sensors_env import AltitudeSensorsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    def _init():
        return AltitudeSensorsEnv(render_mode=False)
    return _init

print("Training altitude control with sensors...")
envs = [make_env() for _ in range(4)]
env = DummyVecEnv(envs)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="../experiments/logs/altitude/",
    device='cuda'
)

print("Training for 50k steps (fast)...")
model.learn(total_timesteps=50_000)

model.save("../experiments/checkpoints/altitude_50k")
print("Training complete")
env.close()

