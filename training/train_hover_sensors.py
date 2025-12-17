import sys
sys.path.append('..')
from envs.hover_sensors_env import HoverSensorsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def make_env(render=False, use_ekf=True):
    def _init():
        return HoverSensorsEnv(render_mode=render, use_ekf=use_ekf)
    return _init

print("Creating vectorized environments with sensors...")
envs = [make_env(render=False, use_ekf=True) for _ in range(2)]
env = DummyVecEnv(envs)

print("Initializing PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="../experiments/logs/hover_sensors/",
    device='cuda'
)

print("Starting training with realistic sensors and EKF...")
print("Agent receives estimated state, not ground truth")
model.learn(total_timesteps=200_000)

print("Saving model...")
os.makedirs("../experiments/checkpoints", exist_ok=True)
model.save("../experiments/checkpoints/hover_sensors_200k")

print("Training complete")
env.close()
