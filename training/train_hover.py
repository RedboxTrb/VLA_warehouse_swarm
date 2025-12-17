import sys
sys.path.append('..')
from envs.hover_env import SimpleHoverEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def make_env(render=False):
    def _init():
        return SimpleHoverEnv(render_mode=render)
    return _init

print("Creating vectorized environments...")
# Reduce to 2 environments to save memory
envs = [make_env(render=False) for _ in range(2)]
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
    tensorboard_log="../experiments/logs/hover/",
    device='cuda'
)

print("Starting training for 100k steps...")  # Reduced from 200k
model.learn(total_timesteps=100_000)

print("Saving model...")
os.makedirs("../experiments/checkpoints", exist_ok=True)
model.save("../experiments/checkpoints/hover_100k")

print("Training complete")
env.close()
