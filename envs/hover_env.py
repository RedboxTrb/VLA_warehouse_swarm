import gymnasium as gym
import numpy as np
from PyFlyt.core import Aviary

class SimpleHoverEnv(gym.Env):
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.aviary = None
        
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -0.3, -0.3, -1.0]),
            high=np.array([0.5, 0.3, 0.3, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.max_steps = 500
        self.current_step = 0
        self.physics_hz = 240
        self.control_hz = 30
        self.steps_per_control = self.physics_hz // self.control_hz
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        if self.aviary is not None:
            del self.aviary
        
        start_pos = self.target_pos + np.random.uniform(-0.3, 0.3, 3)
        start_pos[2] = max(start_pos[2], 0.5)
        
        self.aviary = Aviary(
            start_pos=np.array([start_pos]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            drone_type="quadx",
            render=self.render_mode,
            physics_hz=self.physics_hz
        )
        
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        state = self.aviary.state(0)
        pos = state[0]
        vel = state[1]
        ang = state[2][:3]
        ang_vel = state[3]
        
        obs = np.concatenate([pos, vel, ang, ang_vel]).astype(np.float32)
        return obs
    
    def _compute_motor_commands(self, action, state):
        pos = state[0]
        vel = state[1]
        ang = state[2][:3]
        
        thrust_adjust = action[0]
        roll_cmd = action[1]
        pitch_cmd = action[2]
        yaw_rate = action[3]
        
        hover_thrust = 0.59
        
        kp = 0.3
        kd = 0.2
        
        pos_error = self.target_pos - pos
        
        des_roll = -kp * pos_error[1] - kd * vel[1]
        des_pitch = kp * pos_error[0] + kd * vel[0]
        
        des_roll = np.clip(des_roll, -0.3, 0.3)
        des_pitch = np.clip(des_pitch, -0.3, 0.3)
        
        roll_total = des_roll + roll_cmd * 0.1
        pitch_total = des_pitch + pitch_cmd * 0.1
        
        alt_error = self.target_pos[2] - pos[2]
        thrust = hover_thrust + 0.5 * alt_error + 0.3 * (-vel[2]) + thrust_adjust * 0.2
        thrust = np.clip(thrust, 0.1, 1.0)
        
        motor_0 = thrust - roll_total - pitch_total - yaw_rate * 0.1
        motor_1 = thrust + roll_total - pitch_total + yaw_rate * 0.1
        motor_2 = thrust - roll_total + pitch_total + yaw_rate * 0.1
        motor_3 = thrust + roll_total + pitch_total - yaw_rate * 0.1
        
        motors = np.array([motor_0, motor_1, motor_2, motor_3])
        motors = np.clip(motors, 0.0, 1.0)
        
        return motors
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        state = self.aviary.state(0)
        motors = self._compute_motor_commands(action, state)
        
        for _ in range(self.steps_per_control):
            self.aviary.set_setpoint(0, motors)
            self.aviary.step()
        
        self.current_step += 1
        obs = self._get_obs()
        
        pos = obs[0:3]
        vel = obs[3:6]
        distance = np.linalg.norm(pos - self.target_pos)
        
        reward = 0.0
        reward -= distance
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.01 * np.sum(action**2)
        
        terminated = False
        if distance < 0.2:
            reward += 100
            terminated = True
        
        if pos[2] < 0.1 or pos[2] > 5.0:
            reward -= 100
            terminated = True
        
        if np.abs(pos[0]) > 5.0 or np.abs(pos[1]) > 5.0:
            reward -= 100
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def close(self):
        if self.aviary is not None:
            del self.aviary
