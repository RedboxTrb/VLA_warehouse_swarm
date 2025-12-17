import gymnasium as gym
import numpy as np
from PyFlyt.core import Aviary
import sys
sys.path.append('..')
from utils.sensors import BudgetSensorSuite
from utils.ekf import ExtendedKalmanFilter

class AltitudeSensorsEnv(gym.Env):
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.aviary = None
        self.action_space = gym.spaces.Box(low=np.array([-0.3]), high=np.array([0.3]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.target_alt = 1.0
        self.max_steps = 300
        self.current_step = 0
        self.physics_hz = 240
        self.control_hz = 30
        self.steps_per_control = self.physics_hz // self.control_hz
        self.sensors = BudgetSensorSuite()
        self.ekf = ExtendedKalmanFilter(dt=1.0/self.control_hz)
        self.true_position = None
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self.aviary is not None:
            del self.aviary
        start_alt = self.target_alt + np.random.uniform(-0.3, 0.3)
        start_alt = max(start_alt, 0.3)
        self.aviary = Aviary(start_pos=np.array([[0.0, 0.0, start_alt]]), start_orn=np.array([[0.0, 0.0, 0.0]]), drone_type="quadx", render=self.render_mode, physics_hz=self.physics_hz)
        self.sensors.reset()
        self.ekf.reset(initial_state=np.array([0.0, 0.0, start_alt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.current_step = 0
        self.true_position = np.array([0.0, 0.0, start_alt])
        for _ in range(5):
            true_state = self._get_true_state()
            measurements = self.sensors.measure_all(true_state, dt=1.0/self.control_hz)
            self._update_ekf(measurements)
        obs = self._get_obs()
        return obs, {}
    
    def _get_true_state(self):
        state = self.aviary.state(0)
        return {'position': state[0], 'velocity': state[1], 'acceleration': np.array([0.0, 0.0, 9.81]), 'angular_velocity': state[3], 'motor_throttle': 0.6, 'obstacle_distances': [10.0, 10.0, 10.0, 10.0]}
    
    def _update_ekf(self, measurements):
        self.ekf.predict()
        self.ekf.update_imu_acc(measurements['imu_acc'])
        self.ekf.update_imu_gyro(measurements['imu_gyro'])
        self.ekf.update_barometer(measurements['baro_alt'])
        self.ekf.update_optical_flow(measurements['flow_vel'], measurements['flow_quality'])
        self.ekf.update_tof(measurements['tof_ground'])
    
    def _get_obs(self):
        state = self.ekf.get_state()
        uncertainty = self.ekf.get_uncertainty()
        obs = np.array([state['position'][2], state['velocity'][2], uncertainty[2], uncertainty[5], self.target_alt, self.target_alt - state['position'][2]]).astype(np.float32)
        return obs
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        thrust_adjust = action[0]
        estimated_state = self.ekf.get_state()
        alt = estimated_state['position'][2]
        vz = estimated_state['velocity'][2]
        alt_error = self.target_alt - alt
        kp = 0.8
        kd = 0.5
        base_thrust = 0.59
        thrust = base_thrust + kp * alt_error - kd * vz + thrust_adjust * 0.3
        thrust = np.clip(thrust, 0.2, 0.9)
        motors = np.array([thrust, thrust, thrust, thrust])
        for _ in range(self.steps_per_control):
            self.aviary.set_setpoint(0, motors)
            self.aviary.step()
        true_state = self._get_true_state()
        measurements = self.sensors.measure_all(true_state, dt=1.0/self.control_hz)
        self._update_ekf(measurements)
        self.current_step += 1
        obs = self._get_obs()
        self.true_position = true_state['position']
        alt_error_reward = abs(self.true_position[2] - self.target_alt)
        reward = 0.0
        reward -= alt_error_reward * 2.0
        reward -= 0.1 * abs(true_state['velocity'][2])
        reward -= 0.01 * action[0]**2
        terminated = False
        if alt_error_reward < 0.1:
            reward += 10
        if self.current_step > 50 and alt_error_reward < 0.15:
            reward += 50
            terminated = True
        if self.true_position[2] < 0.05 or self.true_position[2] > 3.0:
            reward -= 100
            terminated = True
        horiz_dist = np.linalg.norm(self.true_position[:2])
        if horiz_dist > 2.0:
            reward -= 50
            terminated = True
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}
    
    def close(self):
        if self.aviary is not None:
            del self.aviary
