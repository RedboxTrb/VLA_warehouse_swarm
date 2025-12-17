import gymnasium as gym
import numpy as np
from PyFlyt.core import Aviary
import sys
sys.path.append('..')
from utils.sensors import BudgetSensorSuite
from utils.ekf import ExtendedKalmanFilter

class HoverSensorsEnv(gym.Env):
    """
    Hover environment using realistic sensors and EKF
    Agent receives estimated state from EKF, not ground truth
    """
    
    def __init__(self, render_mode=False, use_ekf=True):
        self.render_mode = render_mode
        self.use_ekf = use_ekf
        self.aviary = None
        
        # Action: [thrust_adjust, roll, pitch, yaw_rate]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -0.3, -0.3, -1.0]),
            high=np.array([0.5, 0.3, 0.3, 1.0]),
            dtype=np.float32
        )
        
        if use_ekf:
            # Observation: EKF state (9) + uncertainty (9) = 18D
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
            )
        else:
            # Observation: raw sensor measurements (for comparison)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )
        
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.max_steps = 500
        self.current_step = 0
        self.physics_hz = 240
        self.control_hz = 30
        self.steps_per_control = self.physics_hz // self.control_hz
        
        # Sensors and EKF
        self.sensors = BudgetSensorSuite()
        self.ekf = ExtendedKalmanFilter(dt=1.0/self.control_hz)
        
        # Track actual position for reward calculation
        self.true_position = None
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        if self.aviary is not None:
            del self.aviary
        
        # Random start near target
        start_pos = self.target_pos + np.random.uniform(-0.3, 0.3, 3)
        start_pos[2] = max(start_pos[2], 0.5)
        
        self.aviary = Aviary(
            start_pos=np.array([start_pos]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            drone_type="quadx",
            render=self.render_mode,
            physics_hz=self.physics_hz
        )
        
        # Reset sensors and EKF
        self.sensors.reset()
        self.ekf.reset(initial_state=np.concatenate([start_pos, [0, 0, 0, 0, 0, 0]]))
        
        self.current_step = 0
        self.true_position = start_pos.copy()
        
        # Warm up EKF with a few measurements
        for _ in range(5):
            true_state = self._get_true_state()
            measurements = self.sensors.measure_all(true_state, dt=1.0/self.control_hz)
            self._update_ekf(measurements)
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_true_state(self):
        """Get ground truth state from simulator"""
        state = self.aviary.state(0)
        pos = state[0]
        vel = state[1]
        ang = state[2][:3]
        ang_vel = state[3]
        
        # Compute acceleration (approximate from velocity change)
        # For simplicity, assume hovering acceleration
        acc = np.array([0.0, 0.0, 9.81])
        
        return {
            'position': pos,
            'velocity': vel,
            'acceleration': acc,
            'angular_velocity': ang_vel,
            'motor_throttle': 0.6,
            'obstacle_distances': [10.0, 10.0, 10.0, 10.0]
        }
    
    def _update_ekf(self, measurements):
        """Update EKF with sensor measurements"""
        self.ekf.predict()
        self.ekf.update_imu_acc(measurements['imu_acc'])
        self.ekf.update_imu_gyro(measurements['imu_gyro'])
        self.ekf.update_barometer(measurements['baro_alt'])
        self.ekf.update_optical_flow(measurements['flow_vel'], measurements['flow_quality'])
        self.ekf.update_tof(measurements['tof_ground'])
    
    def _get_obs(self):
        """Get observation for agent"""
        if self.use_ekf:
            # EKF state estimate + uncertainty
            state = self.ekf.get_state()
            uncertainty = self.ekf.get_uncertainty()
            
            obs = np.concatenate([
                state['position'],
                state['velocity'],
                state['angles'],
                uncertainty
            ]).astype(np.float32)
        else:
            # Raw sensor measurements (for comparison)
            true_state = self._get_true_state()
            measurements = self.sensors.measure_all(true_state, dt=1.0/self.control_hz)
            
            obs = np.concatenate([
                measurements['imu_acc'],
                measurements['imu_gyro'],
                [measurements['baro_alt']],
                measurements['flow_vel'],
                [measurements['tof_ground']],
                measurements['ultrasonic']
            ]).astype(np.float32)
        
        return obs
    
    def _compute_motor_commands(self, action, estimated_state):
        """Convert actions to motor commands using estimated state"""
        pos = estimated_state['position']
        vel = estimated_state['velocity']
        
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
        
        # Get estimated state for control
        estimated_state = self.ekf.get_state()
        
        # Compute motor commands
        motors = self._compute_motor_commands(action, estimated_state)
        
        # Step physics
        for _ in range(self.steps_per_control):
            self.aviary.set_setpoint(0, motors)
            self.aviary.step()
        
        # Get new measurements and update EKF
        true_state = self._get_true_state()
        measurements = self.sensors.measure_all(true_state, dt=1.0/self.control_hz)
        self._update_ekf(measurements)
        
        self.current_step += 1
        
        # Get observation for agent
        obs = self._get_obs()
        
        # Compute reward using TRUE position (agent doesn't see this)
        self.true_position = true_state['position']
        distance = np.linalg.norm(self.true_position - self.target_pos)
        
        reward = 0.0
        reward -= distance
        reward -= 0.05 * np.linalg.norm(true_state['velocity'])
        reward -= 0.01 * np.sum(action**2)
        
        # Check termination
        terminated = False
        if distance < 0.2:
            reward += 100
            terminated = True
        
        if self.true_position[2] < 0.1 or self.true_position[2] > 5.0:
            reward -= 100
            terminated = True
        
        if np.abs(self.true_position[0]) > 5.0 or np.abs(self.true_position[1]) > 5.0:
            reward -= 100
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def close(self):
        if self.aviary is not None:
            del self.aviary
