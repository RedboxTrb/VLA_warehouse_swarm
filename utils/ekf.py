import numpy as np

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for drone state estimation
    
    State vector (9D):
        [x, y, z, vx, vy, vz, roll, pitch, yaw]
    
    Fuses measurements from:
        - IMU (acceleration, angular velocity)
        - Barometer (altitude)
        - Optical flow (horizontal velocity)
        - ToF (ground distance)
    """
    
    def __init__(self, dt=0.01):
        
        self.dt = dt
        
        # State dimension
        self.n = 9
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.x = np.zeros(9)
        
        # Covariance matrix
        self.P = np.eye(9) * 1.0
        
        # Process noise covariance
        self.Q = np.diag([
            0.01, 0.01, 0.01,  # Position
            0.1, 0.1, 0.1,     # Velocity
            0.01, 0.01, 0.01   # Angles
        ])
        
        # Measurement noise covariances
        self.R_imu_acc = np.eye(3) * 0.05
        self.R_imu_gyro = np.eye(3) * 0.01
        self.R_baro = np.array([[0.2]])
        self.R_flow = np.eye(2) * 0.1
        self.R_tof = np.array([[0.05]])
        
        # Gravity constant
        self.g = 9.81
        
        # Last update time
        self.last_time = 0.0
        
    def predict(self, control_input=None):
        """
        Prediction step using motion model
        
        Args:
            control_input: Not used (IMU provides motion info)
        """
        # Extract state components
        x, y, z = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        roll, pitch, yaw = self.x[6:9]
        
        # State transition (simple kinematic model)
        # Position update
        self.x[0] += vx * self.dt
        self.x[1] += vy * self.dt
        self.x[2] += vz * self.dt
        
        # Velocity and angles updated by measurement updates
        
        # Jacobian of state transition
        F = np.eye(9)
        F[0, 3] = self.dt  # dx/dvx
        F[1, 4] = self.dt  # dy/dvy
        F[2, 5] = self.dt  # dz/dvz
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        
    def update_imu_acc(self, acc_measurement):
        """
        Update with IMU acceleration measurement
        
        Args:
            acc_measurement: [ax, ay, az] in body frame
        """
        # Extract angles
        roll, pitch, yaw = self.x[6:9]
        
        # Rotation matrix from body to world frame (simplified)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Expected acceleration in body frame (gravity rotated)
        h = np.array([
            self.g * sp,
            -self.g * sr * cp,
            -self.g * cr * cp
        ])
        
        # Measurement residual
        y = acc_measurement - h
        
        # Measurement Jacobian (simplified - gravity effect on angles)
        H = np.zeros((3, 9))
        H[0, 7] = self.g * cp  # pitch
        H[1, 6] = -self.g * cr * cp  # roll
        H[1, 7] = self.g * sr * sp  # pitch
        H[2, 6] = self.g * sr * cp  # roll
        H[2, 7] = self.g * cr * sp  # pitch
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_imu_acc
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ y
        
        # Covariance update
        self.P = (np.eye(9) - K @ H) @ self.P
        
    def update_imu_gyro(self, gyro_measurement):
        """
        Update with IMU gyroscope measurement
        
        Args:
            gyro_measurement: [wx, wy, wz] angular velocity
        """
        # Integrate angular velocity to update angles
        # Simple integration (for small angles)
        self.x[6] += gyro_measurement[0] * self.dt  # roll
        self.x[7] += gyro_measurement[1] * self.dt  # pitch
        self.x[8] += gyro_measurement[2] * self.dt  # yaw
        
        # Wrap angles to [-pi, pi]
        self.x[6:9] = np.arctan2(np.sin(self.x[6:9]), np.cos(self.x[6:9]))
        
    def update_barometer(self, baro_measurement):
        """
        Update with barometer altitude measurement
        
        Args:
            baro_measurement: altitude in meters
        """
        # Measurement model: z = altitude
        h = self.x[2]
        
        # Measurement residual
        y = np.array([baro_measurement - h])
        
        # Measurement Jacobian
        H = np.zeros((1, 9))
        H[0, 2] = 1.0  # Measures z directly
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += (K @ y).flatten()
        
        # Covariance update
        self.P = (np.eye(9) - K @ H) @ self.P
        
    def update_optical_flow(self, flow_measurement, quality):
        """
        Update with optical flow velocity measurement
        
        Args:
            flow_measurement: [vx, vy] horizontal velocity
            quality: measurement quality [0-1]
        """
        if quality < 0.2:
            return  # Reject low quality measurements
        
        # Adjust measurement noise based on quality
        R = self.R_flow / quality
        
        # Measurement model: measures vx, vy
        h = self.x[3:5]
        
        # Measurement residual
        y = flow_measurement - h
        
        # Measurement Jacobian
        H = np.zeros((2, 9))
        H[0, 3] = 1.0  # Measures vx
        H[1, 4] = 1.0  # Measures vy
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ y
        
        # Covariance update
        self.P = (np.eye(9) - K @ H) @ self.P
        
    def update_tof(self, tof_measurement):
        """
        Update with ToF ground distance measurement
        
        Args:
            tof_measurement: distance to ground in meters
        """
        # Measurement model: measures altitude
        h = self.x[2]
        
        # Measurement residual
        y = np.array([tof_measurement - h])
        
        # Measurement Jacobian
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_tof
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += (K @ y).flatten()
        
        # Covariance update
        self.P = (np.eye(9) - K @ H) @ self.P
        
    def get_state(self):
        """
        Get current state estimate
        
        Returns:
            Dictionary with state components
        """
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
            'angles': self.x[6:9].copy(),
            'covariance': self.P.copy()
        }
    
    def get_uncertainty(self):
        """
        Get uncertainty estimates
        
        Returns:
            Standard deviations for each state component
        """
        return np.sqrt(np.diag(self.P))
    
    def reset(self, initial_state=None):
        """
        Reset filter state
        
        Args:
            initial_state: Optional initial state vector
        """
        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(9)
        
        self.P = np.eye(9) * 1.0
        self.last_time = 0.0

