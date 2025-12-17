import numpy as np

class IMUSensor:
    """
    Inertial Measurement Unit - measures acceleration and angular velocity
    Models noise, bias, bias drift, and temperature effects
    """
    def __init__(self, acc_noise_std=0.02, gyro_noise_std=0.002,
                 acc_bias_std=0.05, gyro_bias_std=0.01,
                 bias_drift_rate=0.0001, sample_rate=1000):
        
        # Noise parameters
        self.acc_noise_std = acc_noise_std
        self.gyro_noise_std = gyro_noise_std
        
        # Bias parameters
        self.acc_bias = np.random.normal(0, acc_bias_std, 3)
        self.gyro_bias = np.random.normal(0, gyro_bias_std, 3)
        
        # Bias drift
        self.bias_drift_rate = bias_drift_rate
        
        # Sample rate
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Temperature model
        self.temperature = 25.0  # Celsius
        
    def measure(self, true_acc, true_gyro, motor_throttle=0.0):
        """
        Generate noisy IMU measurements
        
        Args:
            true_acc: True acceleration [ax, ay, az] in m/s^2
            true_gyro: True angular velocity [wx, wy, wz] in rad/s
            motor_throttle: Motor throttle [0-1] for temperature effects
            
        Returns:
            measured_acc, measured_gyro
        """
        # Update temperature based on motor usage
        self.temperature += motor_throttle * 0.01 - 0.005
        self.temperature = np.clip(self.temperature, 20, 60)
        
        # Temperature-dependent bias drift
        temp_factor = 1.0 + 0.02 * (self.temperature - 25.0)
        
        # Bias random walk
        self.acc_bias += np.random.normal(0, self.bias_drift_rate * temp_factor, 3)
        self.gyro_bias += np.random.normal(0, self.bias_drift_rate * temp_factor, 3)
        
        # Add measurement noise and bias
        measured_acc = true_acc + self.acc_bias + np.random.normal(0, self.acc_noise_std, 3)
        measured_gyro = true_gyro + self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        
        return measured_acc, measured_gyro
    
    def reset(self):
        """Reset sensor state for new episode"""
        self.acc_bias = np.random.normal(0, 0.05, 3)
        self.gyro_bias = np.random.normal(0, 0.01, 3)
        self.temperature = 25.0


class BarometerSensor:
    """
    Barometer - measures altitude via air pressure
    Models noise, drift, and response lag
    """
    def __init__(self, noise_std=0.1, drift_rate=0.001, response_time=0.05):
        
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        
        # First-order lag filter
        self.tau = response_time
        self.filtered_altitude = 0.0
        
        # Slow drift
        self.drift = 0.0
        
    def measure(self, true_altitude, dt=0.01):
        """
        Generate noisy barometer measurement
        
        Args:
            true_altitude: True altitude in meters
            dt: Time step in seconds
            
        Returns:
            measured_altitude
        """
        # Update drift
        self.drift += np.random.normal(0, self.drift_rate)
        
        # Apply first-order lag
        alpha = dt / (self.tau + dt)
        self.filtered_altitude = alpha * true_altitude + (1 - alpha) * self.filtered_altitude
        
        # Add noise and drift
        measured = self.filtered_altitude + self.drift + np.random.normal(0, self.noise_std)
        
        return measured
    
    def reset(self):
        """Reset sensor state"""
        self.filtered_altitude = 0.0
        self.drift = 0.0


class OpticalFlowSensor:
    """
    Optical flow - estimates velocity from camera
    Quality depends on altitude, texture, lighting, and velocity
    """
    def __init__(self, baseline_noise=0.05):
        
        self.baseline_noise = baseline_noise
        self.min_altitude = 0.2
        self.max_altitude = 5.0
        self.optimal_altitude = 1.5
        
    def compute_quality(self, altitude, velocity_mag, texture_score=1.0, lighting=1.0):
        """
        Compute measurement quality based on conditions
        
        Args:
            altitude: Height above ground in meters
            velocity_mag: Velocity magnitude in m/s
            texture_score: Ground texture quality [0-1]
            lighting: Lighting quality [0-1]
            
        Returns:
            quality [0-1]
        """
        # Altitude quality (optimal around 1-2m)
        if altitude < self.min_altitude:
            alt_quality = 0.0
        elif altitude > self.max_altitude:
            alt_quality = 0.1
        else:
            # Gaussian-like around optimal
            alt_quality = np.exp(-0.5 * ((altitude - self.optimal_altitude) / 1.5) ** 2)
        
        # Velocity quality (motion blur at high speeds)
        vel_quality = np.exp(-velocity_mag / 3.0)
        
        # Combined quality
        quality = alt_quality * vel_quality * texture_score * lighting
        quality = np.clip(quality, 0.0, 1.0)
        
        return quality
    
    def measure(self, true_velocity, altitude, texture_score=1.0, lighting=1.0):
        """
        Generate noisy velocity measurement
        
        Args:
            true_velocity: True velocity [vx, vy] in m/s (horizontal only)
            altitude: Height above ground
            texture_score: Ground texture quality [0-1]
            lighting: Lighting quality [0-1]
            
        Returns:
            measured_velocity [vx, vy], quality
        """
        velocity_mag = np.linalg.norm(true_velocity)
        quality = self.compute_quality(altitude, velocity_mag, texture_score, lighting)
        
        # Noise inversely proportional to quality
        if quality < 0.1:
            # Very poor quality - random measurement
            measured = np.random.normal(0, 1.0, 2)
            quality = 0.0
        else:
            noise_std = self.baseline_noise / quality
            measured = true_velocity + np.random.normal(0, noise_std, 2)
        
        return measured, quality
    
    def reset(self):
        """Reset sensor state"""
        pass


class UltrasonicSensor:
    """
    Ultrasonic rangefinder - measures distance to obstacles
    Models beam width, range limits, and occasional false readings
    """
    def __init__(self, min_range=0.02, max_range=4.0, noise_std=0.05, 
                 beam_width=30.0, false_reading_prob=0.02):
        
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.beam_width = np.deg2rad(beam_width)
        self.false_reading_prob = false_reading_prob
        
    def measure(self, true_distance):
        """
        Generate noisy distance measurement
        
        Args:
            true_distance: True distance to obstacle in meters
            
        Returns:
            measured_distance
        """
        # Random false reading
        if np.random.random() < self.false_reading_prob:
            return np.random.uniform(self.min_range, self.max_range)
        
        # Out of range
        if true_distance < self.min_range or true_distance > self.max_range:
            return self.max_range
        
        # Add noise
        measured = true_distance + np.random.normal(0, self.noise_std)
        measured = np.clip(measured, self.min_range, self.max_range)
        
        return measured
    
    def reset(self):
        """Reset sensor state"""
        pass


class ToFSensor:
    """
    Time-of-Flight sensor - more accurate than ultrasonic
    Used for precise ground distance measurement
    """
    def __init__(self, min_range=0.05, max_range=2.0, noise_std=0.03,
                 material_factor=1.0):
        
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.material_factor = material_factor
        
    def measure(self, true_distance, surface_reflectivity=1.0):
        """
        Generate ToF measurement
        
        Args:
            true_distance: True distance in meters
            surface_reflectivity: Surface property [0-1]
                                 0 = black/absorbing, 1 = white/reflective
            
        Returns:
            measured_distance
        """
        # Poor reflectivity increases noise and reduces range
        effective_noise = self.noise_std / (0.3 + 0.7 * surface_reflectivity)
        effective_range = self.max_range * (0.5 + 0.5 * surface_reflectivity)
        
        if true_distance > effective_range:
            return effective_range
        
        if true_distance < self.min_range:
            return self.min_range
        
        # Add noise
        measured = true_distance + np.random.normal(0, effective_noise)
        measured = np.clip(measured, self.min_range, effective_range)
        
        return measured
    
    def reset(self):
        """Reset sensor state"""
        pass


class BudgetSensorSuite:
    """
    Complete sensor suite for budget drone
    Combines all sensors with realistic characteristics
    """
    def __init__(self):
        
        self.imu = IMUSensor()
        self.barometer = BarometerSensor()
        self.optical_flow = OpticalFlowSensor()
        self.ultrasonics = [UltrasonicSensor() for _ in range(4)]  # Front, back, left, right
        self.tof = ToFSensor()
        
        self.last_update_time = 0.0
        
    def measure_all(self, true_state, dt=0.01):
        """
        Generate measurements from all sensors
        
        Args:
            true_state: Dictionary with keys:
                - position: [x, y, z]
                - velocity: [vx, vy, vz]
                - acceleration: [ax, ay, az]
                - angular_velocity: [wx, wy, wz]
                - motor_throttle: average motor command [0-1]
                - obstacle_distances: [front, back, left, right]
                
        Returns:
            measurements: Dictionary of sensor readings
        """
        measurements = {}
        
        # IMU measurements
        acc_meas, gyro_meas = self.imu.measure(
            true_state['acceleration'],
            true_state['angular_velocity'],
            true_state.get('motor_throttle', 0.0)
        )
        measurements['imu_acc'] = acc_meas
        measurements['imu_gyro'] = gyro_meas
        
        # Barometer
        measurements['baro_alt'] = self.barometer.measure(true_state['position'][2], dt)
        
        # Optical flow
        vel_xy = true_state['velocity'][:2]
        altitude = true_state['position'][2]
        measurements['flow_vel'], measurements['flow_quality'] = self.optical_flow.measure(
            vel_xy, altitude
        )
        
        # Ultrasonics
        obstacle_distances = true_state.get('obstacle_distances', [4.0, 4.0, 4.0, 4.0])
        measurements['ultrasonic'] = [
            sensor.measure(dist) for sensor, dist in zip(self.ultrasonics, obstacle_distances)
        ]
        
        # ToF
        measurements['tof_ground'] = self.tof.measure(altitude)
        
        self.last_update_time += dt
        
        return measurements
    
    def reset(self):
        """Reset all sensors"""
        self.imu.reset()
        self.barometer.reset()
        self.optical_flow.reset()
        for sensor in self.ultrasonics:
            sensor.reset()
        self.tof.reset()
        self.last_update_time = 0.0
