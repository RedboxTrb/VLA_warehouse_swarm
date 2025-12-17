import numpy as np
import sys
sys.path.append('.')
from utils.sensors import BudgetSensorSuite

print("Testing Budget Sensor Suite")
print("=" * 60)

# Create sensor suite
sensors = BudgetSensorSuite()

# Simulate drone hovering at 1m altitude
true_state = {
    'position': np.array([0.0, 0.0, 1.0]),
    'velocity': np.array([0.0, 0.0, 0.0]),
    'acceleration': np.array([0.0, 0.0, 9.81]),  # Gravity compensation
    'angular_velocity': np.array([0.0, 0.0, 0.0]),
    'motor_throttle': 0.6,
    'obstacle_distances': [3.0, 3.0, 3.0, 3.0]
}

print("\nTrue state:")
print(f"  Position: {true_state['position']}")
print(f"  Velocity: {true_state['velocity']}")
print(f"  Altitude: {true_state['position'][2]:.3f} m")

print("\n" + "=" * 60)
print("Running 10 measurements:")
print("=" * 60)

for i in range(10):
    measurements = sensors.measure_all(true_state, dt=0.01)
    
    print(f"\nMeasurement {i+1}:")
    print(f"  IMU accel: {measurements['imu_acc']}")
    print(f"  IMU gyro: {measurements['imu_gyro']}")
    print(f"  Barometer: {measurements['baro_alt']:.3f} m (true: {true_state['position'][2]:.3f})")
    print(f"  Optical flow: {measurements['flow_vel']} m/s (quality: {measurements['flow_quality']:.2f})")
    print(f"  ToF ground: {measurements['tof_ground']:.3f} m")
    print(f"  Ultrasonics: {[f'{d:.2f}' for d in measurements['ultrasonic']]}")

print("\n" + "=" * 60)
print("Sensor test complete")

