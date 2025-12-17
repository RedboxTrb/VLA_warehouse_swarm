import numpy as np
import sys
sys.path.append('.')
from utils.sensors import BudgetSensorSuite
from utils.ekf import ExtendedKalmanFilter

print("Testing Extended Kalman Filter")
print("=" * 60)

# Create sensor suite and EKF
sensors = BudgetSensorSuite()
ekf = ExtendedKalmanFilter(dt=0.01)

# Initialize EKF with approximate starting position
ekf.reset(initial_state=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# True state - hovering drone
true_state = {
    'position': np.array([0.0, 0.0, 1.0]),
    'velocity': np.array([0.0, 0.0, 0.0]),
    'acceleration': np.array([0.0, 0.0, 9.81]),
    'angular_velocity': np.array([0.0, 0.0, 0.0]),
    'motor_throttle': 0.6,
    'obstacle_distances': [3.0, 3.0, 3.0, 3.0]
}

print("\nTrue state:")
print(f"  Position: {true_state['position']}")
print(f"  Velocity: {true_state['velocity']}")

print("\n" + "=" * 60)
print("Running 50 EKF updates:")
print("=" * 60)

for i in range(50):
    # Get sensor measurements
    measurements = sensors.measure_all(true_state, dt=0.01)
    
    # EKF prediction step
    ekf.predict()
    
    # EKF update steps with each sensor
    ekf.update_imu_acc(measurements['imu_acc'])
    ekf.update_imu_gyro(measurements['imu_gyro'])
    ekf.update_barometer(measurements['baro_alt'])
    ekf.update_optical_flow(measurements['flow_vel'], measurements['flow_quality'])
    ekf.update_tof(measurements['tof_ground'])
    
    # Print every 10 iterations
    if (i + 1) % 10 == 0:
        state = ekf.get_state()
        uncertainty = ekf.get_uncertainty()
        
        print(f"\nIteration {i+1}:")
        print(f"  Estimated position: {state['position']}")
        print(f"  True position:      {true_state['position']}")
        print(f"  Position error:     {np.linalg.norm(state['position'] - true_state['position']):.4f} m")
        print(f"  Position uncertainty: [{uncertainty[0]:.3f}, {uncertainty[1]:.3f}, {uncertainty[2]:.3f}]")
        print(f"  Estimated velocity: {state['velocity']}")
        print(f"  Velocity uncertainty: [{uncertainty[3]:.3f}, {uncertainty[4]:.3f}, {uncertainty[5]:.3f}]")

print("\n" + "=" * 60)
final_state = ekf.get_state()
final_error = np.linalg.norm(final_state['position'] - true_state['position'])
print(f"Final position error: {final_error:.4f} m")
print(f"Final position estimate: {final_state['position']}")
print(f"True position: {true_state['position']}")
print("=" * 60)
print("EKF test complete")

