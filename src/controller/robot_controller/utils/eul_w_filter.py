import numpy as np
from scipy.linalg import inv

class EulWFilter:
    def __init__(self, dt):
        self.dt = dt
        self.isIni = False
        
        # State covariance matrix
        self.P = np.eye(6)
        
        # Observation matrix
        self.H = np.eye(6)
        
        # State transition matrix (will be updated in each step)
        self.F = np.eye(6)
        
        # Process noise covariance (adjust these values as needed)
        self.Q = np.eye(6) * 0.01
        
        # Measurement noise covariance (adjust these values as needed)
        self.R = np.eye(6) * 0.1
        
        # State vector [roll, pitch, yaw, wx, wy, wz]
        self.kal_X = np.zeros(6)
        
        # Measurement vector
        self.kal_Z = np.zeros(6)
        
        # Filtered outputs
        self.Eul_filtered = np.zeros(3)
        self.wL_filtered = np.zeros(3)
    
    def run(self, eulIn, wLIn):
        """Run the Kalman filter update step"""
        # Update measurement vector
        self.kal_Z = np.array([eulIn[0], eulIn[1], eulIn[2], 
                              wLIn[0], wLIn[1], wLIn[2]])
        
        if not self.isIni:
            # Initialize state on first run
            self.kal_X = self.kal_Z.copy()
            self.Eul_filtered = self.kal_X[:3].copy()
            self.wL_filtered = self.kal_X[3:6].copy()
            self.isIni = True
            return
        
        # Get previous state
        roll_Old, pitch_Old, yaw_Old = self.kal_X[0], self.kal_X[1], self.kal_X[2]
        
        # Update state transition matrix F
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[0, 4] = self.dt * np.sin(roll_Old) * np.sin(pitch_Old) / np.cos(pitch_Old)
        self.F[0, 5] = self.dt * np.cos(roll_Old) * np.sin(pitch_Old) / np.cos(pitch_Old)
        self.F[1, 4] = self.dt * np.cos(roll_Old)
        self.F[1, 5] = -self.dt * np.sin(roll_Old)
        self.F[2, 4] = self.dt * np.sin(roll_Old) / np.cos(pitch_Old)
        self.F[2, 5] = self.dt * np.cos(roll_Old) / np.cos(pitch_Old)
        
        # Predict step
        self.kal_X = self.F @ self.kal_X
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        kal_Y = self.kal_Z - self.H @ self.kal_X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.kal_X = self.kal_X + K @ kal_Y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Store filtered values
        self.Eul_filtered = self.kal_X[:3].copy()
        self.wL_filtered = self.kal_X[3:6].copy()
    
    def getData(self):
        """Get the filtered Euler angles and angular velocities"""
        return self.Eul_filtered.copy(), self.wL_filtered.copy()