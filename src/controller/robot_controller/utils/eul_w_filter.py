import numpy as np

class Eul_W_filter:
    def __init__(self, dt: float):
        """
        Euler angles and angular velocity Kalman filter
        
        Args:
            dt: Time step for the filter
        """
        self.dt = dt
        self.isIni = False
        
        # State vectors
        self.kal_X = np.zeros(6)  # State vector [eul, wL]
        self.kal_Y = np.zeros(6)  # Innovation
        self.kal_Z = np.zeros(6)  # Measurement vector
        
        # Matrices
        self.F = np.eye(6)  # State transition matrix
        self.H = np.eye(6)  # Output matrix
        self.P = np.eye(6)  # Covariance matrix
        self.Q = np.eye(6)  # Process noise matrix
        self.R = np.eye(6)  # Output noise matrix
        
        # Filtered outputs
        self.Eul_filtered = np.zeros(3)
        self.wL_filtered = np.zeros(3)
    
    def run(self, eulIn: np.ndarray, wLIn: np.ndarray):
        """
        Run one iteration of the Kalman filter
        
        Args:
            eulIn: Input Euler angles [roll, pitch, yaw] in radians
            wLIn: Input angular velocities [wx, wy, wz] in rad/s
        """
        self.kal_Z = np.concatenate([eulIn, wLIn])
        
        if not self.isIni:
            # Initialize state with first measurement
            self.kal_X = self.kal_Z.copy()
            self.Eul_filtered = self.kal_X[:3].copy()
            self.wL_filtered = self.kal_X[3:].copy()
            self.isIni = True
            return
        
        # Get previous state
        roll_Old, pitch_Old, yaw_Old = self.kal_X[:3]
        
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
        self.kal_Y = self.kal_Z - self.H @ self.kal_X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.kal_X = self.kal_X + K @ self.kal_Y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Update filtered outputs
        self.Eul_filtered = self.kal_X[:3].copy()
        self.wL_filtered = self.kal_X[3:].copy()
    
    def getData(self):
        """
        Get the filtered Euler angles and angular velocities
        
        Returns:
            tuple: (Euler angles [roll, pitch, yaw], angular velocities [wx, wy, wz])
        """
        return self.Eul_filtered.copy(), self.wL_filtered.copy()