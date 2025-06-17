import math

class JoyStickInterpreter:
    def __init__(self, dt):
        self.dt = dt  # Time step
        
        # Velocity generators (assuming these are implemented elsewhere)
        self.vxLGen = VelocityGenerator()
        self.vyLGen = VelocityGenerator()
        self.wzLGen = VelocityGenerator()
        
        # State variables
        self.vx_L = 0.0  # Local x velocity
        self.vy_L = 0.0  # Local y velocity
        self.wz_L = 0.0  # Local angular velocity (z-axis)
        
        self.thetaZ = 0.0  # Current orientation (yaw)
        
        # World frame positions and velocities
        self.px_W = 0.0  # World x position
        self.py_W = 0.0  # World y position
        self.pz_W = 0.0  # World z position (fixed for flat terrain)
        self.vx_W = 0.0  # World x velocity
        self.vy_W = 0.0  # World y velocity
        self.vz_W = 0.0  # World z velocity (fixed for flat terrain)
    
    def setVxDesLPara(self, vxDesLIn, timeToReach):
        """Set desired x velocity in local frame with smoothing"""
        self.vxLGen.setPara(vxDesLIn, timeToReach)
    
    def setVyDesLPara(self, vyDesLIn, timeToReach):
        """Set desired y velocity in local frame with smoothing"""
        self.vyLGen.setPara(vyDesLIn, timeToReach)
    
    def setWzDesLPara(self, wzDesLIn, timeToReach):
        """Set desired angular velocity (z-axis) with smoothing"""
        self.wzLGen.setPara(wzDesLIn, timeToReach)
    
    def step(self):
        """Update the interpreter state"""
        # Get smoothed velocity commands
        self.vx_L = self.vxLGen.step()
        self.vy_L = self.vyLGen.step()
        self.wz_L = self.wzLGen.step()
        
        # Update orientation
        self.thetaZ += self.wz_L * self.dt
        
        # Convert local velocities to world frame
        self.vx_W = math.cos(self.thetaZ) * self.vx_L - math.sin(self.thetaZ) * self.vy_L
        self.vy_W = math.sin(self.thetaZ) * self.vx_L + math.cos(self.thetaZ) * self.vy_L
        
        # Update position
        self.px_W += self.vx_W * self.dt
        self.py_W += self.vy_W * self.dt
    
    def dataBusWrite(self, dataBus):
        """Write commands to DataBus"""
        dataBus.js_pos_des[0] = self.px_W
        dataBus.js_pos_des[1] = self.py_W
        dataBus.js_vel_des[0] = self.vx_W
        dataBus.js_vel_des[1] = self.vy_W
        dataBus.js_eul_des[2] = self.thetaZ
        dataBus.js_omega_des[2] = self.wz_L
        
        dataBus.base_pos_des = np.array([self.px_W, self.py_W, self.pz_W])
        dataBus.base_rpy_des[2] = self.thetaZ
        dataBus.base_vel_des = np.array([self.vx_W, self.vy_W, self.vz_W])
        dataBus.base_omega_des[2] = self.wz_L
    
    def reset(self):
        """Reset all commands to zero"""
        self.vxLGen.resetOut(0)
        self.vyLGen.resetOut(0)
        self.wzLGen.resetOut(0)
        
        self.vx_L = 0
        self.vy_L = 0
        self.wz_L = 0
        self.thetaZ = 0
        
        # Note: World positions are not reset here (use setIniPos)
    
    def setIniPos(self, posX, posY, thetaZ, posZ=None):
        """Set initial position and orientation"""
        self.px_W = posX
        self.py_W = posY
        self.thetaZ = thetaZ
        
        if posZ is not None:
            self.pz_W = posZ


class VelocityGenerator:
    """Helper class for smooth velocity transitions"""
    def __init__(self):
        self.current = 0.0
        self.target = 0.0
        self.ramp_time = 0.0
        self.elapsed_time = 0.0
    
    def setPara(self, target, timeToReach):
        """Set target velocity and time to reach it"""
        self.target = target
        self.ramp_time = max(timeToReach, 1e-6)  # Avoid division by zero
        self.elapsed_time = 0.0
    
    def step(self):
        """Update current velocity towards target"""
        if self.elapsed_time < self.ramp_time:
            alpha = self.elapsed_time / self.ramp_time
            self.current = self.current * (1 - alpha) + self.target * alpha
            self.elapsed_time += 1.0/1000  # Assuming 1kHz control loop
        else:
            self.current = self.target
        return self.current
    
    def resetOut(self, value):
        """Reset to specific value"""
        self.current = value
        self.target = value
        self.elapsed_time = 0.0
        self.ramp_time = 0.0