import json
import math
from typing import List

class LPF_Fst:
    """First-order low-pass filter"""
    def __init__(self):
        self.y_prev = 0.0
    
    def setPara(self, fc: float, dt: float):
        """Set filter parameters"""
        self.alpha = 2 * math.pi * fc * dt / (2 * math.pi * fc * dt + 1)
    
    def ftOut(self, x: float) -> float:
        """Filter the input signal"""
        self.y_prev = self.alpha * x + (1 - self.alpha) * self.y_prev
        return self.y_prev

class MotorController:
    def __init__(self, timeStepIn: float, jsonPath: str):
        # Motor names (should be defined elsewhere or passed as parameter)
        self.motorName = ["motor1", "motor2", ...]  # Replace with actual motor names
        
        self.jointNum = len(self.motorName)
        self.timeStep = timeStepIn
        
        # Initialize arrays
        self.tau_out_lpf = [LPF_Fst() for _ in range(self.jointNum)]
        self.motor_vel = [0.0] * self.jointNum
        self.motor_pos_cur = [0.0] * self.jointNum
        self.motor_pos_des_old = [0.0] * self.jointNum
        self.motor_tor_out_link = [0.0] * self.jointNum
        self.motor_tor_out_motor = [0.0] * self.jointNum
        self.pvt_Kp = [0.0] * self.jointNum
        self.pvt_Kd = [0.0] * self.jointNum
        self.maxTor = [400.0] * self.jointNum
        self.maxVel = [50.0] * self.jointNum
        self.maxPos = [3.14] * self.jointNum
        self.minPos = [-3.14] * self.jointNum
        self.PV_enable = [1] * self.jointNum
        self.gear = [1.0] * self.jointNum
        
        # Read joint parameters from JSON
        self._read_joint_params(jsonPath)
    
    def _read_joint_params(self, jsonPath: str):
        """Read joint parameters from JSON file"""
        with open(jsonPath, 'r') as f:
            params = json.load(f)
        
        for i in range(self.jointNum):
            motor = self.motorName[i]
            if motor in params:
                self.pvt_Kp[i] = params[motor].get("kp", 0.0)
                self.pvt_Kd[i] = params[motor].get("kd", 0.0)
                self.maxTor[i] = params[motor].get("maxTorque", 400.0)
                self.maxVel[i] = params[motor].get("maxSpeed", 50.0)
                self.maxPos[i] = params[motor].get("maxPos", 3.14)
                self.minPos[i] = params[motor].get("minPos", -3.14)
                fc = params[motor].get("PVT_LPF_Fc", 10.0)  # Default cutoff 10Hz
                self.gear[i] = params[motor].get("gear", 1.0)
                
                self.tau_out_lpf[i].setPara(fc, self.timeStep)
                self.tau_out_lpf[i].ftOut(0.0)
    
    def dataBusRead(self, busIn):
        """Read data from DataBus"""
        for i in range(self.jointNum):
            self.motor_pos_cur[i] = busIn.motors_pos_cur[i]
            self.motor_vel[i] = busIn.motors_vel_cur[i]
        
        self.motor_pos_des = busIn.motors_pos_des.copy()
        self.motor_vel_des = busIn.motors_vel_des.copy()
        self.motor_tor_des = busIn.motors_tor_des.copy()
    
    def dataBusWrite(self, busIn):
        """Write data to DataBus"""
        busIn.motors_tor_out = self.motor_tor_out_motor.copy()
        busIn.motors_tor_cur = self.motor_tor_out_link.copy()
    
    def setJointPD(self, kp: float, kd: float, jointName: str):
        """Set PD gains for specific joint"""
        try:
            idx = self.motorName.index(jointName)
            self.pvt_Kp[idx] = kp
            self.pvt_Kd[idx] = kd
        except ValueError:
            print(f"{jointName} NOT found!")
    
    def calMotorsPVT(self, deltaP_Lim: float = None):
        """Calculate motor torques using PVT control"""
        for i in range(self.jointNum):
            tauDes = 0.0
            
            if deltaP_Lim is None:
                # Regular PVT control
                pos_error = self.motor_pos_des[i] - self.motor_pos_cur[i]
            else:
                # PVT control with position delta limit
                delta = self.motor_pos_des[i] - self.motor_pos_des_old[i]
                if abs(delta) >= abs(deltaP_Lim):
                    delta = deltaP_Lim * self.sign(delta)
                pos_error = (self.motor_pos_des_old[i] + delta) - self.motor_pos_cur[i]
                self.motor_pos_des_old[i] += delta
            
            # PD control
            vel_error = self.motor_vel_des[i] - self.motor_vel[i]
            tauDes = (self.PV_enable[i] * self.pvt_Kp[i] * pos_error + 
                     self.PV_enable[i] * self.pvt_Kd[i] * vel_error)
            
            # Apply low-pass filter and feedforward torque
            tauDes = self.tau_out_lpf[i].ftOut(tauDes) + self.motor_tor_des[i]
            
            # Torque limiting
            if abs(tauDes) >= abs(self.maxTor[i]):
                tauDes = self.sign(tauDes) * self.maxTor[i]
            
            # Apply gear ratio
            self.motor_tor_out_motor[i] = tauDes / self.gear[i]
            self.motor_tor_out_link[i] = tauDes
            
            if deltaP_Lim is None:
                self.motor_pos_des_old[i] = self.motor_pos_des[i]
    
    @staticmethod
    def sign(x: float) -> float:
        """Sign function"""
        return 1.0 if x >= 0 else -1.0
    
    def enablePV(self, jtId: int = None):
        """Enable PV control for all or specific joint"""
        if jtId is None:
            self.PV_enable = [1] * self.jointNum
        else:
            self.PV_enable[jtId] = 1
    
    def disablePV(self, jtId: int = None):
        """Disable PV control for all or specific joint"""
        if jtId is None:
            self.PV_enable = [0] * self.jointNum
        else:
            self.PV_enable[jtId] = 0