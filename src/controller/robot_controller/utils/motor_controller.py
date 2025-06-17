import json
import math
from typing import List

class LPF_Fst:
    def __init__(self):
        self.fc = 0.0
        self.dt = 0.0
        self.y = 0.0
        self.y_old = 0.0
        self.a = 0.0
        self.b = 0.0
    
    def set_para(self, fc: float, dt: float):
        self.fc = fc
        self.dt = dt
        self.a = 1.0 / (2 * math.pi * fc * dt + 1)
        self.b = 2 * math.pi * fc * dt * self.a
    
    def ft_out(self, x: float) -> float:
        self.y = self.a * x + self.b * self.y_old
        self.y_old = self.y
        return self.y

class PVT_Ctr:
    def __init__(self, time_step: float, json_path: str):
        self.joint_num = len(self.motor_name)
        
        # Initialize vectors
        self.tau_out_lpf = [LPF_Fst() for _ in range(self.joint_num)]
        self.motor_vel = [0.0] * self.joint_num
        self.motor_pos_cur = [0.0] * self.joint_num
        self.motor_pos_des_old = [0.0] * self.joint_num
        self.motor_tor_out_link = [0.0] * self.joint_num
        self.motor_tor_out_motor = [0.0] * self.joint_num
        self.pvt_Kp = [0.0] * self.joint_num
        self.pvt_Kd = [0.0] * self.joint_num
        self.max_tor = [400.0] * self.joint_num
        self.max_vel = [50.0] * self.joint_num
        self.max_pos = [3.14] * self.joint_num
        self.min_pos = [-3.14] * self.joint_num
        self.PV_enable = [1] * self.joint_num
        self.gear = [1.0] * self.joint_num
        
        # Desired values
        self.motor_pos_des = [0.0] * self.joint_num
        self.motor_vel_des = [0.0] * self.joint_num
        self.motor_tor_des = [0.0] * self.joint_num
        
        # Read joint PVT parameters from JSON
        with open(json_path, 'r') as f:
            config = json.load(f)
            
        for i in range(self.joint_num):
            joint_name = self.motor_name[i]
            if joint_name in config:
                joint_config = config[joint_name]
                self.pvt_Kp[i] = joint_config.get("kp", 0.0)
                self.pvt_Kd[i] = joint_config.get("kd", 0.0)
                self.max_tor[i] = joint_config.get("maxTorque", 400.0)
                self.max_vel[i] = joint_config.get("maxSpeed", 50.0)
                self.max_pos[i] = joint_config.get("maxPos", 3.14)
                self.min_pos[i] = joint_config.get("minPos", -3.14)
                fc = joint_config.get("PVT_LPF_Fc", 0.0)
                self.gear[i] = joint_config.get("gear", 1.0)
                
                self.tau_out_lpf[i].set_para(fc, time_step)
                self.tau_out_lpf[i].ft_out(0.0)
    
    # Motor names
    motor_name = [
        "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04", "J_arm_l_05",
        "J_arm_l_06", "J_arm_l_07", "J_arm_r_01", "J_arm_r_02", "J_arm_r_03",
        "J_arm_r_04", "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
        "J_head_yaw", "J_head_pitch", "J_waist_pitch", "J_waist_roll", "J_waist_yaw",
        "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch", "J_knee_l_pitch",
        "J_ankle_l_pitch", "J_ankle_l_roll", "J_hip_r_roll", "J_hip_r_yaw",
        "J_hip_r_pitch", "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"
    ]
    
    def data_bus_read(self, bus):
        for i in range(self.joint_num):
            self.motor_pos_cur[i] = bus.motors_pos_cur[i]
            self.motor_vel[i] = bus.motors_vel_cur[i]
        
        self.motor_pos_des = bus.motors_pos_des
        self.motor_vel_des = bus.motors_vel_des
        self.motor_tor_des = bus.motors_tor_des
    
    def data_bus_write(self, bus):
        bus.motors_tor_out = self.motor_tor_out_motor
        bus.motors_tor_cur = self.motor_tor_out_link
    
    def set_joint_pd(self, kp: float, kd: float, joint_name: str):
        try:
            idx = self.motor_name.index(joint_name)
            self.pvt_Kp[idx] = kp
            self.pvt_Kd[idx] = kd
        except ValueError:
            print(f"{joint_name} NOT found!")
    
    def cal_motors_pvt(self):
        for i in range(self.joint_num):
            tau_des = (self.PV_enable[i] * self.pvt_Kp[i] * (self.motor_pos_des[i] - self.motor_pos_cur[i]) +
                      self.PV_enable[i] * self.pvt_Kd[i] * (self.motor_vel_des[i] - self.motor_vel[i]))
            
            tau_des = self.tau_out_lpf[i].ft_out(tau_des) + self.motor_tor_des[i]
            
            if abs(tau_des) >= abs(self.max_tor[i]):
                tau_des = self.sign(tau_des) * self.max_tor[i]
            
            self.motor_tor_out_motor[i] = tau_des / self.gear[i]
            self.motor_tor_out_link[i] = tau_des
            self.motor_pos_des_old[i] = self.motor_pos_des[i]
    
    def cal_motors_pvt_with_limit(self, delta_p_lim: float):
        for i in range(self.joint_num):
            delta = self.motor_pos_des[i] - self.motor_pos_des_old[i]
            
            if abs(delta) >= abs(delta_p_lim):
                delta = delta_p_lim * self.sign(delta)
            
            p_des = delta + self.motor_pos_des_old[i]
            
            tau_des = (self.PV_enable[i] * self.pvt_Kp[i] * (p_des - self.motor_pos_cur[i]) +
                      self.PV_enable[i] * self.pvt_Kd[i] * (self.motor_vel_des[i] - self.motor_vel[i]))
            
            tau_des = self.tau_out_lpf[i].ft_out(tau_des) + self.motor_tor_des[i]
            
            if abs(tau_des) >= abs(self.max_tor[i]):
                tau_des = self.sign(tau_des) * self.max_tor[i]
            
            self.motor_tor_out_motor[i] = tau_des / self.gear[i]
            self.motor_tor_out_link[i] = tau_des
            self.motor_pos_des_old[i] = p_des
    
    @staticmethod
    def sign(x: float) -> float:
        return 1.0 if x >= 0 else -1.0
    
    def enable_pv(self, jt_id: int = None):
        if jt_id is None:
            self.PV_enable = [1] * self.joint_num
        else:
            self.PV_enable[jt_id] = 1
    
    def disable_pv(self, jt_id: int = None):
        if jt_id is None:
            self.PV_enable = [0] * self.joint_num
        else:
            self.PV_enable[jt_id] = 0