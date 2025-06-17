import math

class RampTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.y = 0.0
        self.yDes = 0.0
        self.yOld = 0.0
        self.k = 0.0
    
    def set_para(self, y_des_in, time_to_reach):
        self.yDes = y_des_in
        if time_to_reach < 0.001:
            time_to_reach = 0.001
        self.k = (y_des_in - self.yOld) / time_to_reach
    
    def set_para_dirt(self, y_des_in, delta_y):
        self.yDes = y_des_in
        self.k = delta_y / self.dt
    
    def step(self):
        self.y = self.yOld + self.k * self.dt
        if abs(self.yDes - self.y) < abs(1.5 * self.k * self.dt):
            self.y = self.yDes
        self.yOld = self.y
        return self.y
    
    def is_reach_des(self):
        return abs(self.yDes - self.y) < abs(self.k * self.dt)
    
    def reset_out(self, y_out):
        self.y = y_out
        self.yOld = y_out

class JoyStickInterpreter:
    def __init__(self, dt):
        self.dt = dt
        self.thetaZ = 0.0
        self.vx_W = 0.0
        self.vy_W = 0.0
        self.vz_W = 0.0
        self.px_W = 0.0
        self.py_W = 0.0
        self.pz_W = 0.0
        self.vx_L = 0.0
        self.vy_L = 0.0
        self.wz_L = 0.0
        
        self.vxLGen = RampTrajectory(dt)
        self.vyLGen = RampTrajectory(dt)
        self.wzLGen = RampTrajectory(dt)
        self.thetazGen = RampTrajectory(dt)
    
    def set_vx_des_l_para(self, vx_des_l_in, time_to_reach):
        self.vxLGen.set_para(vx_des_l_in, time_to_reach)
    
    def set_vy_des_l_para(self, vy_des_l_in, time_to_reach):
        self.vyLGen.set_para(vy_des_l_in, time_to_reach)
    
    def set_wz_des_l_para(self, wz_des_l_in, time_to_reach):
        self.wzLGen.set_para(wz_des_l_in, time_to_reach)
    
    def step(self):
        self.vx_L = self.vxLGen.step()
        self.vy_L = self.vyLGen.step()
        self.wz_L = self.wzLGen.step()
        
        self.thetaZ = self.thetaZ + self.wz_L * self.dt
        self.vx_W = math.cos(self.thetaZ) * self.vx_L - math.sin(self.thetaZ) * self.vy_L
        self.vy_W = math.sin(self.thetaZ) * self.vx_L + math.cos(self.thetaZ) * self.vy_L
        self.px_W += self.vx_W * self.dt
        self.py_W += self.vy_W * self.dt
    
    def data_bus_write(self, data_bus):
        data_bus.js_pos_des[0] = self.px_W
        data_bus.js_pos_des[1] = self.py_W
        data_bus.js_vel_des[0] = self.vx_W
        data_bus.js_vel_des[1] = self.vy_W
        data_bus.js_eul_des[2] = self.thetaZ
        data_bus.js_omega_des[2] = self.wz_L
        data_bus.base_pos_des[:] = [self.px_W, self.py_W, self.pz_W]
        data_bus.base_rpy_des[2] = self.thetaZ
        data_bus.base_vel_des[:] = [self.vx_W, self.vy_W, self.vz_W]
        data_bus.base_omega_des[2] = self.wz_L
    
    def reset(self):
        self.vxLGen.reset_out(0)
        self.vyLGen.reset_out(0)
        self.wzLGen.reset_out(0)
        self.vx_L = 0
        self.vy_L = 0
        self.wz_L = 0
        self.thetaZ = 0
    
    def set_ini_pos(self, pos_x, pos_y, theta_z):
        self.px_W = pos_x
        self.py_W = pos_y
        self.thetaZ = theta_z
    
    def set_ini_pos_3d(self, pos_x, pos_y, pos_z, theta_z):
        self.px_W = pos_x
        self.py_W = pos_y
        self.pz_W = pos_z
        self.thetaZ = theta_z