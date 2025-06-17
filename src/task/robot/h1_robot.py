from src.task.robot.legged_robot import LeggedRobot
import numpy as np
from typing import List
import mujoco
import pinocchio as pin
from src.task.robot.data_bus import DataBus
class H1Robot(LeggedRobot):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_biped = pin.buildModelFromUrdf(self.cfg['urdf_path'], pin.JointModelFreeFlyer())
        self.model_nv = self.model_biped.nv
        self.state = DataBus(self.model_nv)

    def init_state(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        Initialize the MJ_Interface with MuJoCo model and data
        
        Args:
            mj_model: MuJoCo model object
            mj_data: MuJoCo data object
        """
        # Constants
        self.JointName = [
            "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04", "J_arm_l_05",
            "J_arm_l_06", "J_arm_l_07", "J_arm_r_01", "J_arm_r_02", "J_arm_r_03",
            "J_arm_r_04", "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
            "J_head_yaw", "J_head_pitch", "J_waist_pitch", "J_waist_roll", "J_waist_yaw",
            "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch", "J_knee_l_pitch",
            "J_ankle_l_pitch", "J_ankle_l_roll", "J_hip_r_roll", "J_hip_r_yaw",
            "J_hip_r_pitch", "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"
        ]
        self.baseName = "base_link"
        self.orientationSensorName = "baselink-quat"  # in quat, mujoco order is [w,x,y,z], rearranged to [x,y,z,w]
        self.velSensorName = "baselink-velocity"
        self.gyroSensorName = "baselink-gyro"
        self.accSensorName = "baselink-baseAcc"

        self.mj_model = mj_model
        self.mj_data = mj_data
        self.timeStep = mj_model.opt.timestep
        self.jointNum = len(self.JointName)
        
        # Initialize arrays
        self.jntId_qpos = [0] * self.jointNum
        self.jntId_qvel = [0] * self.jointNum
        self.jntId_dctl = [0] * self.jointNum
        self.motor_pos = [0.0] * self.jointNum
        self.motor_vel = [0.0] * self.jointNum
        self.motor_pos_Old = [0.0] * self.jointNum
        
        # Setup joint and actuator mappings
        for i in range(self.jointNum):
            tmpId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.JointName[i])
            if tmpId == -1:
                raise ValueError(f"{self.JointName[i]} not found in the XML file!")
            
            self.jntId_qpos[i] = mj_model.jnt_qposadr[tmpId]
            self.jntId_qvel[i] = mj_model.jnt_dofadr[tmpId]
            
            motorName = "M" + self.JointName[i][1:]
            tmpId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, motorName)
            if tmpId == -1:
                raise ValueError(f"{motorName} not found in the XML file!")
            
            self.jntId_dctl[i] = tmpId
        
        # Setup sensor IDs
        self.baseBodyId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.baseName)
        self.orientataionSensorId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.orientationSensorName)
        self.velSensorId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.velSensorName)
        self.gyroSensorId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.gyroSensorName)
        self.accSensorId = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.accSensorName)
        
        # Initialize sensor data arrays
        self.rpy = np.zeros(3)  # roll, pitch and yaw of baselink
        self.yaw_simgle = 0.0
        self.yaw_N = 0
        self.baseQuat = np.zeros(4)  # in quat, mujoco order is [w,x,y,z], rearranged to [x,y,z,w]
        self.f3d = np.zeros((3, 2))  # 3D foot-end contact force, L for 1st col, R for 2nd col
        self.basePos = np.zeros(3)  # position of baselink, in world frame
        self.baseAcc = np.zeros(3)  # acceleration of baselink, in body frame
        self.baseAngVel = np.zeros(3)  # angular velocity of baselink, in body frame
        self.baseLinVel = np.zeros(3)  # linear velocity of baselink, in body frame
        
        return self.state
    
    def updateSensorValues(self, mj_data: mujoco.MjData):
        self.mj_data = mj_data
        """Update all sensor values from MuJoCo simulation"""
        # Update joint positions and velocities
        for i in range(self.jointNum):
            self.motor_pos_Old[i] = self.motor_pos[i]
            self.motor_pos[i] = self.mj_data.qpos[self.jntId_qpos[i]]
            self.motor_vel[i] = self.mj_data.qvel[self.jntId_qvel[i]]
        
        # Update orientation (convert quaternion to Euler angles)
        for i in range(4):
            self.baseQuat[i] = self.mj_data.sensordata[self.mj_model.sensor_adr[self.orientataionSensorId] + i]
        
        # Reorder quaternion from [w,x,y,z] to [x,y,z,w]
        tmp = self.baseQuat[0]
        self.baseQuat[0] = self.baseQuat[1]
        self.baseQuat[1] = self.baseQuat[2]
        self.baseQuat[2] = self.baseQuat[3]
        self.baseQuat[3] = tmp
        
        # Convert quaternion to Euler angles
        x, y, z, w = self.baseQuat
        self.rpy[0] = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))  # roll
        self.rpy[1] = np.arcsin(2*(w*y - z*x))  # pitch
        self.rpy[2] = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  # yaw
        
        # Handle yaw wrapping
        if (self.rpy[2] - self.yaw_simgle) > np.pi * 0.5:
            self.yaw_N -= 1.0
        elif (self.rpy[2] - self.yaw_simgle) < -np.pi * 0.5:
            self.yaw_N += 1.0
        
        self.yaw_simgle = self.rpy[2]
        self.rpy[2] = self.yaw_simgle + self.yaw_N * 2.0 * np.pi
        
        # Update position, acceleration, angular velocity, and linear velocity
        self.basePos = self.mj_data.xpos[self.baseBodyId]
        for i in range(3):
            posOld = self.basePos[i]
            
            self.baseAcc[i] = self.mj_data.sensordata[self.mj_model.sensor_adr[self.accSensorId] + i]
            self.baseAngVel[i] = self.mj_data.sensordata[self.mj_model.sensor_adr[self.gyroSensorId] + i]
            self.baseLinVel[i] = (self.basePos[i] - posOld) / self.mj_model.opt.timestep
    
    def dataBusWrite(self, busIn):
        """
        Write sensor data to DataBus
        
        Args:
            busIn: DataBus object to write to
        """
        busIn.motors_pos_cur = self.motor_pos.copy()
        busIn.motors_vel_cur = self.motor_vel.copy()
        busIn.rpy[0] = self.rpy[0]
        busIn.rpy[1] = self.rpy[1]
        busIn.rpy[2] = self.rpy[2]
        busIn.fL[0] = self.f3d[0][0]
        busIn.fL[1] = self.f3d[1][0]
        busIn.fL[2] = self.f3d[2][0]
        busIn.fR[0] = self.f3d[0][1]
        busIn.fR[1] = self.f3d[1][1]
        busIn.fR[2] = self.f3d[2][1]
        busIn.baseAcc[0] = self.baseAcc[0]
        busIn.baseAcc[1] = self.baseAcc[1]
        busIn.baseAcc[2] = self.baseAcc[2]
        busIn.baseAngVel[0] = self.baseAngVel[0]
        busIn.baseAngVel[1] = self.baseAngVel[1]
        busIn.baseAngVel[2] = self.baseAngVel[2]
        busIn.updateQ()

    def mujoco_state_adoption(self, mj_data: mujoco.MjData):
        self.updateSensorValues(mj_data)
        self.dataBusWrite(self.state)
        return self.state

    def init_mujoco_state(self, mj_data: mujoco.MjData):
        return mj_data
    
    def mujoco_action_adoption(self, action, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        for i in range(len(self.JointName)):
            mj_data.ctrl[i] = action[i]
            # mj_data.ctrl[i] = 0.0
        return mj_data
