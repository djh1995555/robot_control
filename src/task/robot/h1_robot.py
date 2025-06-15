import math
from src.task.robot.legged_robot import LeggedRobot
from src.task.robot.h1_state import H1State
from pinocchio import pin
import mujoco
import sys
from scipy.spatial.transform import Rotation

class H1Robot(LeggedRobot):
    def __init__(self, cfg):
        super().__init__(cfg)
        urdf_path = cfg['urdf_path']
        self.model_biped = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.state = H1State(self.model_biped.nv)

        self.joint_names=[ 
            "J_arm_l_01","J_arm_l_02","J_arm_l_03","J_arm_l_04","J_arm_l_05","J_arm_l_06","J_arm_l_07",
            "J_arm_r_01","J_arm_r_02","J_arm_r_03","J_arm_r_04","J_arm_r_05","J_arm_r_06","J_arm_r_07",
            "J_head_yaw","J_head_pitch","J_waist_pitch","J_waist_roll", "J_waist_yaw",
            "J_hip_l_roll","J_hip_l_yaw","J_hip_l_pitch","J_hip_r_roll","J_hip_r_yaw","J_hip_r_pitch", 
            "J_ankle_l_pitch", "J_ankle_l_roll", "J_ankle_r_pitch", "J_ankle_r_roll",
            "J_knee_l_pitch","J_knee_r_pitch",]
        self.base_name="base_link"
        self.orientation_sensor_name="baselink-quat"
        self.vel_sensor_name="baselink-velocity"
        self.gyro_sensor_name="baselink-gyro"
        self.acc_sensor_name="baselink-baseAcc"


    def mujoco_init_state(self, model, data):
        # init state using mujoco
        self.joint_qpos_id = []
        self.joint_qvel_id = []
        self.joint_dctl_id = []
        
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                print(f"{joint_name} not found in the XML file!")
                sys.exit()
                
            # 存储关节的位置和速度地址
            self.joint_qpos_id.append(model.jnt_qposadr[joint_id])
            self.joint_qvel_id.append(model.jnt_dofadr[joint_id])
            
            # 构造对应的驱动器名称 (假设驱动器名称是关节名前加'M')
            motor_name = "M" + joint_name[1:] if joint_name.startswith('J') else "M_" + joint_name
            
            # 获取驱动器ID
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
            if actuator_id == -1:
                print(f"{motor_name} not found in the XML file!")
                sys.exit()
                
            self.joint_dctl_id.append(actuator_id)
        self.base_name="base_link"
        self.orientation_sensor_name="baselink-quat"
        self.vel_sensor_name="baselink-velocity"
        self.gyro_sensor_name="baselink-gyro"
        self.acc_sensor_name="baselink-baseAcc"

        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        self.orientataion_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.orientation_sensor_name)
        self.vel_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.vel_sensor_name)
        self.gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.gyro_sensor_name)
        self.acc_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.acc_sensor_name)

        # init mujoco state from cfg
        return data
    
    def mujoco_state_adoption(self, model, data):
        for i in range(len(self.joint_names)):
            self.state.motor_pos_cur[i] = data.qpos[self.joint_qpos_id[i]]
            self.state.motor_vel_cur[i] = data.qpos[self.joint_qvel_id[i]]
        
        for i in range(4):
            self.state.base_quat[i] = data.sensordata[model.sensor_adr[self.orientataion_sensor_id] + i]
        # first_element = base_quat.pop(0)
        # base_quat.append(first_element)

        # self.state.rpy[0] = math.atan2(2 * (base_quat[3] * base_quat[0] + base_quat[1] * base_quat[2]), 1 - 2 * (base_quat[0] * base_quat[0] + base_quat[1] * base_quat[1]))
        # self.state.rpy[1] = math.asin(2 * (base_quat[3] * base_quat[1] + base_quat[0] * base_quat[2]))
        # self.state.rpy[2] = math.atan2(2 * (base_quat[3] * base_quat[2] + base_quat[0] * base_quat[1]), 1 - 2 * (base_quat[1] * base_quat[1] + base_quat[2] * base_quat[2]))
        
        self.state.base_rpy = Rotation.from_quat(self.state.base_quat).as_euler('zyx', degrees=False)
        yaw_N = 0.0
        if ((self.state.base_rpy[2] - self.state.yaw_pre) > 0.5 * math.pi):
            yaw_N -= 1.0
        elif (self.state.base_rpy[2] - self.state.yaw_pre) < -0.5 * math.pi:
            yaw_N += 1.0
        
        self.state.yaw_pre = self.state.base_rpy[2]
        self.state.base_rpy[2] = self.state.yaw_pre + yaw_N * 2.0 * math.pi

        for i in range(3):
            pos_pre = self.state.base_pos[i]
            self.state.base_acc[i] = data.sensordata[model.sensor_adr[self.acc_sensor_id] + i]
            self.state.base_vel[i] = (self.state.base_pos[i] - pos_pre) / (model.opt.timestep)
            self.state.base_omega[i] = data.sensordata[model.sensor_adr[self.gyro_sensor_id] + i]

        self.state.base_pos = data.xpos[self.base_body_id,:]
        self.state.update()
        return self.state
    
    def mujoco_action_adoption(self, action, model, data):
        return data