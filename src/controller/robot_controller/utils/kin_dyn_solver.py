import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import List, Tuple
import json
from enum import Enum, auto

class LegIdx(Enum):
    left = auto()
    right = auto()

@dataclass
class IkRes:
    status: int
    itr: int
    err: np.ndarray
    jointPosRes: np.ndarray

class Pin_KinDyn:
    def __init__(self, urdf_path: str):
        """
        Initialize the Pinocchio kinematics and dynamics model
        
        Args:
            urdf_path: Path to the URDF file describing the robot
        """
        # Initialize models
        root_joint = pin.JointModelFreeFlyer()
        self.model_biped = pin.buildModelFromUrdf(urdf_path, root_joint)
        self.model_biped_fixed = pin.buildModelFromUrdf(urdf_path)
        
        # Create data structures
        self.data_biped = self.model_biped.createData()
        self.data_biped_fixed = self.model_biped_fixed.createData()
        
        # Model parameters
        self.model_nv = self.model_biped.nv
        self.urdf_path = urdf_path
        
        # Initialize matrices and vectors
        self.J_l = np.zeros((6, self.model_nv))
        self.J_r = np.zeros((6, self.model_nv))
        self.J_l_body = np.zeros((6, self.model_biped_fixed.nv))
        self.J_r_body = np.zeros((6, self.model_biped_fixed.nv))
        self.J_hd_l = np.zeros((6, self.model_nv))
        self.J_hd_r = np.zeros((6, self.model_nv))
        self.J_base = np.zeros((6, self.model_nv))
        self.J_hip_link = np.zeros((6, self.model_nv))
        self.dJ_l = np.zeros((6, self.model_nv))
        self.dJ_r = np.zeros((6, self.model_nv))
        self.dJ_hd_l = np.zeros((6, self.model_nv))
        self.dJ_hd_r = np.zeros((6, self.model_nv))
        self.dJ_base = np.zeros((6, self.model_nv))
        
        self.q = np.zeros(self.model_nv + 1)  # +1 for quaternion
        self.dq = np.zeros(self.model_nv)
        self.ddq = np.zeros(self.model_nv)
        self.Rcur = np.eye(3)
        
        # Dynamics matrices
        self.dyn_M = np.zeros((self.model_nv, self.model_nv))
        self.dyn_M_inv = np.zeros((self.model_nv, self.model_nv))
        self.dyn_C = np.zeros((self.model_nv, self.model_nv))
        self.dyn_G = np.zeros(self.model_nv)
        self.dyn_Ag = np.zeros((6, self.model_nv))
        self.dyn_dAg = np.zeros((6, self.model_nv))
        self.dyn_Non = np.zeros(self.model_nv)
        
        # Positions and orientations
        self.fe_r_pos = np.zeros(3)
        self.fe_l_pos = np.zeros(3)
        self.base_pos = np.zeros(3)
        self.fe_r_pos_body = np.zeros(3)
        self.fe_l_pos_body = np.zeros(3)
        self.hd_r_pos = np.zeros(3)
        self.hd_l_pos = np.zeros(3)
        self.fe_r_vel_body = np.zeros(3)
        self.fe_l_vel_body = np.zeros(3)
        self.hd_r_pos_body = np.zeros(3)
        self.hd_l_pos_body = np.zeros(3)
        self.hip_r_pos = np.zeros(3)
        self.hip_l_pos = np.zeros(3)
        self.hip_link_pos = np.zeros(3)
        self.hip_r_pos_body = np.zeros(3)
        self.hip_l_pos_body = np.zeros(3)
        
        # Rotation matrices
        self.hip_link_rot = np.eye(3)
        self.fe_r_rot = np.eye(3)
        self.fe_l_rot = np.eye(3)
        self.base_rot = np.eye(3)
        self.fe_r_rot_body = np.eye(3)
        self.fe_l_rot_body = np.eye(3)
        self.hd_r_rot = np.eye(3)
        self.hd_l_rot = np.eye(3)
        self.hd_r_rot_body = np.eye(3)
        self.hd_l_rot_body = np.eye(3)
        
        # Centroidal dynamics
        self.CoM_pos = np.zeros(3)
        self.inertia = np.eye(3)
        self.Jcom = np.zeros((3, self.model_nv))
        
        # Joint limits and motor parameters
        self.motorName = [
            "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04", "J_arm_l_05",
            "J_arm_l_06", "J_arm_l_07", "J_arm_r_01", "J_arm_r_02", "J_arm_r_03",
            "J_arm_r_04", "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
            "J_head_yaw", "J_head_pitch", "J_waist_pitch", "J_waist_roll", "J_waist_yaw",
            "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch", "J_knee_l_pitch",
            "J_ankle_l_pitch", "J_ankle_l_roll", "J_hip_r_roll", "J_hip_r_yaw",
            "J_hip_r_pitch", "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"
        ]
        
        self.motorReachLimit = [False] * len(self.motorName)
        self.motorMaxTorque = np.zeros(len(self.motorName))
        self.motorMaxPos = np.zeros(len(self.motorName))
        self.motorMinPos = np.zeros(len(self.motorName))
        self.tauJointOld = np.zeros(len(self.motorName))
        
        # Get joint indices
        self.r_ankle_joint = self.model_biped.getJointId("J_ankle_r_roll")
        self.l_ankle_joint = self.model_biped.getJointId("J_ankle_l_roll")
        self.r_hand_joint = self.model_biped.getJointId("J_arm_r_07")
        self.l_hand_joint = self.model_biped.getJointId("J_arm_l_07")
        self.r_hand_joint_fixed = self.model_biped_fixed.getJointId("J_arm_r_07")
        self.l_hand_joint_fixed = self.model_biped_fixed.getJointId("J_arm_l_07")
        self.r_hip_joint = self.model_biped.getJointId("J_hip_r_yaw")
        self.l_hip_joint = self.model_biped.getJointId("J_hip_l_yaw")
        self.r_hip_roll_joint = self.model_biped.getJointId("J_hip_r_roll")
        self.l_hip_roll_joint = self.model_biped.getJointId("J_hip_l_roll")
        self.r_ankle_joint_fixed = self.model_biped_fixed.getJointId("J_ankle_r_roll")
        self.l_ankle_joint_fixed = self.model_biped_fixed.getJointId("J_ankle_l_roll")
        self.r_hip_joint_fixed = self.model_biped_fixed.getJointId("J_hip_r_yaw")
        self.l_hip_joint_fixed = self.model_biped_fixed.getJointId("J_hip_l_yaw")
        self.base_joint = self.model_biped.getJointId("root_joint")
        self.waist_yaw_joint = self.model_biped.getJointId("J_waist_yaw")
        
        # Load joint limits from config file
        with open("/home/djh/robot/robot_control/config/h1/joint_ctrl_config.json", "r") as f:
            config = json.load(f)
            for i, name in enumerate(self.motorName):
                self.motorMaxTorque[i] = config[name]["maxTorque"]
                self.motorMaxPos[i] = config[name]["maxPos"]
                self.motorMinPos[i] = config[name]["minPos"]
    
    def dataBusRead(self, robotState):
        """
        Read data from DataBus into the kinematics/dynamics model
        
        Args:
            robotState: DataBus object containing current robot state
        """
        self.q = robotState.q.copy()
        self.dq = robotState.dq.copy()
        
        # Convert base velocities to local frame
        self.dq[:3] = robotState.base_rot.T @ self.dq[:3]
        self.dq[3:6] = robotState.base_rot.T @ self.dq[3:6]
        
        self.ddq = robotState.ddq.copy()
    
    def dataBusWrite(self, robotState):
        """
        Write kinematics and dynamics data to DataBus
        
        Args:
            robotState: DataBus object to be updated
        """
        robotState.J_l = self.J_l.copy()
        robotState.J_r = self.J_r.copy()
        robotState.J_base = self.J_base.copy()
        robotState.dJ_l = self.dJ_l.copy()
        robotState.dJ_r = self.dJ_r.copy()
        robotState.J_hd_l = self.J_hd_l.copy()
        robotState.J_hd_r = self.J_hd_r.copy()
        robotState.dJ_hd_l = self.dJ_hd_l.copy()
        robotState.dJ_hd_r = self.dJ_hd_r.copy()
        robotState.dJ_base = self.dJ_base.copy()
        robotState.J_hip_link = self.J_hip_link.copy()
        
        # Position and orientation data
        robotState.fe_l_pos_W = self.fe_l_pos.copy()
        robotState.fe_r_pos_W = self.fe_r_pos.copy()
        robotState.fe_l_pos_L = self.fe_l_pos_body.copy()
        robotState.fe_r_pos_L = self.fe_r_pos_body.copy()
        robotState.fe_l_rot_W = self.fe_l_rot.copy()
        robotState.fe_r_rot_W = self.fe_r_rot.copy()
        robotState.fe_l_rot_L = self.fe_l_rot_body.copy()
        robotState.fe_r_rot_L = self.fe_r_rot_body.copy()
        robotState.fe_l_vel_L = self.fe_l_vel_body.copy()
        robotState.fe_r_vel_L = self.fe_r_vel_body.copy()
        robotState.hip_r_pos_L = self.hip_r_pos_body.copy()
        robotState.hip_l_pos_L = self.hip_l_pos_body.copy()
        robotState.hip_r_pos_W = self.hip_r_pos.copy()
        robotState.hip_l_pos_W = self.hip_l_pos.copy()
        robotState.hd_l_pos_L = self.hd_l_pos_body.copy()
        robotState.hd_l_rot_L = self.hd_l_rot_body.copy()
        robotState.hd_l_pos_W = self.hd_l_pos.copy()
        robotState.hd_l_rot_W = self.hd_l_rot.copy()
        robotState.hd_r_pos_L = self.hd_r_pos_body.copy()
        robotState.hd_r_rot_L = self.hd_r_rot_body.copy()
        robotState.hd_r_pos_W = self.hd_r_pos.copy()
        robotState.hd_r_rot_W = self.hd_r_rot.copy()
        robotState.hip_link_pos = self.hip_link_pos.copy()
        robotState.hip_link_rot = self.hip_link_rot.copy()
        
        # Dynamics data
        robotState.dyn_M = self.dyn_M.copy()
        robotState.dyn_M_inv = self.dyn_M_inv.copy()
        robotState.dyn_C = self.dyn_C.copy()
        robotState.dyn_G = self.dyn_G.copy()
        robotState.dyn_Ag = self.dyn_Ag.copy()
        robotState.dyn_dAg = self.dyn_dAg.copy()
        robotState.dyn_Non = self.dyn_Non.copy()
        
        robotState.pCoM_W = self.CoM_pos.copy()
        robotState.Jcom_W = self.Jcom.copy()
        robotState.inertia = self.inertia.copy()
    
    def computeJ_dJ(self):
        """Compute Jacobians and their time derivatives"""
        # Forward kinematics
        pin.forwardKinematics(self.model_biped, self.data_biped, self.q)
        pin.jacobianCenterOfMass(self.model_biped, self.data_biped, self.q, True)
        pin.computeJointJacobiansTimeVariation(self.model_biped, self.data_biped, self.q, self.dq)
        pin.updateGlobalPlacements(self.model_biped, self.data_biped)
        
        # Get Jacobians
        self.J_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hd_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hd_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_base = pin.getJointJacobian(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED)
        
        # Get Jacobian time derivatives
        self.dJ_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_hd_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_hd_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_base = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED)
        
        # Get positions and orientations
        self.fe_l_pos = self.data_biped.oMi[self.l_ankle_joint].translation
        self.fe_l_rot = self.data_biped.oMi[self.l_ankle_joint].rotation
        self.hip_l_pos = self.data_biped.oMi[self.l_hip_joint].translation
        self.fe_r_pos = self.data_biped.oMi[self.r_ankle_joint].translation
        self.fe_r_rot = self.data_biped.oMi[self.r_ankle_joint].rotation
        self.hip_r_pos = self.data_biped.oMi[self.r_hip_joint].translation
        self.base_pos = self.data_biped.oMi[self.base_joint].translation
        self.base_rot = self.data_biped.oMi[self.base_joint].rotation
        self.hd_l_pos = self.data_biped.oMi[self.l_hand_joint].translation
        self.hd_l_rot = self.data_biped.oMi[self.l_hand_joint].rotation
        self.hd_r_pos = self.data_biped.oMi[self.r_hand_joint].translation
        self.hd_r_rot = self.data_biped.oMi[self.r_hand_joint].rotation
        self.hip_link_pos = self.data_biped.oMi[self.waist_yaw_joint].translation
        self.hip_link_rot = self.data_biped.oMi[self.waist_yaw_joint].rotation
        self.Jcom = self.data_biped.Jcom
        
        # Transform Jacobians to world frame
        Mpj = np.eye(self.model_nv)
        Mpj[:3, :3] = self.base_rot.T
        Mpj[3:6, 3:6] = self.base_rot.T
        
        self.J_l = self.J_l @ Mpj
        self.J_r = self.J_r @ Mpj
        self.J_base = self.J_base @ Mpj
        self.dJ_l = self.dJ_l @ Mpj
        self.dJ_r = self.dJ_r @ Mpj
        self.J_hd_l = self.J_hd_l @ Mpj
        self.J_hd_r = self.J_hd_r @ Mpj
        self.dJ_hd_l = self.dJ_hd_l @ Mpj
        self.dJ_hd_r = self.dJ_hd_r @ Mpj
        self.dJ_base = self.dJ_base @ Mpj
        self.J_hip_link = self.J_hip_link @ Mpj
        self.Jcom = self.Jcom @ Mpj
        
        # Compute fixed base model kinematics
        q_fixed = self.q[7:7+self.model_biped_fixed.nv]
        dq_fixed = self.dq[6:6+self.model_biped_fixed.nv]
        
        pin.forwardKinematics(self.model_biped_fixed, self.data_biped_fixed, q_fixed)
        pin.computeJointJacobians(self.model_biped_fixed, self.data_biped_fixed, q_fixed)
        pin.updateGlobalPlacements(self.model_biped_fixed, self.data_biped_fixed)
        
        self.J_r_body = pin.getJointJacobian(self.model_biped_fixed, self.data_biped_fixed, self.r_ankle_joint_fixed, pin.LOCAL_WORLD_ALIGNED)
        self.J_l_body = pin.getJointJacobian(self.model_biped_fixed, self.data_biped_fixed, self.l_ankle_joint_fixed, pin.LOCAL_WORLD_ALIGNED)
        
        self.fe_l_pos_body = self.data_biped_fixed.oMi[self.l_ankle_joint_fixed].translation
        self.fe_r_pos_body = self.data_biped_fixed.oMi[self.r_ankle_joint_fixed].translation
        self.fe_l_rot_body = self.data_biped_fixed.oMi[self.l_ankle_joint_fixed].rotation
        self.fe_r_rot_body = self.data_biped_fixed.oMi[self.r_ankle_joint_fixed].rotation
        self.hip_l_pos_body = self.data_biped_fixed.oMi[self.l_hip_joint_fixed].translation
        self.hip_r_pos_body = self.data_biped_fixed.oMi[self.r_hip_joint_fixed].translation
        self.hd_l_pos_body = self.data_biped_fixed.oMi[self.l_hand_joint_fixed].translation
        self.hd_l_rot_body = self.data_biped_fixed.oMi[self.l_hand_joint_fixed].rotation
        self.hd_r_pos_body = self.data_biped_fixed.oMi[self.r_hand_joint_fixed].translation
        self.hd_r_rot_body = self.data_biped_fixed.oMi[self.r_hand_joint_fixed].rotation
        self.fe_l_vel_body = (self.J_l_body @ dq_fixed)[:3]
        self.fe_r_vel_body = (self.J_r_body @ dq_fixed)[:3]
    
    @staticmethod
    def intQuat(quat: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Integrate quaternion with angular velocity
        
        Args:
            quat: Current quaternion [x,y,z,w]
            w: Angular velocity vector [wx, wy, wz]
            
        Returns:
            Integrated quaternion
        """
        Rcur = pin.Quaternion(quat).normalized().toRotationMatrix()
        theta = np.linalg.norm(w)
        
        if theta > 1e-8:
            w_norm = w / theta
            a = np.array([
                [0, -w_norm[2], w_norm[1]],
                [w_norm[2], 0, -w_norm[0]],
                [-w_norm[1], w_norm[0], 0]
            ])
            Rinc = np.eye(3) + a * np.sin(theta) + a @ a * (1 - np.cos(theta))
        else:
            Rinc = np.eye(3)
            
        Rend = Rcur @ Rinc
        return pin.Quaternion(Rend).coeffs()  # Returns [x,y,z,w]
    
    def integrateDIY(self, qI: np.ndarray, dqI: np.ndarray) -> np.ndarray:
        """
        Integrate the configuration with velocity
        
        Args:
            qI: Initial configuration vector
            dqI: Velocity vector
            
        Returns:
            Integrated configuration vector
        """
        qRes = np.zeros(self.model_nv + 1)
        wDes = dqI[3:6]
        
        quatNow = qI[3:7]  # [x,y,z,w]
        quatNew = self.intQuat(quatNow, wDes)
        
        qRes[:3] = qI[:3] + dqI[:3]  # Position
        qRes[3:7] = quatNew          # Orientation
        qRes[7:] = qI[7:] + dqI[6:]  # Joint positions
        
        return qRes
    
    def computeDyn(self):
        """Compute dynamic parameters (M, Minv, C, G, etc.)"""
        # Compute mass matrix
        pin.crba(self.model_biped, self.data_biped, self.q)
        self.data_biped.M = (self.data_biped.M + self.data_biped.M.T) / 2  # Ensure symmetry
        self.dyn_M = self.data_biped.M
        
        # Compute inverse mass matrix
        pin.computeMinverse(self.model_biped, self.data_biped, self.q)
        self.data_biped.Minv = (self.data_biped.Minv + self.data_biped.Minv.T) / 2
        self.dyn_M_inv = self.data_biped.Minv
        
        # Compute Coriolis matrix
        pin.computeCoriolisMatrix(self.model_biped, self.data_biped, self.q, self.dq)
        self.dyn_C = self.data_biped.C
        
        # Compute gravity vector
        pin.computeGeneralizedGravity(self.model_biped, self.data_biped, self.q)
        self.dyn_G = self.data_biped.g
        
        # Compute centroidal momentum matrix
        pin.dccrba(self.model_biped, self.data_biped, self.q, self.dq)
        pin.computeCentroidalMomentum(self.model_biped, self.data_biped, self.q, self.dq)
        self.dyn_Ag = self.data_biped.Ag
        self.dyn_dAg = self.data_biped.dAg
        
        # Compute nonlinear terms
        self.dyn_Non = self.dyn_C @ self.dq + self.dyn_G
        
        # Compute inertia matrix
        pin.ccrba(self.model_biped, self.data_biped, self.q, self.dq)
        self.inertia = self.data_biped.Ig.inertia
        
        # Compute center of mass
        self.CoM_pos = self.data_biped.com[0]
        
        # Transform dynamics to world frame
        Mpj = np.eye(self.model_nv)
        Mpj_inv = np.eye(self.model_nv)
        Mpj[:3, :3] = self.base_rot.T
        Mpj[3:6, 3:6] = self.base_rot.T
        Mpj_inv[:3, :3] = self.base_rot
        Mpj_inv[3:6, 3:6] = self.base_rot
        
        self.dyn_M = Mpj_inv @ self.dyn_M @ Mpj
        self.dyn_M_inv = Mpj_inv @ self.dyn_M_inv @ Mpj
        self.dyn_C = Mpj_inv @ self.dyn_C @ Mpj
        self.dyn_G = Mpj_inv @ self.dyn_G
        self.dyn_Non = Mpj_inv @ self.dyn_Non
    
    def computeInK_Leg(self, Rdes_L: np.ndarray, Pdes_L: np.ndarray, 
                       Rdes_R: np.ndarray, Pdes_R: np.ndarray) -> IkRes:
        """
        Inverse kinematics for leg posture
        
        Args:
            Rdes_L: Desired rotation matrix for left leg
            Pdes_L: Desired position for left leg
            Rdes_R: Desired rotation matrix for right leg
            Pdes_R: Desired position for right leg
            
        Returns:
            IkRes: Result of IK computation
        """
        oMdesL = pin.SE3(Rdes_L, Pdes_L)
        oMdesR = pin.SE3(Rdes_R, Pdes_R)
        
        # Initial guess
        qIk = np.zeros(self.model_biped_fixed.nv)
        qIk[22] = -0.1  # Left knee
        qIk[28] = -0.1  # Right knee
        
        eps = 1e-4
        IT_MAX = 100
        DT = 7e-1
        damp = 5e-3
        
        JL = np.zeros((6, self.model_biped_fixed.nv))
        JR = np.zeros((6, self.model_biped_fixed.nv))
        JCompact = np.zeros((12, self.model_biped_fixed.nv))
        
        success = False
        errL = np.zeros(6)
        errR = np.zeros(6)
        errCompact = np.zeros(12)
        v = np.zeros(self.model_biped_fixed.nv)
        
        J_Idx_l = self.l_ankle_joint_fixed
        J_Idx_r = self.r_ankle_joint_fixed
        
        for itr_count in range(IT_MAX):
            pin.forwardKinematics(self.model_biped_fixed, self.data_biped_fixed, qIk)
            
            iMdL = self.data_biped_fixed.oMi[J_Idx_l].actInv(oMdesL)
            iMdR = self.data_biped_fixed.oMi[J_Idx_r].actInv(oMdesR)
            
            errL = pin.log6(iMdL).vector
            errR = pin.log6(iMdR).vector
            errCompact[:6] = errL
            errCompact[6:] = errR
            
            if np.linalg.norm(errCompact) < eps:
                success = True
                break
            
            # Compute Jacobians
            JL = pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_l)
            JR = pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_r)
            
            # Apply weights (disable waist joints)
            W = np.eye(self.model_biped_fixed.nv)
            JL[:, 16:19] = 0  # Waist joints
            JR[:, 16:19] = 0
            
            # Compute Jacobian logarithms
            JlogL = pin.Jlog6(iMdL.inverse())
            JlogR = pin.Jlog6(iMdR.inverse())
            
            JL = -JlogL @ JL
            JR = -JlogR @ JR
            
            JCompact[:6] = JL
            JCompact[6:] = JR
            
            # Solve IK
            JJt = JCompact @ W @ JCompact.T
            JJt += damp * np.eye(12)
            v = -W @ JCompact.T @ np.linalg.solve(JJt, errCompact)
            
            qIk = pin.integrate(self.model_biped_fixed, qIk, v * DT)
        
        res = IkRes(
            status=0 if success else -1,
            itr=itr_count,
            err=errCompact,
            jointPosRes=qIk
        )
        
        return res
    
    def computeInK_Hand(self, Rdes_L: np.ndarray, Pdes_L: np.ndarray, 
                        Rdes_R: np.ndarray, Pdes_R: np.ndarray) -> IkRes:
        """
        Inverse kinematics for hand posture
        
        Args:
            Rdes_L: Desired rotation matrix for left hand
            Pdes_L: Desired position for left hand
            Rdes_R: Desired rotation matrix for right hand
            Pdes_R: Desired position for right hand
            
        Returns:
            IkRes: Result of IK computation
        """
        oMdesL = pin.SE3(Rdes_L, Pdes_L)
        oMdesR = pin.SE3(Rdes_R, Pdes_R)
        
        # Initial guess (specific arm positions)
        qIk = np.zeros(self.model_biped_fixed.nv)
        qIk[:7] = np.array([
            0.433153883479341, 1.11739345867607, 1.88491913406236,
            0.802378252758275, 1.22726400279662, 0.0249797771339966, -0.0875282610654057
        ])
        qIk[7:14] = np.array([
            -0.433152540054138, -1.11739347975224, -1.88492038240761,
            0.802375980602373, -1.22726323451626, 0.0249795712262396, 0.0875271396314979
        ])
        
        eps = 1e-4
        IT_MAX = 100
        DT = 6e-1
        damp = 1e-2
        
        JL = np.zeros((6, self.model_biped_fixed.nv))
        JR = np.zeros((6, self.model_biped_fixed.nv))
        JCompact = np.zeros((12, self.model_biped_fixed.nv))
        
        success = False
        errL = np.zeros(6)
        errR = np.zeros(6)
        errCompact = np.zeros(12)
        v = np.zeros(self.model_biped_fixed.nv)
        
        J_Idx_l = self.l_hand_joint_fixed
        J_Idx_r = self.r_hand_joint_fixed
        
        for itr_count in range(IT_MAX):
            pin.forwardKinematics(self.model_biped_fixed, self.data_biped_fixed, qIk)
            
            iMdL = self.data_biped_fixed.oMi[J_Idx_l].actInv(oMdesL)
            iMdR = self.data_biped_fixed.oMi[J_Idx_r].actInv(oMdesR)
            
            errL = pin.log6(iMdL).vector
            errR = pin.log6(iMdR).vector
            errCompact[:6] = errL
            errCompact[6:] = errR
            
            if np.linalg.norm(errCompact) < eps:
                success = True
                break
            
            # Compute Jacobians
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_l, JL)
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_r, JR)
            
            # Compute Jacobian logarithms
            JlogL = pin.Jlog6(iMdL.inverse())
            JlogR = pin.Jlog6(iMdR.inverse())
            
            JL = -JlogL @ JL
            JR = -JlogR @ JR
            
            JCompact[:6] = JL
            JCompact[6:] = JR
            
            # Solve IK
            JJt = JCompact @ JCompact.T
            JJt += damp * np.eye(12)
            v = -JCompact.T @ np.linalg.solve(JJt, errCompact)
            
            qIk = pin.integrate(self.model_biped_fixed, qIk, v * DT)
        
        res = IkRes(
            status=0 if success else -1,
            itr=itr_count,
            err=errCompact,
            jointPosRes=qIk
        )
        
        return res
    
    def workspaceConstraint(self, qFT: np.ndarray, tauJointFT: np.ndarray):
        """
        Apply joint position limits
        
        Args:
            qFT: Configuration vector to be constrained
            tauJointFT: Torque vector to be constrained
        """
        for i in range(len(self.motorName)):
            if qFT[i + 7] > self.motorMaxPos[i]:
                qFT[i + 7] = self.motorMaxPos[i]
                self.motorReachLimit[i] = True
                tauJointFT[i] = self.tauJointOld[i]
            elif qFT[i + 7] < self.motorMinPos[i]:
                qFT[i + 7] = self.motorMinPos[i]
                self.motorReachLimit[i] = True
                tauJointFT[i] = self.tauJointOld[i]
            else:
                self.motorReachLimit[i] = False
        
        self.tauJointOld = tauJointFT.copy()