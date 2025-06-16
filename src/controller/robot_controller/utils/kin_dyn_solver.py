import math
from pinocchio import pin
import numpy as np
from scipy.spatial.transform import Rotation
class IkRes:
    def __init__(self):
        self.err = None
        self.itr = 0
        self.status = 0
        self.jointPosRes = None
class KinDynSolver():
    def __init__(self, cfg):
        urdf_path = cfg['urdf_path']

        self.model_biped = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data_biped = self.model_biped.createData()
        self.model_nv = self.data_biped.nv

        self.r_ankle_joint = self.model_biped.getJointId("J_ankle_r_roll")
        self.l_ankle_joint = self.model_biped.getJointId("J_ankle_l_roll")
        self.r_hand_joint = self.model_biped.getJointId("J_arm_r_07")
        self.l_hand_joint = self.model_biped.getJointId("J_arm_l_07")
        self.r_hip_yaw_joint = self.model_biped.getJointId("J_hip_r_yaw")
        self.l_hip_yaw_joint = self.model_biped.getJointId("J_hip_l_yaw")
        self.r_hip_roll_joint = self.model_biped.getJointId("J_hip_r_roll")
        self.l_hip_roll_joint = self.model_biped.getJointId("J_hip_l_roll")
        self.waist_yaw_joint = self.model_biped.getJointId("J_waist_yaw")
        self.base_joint = self.model_biped.getJointId("root_joint")

        self.model_biped_fixed = pin.buildModelFromUrdf(urdf_path)
        self.data_biped_fixed = self.model_biped_fixed.createData()
        self.r_ankle_joint_fixed = self.model_biped_fixed.getJointId("J_ankle_r_roll")
        self.l_ankle_joint_fixed = self.model_biped_fixed.getJointId("J_ankle_l_roll")
        self.r_hand_joint_fixed = self.model_biped_fixed.getJointId("J_arm_r_07")
        self.l_hand_joint_fixed = self.model_biped_fixed.getJointId("J_arm_l_07")
        self.r_hip_yaw_joint_fixed = self.model_biped_fixed.getJointId("J_hip_r_yaw")
        self.l_hip_yaw_joint_fixed = self.model_biped_fixed.getJointId("J_hip_l_yaw")

    def update(self, state):
        self.state = state
        self.dq = state.dq
        self.dq[0:3] = state.R_B2W.T @ self.dq[0:3].T
        self.dq[3:6] = state.R_B2W.T @ self.dq[3:6].T

    def compute_Jacobians(self):
        pin.forwardKinematics(self.model_biped, self.data_biped, self.state.q)
        pin.jacobianCenterOfMass(self.model_biped, self.data_biped, self.state.q, True)
        # pin.computeJointJacobians(self.model_biped, self.data_biped, self.state.q)
        pin.computeJointJacobiansTimeVariation(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        pin.updateGlobalPlacements(self.model_biped, self.data_biped)

        # transform into world frame
        R_B2W_compact = np.identity(self.model_nv)
        R_B2W_compact[0:3,0:3] = self.state.R_B2W
        R_B2W_compact[3:6,3:6] = self.state.R_B2W

        self.J_ankle_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.J_ankle_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.J_hand_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.J_hand_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.J_base = pin.getJointJacobian(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact


        self.dJ_ankle_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.dJ_ankle_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.dJ_hand_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.dJ_hand_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.dJ_base = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact

        self.Jcom = self.data_biped.Jcom @ R_B2W_compact
        self.J_waist_yaw = pin.getJointJacobian(self.model_biped, self.data_biped, self.waist_yaw_joint, pin.LOCAL_WORLD_ALIGNED) @ R_B2W_compact
        self.J_hip_roll_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_hip_roll_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hip_roll_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_hip_roll_joint, pin.LOCAL_WORLD_ALIGNED)

        self.fe_l_pos = self.data_biped.oMi[self.l_ankle_joint].translation
        self.fe_l_rot = self.data_biped.oMi[self.l_ankle_joint].rotation
        self.hip_l_pos = self.data_biped.oMi[self.l_hip_yaw_joint].translation
        self.fe_r_pos = self.data_biped.oMi[self.r_ankle_joint].translation
        self.fe_r_rot = self.data_biped.oMi[self.r_ankle_joint].rotation
        self.hip_r_pos = self.data_biped.oMi[self.r_hip_yaw_joint].translation
        self.base_pos = self.data_biped.oMi[self.base_joint].translation
        self.base_rot = self.data_biped.oMi[self.base_joint].rotation
        self.hd_l_pos = self.data_biped.oMi[self.l_hand_joint].translation
        self.hd_l_rot = self.data_biped.oMi[self.l_hand_joint].rotation
        self.hd_r_pos = self.data_biped.oMi[self.r_hand_joint].translation
        self.hd_r_rot = self.data_biped.oMi[self.r_hand_joint].rotation
        self.hip_link_pos = self.data_biped.oMi[self.waist_yaw_joint].translation
        self.hip_link_rot = self.data_biped.oMi[self.waist_yaw_joint].rotation
        
        # fixed part
        q_fixed = self.state.q[7:7 + self.model_biped_fixed.nv,:]
        dq_fixed = self.state.dq[6:6 + self.model_biped_fixed.nv,:]

        pin.forwardKinematics(self.model_biped_fixed, self.data_biped_fixed, q_fixed)
        pin.computeJointJacobians(self.model_biped_fixed, self.data_biped_fixed, q_fixed)
        pin.updateGlobalPlacements(self.model_biped_fixed, self.data_biped_fixed)
        self.J_ankle_l_fixed = pin.getJointJacobian(self.model_biped_fixed, self.data_biped_fixed, self.l_ankle_joint_fixed, pin.LOCAL_WORLD_ALIGNED)
        self.J_ankle_r_fixed = pin.getJointJacobian(self.model_biped_fixed, self.data_biped_fixed, self.r_ankle_joint_fixed, pin.LOCAL_WORLD_ALIGNED)
        self.fe_l_pos_fixed = self.data_biped_fixed.oMi[self.l_ankle_joint_fixed].translation
        self.fe_r_pos_fixed = self.data_biped_fixed.oMi[self.r_ankle_joint_fixed].translation
        self.fe_l_rot_fixed = self.data_biped_fixed.oMi[self.l_ankle_joint_fixed].rotation
        self.fe_r_rot_fixed = self.data_biped_fixed.oMi[self.r_ankle_joint_fixed].rotation
        self.hip_l_pos_fixed = self.data_biped_fixed.oMi[self.l_hip_joint_fixed].translation
        self.hip_r_pos_fixed = self.data_biped_fixed.oMi[self.r_hip_joint_fixed].translation
        self.hd_l_pos_fixed = self.data_biped_fixed.oMi[self.l_hand_joint_fixed].translation
        self.hd_l_rot_fixed = self.data_biped_fixed.oMi[self.l_hand_joint_fixed].rotation
        self.hd_r_pos_fixed = self.data_biped_fixed.oMi[self.r_hand_joint_fixed].translation
        self.hd_r_rot_fixed = self.data_biped_fixed.oMi[self.r_hand_joint_fixed].rotation
        self.fe_l_vel_fixed = (self.J_ankle_l_fixed @ dq_fixed)[0:3]
        self.fe_r_vel_fixed = (self.J_ankle_r_fixed @ dq_fixed)[0:3]

    def compute_dynamic_matrix(self):
        # Calculate M
        pin.crba(self.model_biped, self.data_biped, self.state.q)
        # Pinocchio only gives half of the M, needs to restore it here
        self.data_biped.M[np.tril_indices(self.model_biped.nv, -1)] = self.data_biped.M.T[np.tril_indices(self.model_biped.nv, -1)]
        dyn_M = self.data_biped.M.copy()

        # Calculate Minv
        pin.computeMinverse(self.model_biped, self.data_biped, self.state.q)
        self.data_biped.Minv[np.tril_indices(self.model_biped.nv, -1)] = self.data_biped.Minv.T[np.tril_indices(self.model_biped.nv, -1)]
        dyn_M_inv = self.data_biped.Minv.copy()

        # Calculate C
        pin.computeCoriolisMatrix(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        dyn_C = self.data_biped.C.copy()

        # Calculate G
        pin.computeGeneralizedGravity(self.model_biped, self.data_biped, self.state.q)
        dyn_G = self.data_biped.g.copy()

        # Calculate Ag (Centroidal Momentum Matrix)
        pin.dccrba(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        pin.computeCentroidalMomentum(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        dyn_Ag = self.data_biped.Ag.copy()
        dyn_dAg = self.data_biped.dAg.copy()

        # Calculate nonlinear term
        dyn_Non = dyn_C @ self.state.dq + dyn_G

        # Calculate I
        pin.ccrba(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        inertia = self.data_biped.Ig.inertia().matrix()

        # Calculate CoM
        CoM_pos = self.data_biped.com[0]

        # Transform into world frame
        Mpj = np.eye(self.model_biped.nv)
        Mpj_inv = np.eye(self.model_biped.nv)
        Mpj[:3, :3] = self.ase_rot.T
        Mpj[3:6, 3:6] = self.base_rot.T
        Mpj_inv[:3, :3] = self.base_rot
        Mpj_inv[3:6, 3:6] = self.base_rot

        dyn_M = Mpj_inv @ dyn_M @ Mpj
        dyn_M_inv = Mpj_inv @ dyn_M_inv @ Mpj
        dyn_C = Mpj_inv @ dyn_C @ Mpj
        dyn_G = Mpj_inv @ dyn_G
        dyn_Non = Mpj_inv @ dyn_Non

    def computeInK_Leg(self, R_l_leg, pos_l_leg, R_r_leg, pos_r_leg):
        """Inverse kinematics for leg posture"""
        oMdesL = pin.SE3(R_l_leg, pos_l_leg)
        oMdesR = pin.SE3(R_r_leg, pos_r_leg)
        
        # Initial guess (arm-l: 0-6, arm-r: 7-13, head: 14,15 waist: 16-18, leg-l: 19-24, leg-r: 25-30)
        qIk = np.zeros(self.model_biped_fixed.nv)
        qIk[22] = -0.1
        qIk[28] = -0.1

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
        
        for itr_count in range(IT_MAX + 1):
            pin.forwardKinematics(self.model_biped_fixed, self.data_biped_fixed, qIk)
            iMdL = self.data_biped_fixed.oMi[J_Idx_l].actInv(oMdesL)
            iMdR = self.data_biped_fixed.oMi[J_Idx_r].actInv(oMdesR)
            
            errL = pin.log6(iMdL).vector  # in joint frame
            errR = pin.log6(iMdR).vector  # in joint frame
            
            errCompact[:6] = errL
            errCompact[6:] = errR
            
            if np.linalg.norm(errCompact) < eps:
                success = True
                break
                
            if itr_count >= IT_MAX:
                success = False
                break

            # Compute Jacobians
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_l, JL)
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_r, JR)
            
            # Weight matrix (optional)
            W = np.eye(self.model_biped_fixed.nv)
            # W[16:19, 16:19] = 0.001  # Smaller weights for waist joints
            
            # Zero out waist joints
            JL[:, 16:19] = 0
            JR[:, 16:19] = 0
            
            # Compute Jlog
            JlogL = pin.Jlog6(iMdL.inverse())
            JlogR = pin.Jlog6(iMdR.inverse())
            
            JL = -JlogL @ JL
            JR = -JlogR @ JR
            
            JCompact[:6, :] = JL
            JCompact[6:, :] = JR
            
            # Damped least squares
            JJt = JCompact @ W @ JCompact.T
            JJt += damp * np.eye(12)
            
            v = -W @ JCompact.T @ np.linalg.solve(JJt, errCompact)
            qIk = pin.integrate(self.model_biped_fixed, qIk, v * DT)
            
        res = IkRes()
        res.err = errCompact
        res.itr = itr_count
        res.status = 0 if success else -1
        res.jointPosRes = qIk
        
        return res

    def computeInK_Hand(self, R_l_hand, pos_l_hand, R_r_hand, pos_r_hand):
        """Inverse kinematics for hand posture"""
        oMdesL = pin.SE3(R_l_hand, pos_l_hand)
        oMdesR = pin.SE3(R_r_hand, pos_r_hand)
        
        # Initial guess (arm-l: 0-6, arm-r: 7-13, head: 14,15 waist: 16-18, leg-l: 19-24, leg-r: 25-30)
        qIk = np.zeros(self.model_biped_fixed.nv)
        qIk[0:7] = [0.433153883479341, 1.11739345867607, 1.88491913406236,
                    0.802378252758275, 1.22726400279662, 0.0249797771339966, -0.0875282610654057]
        qIk[7:14] = [-0.433152540054138, -1.11739347975224, -1.88492038240761,
                     0.802375980602373, -1.22726323451626, 0.0249795712262396, 0.0875271396314979]

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
        
        for itr_count in range(IT_MAX + 1):
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
                
            if itr_count >= IT_MAX:
                success = False
                break

            # Compute Jacobians
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_l, JL)
            pin.computeJointJacobian(self.model_biped_fixed, self.data_biped_fixed, qIk, J_Idx_r, JR)
            
            # Compute Jlog
            JlogL = pin.Jlog6(iMdL.inverse())
            JlogR = pin.Jlog6(iMdR.inverse())
            
            JL = -JlogL @ JL
            JR = -JlogR @ JR
            
            JCompact[:6, :] = JL
            JCompact[6:, :] = JR
            
            # Damped least squares
            JJt = JCompact @ JCompact.T
            JJt += damp * np.eye(12)
            
            v = -JCompact.T @ np.linalg.solve(JJt, errCompact)
            qIk = pin.integrate(self.model_biped_fixed, qIk, v * DT)
            
        res = IkRes()
        res.err = errCompact
        res.itr = itr_count
        res.status = 0 if success else -1
        res.jointPosRes = qIk
        
        return res
    
    def update_q(self, q, dq):
        omega = np.array[dq[3], dq[4], dq[5]]
        quat = np.array[q[3], q[4], q[5], q[6]]
        quat_next = self.update_quat(quat, omega)

        q_next = q
        q_next[0] += dq[0]
        q_next[1] += dq[1]
        q_next[2] += dq[2]
        q_next[3] = quat_next[0]
        q_next[4] = quat_next[1]
        q_next[5] = quat_next[2]
        q_next[6] = quat_next[3]
        for i in range(self.model_nv - 6):
            q_next[7+i] += dq[6 + i]
        return q_next
    
    # quat：四元数 w：角速度
    def update_quat(self, quat, w):
        R = Rotation.from_quat(quat).as_matrix()
        R_increase = np.identity(3)
        theta = np.linalg.norm(w)
        if(theta > 1e-8):
            w_norm = w / theta
            a = np.array[
                [0, -w_norm[2], w_norm[1]],
                [w_norm[0], 0, -w_norm[0]],
                [-w_norm[1], w_norm[0], 0]
            ]
            R_increase = np.identity(3) + a * math.sin(theta) + a @ a * (1 - math.cos(theta))
        
        return R @ R_increase
    
