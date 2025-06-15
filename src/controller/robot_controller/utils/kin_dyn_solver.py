from pinocchio import pin

class KinDynSolver():
    def __init__(self, cfg):
        urdf_path = cfg['urdf_path']

        self.model_biped = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data_biped = self.model_biped.createData()

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
        pin.computeJointJacobiansTimeVariation(self.model_biped, self.data_biped, self.state.q, self.state.dq)
        pin.updateGlobalPlacements(self.model_biped, self.data_biped)
        self.J_ankle_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_ankle_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hand_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hand_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_base = pin.getJointJacobian(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hip_roll_l = pin.getJointJacobian(self.model_biped, self.data_biped, self.l_hip_roll_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_hip_roll_r = pin.getJointJacobian(self.model_biped, self.data_biped, self.r_hip_roll_joint, pin.LOCAL_WORLD_ALIGNED)
        self.J_waist_yaw = pin.getJointJacobian(self.model_biped, self.data_biped, self.waist_yaw_joint, pin.LOCAL_WORLD_ALIGNED)

        self.dJ_ankle_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_ankle_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_ankle_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_hand_r = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.r_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_hand_l = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.l_hand_joint, pin.LOCAL_WORLD_ALIGNED)
        self.dJ_base = pin.getJointJacobianTimeVariation(self.model_biped, self.data_biped, self.base_joint, pin.LOCAL_WORLD_ALIGNED)

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
        self.Jcom = self.data_biped.Jcom