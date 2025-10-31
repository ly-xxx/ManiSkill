from copy import deepcopy

import numpy as np
import sapien.core as sapien
import torch
from mani_skill.utils.structs import Actor

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_agent(asset_download_ids=["xarm6", "fixed_inspire_hand_right"])
class XArm6InspireHandRight(BaseAgent):
    uid = "xarm6_inspire_hand_right"
    urdf_path = f"{ASSET_DIR}/robots/xarm6_inspire_hand_right/xarm6_inspire_hand_right.urdf"
    disable_self_collisions = True  # Disable self-collisions to prevent instability

    # XArm6 keyframes
    arm_keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    1.56280772e-03,
                    -1.10912404e00,
                    -9.71343926e-02,
                    1.52969832e-04,
                    1.20606723e00,
                    1.66234924e-03,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j1=Keyframe(
            qpos=np.array([np.pi / 2, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j2=Keyframe(
            qpos=np.array([0, np.pi / 2, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j3=Keyframe(
            qpos=np.array([0, 0, np.pi / 2, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j4=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j5=Keyframe(
            qpos=np.array([0, 0, 0, 0, np.pi / 2, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j6=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, np.pi / 2]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    # Inspire hand keyframes
    hand_keyframes = dict(
        palm_side=Keyframe(
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.03,  # thumb_CMC_yaw: well below upper limit (0.0) with noise
                    -0.16734816,  # index_MCP
                    -0.16734803,  # middle_MCP
                    -0.16734798,  # ring_MCP
                    -0.167348,     # pinky_MCP
                    -0.03,  # thumb_CMC_pitch: well below upper limit (0.0) with noise
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=[0, 0, 0, 1]),
        ),
        palm_up=Keyframe(
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.16734816,
                    -0.16734803,
                    -0.16734798,
                    -0.167348,
                    -0.08147363,
                    -0.07234851,
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=[0.70710678, 0, 0, 0.70710678]),
        ),
    )

    # Combined keyframes (arm 6 joints + wrist 2 joints + fingers 6 joints = 14 joints total)
    keyframes = {}
    for k, v in arm_keyframes.items():
        # Use palm_side hand configuration for all keyframes
        hand_qpos = hand_keyframes["palm_side"].qpos[0]
        combined_qpos = np.concatenate([v.qpos, hand_qpos])
        keyframes[k] = Keyframe(
            qpos=combined_qpos,
            pose=v.pose,
        )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    arm_stiffness = 1e4  # Higher stiffness like xarm6_robotiq
    arm_damping = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3]  # Higher damping like xarm6_robotiq
    arm_friction = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "link6"
        )

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the inspire hand is grasping an object using multiple finger contacts."""
        # Get contact forces from multiple finger links
        finger_links = [
            "right_hand_thumb_distal",
            "right_hand_index_middle",
            "right_hand_middle_middle",
            "right_hand_ring_middle",
            "right_hand_pinky_middle"
        ]

        total_force = 0
        contact_count = 0

        for link_name in finger_links:
            link = self.robot.links_map[link_name]
            contact_forces = self.scene.get_pairwise_contact_forces(link, object)
            force_magnitude = torch.linalg.norm(contact_forces, axis=1)
            total_force += force_magnitude
            contact_count += (force_magnitude > min_force).float()

        # Require at least 2 fingers to have sufficient contact force
        return contact_count >= 2

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        pd_joint_target_delta_pos = deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # PD ee position
        pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="link6",
            urdf_path=self.urdf_path,
        )
        pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="link6",
            urdf_path=self.urdf_path,
        )
        pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="link6",
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        pd_ee_target_delta_pos = deepcopy(pd_ee_delta_pos)
        pd_ee_target_delta_pos.use_target = True
        pd_ee_target_delta_pose = deepcopy(pd_ee_delta_pose)
        pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        wrist_joint_pos = PDJointPosControllerConfig(
            joint_names=["right_hand_wrist_pitch_joint", "right_hand_wrist_yaw_joint"],
            lower=None,
            upper=None,
            stiffness=1e4,  # Higher stiffness
            damping=[1e3, 1e3],  # Higher damping
            force_limit=100,
            normalize_action=False,
        )
        fingers_joint_pos = PDJointPosControllerConfig(
            joint_names=[
                "right_hand_thumb_CMC_yaw_joint",
                "right_hand_thumb_CMC_pitch_joint",
                "right_hand_index_MCP_joint",
                "right_hand_middle_MCP_joint",
                "right_hand_ring_MCP_joint",
                "right_hand_pinky_MCP_joint",
            ],
            lower=None,
            upper=None,
            stiffness=1e4,  # Higher stiffness
            damping=[1e3, 1e3, 1e3, 1e3, 1e3, 1e3],  # Higher damping
            force_limit=20,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            joint_names=[
                "right_hand_thumb_MCP_joint",
                "right_hand_thumb_IP_joint",
                "right_hand_index_PIP_joint",
                "right_hand_middle_PIP_joint",
                "right_hand_ring_PIP_joint",
                "right_hand_pinky_PIP_joint",
            ],
            damping=0.001,
            force_limit=20,
        )

        wrist_joint_delta_pos = deepcopy(wrist_joint_pos)
        wrist_joint_delta_pos.use_delta = True
        wrist_joint_delta_pos.normalize_action = True
        wrist_joint_delta_pos.lower = -0.1
        wrist_joint_delta_pos.upper = 0.1
        wrist_joint_delta_pos.damping = [2e3, 2e3]  # Higher damping for delta control

        fingers_joint_delta_pos = deepcopy(fingers_joint_pos)
        fingers_joint_delta_pos.use_delta = True
        fingers_joint_delta_pos.normalize_action = True
        fingers_joint_delta_pos.lower = -0.1
        fingers_joint_delta_pos.upper = 0.1
        fingers_joint_delta_pos.damping = [2e3, 2e3, 2e3, 2e3, 2e3, 2e3]  # Higher damping for delta control

        # Combined controller for full robot control (arm + wrist + fingers)
        # Joint names in the correct order matching qpos indices
        controlled_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",  # arm (6) -> qpos 0-5
            "right_hand_wrist_pitch_joint", "right_hand_wrist_yaw_joint",  # wrist (2) -> qpos 6-7
            "right_hand_thumb_CMC_yaw_joint",  # thumb yaw (1) -> qpos 8
            "right_hand_index_MCP_joint", "right_hand_middle_MCP_joint",  # index & middle (2) -> qpos 9-10
            "right_hand_ring_MCP_joint", "right_hand_pinky_MCP_joint",  # ring & pinky (2) -> qpos 11-12
            "right_hand_thumb_CMC_pitch_joint"  # thumb pitch (1) -> qpos 13
        ]  # Total: 14 joints

        combined_joint_pos = PDJointPosControllerConfig(
            joint_names=controlled_joint_names,
            lower=None,  # Use joint limits
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping + [2e3, 2e3] + [5e3, 5e3, 5e3, 5e3, 5e3, 5e3],  # Much higher damping for fingers
            friction=self.arm_friction + [0.1, 0.1] + [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # arm(6) + wrist(2) + fingers(6)
            force_limit=self.arm_force_limit,
            use_delta=False,  # Use absolute position control
            normalize_action=False,  # Don't normalize actions
        )

        controller_configs = dict(
            # Combined controller (set as first for default control mode)
            pd_joint_pos=combined_joint_pos,
            # Arm controllers
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_ee_delta_pos=pd_ee_delta_pos,
            pd_ee_delta_pose=pd_ee_delta_pose,
            pd_ee_pose=pd_ee_pose,
            pd_ee_target_delta_pos=pd_ee_target_delta_pos,
            pd_ee_target_delta_pose=pd_ee_target_delta_pose,
            pd_joint_vel=pd_joint_vel,
            pd_joint_pos_vel=pd_joint_pos_vel,
            pd_joint_delta_pos_vel=pd_joint_delta_pos_vel,
            # Hand controllers
            pd_joint_pos_wrist=wrist_joint_pos,
            pd_joint_pos_fingers=fingers_joint_pos,
            pd_joint_delta_pos_wrist=wrist_joint_delta_pos,
            pd_joint_delta_pos_fingers=fingers_joint_delta_pos,
            passive=passive,
        )

        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)


@register_agent()
class XArm6InspireHandRightWristCamera(XArm6InspireHandRight):
    uid = "xarm6_inspire_hand_right_wristcam"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
