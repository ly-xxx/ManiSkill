from copy import deepcopy

import numpy as np
import sapien.core as sapien
import torch

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["xarm6"])
class XArm6Gripper(BaseAgent):
    uid = "xarm6_gripper"
    urdf_path = f"{ASSET_DIR}/robots/xarm6/xarm6/xarm6_with_gripper.urdf"

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    1.56280772e-03,
                    -1.10912404e00,
                    -9.71343926e-02,
                    1.52969832e-04,
                    1.20606723e00,
                    1.66234924e-03,
                    0,
                    0,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j1=Keyframe(
            qpos=np.array([np.pi / 2, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j2=Keyframe(
            qpos=np.array([0, np.pi / 2, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j3=Keyframe(
            qpos=np.array([0, 0, np.pi / 2, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j4=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j5=Keyframe(
            qpos=np.array([0, 0, 0, 0, np.pi / 2, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j6=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, np.pi / 2, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    arm_stiffness = 1e4
    arm_damping = 1e3
    arm_friction = 0.1
    arm_force_limit = 100

    gripper_stiffness = 1e5
    gripper_damping = 2000
    gripper_force_limit = 0.1
    gripper_friction = 1
    ee_link_name = "link6"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.01,
            upper=0.01,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.01,
            pos_upper=0.01,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.01,
            pos_upper=0.01,
            rot_lower=-0.01,
            rot_upper=0.01,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.01,
            0.01,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #

        # For the xarm_gripper, we only need one active joint for control
        # The other finger is driven by a mimic joint in the URDF
        finger_joint_name = ["drive_joint"]  # The main drive joint

        # For a simple parallel gripper, we use a single joint controller
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            finger_joint_name,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )

        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=finger_joint_name,
            lower=-0.15,
            upper=0.15,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_delta_pos,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_delta_pos,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)

    def _after_loading_articulation(self):
        # For the xarm_gripper, we don't need to create additional drives 
        # as the mimic joint is already defined in the URDF
        
        # Disable impossible collisions
        gripper_links = [
            "link_base",
            "left_finger",
            "right_finger",
            "link6",  # not gripper link but is adjacent to the gripper part
        ]
        for link_name in gripper_links:
            if link_name in self.robot.links_map:
                link = self.robot.links_map[link_name]
                link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        
        # 添加调试信息和错误处理
        all_links = [link.name for link in self.robot.get_links()]
        print(f"Available links: {all_links}")
        if self.tcp is None:
            print(f"Warning: Could not find link with name {self.ee_link_name}")
            # 尝试使用link6作为备选
            self.tcp = sapien_utils.get_obj_by_name(
                self.robot.get_links(), "link6"
            )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class XArm6GripperWristCamera(XArm6Gripper):
    uid = "xarm6_gripper_wristcam"

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