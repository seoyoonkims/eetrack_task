import math
import numpy as np
import torch as th
from dataclasses import MISSING
from typing import TYPE_CHECKING, List, Literal, Union
from collections.abc import Sequence

import omni.isaac.lab_tasks.manager_based.eetrack_task.mdp as mdp


import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    Articulation,
    RigidObject,
    RigidObjectCfg
)
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, ManagerBasedEnv
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CommandTermCfg, CommandTerm
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from .g1_spawn_info import G1_29_FIXED_HAND_CFG

import os
spawn_obstacles = os.environ.get('SPAWN_OBSTACLES', 'False').lower() in ('true', '1', 't')

# import pkm_utils.src.pkm
# pkm.util.math_util import xyzw2wxyz

# import domi.env.help.zmp as zmp
# from icecream import ic


def xyzw2wxyz(q_xyzw: th.Tensor, dim: int = -1):
    return th.roll(q_xyzw, 1, dims=dim)


def foot_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_foot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names="left_ankle_roll_link"
    ),
    right_foot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names="right_ankle_roll_link"
    ),
) -> th.Tensor:
    """The position of the object in the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_ids = asset.find_bodies(left_foot_cfg.body_names)[0][0]
    right_foot_ids = asset.find_bodies(right_foot_cfg.body_names)[0][0]
    foot_pos_left_b, foot_quat_left_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_foot_ids, :3],
        asset.data.body_state_w[:, left_foot_ids, 3:7],
    )
    foot_pos_right_b, foot_quat_right_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_foot_ids, :3],
        asset.data.body_state_w[:, right_foot_ids, 3:7],
    )
    foot_axa_left_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(foot_quat_left_b)
    )
    foot_axa_right_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(foot_quat_right_b)
    )

    foot_pose_b = th.cat(
        (foot_pos_left_b, foot_pos_right_b, foot_axa_left_b, foot_axa_right_b), dim=-1
    )

    return foot_pose_b


def hand_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names="left_hand_palm_link"
    ),
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names="right_hand_palm_link"
    ),
) -> th.Tensor:
    """The position of the object in the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_hand_ids = asset.find_bodies(left_hand_cfg.body_names)[0][0]
    right_hand_ids = asset.find_bodies(right_hand_cfg.body_names)[0][0]
    hand_pos_left_b, hand_quat_left_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_hand_ids, :3],
        asset.data.body_state_w[:, left_hand_ids, 3:7],
    )
    hand_pos_right_b, hand_quat_right_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_hand_ids, :3],
        asset.data.body_state_w[:, right_hand_ids, 3:7],
    )
    hand_axa_left_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(hand_quat_left_b)
    )
    hand_axa_right_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(hand_quat_right_b)
    )

    hand_pose_b = th.cat(
        (hand_pos_left_b, hand_pos_right_b, hand_axa_left_b, hand_axa_right_b), dim=-1
    )

    return hand_pose_b


def pelvis_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[..., 2:3]


def pelvis_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sigma: float = 50.0,
) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = th.abs(command.pelvis_command - asset.data.root_state_w[..., 2:3])[
        ..., 0
    ]

    rew = th.exp(-sigma * th.square(pos_error))

    # Zero-mask reward for the moving command envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def compute_com(asset: Articulation, device: str, body_ids=None) -> th.Tensor:
    """
    Returns the COM in the world frame given the articulation
    assets.
    """
    link_com_pose_b = asset.root_physx_view.get_coms().clone().to(device)
    link_pose = asset.root_physx_view.get_link_transforms().clone()
    link_mass = asset.data.default_mass.clone().to(device)

    if body_ids is not None:
        link_com_pose_b = link_com_pose_b[:, body_ids, :]
        link_pose = link_pose[:, body_ids, :]
        link_mass = link_mass[:, body_ids]

    link_com_pos_w, link_com_quat_w = math_utils.combine_frame_transforms(
        link_pose[..., :3],
        xyzw2wxyz(link_pose[..., 3:7]),
        link_com_pose_b[..., :3],
        xyzw2wxyz(link_com_pose_b[..., 3:7]),
    )
    com_pos_w = (link_com_pos_w * link_mass.unsqueeze(-1)).sum(dim=1) / link_mass.sum(
        dim=-1
    ).unsqueeze(-1)

    return com_pos_w


def relative_arm_com(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    arm_com_w = compute_com(asset, env.device, asset_cfg.body_ids)
    arm_com_b = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w, arm_com_w - asset.data.root_pos_w
    )

    return arm_com_b


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # obstacles
    if spawn_obstacles:
        obstacle1 = mdp.ObstacleCfg(
            prim_path="{ENV_REGEX_NS}/obstacle1",
            size=(0.2, 0.01, 0.5)
        )
        obstacle2 = mdp.ObstacleCfg(
            prim_path="{ENV_REGEX_NS}/obstacle2",
            size=(0.2, 0.01, 0.5)
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "right_shoulder_.*",
            "right_elbow_joint",
            "right_wrist_.*",
            "waist_.*",
        ],
        scale=0.5,
        use_default_offset=True,
    )
    # NOTE if I use just residual policy, you don't reach the goal pose, and if I just use IK controller,
    #     you get into a joint lock
    left_arm = mdp.PassiveIKActionCfg(
        asset_name="robot",
        command_name="hands_pose",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_.*",
        ],
        body_name="left_hand_palm_link",
        control_arm="left",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
            use_weighted_jacobian=False,
            use_max_clipping=True,
            max_delta_pos=0.5,
            weight_pos=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            weight_ori=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        scale=1.0,
        compensate_gravity=True,
    )
    left_arm_res = mdp.ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_.*",
        ],
        scale=0.3,
        use_clipping=True,
        # clip_range=(-0.5, 0.5)
        clip_range=(-0.2, 0.2),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        foot_pose = ObsTerm(func=foot_pose_in_robot_root_frame)
        hand_pose = ObsTerm(func=hand_pose_in_robot_root_frame)

        # TODO Is projected CoM really used?
        projected_com = ObsTerm(
            func=mdp.zmp_rwd_computation_helper.projected_coms,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        projected_zmp = ObsTerm(
            func=mdp.zmp_rwd_computation_helper.projected_zmps,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "left_foot_sensor_cfg": SceneEntityCfg("contact_left_foot"),
                "right_foot_sensor_cfg": SceneEntityCfg("contact_right_foot"),
            },
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        hands_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "hands_pose"}
        )
        right_arm_com = ObsTerm(
            func=relative_arm_com,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "right_shoulder_pitch_link",
                        "right_shoulder_roll_link",
                        "right_shoulder_yaw_link",
                        "right_elbow_link",
                        "right_wrist_.*",
                    ],
                ),
            },
            scale=1.0,
        )
        left_arm_com = ObsTerm(
            func=relative_arm_com,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_shoulder_pitch_link",
                        "left_shoulder_roll_link",
                        "left_shoulder_yaw_link",
                        "left_elbow_link",
                        "left_wrist_.*",
                    ],
                ),
            },
            scale=1.0,
        )
        pelvis_height = ObsTerm(
            func=pelvis_height,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # for some reason, without this, we can't initalize the robot to the init_config written in CFG file.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            # "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_eetrack_sg_index = EventTerm(func=mdp.reset_eetrack_sg, mode="reset")

    """
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        # mode="reset",
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (1000.0, 1000.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # interval push
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # elbow joint limit
    robot_joint_limits_elbow = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names="right_elbow_joint"),
            "lower_limit_distribution_params": (-1., -1.),
            "upper_limit_distribution_params": (1.5, 1.5),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_limits_shoulder = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names="right_shoulder_yaw_joint"),
            "lower_limit_distribution_params": (-0.5, -0.5),
            "upper_limit_distribution_params": (2.6, 2.6),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    """


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    eetrack_subgoal_maxxed = DoneTerm(func=mdp.eetrack_subgoal_index_maxxed)

    """
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                # body_names=[".*_hip_.*", "head_link"]), 
                body_names=["head_link"]), 
            "threshold": 1.0},
    )
    """

    torso_height = DoneTerm(
        func=mdp.eetrack_root_height_below_minimum, params={"minimum_height": 0.25}
    )

    bad_ori = DoneTerm(
        func=mdp.eetrack_bad_ori, params={"limit_euler_angle": [0.9, 1.0]}
    )


@configclass
class EETrackCommandsCfg:
    # note about the resampling
    # Currently, we say we have 4 seconds to achieve the next pose
    # The environment step dt (not the physics engine dt) is set to 0.02s.
    # If this is not satisfied, we move on to the new goal.
    # This needs to be changed;
    hands_pose = mdp.EETrackCommandCfg(
        class_type=mdp.EETrackCommand,
        asset_name="robot",
        # resampling_time_range=(0.1, 0.1),  #TODO the com term uses this in compute function; inherit it and redefine it. So that it works in accordance with the description below.
        moving_time=3.0,  # what's this moving time?
        left_hand_body_name="left_hand_palm_link",
        right_hand_body_name="right_hand_palm_link",
        left_foot_body_name="left_ankle_roll_link",
        right_foot_body_name="right_ankle_roll_link",
        torso_body_name="torso_link",
        mode="pelvis-cart",
        frame="foot",
        debug_vis=True,
        eetrack_line_length=0.3,  # 1m
        eetrack_vel=0.1,  # 0.1m per second
        eetrack_segment_length=0.01,
        ranges=mdp.EETrackCommandCfg.Ranges(
            # not used at the moment.
            eetrack_start=(-1.0, 0.3, 0.75),
            eetrack_end=(0.0, 0.3, 0.75),
        ),
    )


@configclass
class G1EETrackEnvCfg(ManagerBasedRLEnvCfg):
    rewards: mdp.G1EETrackRewards = mdp.G1EETrackRewards()
    commands: EETrackCommandsCfg = EETrackCommandsCfg()
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # self.episode_length_s = 10.
        # self.episode_length_s = 3.
        # self.episode_length_s = 8.
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Scene
        self.scene.robot = G1_29_FIXED_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None

        self.events.reset_base.params = {
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # TODO I've fixed this, but it seems like we are having trouble initializing the joint positions?
            # I want that to be deterministic too
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Contact sensors
        self.scene.contact_left_foot = mdp.ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096,
        )
        self.scene.contact_right_foot = mdp.ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096,
        )
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


class G1EETrackEnvCfgPlay(G1EETrackEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
