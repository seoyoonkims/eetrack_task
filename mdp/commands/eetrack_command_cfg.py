# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg

from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import (
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
    RAY_CASTER_MARKER_CFG,
    POSITION_GOAL_MARKER_CFG,
)


from omni.isaac.lab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from omni.isaac.lab.utils import configclass

"""
from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand
from .eetrack_command import EETrackCommand

"""

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal
import omni.isaac.lab.sim as sim_utils


@configclass
class EETrackCommandCfg(CommandTermCfg):
    """Configuration for humanoid pose command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_hand_body_name: str = MISSING
    """Name of the left hand body in the asset for which the commands are generated."""

    right_hand_body_name: str = MISSING
    """Name of the right hand body in the asset for which the commands are generated."""

    left_foot_body_name: str = MISSING
    right_foot_body_name: str = MISSING
    torso_body_name: str = MISSING

    make_quat_unique: bool = False

    vis_left: bool = True
    vis_right: bool = True

    mode: Literal["cylinder", "cart", "pelvis-cart"] = MISSING
    frame: Literal["torso", "z-inv", "foot"] = MISSING
    moving_time: float = MISSING

    eetrack_line_length: float = MISSING  # hyperparameter
    eetrack_vel: float = MISSING  # hyperparameter
    eetrack_segment_length: float = MISSING
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """
        # Ranges for the commands in cylindrical coordinates
        r_range: tuple[float, float] = MISSING  # min, max [m]
        theta_range_left: tuple[float, float] = MISSING  # min, max [rad]
        theta_range_right: tuple[float, float] = MISSING  # min, max [rad]
        # Ranges for cartesian coord sampling
        y_left_range: tuple[float, float] = MISSING  # min, max [m]
        y_right_range: tuple[float, float] = MISSING  # min, max [m]
        x_range: tuple[float, float] = MISSING  # min, max [m]

        z_range: tuple[float, float] = MISSING  # min, max [m]

        pelvis_z_range: tuple[float, float] = (0.35, 0.75)
        added_z_range: tuple[float, float] = (0.0, 0.2)

        # Ranges for the angle noise
        noise_roll_range: tuple[float, float] = (-0.0, 0.0)
        noise_pitch_range: tuple[float, float] = (-0.0, 0.0)
        noise_yaw_range: tuple[float, float] = (-0.0, 0.0)

        # Ranges for the z noise
        noise_z_range: tuple[float, float] = (-0.0, 0.0)
        """

        eetrack_start: tuple[float, float, float] = MISSING
        eetrack_end: tuple[float, float, float] = MISSING

    ranges: Ranges = MISSING

    """
    # Configuration parameters for shifts and angle deltas
    angle_noise: float = 0.001 # Added angle noise in degrees
    # angle_noise: float = 30. # Added angle noise in degrees
    spherical_z: float = 1.0 # Height of the spherical coordinate
    """
    eetrack_line_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pelvis_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/pelvis_current",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            )
        },
    )
    goal_pelvis_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/pelvis_goal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            )
        },
    )

    vis_hand_bbox: bool = True
    # hand_bbox_file: str = '/home/user/dex-clutter/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/commands/right_hand.npy'
    hand_bbox = [
        [0.0000, -0.0214, -0.0439],
        [0.0000, -0.0214, 0.0439],
        [0.0000, 0.0728, -0.0439],
        [0.0000, 0.0728, 0.0439],
        [0.1038, -0.0214, -0.0439],
        [0.1038, -0.0214, 0.0439],
        [0.1038, 0.0728, -0.0439],
        [0.1038, 0.0728, 0.0439],
    ]

    bbox_vis_ref_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path=f"/Visuals/Command/bbox")
    bbox_vis_ref_cfg.markers["hit"].radius = 0.01
    # goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    resampling_time_range = None
