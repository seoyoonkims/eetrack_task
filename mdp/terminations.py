# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def eetrack_subgoal_index_maxxed(env: ManagerBasedRLEnv) -> th.Tensor:
    eetrack_command_term = env.command_manager._terms["hands_pose"]
    reached_the_end = (
        eetrack_command_term.current_eetrack_sg_index
        == eetrack_command_term.number_of_subgoals
    )
    return reached_the_end


def eetrack_root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    is_below = asset.data.root_pos_w[:, 2] < minimum_height
    # this is a bit complicated... I need to index into the environments that has terminated
    # eetrack_command_term = env.command_manager._terms['hands_pose']
    # if th.any(is_below):
    #    print(f"is_below {is_below}")
    return is_below


def eetrack_bad_ori(
    env: ManagerBasedRLEnv,
    limit_euler_angle: List[float] = [0.5, 1.5],
    torso_body_name="torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Terminate when the asset's orientation is out of predefined range

    Args:
        limit_euler_angle: euler angle threshold [roll, pitch]. Episode
            will be terminated if the abs of the root euler angle will
            exceed this threshold
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    torso_idx = asset.find_bodies(torso_body_name)[0][0]
    euler = math_utils.wrap_to_pi(
        th.stack(
            # math_utils.euler_xyz_from_quat(asset.data.root_quat_w), dim=-1))
            math_utils.euler_xyz_from_quat(asset.data.body_state_w[:, torso_idx, 3:7]),
            dim=-1,
        )
    )
    out_of_limit = th.logical_or(
        th.abs(euler[..., 0]) > limit_euler_angle[0],
        th.abs(euler[..., 1]) > limit_euler_angle[1],
    )

    """
    if out_of_limit:
        eetrack_command_term = env.command_manager._terms['hands_pose']
        eetrack_command_term.eetrack_line_has_not_been_defined = True
    """
    # if th.any(out_of_limit):
    #    print(f"bad_ori {out_of_limit}")

    return out_of_limit
