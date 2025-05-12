# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.lab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
import torch as th

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_eetrack_sg(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    eetrack_command_term = env.command_manager._terms["hands_pose"]
    eetrack_command_term.current_eetrack_sg_index[env_ids] = 0.0
    eetrack_command_term.time_left[env_ids] = 0.0
