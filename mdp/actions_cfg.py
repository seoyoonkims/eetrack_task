from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)

from .actions import PassiveIKAction


@configclass
class PassiveIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    class_type: type[ActionTerm] = PassiveIKAction

    # this used to be task_space_actions.DifferentialInverseKinematicsAction.
    # I am overriding it with PassiveIKAction.
    # if I do this, the action dim 7 -> 0. Why?
    # the reason is that PassiveIKAction overrides DifferentialInverseKinematicsAction's action_dim, and returns 0.
    # This is odd. Is this action term even being used?

    command_name: str = MISSING
    control_arm: Literal["left", "right"] = MISSING
