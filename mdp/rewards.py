# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs import mdp
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers.manager_base import ManagerTermBase

from . import zmp_rwd_computation_helper as zmp

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

import os
spawn_obstacles = os.environ.get('SPAWN_OBSTACLES', 'False').lower() in ('true', '1', 't')

"""
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    #Penalize joint torques applied on the articulation using L2 squared kernel

    #NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return th.sum(th.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
"""


def action_limits(
    env, action_name: str, command_name: str, limits: List[float] = [-0.5, 0.5]
):
    action = env.action_manager.get_term(action_name)
    # compute out of limits constraints
    out_of_limits = -(action.raw_actions - limits[0]).clip(max=0.0)
    out_of_limits += (action.raw_actions - limits[1]).clip(min=0.0)

    rew = th.sum(out_of_limits, dim=1)
    # command: mdp.EETrackCommand \
    #     = env.command_manager.get_term(command_name)
    # # Zero-mask reward for the moving command envs
    # moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    # rew[moving_env_ids] = 0.

    return rew


def track_lin_vel_xy_yaw_frame_exp_v2(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of base linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = th.sum(
        th.square(command.vel_command_b[:, :2] - vel_yaw[:, :2]), dim=1
    )
    rew = th.exp(-lin_vel_error / std**2)

    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def track_ang_vel_z_world_exp_v2(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of base angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    ang_vel_error = th.square(
        command.vel_command_b[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )

    rew = th.exp(-ang_vel_error / std**2)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def air_time_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    close_threshold: float = 0.2,
    threshold: float = 0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=".*_ankle_roll_link"
    ),
) -> th.Tensor:

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = th.where(in_contact, contact_time, air_time)
    single_stance = th.sum(in_contact.int(), dim=1) == 1
    rew = th.min(th.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    rew = th.clamp(rew, max=threshold)

    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    if "Air_time" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Air_time"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Air_time"] += rew

    return rew


def air_time_v4(
    env: ManagerBasedRLEnv,
    command_name: str,
    close_threshold: float = 0.2,
    threshold: float = 0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=".*_ankle_roll_link"
    ),
) -> th.Tensor:

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = th.where(in_contact, contact_time, air_time)
    single_stance = th.sum(in_contact.int(), dim=1) == 1
    rew = th.min(th.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    rew = th.where(rew < threshold, rew, 2 * threshold - rew)
    rew = th.clamp(rew, min=0.0)

    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def zmp_supp_dist_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> th.Tensor:
    # """
    # Computes the distance/margin between the support polygon and the zmp
    # """
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]

    signed_zmp_dist = zmp.hull_point_signed_dist(
        asset.data.hull_points, asset.data.hull_idx, asset.data.zmp_pos_w[..., :2]
    )
    rew = th.exp(sigma * -th.clip(signed_zmp_dist, max=0)) - 1

    # Zero-mask reward for the moving command envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    rew = th.sum(th.abs(angle), dim=1)

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def joint_deviation_arm(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    default_arm_pos = th.tensor(
        [
            0.3500,
            0.3500,
            0.1600,
            -0.1600,
            0.0000,
            0.0000,
            0.8700,
            0.8700,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],
        device=env.device,
    )
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_arm_pos
    rew = th.sum(th.abs(angle), dim=1)

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    # Zero-mask reward for the standing command envs
    standing_env_ids = command.is_standing_env.nonzero(as_tuple=False).flatten()
    rew[standing_env_ids] = 0.0

    return rew


def action_norm_l2(env, action_name: str):
    action = env.action_manager.get_term(action_name)

    if "Residual_action_norm" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Raw_Residual_action_norm"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )
        env.reward_manager.episode_stat_sums["Residual_action_norm"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Residual_action_norm"] += th.norm(
        action.processed_actions, dim=-1
    )

    env.reward_manager.episode_stat_sums["Raw_Residual_action_norm"] += th.norm(
        action.raw_actions, dim=-1
    )

    # return th.sum(th.square(action.processed_actions), dim=-1)
    return th.sum(th.square(action.raw_actions), dim=-1)


def rel_joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rel_torque = (
        asset.data.applied_torque
        / asset.root_physx_view.get_dof_max_forces().clone().to(env.device)
    )
    rew = th.sum(th.square(rel_torque[:, asset_cfg.joint_ids]), dim=1)
    # extract the used quantities (to enable type-hinting)
    return rew


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> th.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = th.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= th.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> th.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = th.where(in_contact, contact_time, air_time)
    single_stance = th.sum(in_contact.int(), dim=1) == 1
    reward = th.min(th.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = th.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= th.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # checking if contacts exist
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # retrieve velocity of feet
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    
    # compute the reward as: velocity norm if there is a contact, otherwise zero.
    reward = th.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = th.sum(
        th.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return th.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = th.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return th.exp(-ang_vel_error / std**2)


def energy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    energy = th.clip(
        asset.data.joint_vel[:, asset_cfg.joint_ids]
        * asset.data.applied_torque[:, asset_cfg.joint_ids],
        min=0,
    )
    if "Energy" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Energy"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Energy"] += th.sum(energy, dim=-1)

    return th.sum(energy, dim=-1)


def residual_action_rate_l2(env: ManagerBasedRLEnv) -> th.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return th.sum(
        th.square(env.action_manager.action - env.action_manager.prev_action)[..., -7:],
        dim=1,
    )


def action_rate_l2(env: ManagerBasedRLEnv) -> th.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return th.sum(
        th.square(env.action_manager.action - env.action_manager.prev_action)[..., :22],
        dim=1,
    )


def position_command_error(env: ManagerBasedRLEnv, command_name: str) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command = env.command_manager.get_command(command_name)
    pos_error = command[..., :3].norm(dim=-1)

    if "Right_hand_target_dist" not in env.reward_manager.episode_stat_sums.keys():
        # env.reward_manager.episode_stat_sums["Left_hand_target_dist"] = \
        #     th.zeros(env.num_envs, dtype=th.float, device=env.device)
        env.reward_manager.episode_stat_sums["Right_hand_target_dist"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    # env.reward_manager.episode_stat_sums["Left_hand_target_dist"] += \
    #     command[..., :3].norm(dim=-1)
    env.reward_manager.episode_stat_sums["Right_hand_target_dist"] += command[
        ..., :3
    ].norm(dim=-1)

    return pos_error


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command = env.command_manager.get_command(command_name)
    ori_error = command[..., 3:6].norm(dim=-1)

    if "Right_hand_target_ori" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Right_hand_target_ori"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Right_hand_target_ori"] += command[
        ..., 3:6
    ].norm(dim=-1)

    return ori_error


def approaching_pose_v2_for_eetrack(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    max_dist: float = 0.2,
    sigma: float = 50.0,
    penalize_joint_limit: bool = True,
    arm: Literal["left", "right"] = "right",
) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    command_name = f"get_bbox_{arm}"
    func = getattr(command, command_name, None)
    curr_hand_bbox, target_hand_bbox = func()
    diff = curr_hand_bbox - target_hand_bbox
    bbox_key_point_pos_error = th.norm(diff, dim=-1).mean(-1)

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = th.logical_or(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        < asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        > asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )
    out_of_limits = th.any(out_of_limits, dim=-1)

    if penalize_joint_limit:
        bbox_key_point_pos_error = th.where(
            out_of_limits, th.ones_like(pos_error), pos_error
        )
    rew = th.exp(-sigma * th.square(bbox_key_point_pos_error))
    if "Arm_joint_limit" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Arm_joint_limit"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Arm_joint_limit"] += out_of_limits.float()

    # Zero-mask reward for the moving command envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def subgoal_achievement(
    env: ManagerBasedRLEnv, command_name: str, arm: Literal["left", "right"] = "left"
) -> th.Tensor:

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    command_name = f"get_bbox_{arm}"

    func = getattr(command, command_name, None)
    curr_hand_bbox, target_hand_bbox = func()
    diff = curr_hand_bbox - target_hand_bbox
    bbox_key_point_pos_error = th.norm(diff, dim=-1).mean(-1)

    rwd = bbox_key_point_pos_error < 0.01
    return rwd


def subgoal_achievement_obstacle(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    arm: Literal["left", "right"] = "left",
    distance: float = 0.15,
) -> th.Tensor:
    """
    Reward for achieving subgoal with obstacle. 
    
    It is given when the hand is close enough to the target considering volume of hand (smaller than distance),
    and simultaneously when track error is small.
    
    If eetrack line is horizontal, it checks the distance in x-axis. 
    Otherwise, it checks the distance in z-axis.
    """
    
    # command
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    command_name = f"get_bbox_{arm}"
    
    # bounding box
    func = getattr(command, command_name, None)
    curr_hand_bbox, target_hand_bbox = func()
    diff = curr_hand_bbox - target_hand_bbox
    bbox_key_point_pos_error = th.norm(diff, dim=-1).mean(-1)
    rwd_distance = bbox_key_point_pos_error < distance
    
    # compare hand position
    curr_hand_pos = curr_hand_bbox.mean(dim=-2)
    target_hand_pos = target_hand_bbox.mean(dim=-2)
    diff_pos = th.abs(curr_hand_pos - target_hand_pos)
    
    # eetrack line
    eetrack_start = command.eetrack_start
    eetrack_end = command.eetrack_end
    
    # check whether track error is small
    is_horizontal = (eetrack_start[:, 2] == eetrack_end[:, 2])
    close_horizontal = th.logical_and(diff_pos[:, 0] < 0.01, is_horizontal)
    close_vertical = th.logical_and(diff_pos[:, 2] < 0.01, ~is_horizontal)
    rwd_track = th.logical_or(close_horizontal, close_vertical).int()
    
    # give reward when both conditions are satisfied
    rwd = th.logical_and(rwd_distance, rwd_track)
    return rwd


def maintain_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    pos_error = command[..., :3].norm(dim=-1)
    ori_error = command[..., 3:6].norm(dim=-1)
    joint_vel = th.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    max_joint_vel, _ = th.max(joint_vel, dim=-1)

    maintain_target = (pos_error <= 0.2) & (ori_error <= 0.2) & (max_joint_vel <= 1.0)

    # ic(pos_error, ori_error, max_joint_vel, maintain_target)

    if "Max_joint_vel" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Max_joint_vel"] = th.zeros(
            env.num_envs, dtype=th.float, device=env.device
        )

    env.reward_manager.episode_stat_sums["Max_joint_vel"] += max_joint_vel

    return maintain_target.float()


def standing_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=".*_ankle_roll_link"
    ),
) -> th.Tensor:
    """
    Penalizes zero stance or single stance for stationaly command
    """

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    double_stance = th.sum(in_contact.int(), dim=1) == 2
    rew = (~double_stance).float()

    # Zero-mask reward for the moving command envs
    # Gets zero for moving environment; otherwise, you get 0 rwd if both feet are on the ground, 1 if only one foot is ground
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew


def contact_obstacles(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=""
    ),
) -> th.Tensor:
    """
    Penalizes leaning to obstacles
    """

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    num_contact = th.sum(in_contact.int(), dim=1)
    rew = num_contact.int()

    # Zero-mask reward for the moving command envs
    # Gets zero for moving environment; otherwise, you get 0 rwd if both feet are on the ground, 1 if only one foot is ground
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0
    return rew


def approaching_above(
    env: ManagerBasedRLEnv,
    command_name: str,
    height: float = 1.0,
    arm: Literal["left", "right"] = "left",
) -> th.Tensor:
    """
    Reward for approaching above the target when the eetrack line is horizontal & not too high (small than height)
    """

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    command_name = f"get_bbox_{arm}"
    func = getattr(command, command_name, None)
    curr_hand_bbox, target_hand_bbox = func()
    
    curr_hand_pos = curr_hand_bbox.mean(dim=-2)         # Average over key points
    target_hand_pos = target_hand_bbox.mean(dim=-2)     # Average over key points
    
    approaching_above = (curr_hand_pos[:, 2] - target_hand_pos[:, 2]) > 0.0
    target_too_high = target_hand_pos[:, 2] > height
    
    # eetrack line
    eetrack_start = command.eetrack_start
    eetrack_end = command.eetrack_end

    # conditions
    is_horizontal = (eetrack_start[:, 2] == eetrack_end[:, 2])
    
    approaching_above = th.logical_and(approaching_above, is_horizontal)
    approaching_above = th.logical_and(approaching_above, ~target_too_high)
    
    rew = approaching_above.int()
    return rew


def approaching_closer(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm: Literal["left", "right"] = "left",
) -> th.Tensor:
    """
    Reward for approaching to the target in a closer way (prevent approaching in the opposite direction)
    """
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)
    command_name = f"get_bbox_{arm}"
    func = getattr(command, command_name, None)
    curr_hand_bbox, target_hand_bbox = func()
    
    curr_hand_pos = curr_hand_bbox.mean(dim=-2)         # Average over key points
    target_hand_pos = target_hand_bbox.mean(dim=-2)     # Average over key points
    
    approaching_closer = (curr_hand_pos[:, 1] - target_hand_pos[:, 1]) < 0.0
    rew = approaching_closer.int()
    return rew


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.
    Migrated here from omni.isaac.lab.envs.mdp.rewards because it is used in our mdp

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(
        self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*"
    ) -> th.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = th.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


@configclass
class G1EETrackRewards:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # -- task

    ### Regualization terms
    rel_torques_l2 = RewTerm(
        func=rel_joint_torques_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    energy = RewTerm(
        func=energy,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        weight=-0.0002,
        # weight=0.
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    residual_action_rate_l2 = RewTerm(func=residual_action_rate_l2, weight=-0.005)
    action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.005)

    ### joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )

    right_arm_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_.*",
                ],
            )
        },
    )
    left_arm_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        # weight=-1.0,
        # weight=-20.0,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            )
        },
    )

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    termination_penalty = RewTerm(
        func=is_terminated_term,
        params={"term_keys": ["torso_height", "bad_ori"]},
        weight=-50.0,
    )

    ### locomotion reward
    feet_slide = RewTerm(
        func=feet_slide,
        weight=-0.2,
        # weight=0.,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    standing_still = RewTerm(
        func=standing_still,
        weight=-0.4,
        params={
            "command_name": "hands_pose",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
        },
    )

    """
    feet_air_time_v4 = RewTerm(
        func=air_time_v4,
        weight=0.2,
        params={
            "command_name": "hands_pose",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    """

    ### ZMP reward
    zmp_supp_dist_v2 = RewTerm(
        func=zmp_supp_dist_v2,
        weight=0.6,
        # weight=0.0,
        # weight=0.,
        params={
            "sigma": 10,
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    ### Hand reaching reward
    approaching_left = RewTerm(
        func=approaching_pose_v2_for_eetrack,
        weight=0.3,
        # weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            ),
            "command_name": "hands_pose",
            "penalize_joint_limit": False,
            "arm": "left",
            "sigma": 20.0,
        },
    )

    subgoal_achievement = RewTerm(
        func=subgoal_achievement,
        # weight=5,
        weight=0.0,
        params={"command_name": "hands_pose"},
    )

    residual_action_limit_left = RewTerm(
        func=action_limits,
        weight=-0.1,
        params={"action_name": "left_arm_res", "command_name": "hands_pose"},
    )
    
    if spawn_obstacles:
        subgoal_achievement_reward = RewTerm(
            func=subgoal_achievement_obstacle,
            # weight=5,
            weight=3.0,
            params={"command_name": "hands_pose"},
        )
        
        contact_obstacles_large_penalty = RewTerm(
            func=contact_obstacles,
            weight=-1.0,
            params={"command_name": "hands_pose",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["head_link", "torso_link", ".*_shoulder_.*_link", "pelvis_contour_link"])},
        )
        
        contact_obstacles_medium_penalty = RewTerm(
            func=contact_obstacles,
            weight=-0.5,
            params={"command_name": "hands_pose",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_elbow_link", ".*_knee_link", ".*_hip_.*_link"])},
        )
        
        contact_obstacles_small_penalty = RewTerm(
            func=contact_obstacles,
            weight=-0.3,
            params={"command_name": "hands_pose",
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wrist_.*_link"])}, # ".*_hand_.*_link"
        )
        
        approaching_above = RewTerm(
            func=approaching_above,
            weight=0.5,
            params={"command_name": "hands_pose", "arm": "left", "height": 1.0},
        )
        
        approaching_closer = RewTerm(
            func=approaching_closer,
            weight=0.3,
            params={"command_name": "hands_pose", "arm": "left"},
        )

    ### joint deviation from nominal poses

    """
    joint_deviation_hip_yaw = RewTerm(
        func=joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*_hip_yaw_joint"])
        },
    )

    joint_deviation_hip_roll = RewTerm(
        func=joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*_hip_roll_joint"])
        },
    )

    joint_deviation_hip_pitch = RewTerm(
        func=joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*_hip_pitch_joint"])
        },
    )
    joint_deviation_waist = RewTerm(
        func=joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["waist_.*"])
        },
    )
    """
