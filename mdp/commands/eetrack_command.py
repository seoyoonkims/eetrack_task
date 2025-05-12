# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch as th

import omni.isaac.lab.utils.math as math_utils
import numpy as np

from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import (
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
    RAY_CASTER_MARKER_CFG,
)

import itertools
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.managers import CommandTerm

import omni.isaac.lab.sim as sim_utils

import random as rd

if TYPE_CHECKING:
    from .eetrack_commands_cfg import EETrackCommandCfg

import os
spawn_obstacles = os.environ.get('SPAWN_OBSTACLES', 'False').lower() in ('true', '1', 't')


def interpolate_pose(p1: th.Tensor, p2: th.Tensor, t: th.Tensor):
    """
    Interpolate between two poses p1 and p2.

    Parameters:
        p1: Tensor of shape [N,7], each row is [x, y, z, qx, qy, qz, qw]
        p2: Tensor of shape [N,7], same as p1
        t: Tensor of shape [N], with values between 0 and 1

    Returns:
        Interpolated pose of shape [N,7]
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)

    pos1 = p1[:, :3]
    pos2 = p2[:, :3]
    q1 = p1[:, 3:]
    q2 = p2[:, 3:]

    # Linear interpolation of position
    interp_pos = pos1 + (pos2 - pos1) * t

    # slerp - spherical linear interpolation; interpolates between two quaternions
    interp_q = math_utils.slerp(q1, q2, t.squeeze())

    # Concatenate interpolated position and quaternion
    interp_pose = th.cat([interp_pos, interp_q], dim=1)
    return interp_pose


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


class EETrackCommand(CommandTerm):
    """Command generator that does nothing.

    This command generator does not generate any commands. It is used for environments that do not
    require any commands.
    """

    cfg: EETrackCommandCfg  # type annotation. doesnt enforce type checking, but provides clarity
    """Configuration for the command generator."""

    def __init__(self, cfg: EETrackCommandCfg, env: ManagerBased):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # save the frequently-used robot link indices
        self.left_hand_idx = self.robot.find_bodies(cfg.left_hand_body_name)[0][0]
        self.right_hand_idx = self.robot.find_bodies(cfg.right_hand_body_name)[0][0]

        self.left_foot_idx = self.robot.find_bodies(cfg.left_foot_body_name)[0][0]
        self.right_foot_idx = self.robot.find_bodies(cfg.right_foot_body_name)[0][0]
        self.torso_idx = self.robot.find_bodies(cfg.torso_body_name)[0][0]

        # Current command wrt torso
        self.curr_command_s_left = th.zeros(self.num_envs, 7, device=self.device)
        self.curr_command_s_right = th.zeros(self.num_envs, 7, device=self.device)

        # Next command wrt torso
        self.next_command_s_left = th.zeros(self.num_envs, 7, device=self.device)
        self.next_command_s_left[:, 3] = 1.0
        # self.next_command_s_right = th.zeros(self.num_envs, 7, device=self.device)
        # self.next_command_s_right[:, 3] = 1.0

        # Lerped (linear interpolated) hand pose command in **world frame**
        self.lerp_command_w_left = th.zeros_like(self.next_command_s_left)
        # self.lerp_command_w_right = th.zeros_like(self.next_command_s_left)

        # Learped hand pose in base frame
        self.lerp_command_b_left = th.zeros_like(self.next_command_s_left)
        # self.lerp_command_b_right = th.zeros_like(self.next_command_s_left)

        # Cylindrical frame wrt **world frame**
        self.ref_pos_w = th.zeros(self.num_envs, 3, device=self.device)
        self.ref_quat_w = th.zeros(self.num_envs, 4, device=self.device)

        # Hands pose wrt **reference frame**
        self.pos_hand_s_left = th.zeros(self.num_envs, 3, device=self.device)
        # self.pos_hand_s_right = th.zeros(self.num_envs, 3, device=self.device)
        self.quat_hand_s_left = th.zeros(self.num_envs, 4, device=self.device)
        # self.quat_hand_s_right = th.zeros(self.num_envs, 4, device=self.device)

        # Hands pose wrt **root frame**
        self.pos_hand_b_left = th.zeros(self.num_envs, 3, device=self.device)
        # self.pos_hand_b_right = th.zeros(self.num_envs, 3, device=self.device)
        self.quat_hand_b_left = th.zeros(self.num_envs, 4, device=self.device)
        # self.quat_hand_b_right = th.zeros(self.num_envs, 4, device=self.device)

        # Target pelvis height -- might get used in the future
        #self.pelvis_command = th.zeros(self.num_envs, 1, device=self.device)

        # Binary indicator for the standing envs
        self.is_standing_env = th.ones(self.num_envs, dtype=th.bool, device=self.device)

        hand_bboxes = th.as_tensor(cfg.hand_bbox, dtype=th.float, device=self.device)
        self._hand_bboxes = hand_bboxes[None].repeat(self.num_envs, 1, 1)

        # Metrics
        self.metrics["position_error_left"] = th.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["orientation_error_left"] = th.zeros(
            self.num_envs, device=self.device
        )

        segment_length = self.cfg.eetrack_segment_length  # each segment is, say, 0.01m
        robot_hand_velocity = self.cfg.eetrack_vel  # the hand moves at 0.1m per second
        self.first_subgoal_sampling_time = 1
        self.non_first_subgoal_sampling_time = segment_length / robot_hand_velocity

        self.current_eetrack_sg_index = th.zeros(
            self.num_envs, device=self.device, dtype=int
        )
        self.eetrack_line_has_not_been_defined = True
        self.number_of_subgoals = int(
            self.cfg.eetrack_line_length / self.cfg.eetrack_segment_length
        )
        self.eetrack_subgoals = th.zeros(self.num_envs, self.number_of_subgoals + 1, 7, device=self.device)
        self.eetrack_midpt = th.zeros(self.num_envs, 3, device=self.device)
        self.eetrack_start = th.zeros(self.num_envs, 3, device=self.device)
        self.eetrack_end = th.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        msg = "EETrackCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def get_eetrack_line(self):
        raise NotImplementedError

    def get_current_hand_pose_wrt_base(self, hand_idx):
        hand_position, hand_orientation = math_utils.subtract_frame_transforms(
            self.robot.data.root_state_w[..., :3],
            self.robot.data.root_state_w[..., 3:7],
            self.robot.data.body_state_w[:, hand_idx, :3],
            self.robot.data.body_state_w[:, hand_idx, 3:7],
        )
        return hand_position, hand_orientation

    """
    Properties
    """

    @property
    def command(self):
        """Returns the pose delta in the **world frame**. Shape is (num_envs, 12)."""
        # NOTE: this function is used by observation manager to get the first action
        # The problem now is that the first command is initialized to 0, and does not get updated
        # until _resample is called. Is robot config initailized at this point?
        # No; but then isn't it getting a wrong initial state? It should be getting the robot joint angels
        # *after* it has been initialized to the correct joint position.
        # why don't i use the default joint pos?
        # TODO Okay but where the heck does initializing the robot to the default value take place???
        # It is happening inside the env.reset in rslwrapper.
        self._update_command()

        # Hands pose wrt **root frame**
        """
        self.pos_hand_b_left, self.quat_hand_b_left = math_utils.subtract_frame_transforms(
            self.robot.data.root_state_w[..., :3],
            self.robot.data.root_state_w[..., 3:7],
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7]
        )
        """

        """
        self.pos_hand_b_right, self.quat_hand_b_right = math_utils.subtract_frame_transforms(
            self.robot.data.root_state_w[..., :3],
            self.robot.data.root_state_w[..., 3:7],
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7]
        )
        """
        (
            self.pos_hand_b_left,
            self.quat_hand_b_left,
        ) = self.get_current_hand_pose_wrt_base(self.left_hand_idx)

        # Compute the position error wrt linear-interpolated command
        # okay what's the implication of this?
        # what I want is the pose delta to the next target.
        # My guess is that they recompute the linear interpolation
        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            self.pos_hand_b_left,
            self.quat_hand_b_left,
            self.lerp_command_b_left[:, :3],
            self.lerp_command_b_left[:, 3:],
        )
        axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)

        hand_command = th.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)
        return hand_command

    """
    Operations.
    """

    def create_direction(self, env_ids: Sequence[int]):
        num_envs = len(env_ids)
        angle_from_eetrack_line = th.rand(num_envs, device=self.device) * np.pi
        angle_from_xyplane_in_global_frame  = th.rand(num_envs, device=self.device) * np.pi - np.pi/2
        roll = th.zeros(num_envs, device=self.device)
        pitch = angle_from_xyplane_in_global_frame 
        yaw = angle_from_eetrack_line
        euler = th.stack([roll, pitch, yaw], dim=1)
        quat = math_utils.quat_from_euler_xyz(euler[:,0], euler[:,1], euler[:,2])
        return quat
  
    def create_eetrack_obstacles(
        self,
        env: ManagerBasedRLEnv,
        env_ids: th.Tensor,
        asset_cfg: SceneEntityCfg,
    ):
        eetrack_command_term = env.command_manager._terms['hands_pose']
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # obstacle root state
        obstacle_root_states = asset.data.default_root_state[env_ids].clone()

        # robot root state
        robot_root_state = self.robot.data.root_state_w[env_ids].clone()
        robot_position = robot_root_state[:, :3]

        # eetrack line
        eetrack_start = eetrack_command_term.eetrack_start[env_ids]
        eetrack_end = eetrack_command_term.eetrack_end[env_ids]

        # conditions
        is_horizontal = eetrack_start[:, 2] == eetrack_end[:, 2]
        is_object1 = asset_cfg.name == "obstacle1"

        # obstacle base position
        positions = (eetrack_start + eetrack_end) / 2
        orientations = obstacle_root_states[:, 3:7]
        device = orientations.device

        # object information
        object_size_x = asset.cfg.size[0]
        angle_between_two_objects = 90  # degrees

        # Rotation quaternions
        rotate_for_alignment = math_utils.quat_from_euler_xyz(
            th.zeros_like(env_ids, dtype=th.float32, device=device),
            th.full_like(env_ids, 90.0, dtype=th.float32, device=device) * th.pi / 180,
            th.zeros_like(env_ids, dtype=th.float32, device=device),
        )
        rotate_object1 = math_utils.quat_from_euler_xyz(
            th.zeros_like(env_ids, dtype=th.float32, device=device),
            th.zeros_like(env_ids, dtype=th.float32, device=device),
            th.zeros_like(env_ids, dtype=th.float32, device=device),
        )
        rotate_object2 = math_utils.quat_from_euler_xyz(
            th.zeros_like(env_ids, dtype=th.float32, device=device),
            th.zeros_like(env_ids, dtype=th.float32, device=device),
            th.full_like(env_ids, 180 + angle_between_two_objects, dtype=th.float32, device=device) * th.pi / 180,
        )

        # Horizontal eetrack processing
        if is_horizontal.any():
            hor_idx = is_horizontal.nonzero(as_tuple=True)[0]
            orientations[hor_idx] = math_utils.quat_mul(orientations[hor_idx], rotate_for_alignment[hor_idx])

            if is_object1:
                orientations[hor_idx] = math_utils.quat_mul(orientations[hor_idx], rotate_object1[hor_idx])
                high_eetrack_mask = positions[hor_idx, 2] > 1.0
                positions[hor_idx, 2] = th.where(high_eetrack_mask, eetrack_start[hor_idx, 2] - object_size_x / 2, eetrack_start[hor_idx, 2] + object_size_x / 2)
            else:
                orientations[hor_idx] = math_utils.quat_mul(orientations[hor_idx], rotate_object2[hor_idx])
                positions[hor_idx, 1] -= object_size_x * th.sin(th.tensor(angle_between_two_objects, device=device) * th.pi / 180) / 2
                positions[hor_idx, 2] += object_size_x * th.cos(th.tensor(angle_between_two_objects, device=device) * th.pi / 180) / 2

        # Vertical eetrack processing
        vert_idx = (~is_horizontal).nonzero(as_tuple=True)[0]
        if vert_idx.numel() > 0:
            if is_object1:
                orientations[vert_idx] = math_utils.quat_mul(orientations[vert_idx], rotate_object1[vert_idx])
                x_offset_mask = robot_position[vert_idx, 0] - positions[vert_idx, 0] > 0.25
                positions[vert_idx, 0] = th.where(x_offset_mask, eetrack_start[vert_idx, 0] + object_size_x / 2, eetrack_start[vert_idx, 0] - object_size_x / 2)
            else:
                orientations[vert_idx] = math_utils.quat_mul(orientations[vert_idx], rotate_object2[vert_idx])
                positions[vert_idx, 0] -= object_size_x * th.cos(th.tensor(angle_between_two_objects, device=device) * th.pi / 180) / 2
                positions[vert_idx, 1] -= object_size_x * th.sin(th.tensor(angle_between_two_objects, device=device) * th.pi / 180) / 2

        asset.write_root_pose_to_sim(th.cat([positions, orientations], dim=-1), env_ids=env_ids)

    def create_eetrack(self, env_ids: Sequence[int]):
        is_horizontal = rd.choices([True, False], k=len(env_ids))
        for env_id, is_hor in zip(env_ids, is_horizontal):
            self.eetrack_start[env_id] = self.eetrack_midpt[env_id].clone()
            self.eetrack_end[env_id] = self.eetrack_midpt[env_id].clone()
            eetrack_offset = rd.uniform(-0.5, 0.5)
            if is_hor:
                self.eetrack_start[env_id, 2] += eetrack_offset
                self.eetrack_end[env_id, 2] += eetrack_offset
                self.eetrack_start[env_id, 0] -= (self.cfg.eetrack_line_length) / 2.
                self.eetrack_end[env_id, 0] += (self.cfg.eetrack_line_length) / 2.
            else:
                self.eetrack_start[env_id, 0] += eetrack_offset
                self.eetrack_end[env_id, 0] += eetrack_offset
                self.eetrack_start[env_id, 2] += (self.cfg.eetrack_line_length) / 2.
                self.eetrack_end[env_id, 2] -= (self.cfg.eetrack_line_length) / 2.
        return self.eetrack_start, self.eetrack_end

    def reset(self, env_ids: Sequence[int]) -> dict[str, float]:
        if self.eetrack_line_has_not_been_defined:
            self.eetrack_midpt = self.robot.data.root_state_w[..., :3]
            self.eetrack_midpt[:, 1] += 0.3
            self.eetrack_line_has_not_been_defined = False
        self.create_eetrack(env_ids)
        if self._env.sim.has_gui():
            self.draw_eetrack_line()
        if spawn_obstacles:
            self.create_eetrack_obstacles(self._env, env_ids, SceneEntityCfg("obstacle1"))
            self.create_eetrack_obstacles(self._env, env_ids, SceneEntityCfg("obstacle2"))
        self.eetrack_subgoals[env_ids] = self.create_eetrack_subgoals(env_ids)

        # logs after a reset
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = self._average_metric_value_over_environments(metric_value, env_ids)

            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras

    """
    Implementation specific functions.
    """
    def _average_metric_value_over_environments(self, metric_value, env_ids):
        return th.mean(metric_value[env_ids]).item()

    def _update_metrics(self):
        # Compute the error for left hand
        pos_error_left, rot_error_left = math_utils.compute_pose_error(
            self.lerp_command_w_left[:, :3],
            self.lerp_command_w_left[:, 3:],
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
        )
        self.metrics["position_error_left"] = th.norm(pos_error_left, dim=-1)
        self.metrics["orientation_error_left"] = th.norm(rot_error_left, dim=-1)

        # pelvis_delta = self.pelvis_command - self.robot.data.root_pos_w[..., 2:3]
        # self.metrics["pelvis_error_z"] = th.abs(pelvis_delta[..., 0])

    def _update_ref_frame(self):
        if self.cfg.frame == "torso":
            self.ref_pos_w = self.robot.data.body_state_w[:, self.torso_idx, :3]
            self.ref_quat_w = self.robot.data.body_state_w[:, self.torso_idx, 3:7]
        elif self.cfg.frame == "z-inv":
            self.ref_pos_w[..., :2] = self.robot.data.root_pos_w[..., :2]
            self.ref_quat_w = math_utils.yaw_quat(self.robot.data.root_quat_w)
        elif self.cfg.frame == "foot":
            self.ref_pos_w[..., :2] = 0.5 * (
                self.robot.data.body_state_w[:, self.left_foot_idx, :2]
                + self.robot.data.body_state_w[:, self.right_foot_idx, :2]
            )

            _, _, left_euler = math_utils.euler_xyz_from_quat(
                math_utils.yaw_quat(
                    self.robot.data.body_state_w[:, self.left_foot_idx, 3:7]
                )
            )
            _, _, right_euler = math_utils.euler_xyz_from_quat(
                # self.robot.data.body_state_w[:, self.right_foot_idx, 3:7]
                math_utils.yaw_quat(
                    self.robot.data.body_state_w[:, self.right_foot_idx, 3:7]
                )
            )
            # Compute average angle of two euler angles
            avg_cos = (th.cos(left_euler) + th.cos(right_euler)) / 2
            avg_sin = (th.sin(left_euler) + th.sin(right_euler)) / 2

            self.ref_quat_w = math_utils.quat_from_euler_xyz(
                th.zeros_like(left_euler),
                th.zeros_like(left_euler),
                th.atan2(avg_sin, avg_cos),
            )

    def _resample(self, env_ids: Sequence[int]):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        # resample the time left before resampling
        # print(f"Subgoal idx {self.current_eetrack_sg_index}")
        # print(f"Env IDs to resample {env_ids}")
        # print(f"Time left before the fix {self.time_left}")
        if len(env_ids) != 0:
            sg_idxs_of_env_to_resample = self.current_eetrack_sg_index[env_ids]
            time_left = self.time_left[env_ids]  # hmm.. weird np/th pointer issue
            time_left[
                sg_idxs_of_env_to_resample == 0
            ] = self.first_subgoal_sampling_time
            time_left[
                sg_idxs_of_env_to_resample != 0
            ] = self.non_first_subgoal_sampling_time
            self.time_left[env_ids] = time_left

            # increment the command counter
            self.command_counter[env_ids] += 1
            # resample the command
            self._resample_command(env_ids)
        # print(f"Time left after the fix {self.time_left}")

    def draw_eetrack_line(self):
        from omni.isaac.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        eetrack_start_for_drawing = [
            (s[0], s[1], s[2]) for s in self.eetrack_start.tolist()
        ]
        eetrack_end_for_drawing = [
            (s[0], s[1], s[2]) for s in self.eetrack_end.tolist()
        ]
        n_env = self._env.num_envs
        colors = [(0, 1, 0, 1) for _ in range(n_env)]
        sizes = [5 for _ in range(n_env)]
        draw.draw_lines(
            eetrack_start_for_drawing, eetrack_end_for_drawing, colors, sizes
        )

    def create_eetrack_subgoals(self, env_ids: Sequence[int]):
        eetrack_subgoals = interpolate_position(
            self.eetrack_start[env_ids], self.eetrack_end[env_ids], self.number_of_subgoals
        )
        #eetrack_subgoals = [th.tensor(l,device=self.device, dtype=th.float32) for l in eetrack_subgoals]
        eetrack_subgoals = [
            (
                l.clone().to(self.device, dtype=th.float32)
                if isinstance(l, th.Tensor) 
                else th.tensor(l, device=self.device, dtype=th.float32)
            )
            for l in eetrack_subgoals
        ]
        eetrack_subgoals = th.stack(eetrack_subgoals,axis=1)
        eetrack_ori = self.create_direction(env_ids).unsqueeze(1).repeat(1, self.number_of_subgoals + 1, 1)
        # welidng_subgoals -> Nenv x Npoints x (3 + 4)
        return th.cat([eetrack_subgoals, eetrack_ori], dim=2)


    def _resample_command(self, env_ids: Sequence[int]):
        self.next_command_s_left[env_ids] = self.eetrack_subgoals[
            env_ids, self.current_eetrack_sg_index[env_ids], :
        ]
        self.current_eetrack_sg_index[env_ids] += 1

        # print(f"cmd L {self.next_command_s_left[0:3].tolist()}")
        # print(f"cmd R {self.next_command_s_right[0:3].tolist()}")
        # print(f"torso{self.next_command_s_right[0:3].tolist()}")

    def _update_command(self):
        """
        This is different from _resample_command method. Since the robot
        poses varies as it moves, we need to refresh the target pose
        considering the robot's movement.
        """
        self._update_ref_frame()

        # my assumption: the command is in the world frame. You can use it as is.
        self.lerp_command_w_left = self.next_command_s_left

        #### Maintain the right hand command given in the initial pose
        # The description of this snippet
        # Note: root=torso
        # Let 0 = world frame, 1 = torso wrt world, 2=hand loc wrt torso
        # self.ref_pos_w, self.ref_quat_w -> torso in world frame
        # rh pose -> in torso's frame
        # Output = the hand's pose in world frame

        # Compute the IK target for the Numerical IK solver
        # 0 = world
        # 01 = torso wrt world
        # 02 = hand command wrt world
        # output = lerp command wrt torso
        # This output is then used by PassiveIKActionTerm to be processed
        # Verify: is the IK target self.lerp_command_b_left and right?
        # Yes. The functions at L356 are used to process them to be used with IK solver
        # NOTE: for RH, we don't have to do this. Later, find a way to rewrite this to make it more explicit.
        (
            self.lerp_command_b_left[..., :3],
            self.lerp_command_b_left[..., 3:7],
        ) = math_utils.subtract_frame_transforms(
            self.robot.data.root_state_w[..., 0:3],
            self.robot.data.root_state_w[..., 3:7],
            self.lerp_command_w_left[:, 0:3],
            self.lerp_command_w_left[:, 3:7],
        )

        # Note: the reason you first put the hand in the world frame is because this code is
        #        meant to support different reference frames

        # One thing that I still don't get is why we linearly interpolate

    @property
    def ik_target_left(self) -> th.Tensor:
        # IK position target(input of the PassiveIKAction)
        # Concat of [binary standing indicator(1), target EE pos(3), target EE quat(4)]
        return th.cat(
            [self.is_standing_env.float()[..., None], self.lerp_command_b_left], dim=-1
        )

    def get_bbox_left(self):
        left_hand_pose_world_frame = self.robot.data.body_state_w[:, self.left_hand_idx]

        # hand bboxes are hand bounding box points, and you transform them to the given pose
        curr_hand_bbox = math_utils.transform_points(
            self._hand_bboxes,
            left_hand_pose_world_frame[..., :3],
            left_hand_pose_world_frame[..., 3:7],
        )
        target_hand_bbox = math_utils.transform_points(
            self._hand_bboxes,
            self.lerp_command_w_left[..., :3],
            self.lerp_command_w_left[..., 3:7],
        )
        return curr_hand_bbox, target_hand_bbox

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer_right"):
                if self.cfg.vis_left:
                    self.goal_pose_visualizer_left = VisualizationMarkers(
                        self.cfg.goal_pose_visualizer_cfg
                    )
                    self.current_pose_visualizer_left = VisualizationMarkers(
                        self.cfg.current_pose_visualizer_cfg
                    )
                if self.cfg.vis_hand_bbox:
                    if not hasattr(self, "bbox_visualizer"):
                        colors = list(itertools.product([0.0, 1.0], repeat=3))
                        if hasattr(self, "_hand_bboxes"):
                            n_points = self._hand_bboxes.shape[1]
                        else:
                            n_points = len(self.cfg.hand_bbox)
                            pass
                            # bbox = np.load(self.cfg.hand_bbox_file)
                            # n_points = bbox.shape[0]
                        self.bbox_visualizer = []
                        for i in range(n_points):
                            cc = self.cfg.bbox_vis_ref_cfg.replace(
                                prim_path=f"/Visuals/Command/bbox_{i}"
                            )
                            cc.markers["hit"].visual_material.diffuse_color = colors[i]
                            self.bbox_visualizer.append(VisualizationMarkers(cc))
                    for v in self.bbox_visualizer:
                        v.set_visibility(True)

            self.goal_pose_visualizer_left.set_visibility(False)
            # self.goal_pose_visualizer_right.set_visibility(True)

            self.current_pose_visualizer_left.set_visibility(True)
            # self.current_pose_visualizer_right.set_visibility(True)

            # self.current_pelvis_visualizer.set_visibility(True)
            # self.goal_pelvis_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pose_visualizer_left"):
                self.goal_pose_visualizer_left.set_visibility(True)
                # self.goal_pose_visualizer_right.set_visibility(True)

                # self.goal_pose_visualizer_left.set_visibility(False)
                # self.goal_pose_visualizer_right.set_visibility(False)
                # self.current_pose_visualizer_left.set_visibility(False)
                # self.current_pose_visualizer_right.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        # is the command supposed to be in the world frame?
        # print(f"next cmd L {self.next_command_s_left[0:3].tolist()} R {self.next_command_s_right[0:3].tolist()}")
        # print(f"curr cmd L {self.lerp_command_w_left[0:3].tolist()} R {self.lerp_command_w_right[0:3].tolist()}")
        if self.cfg.vis_left:
            self.goal_pose_visualizer_left.visualize(
                self.lerp_command_w_left[:, :3], self.lerp_command_w_left[:, 3:]
            )
            body_pose_w_left = self.robot.data.body_state_w[:, self.left_hand_idx]
            self.current_pose_visualizer_left.visualize(
                body_pose_w_left[:, :3], body_pose_w_left[:, 3:7]
            )

        if self.cfg.vis_hand_bbox:
            # right hand
            # rhb, rgb = self.get_bbox_right()
            lhb, lgb = self.get_bbox_left()

            vis_bbox = []
            if self.cfg.vis_left:
                vis_bbox.append(lhb)
                vis_bbox.append(lgb)
            # if self.cfg.vis_right:
            #    vis_bbox.append(rhb)
            #    vis_bbox.append(rgb)

            bb = th.cat(vis_bbox, dim=0)
            for idx, vis in enumerate(self.bbox_visualizer):
                vis.visualize(bb[..., idx, :])
