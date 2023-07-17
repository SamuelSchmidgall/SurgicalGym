# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from surgicalgym.tasks.base.rl_task import RLTask
from surgicalgym.robots.articulations.ecm import ECM
from surgicalgym.robots.articulations.views.ecm_view import ECMView

from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere

import torch.nn.functional as F
from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom


class ECMTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.task_id = self._task_cfg["env"]["task"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.distX_offset = 0.04

        if self.task_id == "target_reach":
            self.dt = 0.005
            self.decimation = 2
            self._num_actions = 6
            self._num_observations = 24
            self._ball_position = torch.tensor([0.2, 0, 0.5])#.to(self._device)

        RLTask.__init__(self, name, env)

        self.time = 0
        self.env_ids_int32 = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        #self.lower = torch.tensor([-90, -45, -66, -45,  0.0,  -90]).to(self._device)
        #self.upper = torch.tensor([90,   66,  45,  66,  0.254, 90]).to(self._device)
        self.lower = torch.tensor([-1, -1, -1, -1, -1, -1]).to(self._device)
        self.upper = torch.tensor([ 1,  1,  1,  1,  1,  1]).to(self._device)

        self.scale = torch.tensor( [1.0,  1.0, 1.0, 1.0, 0.254/2, 1.0]).to(self._device)
        self.offset = torch.tensor([0.0,  0.0, 0.0, 0.0, 0.254/2, 0.0]).to(self._device)
        self.scale = self.scale.unsqueeze(0).repeat(self._num_envs, 1)
        self.offset = self.offset.unsqueeze(0).repeat(self._num_envs, 1)

        return

    def set_up_scene(self, scene) -> None:
        self.get_ecm()

        if self.task_id == "target_reach":
            self.get_target()
        
        super().set_up_scene(scene)
        
        self._ecms = ECMView(prim_paths_expr="/World/envs/.*/ecm", name="ecm_view")

        if self.task_id == "target_reach":
            self._ball_target = RigidPrimView(prim_paths_expr="/World/envs/.*/ball_target")
        
        self._env_indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        scene.add(self._ecms)
        scene.add(self._ecms._ecm_end_link)
        scene.add(self._ecms._ecm_tool_link)

        if self.task_id == "target_reach":
            scene.add(self._ball_target)

        self.init_data()
        return

    def get_ecm(self):
        ecm = ECM(prim_path=self.default_zero_env_path + "/ecm", name="ecm")
        self._sim_config.apply_articulation_settings("ecm", get_prim_at_path(ecm.prim_path), self._sim_config.parse_actor_config("ecm"))

    def init_data(self) -> None:
        #self.ecm_default_dof_pos = torch.tensor([0.0] * 6, device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        self.ecm_dof_targets = torch.zeros_like(self.actions)
        self.initial_root_pos, self.initial_root_rot = self._ecms.get_world_poses(clone=False)
        self._ecms.set_joint_position_targets(self.ecm_dof_targets, indices=self._env_indices)
        if self.task_id == "target_reach":
            self.init_ball_pos, self.init_ball_rot = self._ball_target.get_world_poses(clone=False)
            self.init_ball_pos = self.init_ball_pos.clone()
            self.init_ball_pos[:, 2] += 0.5
            self.ball_offset = torch.randn_like(self.init_ball_pos)* 0.1
            self._ball_target.set_world_poses(
                (self.init_ball_pos+self.ball_offset), 
                self.init_ball_rot, 
                indices=self._env_indices
            )

        self.ecm_tool_link_pos, self.ecm_tool_link_rot = self._ecms._ecm_tool_link.get_world_poses(clone=False)
        self.ecm_tool_link_tip = self.ecm_tool_link_pos.clone()
        
    def get_target(self):
        radius = 0.015
        color = torch.tensor([1, 0, 0])
        ball_target = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball_target",
            translation=self._ball_position,
            name="target_1",
            radius=radius,
            color=color)
        self._sim_config.apply_articulation_settings(
            "ball_target", get_prim_at_path(ball_target.prim_path),
            self._sim_config.parse_actor_config("ball_target"))
        ball_target.set_collision_enabled(False)
    
    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._ecms.get_world_poses(clone=False)
        self.ecm_dof_pos = self._ecms.get_joint_positions(clone=False)
        self.ecm_dof_vel = self._ecms.get_joint_velocities(clone=False)

        self.ecm_tool_end_pos, self.ecm_end_link_rot = self._ecms._ecm_end_link.get_world_poses(clone=False)
        self.ecm_tool_link_pos, self.ecm_tool_link_rot = self._ecms._ecm_tool_link.get_world_poses(clone=False)
        
        def quaternion_to_rotation_matrix(q):
            # Normalize quaternion
            q = q / q.norm(dim=-1, keepdim=True)
            w, x, y, z = q.unbind(-1)
            # Form the rotation matrix
            rotation_matrix = torch.stack([
                1 - 2*(y**2) - 2*(z**2),     2*x*y - 2*z*w,     2*x*z + 2*y*w,
                2*x*y + 2*z*w,     1 - 2*(x**2) - 2*(z**2),     2*y*z - 2*x*w,
                2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*(x**2) - 2*(y**2)
            ], dim=-1).reshape(q.shape[0], 3, 3)
            return rotation_matrix
        
        rotation_matrices = quaternion_to_rotation_matrix(self.ecm_end_link_rot)
        initial_pole_tip = torch.tensor([0.0, 0.375, 0.0]).expand_as(self.ecm_tool_link_pos).to(self._device)
        rotated_pole_tip = torch.bmm(rotation_matrices, initial_pole_tip.unsqueeze(-1)).squeeze(-1)
        tip = self.ecm_tool_link_pos + rotated_pole_tip
        self.ecm_tool_link_tip = tip.clone()

        #print(self.ecm_dof_pos.cpu().numpy())
        if self.task_id == "target_reach":
            self.obs_buf = torch.cat((
                    self.ecm_dof_pos,
                    self.ecm_dof_vel,
                    self.ecm_tool_link_tip,
                    self.ecm_dof_targets,
                    self.ecm_tool_link_tip - (
                        self.init_ball_pos.clone() + self.ball_offset)
                ), dim=-1,
            )
        # tool tip relative to base at default = x=0.615 y=0 z=0.1
        observations = {
            self._ecms.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            
            indices = reset_env_ids.to(dtype=torch.int32)
            num_indices = len(indices)
            self._ecms.set_joint_positions(
                torch.zeros((num_indices, self._ecms.num_dof), device=self._device), #self.ecm_dof_pos, 
                indices=indices
            )
            self._ecms.set_joint_velocities(
                torch.zeros((num_indices, self._ecms.num_dof), device=self._device), #self.ecm_dof_vel, 
                indices=indices
            )
            self._ecms.set_joint_position_targets(
                torch.zeros((num_indices, self._ecms.num_dof), device=self._device), 
                indices=indices
            )
            root_pos, root_rot = self.initial_root_pos[reset_env_ids], self.initial_root_rot[reset_env_ids]
            root_vel = torch.zeros((num_indices, 6), device=self._device)
            self._ecms.set_world_poses(
                root_pos, 
                root_rot, 
                indices=indices
            )
            self._ecms.set_velocities(
                root_vel, 
                indices=indices
            )
            # bookkeeping
            self.reset_buf[reset_env_ids] = 0
            self.progress_buf[reset_env_ids] = 0
            
            if self.task_id == "target_reach":
                self.ball_offset = torch.randn_like(self.init_ball_pos) * 0.1
                self._ball_target.set_world_poses(
                    (self.init_ball_pos+self.ball_offset), 
                    self.init_ball_rot, 
                    indices=indices
                )
            self.time = 0
            
        self.actions = actions.clone().to(self._device)
        for _ in range(self.decimation):
            self.prev_actions = self.actions.clone()
            self.ecm_dof_targets = self.ecm_dof_targets + self.dt * self.actions * self.action_scale
            self.ecm_dof_targets = torch.clip(
                self.ecm_dof_targets, min=self.lower, max=self.upper)
            scaled_ecm_dof_targets = self.ecm_dof_targets * self.scale + self.offset
            self._ecms.set_joint_position_targets(scaled_ecm_dof_targets, indices=self.env_ids_int32)
        self.time += 1


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)
        self.ecm_dof_pos = torch.zeros((num_indices, self._ecms.num_dof), device=self._device)
        self.ecm_dof_vel = torch.zeros((num_indices, self._ecms.num_dof), device=self._device)
        self.ecm_dof_targets = torch.zeros_like(self.actions)

    def post_reset(self):
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        if self.task_id == "target_reach":
            #print((self.ecm_tool_link_tip - (
            #        self.init_ball_pos.clone() + self.ball_offset)).cpu().numpy())
            self.rew_buf[:] = -abs(
                self.ecm_tool_link_tip - (
                    self.init_ball_pos.clone() + self.ball_offset)
                ).sum(-1)

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
