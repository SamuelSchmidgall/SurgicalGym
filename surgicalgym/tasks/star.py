"""
Author: Samuel Schmidgall
Institution: Johns Hopkins University
"""

from surgicalgym.tasks.base.rl_task import RLTask
from surgicalgym.robots.articulations.star import STAR
from surgicalgym.robots.articulations.views.star_view import STARView

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


class STARTask(RLTask):
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

        if self.task_id in ["active_tracking", "target_reach"]:
            self.dt = 0.0025
            self.decimation = 4
            self._num_actions = 8
            self._num_observations = 27
            self._ball_position = torch.tensor([0.0, 0.0, 0.8])

        RLTask.__init__(self, name, env)

        self.time = 0
        self.env_ids_int32 = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        self.lower = torch.tensor([-1 for _ in range(8)]).to(self._device)
        self.upper = torch.tensor([ 1 for _ in range(8)]).to(self._device)

        self.scale = torch.tensor(
            [2.0,  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ).to(self._device)
        self.offset = torch.tensor(
            [0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).to(self._device)
        self.scale = self.scale.unsqueeze(0).repeat(self._num_envs, 1)
        self.offset = self.offset.unsqueeze(0).repeat(self._num_envs, 1)
        
        self.default_pos = torch.tensor([0.0, 0.3, 0.0, -0.3, 0.0, 0.5, 0.0,  0.0]).to(self._device)
        self.default_pos = self.default_pos.unsqueeze(0).repeat(self._num_envs, 1)

        return

    def set_up_scene(self, scene) -> None:
        self.get_star()

        if self.task_id in ["active_tracking", "target_reach"]:
            self.get_target()
        
        super().set_up_scene(scene)
        
        self._stars = STARView(prim_paths_expr="/World/envs/.*/star_endo360", name="star_view")

        if self.task_id in ["active_tracking", "target_reach"]:
            self._ball_target = RigidPrimView(prim_paths_expr="/World/envs/.*/ball_target")
        
        self._env_indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        scene.add(self._stars)
        scene.add(self._stars._star_end_link)

        if self.task_id in ["active_tracking", "target_reach"]:
            scene.add(self._ball_target)

        self.init_data()
        return

    def get_star(self):
        star = STAR(prim_path=self.default_zero_env_path + "/star_endo360", name="star")
        self._sim_config.apply_articulation_settings("star", get_prim_at_path(star.prim_path), self._sim_config.parse_actor_config("star"))

    def init_data(self) -> None:
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        self.star_dof_targets = self.default_pos
        self.initial_root_pos, self.initial_root_rot = self._stars.get_world_poses(clone=False)
        self._stars.set_joint_position_targets(self._stars.num_dof, indices=self._env_indices)
        if self.task_id in ["active_tracking", "target_reach"]:
            self.init_ball_pos, self.init_ball_rot = self._ball_target.get_world_poses(clone=False)
            self.init_ball_pos = self.init_ball_pos.clone()
            self.ball_offset = torch.randn_like(self.init_ball_pos)*0.15
            self._ball_target.set_world_poses(
                (self.init_ball_pos+self.ball_offset), 
                self.init_ball_rot, 
                indices=self._env_indices
            )
            
        if self.task_id in ["active_tracking"]:
            self.ball_momentum = torch.zeros_like(self.ball_offset)
        self.star_tool_end_pos, self.star_end_link_rot = self._stars._star_end_link.get_world_poses(clone=False)

    def get_target(self):
        radius = 0.0225
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
        self.root_pos, self.root_rot = self._stars.get_world_poses(clone=False)
        self.star_dof_pos = self._stars.get_joint_positions(clone=False)
        self.star_dof_vel = self._stars.get_joint_velocities(clone=False)
        
        self.star_tool_end_pos, self.star_end_link_rot = self._stars._star_end_link.get_world_poses(clone=False)
        self.star_tool_link_tip = self.star_tool_end_pos

        #print(self.star_dof_pos.cpu().numpy())
        if self.task_id in ["active_tracking", "target_reach"]:
            self.obs_buf = torch.cat((
                    self.star_dof_pos,
                    self.star_dof_vel,
                    self.star_dof_targets,
                    self.star_tool_link_tip - (
                        self.init_ball_pos.clone() + self.ball_offset)
                ), dim=-1,
            )
        # tool tip relative to base at default = x=0.615 y=0 z=0.1
        observations = {
            self._stars.name: {
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
            self._stars.set_joint_positions(self.default_pos*self.scale, indices=indices)
            self._stars.set_joint_velocities(
                torch.zeros((num_indices, self._stars.num_dof), device=self._device), 
                indices=indices
            )
            self._stars.set_joint_position_targets(self.default_pos*self.scale, indices=indices)
            root_pos, root_rot = self.initial_root_pos[reset_env_ids], self.initial_root_rot[reset_env_ids]
            root_vel = torch.zeros((num_indices, 6), device=self._device)
            self._stars.set_world_poses(
                root_pos, 
                root_rot, 
                indices=indices
            )
            self._stars.set_velocities(
                root_vel, 
                indices=indices
            )
            # bookkeeping
            self.reset_buf[reset_env_ids] = 0
            self.progress_buf[reset_env_ids] = 0
            
            if self.task_id in ["active_tracking", "target_reach"]:
                self.ball_offset = torch.randn_like(self.init_ball_pos) * 0.15
                self._ball_target.set_world_poses(
                    (self.init_ball_pos+self.ball_offset), 
                    self.init_ball_rot, 
                    indices=indices
                )
            if self.task_id in ["active_tracking"]:
                self.ball_momentum = torch.zeros_like(self.ball_offset)
            self.time = 0
        
        if self.task_id in ["active_tracking"]:
            self.ball_momentum = self.ball_momentum + 0.0001*torch.randn_like(self.init_ball_pos)
            self.ball_offset = torch.clip(self.ball_momentum + self.ball_offset, min=-0.2, max=0.2)
            self._ball_target.set_world_poses(
                (self.init_ball_pos+self.ball_offset), 
                self.init_ball_rot, 
                indices=self.env_ids_int32
            )
        self.actions = actions.clone().to(self._device)
        for _ in range(self.decimation):
            self.prev_actions = self.actions.clone()
            self.star_dof_targets = self.star_dof_targets + self.dt * self.actions * self.action_scale
            self.star_dof_targets = torch.clip(
                self.star_dof_targets, min=self.lower, max=self.upper)
            scaled_star_dof_targets = self.star_dof_targets * self.scale + self.offset
            self._stars.set_joint_position_targets(scaled_star_dof_targets, indices=self.env_ids_int32)
        self.time += 1

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)
        self.star_dof_pos = torch.zeros((num_indices, self._stars.num_dof), device=self._device)
        self.star_dof_vel = torch.zeros((num_indices, self._stars.num_dof), device=self._device)
        self.star_dof_targets = self.default_pos

    def post_reset(self):
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        if self.task_id in ["active_tracking", "target_reach"]:
            self.rew_buf[:] = -abs(
                self.star_tool_link_tip - (
                    self.init_ball_pos.clone() + self.ball_offset)).sum(-1) - 0.001*abs(self.star_dof_vel).sum(-1)

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

