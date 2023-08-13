"""
============
Modified by
============
Author: Samuel Schmidgall
Institution: Johns Hopkins University


===================
Original Copyright
===================
Copyright (c) 2018-2022, NVIDIA Corporation
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from surgicalgym.utils.hydra_cfg.hydra_utils import *
from surgicalgym.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from surgicalgym.utils.task_util import initialize_task
from surgicalgym.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    time_testing = False

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    import time
    total_data = 0
    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            if time_testing:
                actions = torch.rand((env.num_envs, 9))
            else:
                actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
            env._task.pre_physics_step(actions)
            env._world.step(render=render)
            env.sim_frame_count += 1
            if time_testing:
                total_data += env.num_envs
                if env.sim_frame_count == 1: 
                    initial_time = time.time()
                if env.sim_frame_count == 100: 
                    print("TOTAL TIME", time.time() - initial_time, total_data)
                    break
            env._task.post_physics_step()
        else:
            env._world.step(render=render)

    env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()
