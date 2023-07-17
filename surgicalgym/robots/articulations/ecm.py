# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from surgicalgym.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

class ECM(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "ecm",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        self._name = name
        self._usd_path = usd_path

        self._position = torch.tensor([1.0, 0.0, -0.5]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = "C:/Users/sschmidgall/Documents/ecm_mod_fu.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "ecm_base_link/ecm_yaw_joint",
            "ecm_yaw_link/ecm_pitch_front_joint",
            "ecm_pitch_front_link/ecm_pitch_bottom_joint",
            "ecm_pitch_bottom_link/ecm_pitch_end_joint",
            "ecm_pitch_end_link/ecm_main_insertion_joint",
            "ecm_main_insertion_link/ecm_tool_joint"
        ]
        # ecm_yaw_joint             [-91.4, 90.62]  angular
        # ecm_pitch_front_joint     [-44.9, 66.32]  angular
        # ecm_pitch_bottom_joint    [-66.3, 44.9 ]  angular
        # ecm_pitch_end_joint       [-44.9, 66.3 ]  angular
        # ecm_main_insertion_joint  [0.0,   0.254]  prismatic
        # ecm_tool_joint            [-90.0, 90.0 ]  angular
        drive_type = ["angular", "angular", "angular", "angular", "prismatic", "angular"]
        damping =   [1000,  1000,  1000,  1000,  1e4, 1000]
        stiffness = [10000, 10000, 10000, 10000, 1e5, 10000]
        for i, dof in enumerate(dof_paths):
            #print(self.prim_path, dof)
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=0.0,
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=10e7
            )
            #PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])
        
        #default_dof_pos = [0.0,0.0,0.0,0.0,0.0,0.0]?
        #max_forces = [1000, 1000, 1000, 1000, 50, 1000]?