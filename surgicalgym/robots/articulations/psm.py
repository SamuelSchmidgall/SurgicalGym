"""
Author: Samuel Schmidgall
Institution: Johns Hopkins University
"""

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

class PSM(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "psm",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        self._name = name
        self._usd_path = usd_path

        self._position = torch.tensor([1.0, 0.0, 0.5]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = "C:/Users/sschmidgall/SurgicalGym/surgicalgym/models/psm.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "psm_base_link/psm_yaw_joint",                  # 0  angular
            "psm_yaw_link/psm_pitch_back_joint",            # 1  angular
            "psm_pitch_back_link/psm_pitch_bottom_joint",   # 2  angular
            "psm_pitch_bottom_link/psm_pitch_end_joint",    # 3  angular
            "psm_pitch_end_link/psm_main_insertion_joint",  # 4  prismatic
            "psm_main_insertion_link/psm_tool_roll_joint",  # 5  angular
            "psm_tool_roll_link/psm_tool_pitch_joint",      # 6  angular
            "psm_tool_pitch_link/psm_tool_yaw_joint",       # 7  angular
            "psm_tool_yaw_link/psm_tool_gripper1_joint",    # 8  angular     [0, 60] degrees
            "psm_tool_yaw_link/psm_tool_gripper2_joint",    # 9  angular     [0, 60] degrees
        ]
        drive_type = ["angular", "angular", "angular", "angular", "prismatic", "angular", "angular", "angular", "angular", "angular"]
        damping =   [1000 for _ in range(len(dof_paths))]
        stiffness = [10000 for _ in range(len(dof_paths))]
        
        for i, dof in enumerate(dof_paths):
            print(dof)
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=0.0,
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=10e9#7
            )