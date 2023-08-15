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

class STAR(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "star",
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
            self._usd_path = "C:/Users/sschmidgall/SurgicalGym/surgicalgym/models/star.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "star_link_0/star_joint_1", # 0  angular
            "star_link_1/star_joint_2",          # 0  angular
            "star_link_2/star_joint_3",          # 1  angular
            "star_link_3/star_joint_4",          # 2  angular
            "star_link_4/star_joint_5",          # 3  angular
            "star_link_5/star_joint_6",          # 4  prismatic
            "star_link_6/star_joint_7",          # 5  angular
            "endo360_link_0/endo360_joint_1",          # 5  angular
        ]
        drive_type = ["angular", "angular", "angular", "angular", "angular", "angular", "angular", "angular"]
        damping =   [12 for _ in range(len(dof_paths))]
        stiffness = [200 for _ in range(len(dof_paths))]
        
        for i, dof in enumerate(dof_paths):
            print(dof)
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=0.0,
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=10e9 #7
            )