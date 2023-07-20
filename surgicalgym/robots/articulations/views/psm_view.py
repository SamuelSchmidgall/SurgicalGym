"""
Author: Samuel Schmidgall
Institution: Johns Hopkins University
"""

from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class PSMView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "PSMView",
    ) -> None:
        """[summary]
        """
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )
        self._psm_end_link  = RigidPrimView(prim_paths_expr="/World/envs/.*/psm/psm_pitch_end_link", name="end_link", reset_xform_properties=False)
        self._psm_tool_link  = RigidPrimView(prim_paths_expr="/World/envs/.*/psm/psm_tool_roll_link", name="tool_link", reset_xform_properties=False)
        

        
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        
