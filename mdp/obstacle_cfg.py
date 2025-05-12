import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    RigidObjectCfg
)

class ObstacleCfg(RigidObjectCfg):
    def __init__(self, prim_path, size, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8), metallic=0.2),
                activate_contact_sensors=True
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=rot,
            )
        )
        self.size = size
