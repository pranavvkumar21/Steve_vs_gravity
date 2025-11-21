import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg
from pathlib import Path
import yaml
from isaaclab_assets import HUMANOID_CFG

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

# Create pattern without the $ anchor
pattern = "(" + "|".join(steve_config["scene"]["body_contact_links"]) + ")"

class Steve_SceneCfg(InteractiveSceneCfg):
    """Configuration for the Steve scene."""
    
    # Make sure activate_contact_sensors is True
    steve = HUMANOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Steve",
        spawn=HUMANOID_CFG.spawn.replace(activate_contact_sensors=True),
        actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    damping=10,
                    stiffness=500,
                    effort_limit=80,
                ),
            },
    )
    
    ground = AssetBaseCfg(prim_path="/World/terrain", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # Body contact group - use .* for regex matching
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Steve/.*" + pattern,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )

    l_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Steve/left_foot",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )

    r_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Steve/right_foot",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )
