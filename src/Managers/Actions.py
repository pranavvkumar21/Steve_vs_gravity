import isaaclab.envs.mdp as mdp   
from isaaclab.utils import configclass
from isaaclab.managers import (ActionTerm,
    ActionTermCfg as ActionTerm,
    SceneEntityCfg)
import yaml
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

@configclass
class ActionsCfg:
    pass
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name = "steve",
    #     joint_names = steve_config["scene"]["joint_names"],
    #     scale=8*torch.pi/180,
    # )

