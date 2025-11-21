from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils import configclass
import torch
import yaml
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "steve_reward_config.yaml", "r") as f:
    config = yaml.safe_load(f)

def velocity_tracking(env, key="x", slope=-3):
    direction = {"x": 0, "y": 1, "z": 2}
    if key=="z":
        vel = env.scene["steve"].data.root_ang_vel_b[:, direction[key]]
    else:
        vel = env.scene["steve"].data.root_lin_vel_b[:, direction[key]]
    cmd_vel = mdp.generated_commands(env, command_name="velocity_command")[:, direction[key]]
    c_v = 1.0 / cmd_vel.abs().clamp_min(1e-3)
    vel_reward = torch.exp(slope * c_v * (vel - cmd_vel) ** 2)
    return vel_reward

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    forward_vel = RewTerm(func=velocity_tracking, params={"key": "x", "slope": config["forward_velocity"]["slope"]}, weight=config["forward_velocity"]["weight"])
    lateral_vel = RewTerm(func=velocity_tracking, params={"key": "y", "slope": config["lateral_velocity"]["slope"]}, weight=config["lateral_velocity"]["weight"])
    angular_vel = RewTerm(func=velocity_tracking, params={"key": "z", "slope": config["angular_velocity"]["slope"]}, weight=config["angular_velocity"]["weight"])
    # balance reward
    