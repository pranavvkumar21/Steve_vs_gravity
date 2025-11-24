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

def joint_position_tracking(env):
    joint_names = env.scene["steve"].data.joint_names
    joint_ids = [env.scene["steve"].data.joint_names.index(name) for name in joint_names]
    
    current_joint_pos = env.scene["steve"].data.joint_pos[:, joint_ids]
    cmd_joint_pos = env.cmd["joint_position"][:, joint_ids]
    #clamp cmd_joint_pos to steve's joint limits
    
    position_error_sq = torch.sum((current_joint_pos - cmd_joint_pos) ** 2, dim=1)
    
    reward = torch.exp(-2.0 * position_error_sq)
    
    return reward

def joint_velocity_tracking(env):
    joint_names = env.scene["steve"].data.joint_names
    joint_ids = [env.scene["steve"].data.joint_names.index(name) for name in joint_names]
    
    current_joint_vel = env.scene["steve"].data.joint_vel[:, joint_ids]
    cmd_joint_vel = env.cmd["joint_velocity"][:, joint_ids]
    
    velocity_error_sq = torch.sum((current_joint_vel - cmd_joint_vel) ** 2, dim=1)
    
    reward = torch.exp(-0.1 * velocity_error_sq)
    
    return reward

def root_orientation_tracking(env):
    root_quat = env.scene["steve"].data.root_link_pose_w[:,3:]
    cmd_quat = env.cmd["root_orientation"]  # [E, 4]
    tracking_error = torch.sum((root_quat - cmd_quat) ** 2, dim=1)
    t = torch.exp(-10.0 * tracking_error)
    return t


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # forward_vel = RewTerm(func=velocity_tracking, params={"key": "x", "slope": config["forward_velocity"]["slope"]}, weight=config["forward_velocity"]["weight"])
    # lateral_vel = RewTerm(func=velocity_tracking, params={"key": "y", "slope": config["lateral_velocity"]["slope"]}, weight=config["lateral_velocity"]["weight"])
    # angular_vel = RewTerm(func=velocity_tracking, params={"key": "z", "slope": config["angular_velocity"]["slope"]}, weight=config["angular_velocity"]["weight"])
    # balance reward
    joint_pos_track = RewTerm(func=joint_position_tracking, weight=config["joint_position_tracking"]["weight"])
    joint_vel_track = RewTerm(func=joint_velocity_tracking, weight=config["joint_velocity_tracking"]["weight"])
    root_orient_track = RewTerm(func=root_orientation_tracking, weight=config["root_orientation_tracking"]["weight"])