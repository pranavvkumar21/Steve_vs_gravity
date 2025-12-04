from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
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
    joint_ids = env.motion_manager.motions["walk"]['joint_indices']
    
    current_joint_pos = env.scene["steve"].data.joint_pos[:, joint_ids]
    cmd_joint_pos = env.cmd["joint_position"]
    joint_weights = env.cmd["joint_weights"]
    
    # position_error_sq = torch.sum((current_joint_pos - cmd_joint_pos) ** 2, dim=1)
    joint_error = torch.sum((current_joint_pos - cmd_joint_pos) ** 2 * joint_weights.unsqueeze(0), dim=1)
    mean_joint_error = torch.mean((current_joint_pos - cmd_joint_pos))
    #debug
    # print("Joint error:", joint_error)
    # print("mean error:", mean_joint_error)
    # root orientation error
    root_quat = env.scene["steve"].data.root_link_pose_w[:,3:]
    cmd_quat = env.cmd["root_orientation"]  # [E, 4]
    dot = torch.abs(torch.sum(root_quat * cmd_quat, dim=1)).clamp(-1, 1)
    root_error = 2 * torch.acos(dot)
    l2_error = joint_error + env.cmd["root_weight"] * root_error
    num_joints = len(joint_ids)
    alpha = config["joint_position_tracking"]["alpha"] * num_joints /15.0


    reward = torch.exp(-alpha * l2_error)
    # print("Joint position tracking ?reward:", reward)
    
    return reward

def root_position_tracking(env):
    root_pos = env.scene["steve"].data.root_link_pose_w[:, :3]
    cmd_pos = env.cmd["root_position"]  # [E, 3]
    pos_err = torch.sum((root_pos - cmd_pos) ** 2, dim=1)
    alpha = config["root_position_tracking"]["alpha"]
    reward = torch.exp(-alpha * pos_err)
    return reward

def end_effector_tracking(env):
    end_effector_names = config["end_effector_tracking"]["end_effector_links"]
    body_names = env.scene["steve"].data.body_names
    cmd_body_names = env.motion_manager.motions["walk"]['link_body_names']
    end_effector_robot_ids = [body_names.index(name) for name in end_effector_names]
    end_effector_mocap_ids = [cmd_body_names.index(name) for name in end_effector_names]
    robot_global_positions = env.scene["steve"].data.body_pos_w[:, end_effector_robot_ids, :]
    robot_local_positions = robot_global_positions - env.scene["steve"].data.root_link_pose_w[:, :3].unsqueeze(1)
    cmd_local_positions  = env.cmd["local_body_position"][:, end_effector_mocap_ids, :]
    # print("robot local positions:", robot_local_positions.shape)
    # print("cmd local positions:", cmd_local_positions.shape)
    alpha = config["end_effector_tracking"]["alpha"]
    pos_err = torch.sum((robot_local_positions - cmd_local_positions) ** 2, dim=(1,2))
    reward = torch.exp(-alpha * pos_err)

    return reward

def joint_velocity_tracking(env):
    joint_ids = env.motion_manager.motions["walk"]['joint_indices']
    current_joint_vel = env.scene["steve"].data.joint_vel[:, joint_ids]
    cmd_joint_vel = env.cmd["joint_velocity"]    
    joint_weights = env.cmd["joint_weights"]
    # Compute squared velocity difference per joint, weighted by joint_weights
    diff = (current_joint_vel - cmd_joint_vel) ** 2
    weighted_diff = diff * joint_weights.unsqueeze(0)  # broadcast weights over batch

    # Sum weighted squared error over joints per timestep
    vel_err = torch.sum(weighted_diff, dim=1)

    # Scale factor from DeepMimic: 0.1 / 15 * num_joints
    num_joints = len(joint_ids)
    vel_scale = 0.1 / 15 * num_joints
    alpha = config["joint_velocity_tracking"]["alpha"]

    # Compute velocity reward as exponent of scaled error
    reward = torch.exp(-alpha * vel_scale * vel_err)

    return reward

 


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # height = RewTerm(func=mdp.flat_orientation_l2, params={"asset_cfg": SceneEntityCfg("steve")}, weight=1.0)
    forward_vel = RewTerm(func=velocity_tracking, params={"key":"x", "slope":-3}, weight=1.0)
    joint_pos_reward = RewTerm(func=joint_position_tracking, weight=config["joint_position_tracking"]["weight"])
    joint_vel_reward = RewTerm(func=joint_velocity_tracking, weight=config["joint_velocity_tracking"]["weight"])
    root_pos_reward = RewTerm(func=root_position_tracking, weight=config["root_position_tracking"]["weight"])
    end_effector_reward = RewTerm(func=end_effector_tracking, weight=config["end_effector_tracking"]["weight"])