#!/usr/bin/env python3
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import MassPropertiesCfg, modify_mass_properties
import torch
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT / "scripts"))
from joint_mappings import JOINT_MAPPING
from motion_manager import MotionManager

def init_cmd(env, env_ids):
    E = env.scene.num_envs
    # num_joints = env.scene["steve"].num_joints
    d = env.device
    env.cmd = {}
    env.motion_manager = MotionManager(device='cuda:0')
    env.motion_name = "walk"
    env.motion_manager.load_motion("walk", str(ROOT / "data/retargeted_lafan_h1/walk1_sub1.pkl"),is_cyclic=True)
    # env.motion_manager.reorder_joints("walk", env.scene["steve"].data.joint_names, env.motion_manager.motions["walk"]['joint_names_ordered'])
    robot_joint_names = env.scene["steve"].data.joint_names
    joint_indices = env.motion_manager.get_joint_indices("walk", robot_joint_names)
    print(joint_indices)
    # env.motion_manager.clamp_to_joint_limits("walk", env.scene["steve"].data.default_joint_pos_limits)
    print("shape of joint positions for walk motion:",env.motion_manager.motions["walk"]['joint_positions'].shape)
    num_joints = env.motion_manager.motions["walk"]['joint_positions'].shape[1]
    env.cmd["joint_position"] = torch.zeros(E, num_joints, device=d)

    print("Initialized joint position command with shape:", env.cmd["joint_position"].shape)
    print(env.motion_manager.motions["walk"]['joint_names_ordered'])
    env.cmd["joint_velocity"] = torch.zeros(E, num_joints, device=d)
    env.cmd["root_orientation"] = torch.zeros(E, 4, device=d)
    env.cmd["phase"] = torch.zeros(E,1, device=d)
    env.cmd["frame_idx"] = torch.zeros(E,1, device=d)
    # env.cmd["done"] = torch.zeros(E,1, device=d)

    # joint_pos, joint_vel, phase, frame_idx = env.motion_manager.sample("walk", E)
    
    # env.cmd["joint_position"] = joint_pos
    # env.cmd["joint_velocity"] = joint_vel
    # env.cmd["phase"] = phase
    # env.cmd["frame_idx"] = frame_idx


def reset_cmd(env, env_ids):
    # env_ids is a tensor of env ids to reset
    mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    # print("Resetting env ids:", env_ids)
    # # Sample initial motion frame for all env_ids at once
    joint_pos, joint_vel, root_orient, phase, frame_idx = env.motion_manager.sample("walk", len(env_ids))
    
    env.cmd["joint_position"][env_ids, :] = joint_pos
    env.cmd["joint_velocity"][env_ids, :] = joint_vel
    env.cmd["root_orientation"][env_ids, :] = root_orient   
    env.cmd["phase"][env_ids, 0] = phase
    env.cmd["frame_idx"][env_ids, 0] = frame_idx.float()
    # print("Reset frame idx for env ids", env_ids, "to", frame_idx.float())
    # Write to sim
    # get joinit index  of all joints in motion manager order from env scene joint names
    # print(f"len of motion manager joinit names ordered: {len(env.motion_manager.motions['walk']['joint_names_ordered'])}")
    joint_ids = env.motion_manager.motions["walk"]['joint_indices']
    env.scene["steve"].write_joint_position_to_sim(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    env.scene["steve"].write_joint_velocity_to_sim(joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    # root_pose = env.scene["steve"].data.root_pose_w[env_ids, :].clone()
    # root_pose[:, 3:] = root_orient
    # env.scene["steve"].write_root_pose_to_sim(root_pose, env_ids=env_ids)
    #write root orientation to sim

    # print("resetting env ids", env_ids, "pos:", pos[env_ids, :])


    
@configclass
class EventsCfg:
    startup_cmd = EventTerm(func=init_cmd, mode="startup")
    reset_cmd = EventTerm(func=reset_cmd, mode="reset", params={})
    # post_init = EventTerm(func=post_init, mode="reset", params={})
    # pre_startup = EventTerm(func=prestartup, mode="prestartup", params={})