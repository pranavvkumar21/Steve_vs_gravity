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
    num_joints = env.scene["steve"].num_joints
    d = env.device
    env.cmd = {}
    env.motion_manager = MotionManager(device='cuda:0')
    env.motion_name = "walk"
    env.motion_manager.load_motion("walk", "../data/mocap_data/07/01/07_01_full_data.npz",is_cyclic=True)
    env.motion_manager.reorder_joints("walk", env.scene["steve"].data.joint_names, list(JOINT_MAPPING.keys()))
    env.motion_manager.clamp_to_joint_limits("walk", env.scene["steve"].data.default_joint_pos_limits)
    env.cmd["joint_position"] = torch.zeros(E, num_joints, device=d)
    env.cmd["joint_velocity"] = torch.zeros(E, num_joints, device=d)
    env.cmd["root_orientation"] = torch.zeros(E, 4, device=d)
    env.cmd["phase"] = torch.zeros(E,1, device=d)
    env.cmd["frame_idx"] = torch.zeros(E,1, device=d)
    env.cmd["done"] = torch.zeros(E,1, device=d)

    # joint_pos, joint_vel, phase, frame_idx = env.motion_manager.sample("walk", E)
    
    # env.cmd["joint_position"] = joint_pos
    # env.cmd["joint_velocity"] = joint_vel
    # env.cmd["phase"] = phase
    # env.cmd["frame_idx"] = frame_idx


def reset_cmd(env, env_ids):
    # env_ids is a tensor of env ids to reset
    mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    # print("Resetting env ids:", env_ids)
    # Sample initial motion frame for all env_ids at once
    joint_pos, joint_vel, root_orient, phase, frame_idx = env.motion_manager.sample("walk", len(env_ids))
    
    env.cmd["joint_position"][env_ids, :] = joint_pos
    env.cmd["joint_velocity"][env_ids, :] = joint_vel
    env.cmd["root_orientation"][env_ids, :] = root_orient   
    env.cmd["phase"][env_ids, 0] = phase
    env.cmd["frame_idx"][env_ids, 0] = frame_idx.float()
    # print("Reset frame idx for env ids", env_ids, "to", frame_idx.float())
    # Write to sim
    env.scene["steve"].write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    env.scene["steve"].write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
    root_pose = env.scene["steve"].data.root_pose_w[env_ids, :].clone()
    root_pose[:, 3:] = root_orient
    env.scene["steve"].write_root_pose_to_sim(root_pose, env_ids=env_ids)
    #write root orientation to sim

    # print("resetting env ids", env_ids, "pos:", pos[env_ids, :])


    
@configclass
class EventsCfg:
    startup_cmd = EventTerm(func=init_cmd, mode="startup")
    reset_cmd = EventTerm(func=reset_cmd, mode="reset", params={})
    # post_init = EventTerm(func=post_init, mode="reset", params={})
    # pre_startup = EventTerm(func=prestartup, mode="prestartup", params={})