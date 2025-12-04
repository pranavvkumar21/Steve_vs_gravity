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
sys.path.append(str(ROOT))

from joint_mappings import JOINT_MAPPING
from motion_manager import MotionManager
from utils import get_normalised_joint_weights

def init_cmd(env, env_ids):
    E = env.scene.num_envs
    # num_joints = env.scene["steve"].num_joints
    d = env.device
    env.cmd = {}

    env.motion_manager = MotionManager(device='cuda:0')
    env.motion_name = "walk"
    env.motion_manager.load_motion("walk", str(ROOT / "data/retargeted_lafan_h1/walk1_sub1.pkl"),is_cyclic=True)
    
    robot_joint_names = env.scene["steve"].data.joint_names

    joint_indices = env.motion_manager.get_joint_indices("walk", robot_joint_names)
    env.motion_manager.move_reference_root_to_origin("walk", env.scene["steve"].data.default_root_state[0][:3])

    print(joint_indices)
    print("shape of joint positions for walk motion:",env.motion_manager.motions["walk"]['joint_positions'].shape)
    
    mocap_num_joints = env.motion_manager.motions["walk"]['joint_positions'].shape[1]
    mocap_num_bodies = len(env.motion_manager.motions["walk"]['link_body_names'])

    root_weight, joint_weights = get_normalised_joint_weights()

    print("Joint weights:", joint_weights)
    env.cmd["joint_weights"] = torch.from_numpy(joint_weights).to(device=d)
    env.cmd["root_weight"] = torch.tensor(root_weight, device=d)

    # print("Initialized joint position command with shape:", env.cmd["joint_position"].shape)
    print(env.motion_manager.motions["walk"]['joint_names'])
    env.cmd["joint_position"] = torch.zeros(E, mocap_num_joints, device=d)
    env.cmd["joint_velocity"] = torch.zeros(E, mocap_num_joints, device=d)
    env.cmd["root_orientation"] = torch.zeros(E, 4, device=d)
    env.cmd["root_position"] = torch.zeros(E, 3, device=d)
    env.cmd["local_body_position"] = torch.zeros(E, mocap_num_bodies, 3, device=d)
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
    
    # Reset the action manager's root offset tracking for these environments
    if hasattr(env, 'action_manager') and hasattr(env.action_manager, 'reset'):
        env.action_manager.reset(env_ids)
    
    # print("Resetting env ids:", env_ids)
    # # Sample initial motion frame for all env_ids at once
    joint_pos, joint_vel, root_position, root_orient, local_body_position, phase, frame_idx = env.motion_manager.sample("walk", len(env_ids))
    
    env.cmd["joint_position"][env_ids, :] = joint_pos.clone()
    env.cmd["joint_velocity"][env_ids, :] = joint_vel.clone()
    env.cmd["root_orientation"][env_ids, :] = root_orient.clone()  
    env.cmd["root_position"][env_ids, :] = root_position.clone()
    env.cmd["local_body_position"][env_ids, :, :] = local_body_position.clone()
    env.cmd["phase"][env_ids, 0] = phase.clone()
    env.cmd["frame_idx"][env_ids, 0] = frame_idx.float()
    # print("Reset frame idx for env ids", env_ids, "to", frame_idx.float())
    # Write to sim
    # get joinit index  of all joints in motion manager order from env scene joint names
    # print(f"len of motion manager joinit names ordered: {len(env.motion_manager.motions['walk']['joint_names'])}")
    
    joint_ids = env.motion_manager.motions["walk"]['joint_indices']
    env.scene["steve"].write_joint_position_to_sim(joint_pos.clone(), joint_ids=joint_ids, env_ids=env_ids)
    env.scene["steve"].write_joint_velocity_to_sim(joint_vel.clone(), joint_ids=joint_ids, env_ids=env_ids)
    root_pose = env.scene["steve"].data.root_pose_w[env_ids, :].clone()
    root_pose[:, :3] = root_position.clone()
    root_pose[:, 3:] = root_orient.clone()
    env.scene["steve"].write_root_pose_to_sim(root_pose, env_ids=env_ids)
    #write root orientation to sim

    # print("resetting env ids", env_ids, "pos:", pos[env_ids, :])


    
@configclass
class EventsCfg:
    startup_cmd = EventTerm(func=init_cmd, mode="startup")
    reset_cmd = EventTerm(func=reset_cmd, mode="reset", params={})
    # post_init = EventTerm(func=post_init, mode="reset", params={})
    # pre_startup = EventTerm(func=prestartup, mode="prestartup", params={})