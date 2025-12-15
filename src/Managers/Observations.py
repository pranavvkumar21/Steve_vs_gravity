
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import  (ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg)
import yaml
import torch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "steve_config.yaml", "r") as f:
    config = yaml.safe_load(f)

joint_names = config["joint_names"]



def phase_obs(env, key="phase", use_trig=False):
    if not hasattr(env, "cmd") or key not in env.cmd:
        print("Warning: phase not initialized yet!")
        env.cmd = {}
        return torch.zeros((env.scene.num_envs, 1), device=env.device)
    ph = env.cmd[key]
    return torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1) if use_trig else ph

def target_obs(env):
    if not hasattr(env, "motion_manager") :
        print("Warning: target joint positions not initialized yet!")
        return torch.zeros((env.scene.num_envs, len(config["joint_names"])), device=env.device)
    joint_names = env.scene["steve"].data.joint_names
    joint_ids = env.motion_manager.motions[env.motion_name]['joint_indices']
    
    current_joint_pos = env.scene["steve"].data.joint_pos[:, joint_ids]
    target = env.cmd["joint_position"]

    return (target - current_joint_pos)

def body_pos_b(env):
    #keep orientation untouched and subtract root position from body positions
    body_pos_w = env.scene["steve"].data.body_link_pose_w[:]
    root_pos = env.scene["steve"].data.root_link_pose_w[:, :3]
    
    return body_pos_w[:, :3] - root_pos


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})  # must match CommandsCfg field name
        root_linear_velocity  = ObsTerm(func=mdp.base_lin_vel,  params={"asset_cfg": SceneEntityCfg("steve")}) #3
        root_angular_velocity = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("steve")}) #6
        root_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("steve")}) #3
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("steve",joint_names=joint_names)}) #29
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("steve",joint_names=joint_names)}) #29
        target_joint_pos = ObsTerm(func=target_obs)

        phase = ObsTerm(func=phase_obs, )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

