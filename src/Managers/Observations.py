
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

joint_names = config["scene"]["joint_names"]



def phase_obs(env, key="phase", use_trig=False):
    if not hasattr(env, "cmd") or key not in env.cmd:
        print("Warning: phase not initialized yet!")
        env.cmd = {}
        return torch.zeros((env.scene.num_envs, 1), device=env.device)
    ph = env.cmd[key]
    return torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1) if use_trig else ph

def target_obs(env):
    joint_ids = [env.scene["steve"].data.joint_names.index(name) for name in joint_names]
    joints = env.scene["steve"].data.joint_pos[:,joint_ids]
    target = env.scene["steve"].data.joint_pos_target[:,joint_ids]
    return (target - joints)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})  # must match CommandsCfg field name
        root_linear_velocity  = ObsTerm(func=mdp.base_lin_vel,  params={"asset_cfg": SceneEntityCfg("steve")})
        root_angular_velocity = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("steve")})
        root_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("steve")})
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("steve",joint_names=joint_names)})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("steve",joint_names=joint_names)})
        target_joint_pos = ObsTerm(func=target_obs)
        phase = ObsTerm(func=phase_obs, )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

