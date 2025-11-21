#!/usr/bin/env python3
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import MassPropertiesCfg, modify_mass_properties

import torch

def init_cmd(env, env_ids, n_phase=4, n_joints=12):
    E = env.scene.num_envs
    d = env.device
    env.cmd = {}
    env.cmd["phase"] = torch.zeros(E, n_phase, device=d)
    env.cmd["target"] = env.scene["steve"].data.joint_pos.clone()
    env.cmd["frequency"] = torch.zeros(E, n_phase, device=d)
    env.cmd["prev_policy"] = torch.zeros(E, n_joints + n_phase, device=d)


def reset_cmd(env, env_ids):
    mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    pos = mdp.root_pos_w(env, SceneEntityCfg("steve"))

    # print("resetting env ids", env_ids, "pos:", pos[env_ids, :])


    
@configclass
class EventsCfg:
    startup_cmd = EventTerm(func=init_cmd, mode="startup", params={"n_phase": 4})
    # reset_cmd = EventTerm(func=reset_cmd, mode="reset", params={})
    # post_init = EventTerm(func=post_init, mode="reset", params={})
    # pre_startup = EventTerm(func=prestartup, mode="prestartup", params={})