#! /usr/bin/env python3


import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from steve_env import Steve_EnvCfg
import torch

def register_envs():
    gym.register(
        id="Steve-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"cfg": Steve_EnvCfg()},

    )