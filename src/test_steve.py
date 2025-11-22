#! /usr/bin/env python3
if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, livestream=2).app
    import omni.replicator.core as rep

import torch
import time
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from Managers.Observations import ObservationsCfg
from Managers.Rewards import RewardsCfg
from Managers.Terminations import TerminationsCfg
from Managers.Actions import ActionsCfg
from Managers.Events import EventsCfg
from Managers.Commands import CommandsCfg
from Scene import Steve_SceneCfg
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent

# Updated joint mapping for HUMANOID_28_CFG
JOINT_MAPPING = {
    'abdomen_x': ('lowerback', 'rx'),
    'abdomen_y': ('lowerback', 'ry'),
    'abdomen_z': ('lowerback', 'rz'),
    'neck_x': ('lowerneck', 'rx'),
    'neck_y': ('lowerneck', 'ry'),
    'neck_z': ('lowerneck', 'rz'),
    'right_shoulder_x': ('rhumerus', 'rx'),
    'right_shoulder_y': ('rhumerus', 'ry'),
    'right_shoulder_z': ('rhumerus', 'rz'),
    'left_shoulder_x': ('lhumerus', 'rx'),
    'left_shoulder_y': ('lhumerus', 'ry'),
    'left_shoulder_z': ('lhumerus', 'rz'),
    'right_elbow': ('rradius', 'rx'),
    'left_elbow': ('lradius', 'rx'),
    'right_hip_x': ('rfemur', 'rx'),
    'right_hip_y': ('rfemur', 'ry'),
    'right_hip_z': ('rfemur', 'rz'),
    'left_hip_x': ('lfemur', 'rx'),
    'left_hip_y': ('lfemur', 'ry'),
    'left_hip_z': ('lfemur', 'rz'),
    'right_knee': ('rtibia', 'rx'),
    'left_knee': ('ltibia', 'rx'),
    'right_ankle_x': ('rfoot', 'rx'),
    'right_ankle_y': None,  # Not in mocap data
    'right_ankle_z': ('rfoot', 'rz'),
    'left_ankle_x': ('lfoot', 'rx'),
    'left_ankle_y': None,  # Not in mocap data
    'left_ankle_z': ('lfoot', 'rz'),
}


def map_range_torch(x, in_min, in_max, out_min, out_max):
    x_clipped = torch.clamp(x, in_min, in_max)
    return out_min + (x_clipped - in_min) * (out_max - out_min) / (in_max - in_min)


@configclass
class Steve_EnvCfg(ManagerBasedRLEnvCfg):
    scene = Steve_SceneCfg(num_envs=4, env_spacing=5.0)
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    actions = ActionsCfg()
    events = EventsCfg()
    commands = CommandsCfg()

    def __post_init__(self):
        # 30 hz
        self.decimation = 5
        self.sim.dt = 1.0 / (30 * self.decimation)
        self.sim.render_interval = self.decimation
        self.max_episode_length = 1000
        self.episode_length_s = 10
        self.viewer.enable = True
        self.viewer.resolution = (1280, 720)
        self.viewer.eye = (8, 8, 8)
        self.viewer.lookat = (0.0, 0.0, 0.5)


def main():
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        rp_path = vp.get_render_product_path()
    except Exception:
        import omni.kit.viewport_legacy as vp_legacy
        vp_win = vp_legacy.get_viewport_interface().get_viewport_window()
        rp_path = vp_win.get_render_product_path()

    # Attach rgb annotator
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp_path])

    # Video writer
    writer = imageio.get_writer(str(ROOT / "videos" / "eval_run.mp4"), fps=30)

    try:
        cfg = Steve_EnvCfg()
        env = ManagerBasedRLEnv(cfg)
        obs = env.reset()

        #get joint names, joint limits
        joint_names = env.scene["steve"].data.joint_names
        joint_limits = env.scene["steve"].data.default_joint_pos_limits[0]
        print("="*20+" Joint Limits "+"="*20)
        for name, limits in zip(joint_names, joint_limits):
            print(f"{name}: {limits[0]:.2f} to {limits[1]:.2f}")

        # Playback loop
        for step in range(1000):
            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))  # Run for many steps


    except Exception as e:
        import traceback
        print("An exception occurred:", e)
        traceback.print_exc()
    finally:
        writer.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
