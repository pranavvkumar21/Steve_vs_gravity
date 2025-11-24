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
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from joint_mappings import JOINT_MAPPING
from motion_manager import MotionManager


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
        self.decimation = 4
        self.sim.dt = 1.0 / (30 * self.decimation)
        self.sim.render_interval = self.decimation
        self.max_episode_length = 10 * 30  # 10 seconds
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
        
        print("mocap order")
        print(list(JOINT_MAPPING.keys()))
        print("robot joint order")
        print(env.scene["steve"].data.joint_names)
        default_root_pose = env.scene["steve"].data.root_link_pose_w
        # Playback loop
        for step in range(500):  # Run for many steps
            
            frame_idx = env.cmd["frame_idx"]
            # print(frame_idx)
            # Get current frame index from env commands


            # print("Current frame idx:", frame_idx)
            
            # Step simulation
            env.scene["steve"].write_root_pose_to_sim(default_root_pose)
            joint_pos = env.cmd["joint_position"]
            # joint_vel = torch.zeros_like(env.cmd["joint_velocity"])
            joint_vel = env.cmd["joint_velocity"]
            env.scene["steve"].write_joint_position_to_sim(joint_pos)
            env.scene["steve"].write_joint_velocity_to_sim(joint_vel)

            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))
            
            # Capture frame
            frame_img = rgb.get_data()
            writer.append_data(frame_img)
            
            # Print progress
            if step % 30 == 0:
                actual_pos = env.scene["steve"].data.joint_pos[0]
                # print(f"Step {step}, Reward: {rewards.item():.4f}, Actual Pos[0]: {actual_pos[0]:.4f}")
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
