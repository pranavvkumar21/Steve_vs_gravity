#! /usr/bin/env python3
if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, livestream=0).app
    

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from Scene import Steve_SceneCfg
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import math
import sys
import yaml
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from joint_mappings import JOINT_MAPPING
from motion_manager import MotionManager
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

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
        self.decimation = 20
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
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        rp_path = vp.get_render_product_path()
        pyfiglet.figlet_format("Steve Demo", font="slant")
    except:
        print("Could not get replicator viewport. Running in headless mode.")


    # # Attach rgb annotator
    # import omni.replicator.core as rep
    # rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    # rgb.attach([rp_path])

    # # Video writer
    # writer = imageio.get_writer(str(ROOT / "videos" / "eval_run.mp4"), fps=30)

    try:
        cfg = Steve_EnvCfg()
        env = ManagerBasedRLEnv(cfg)
        default_root_pose = env.scene["steve"].data.root_link_pose_w
        obs = env.reset()
        
        print("mocap order")
        # print(list(JOINT_MAPPING.keys()))
        # print("robot joint order")s
        print('body names:')
        print(env.scene["steve"].data.body_names)
        print('joint names:')
        # print(env.scene["steve"].data.joint_names)
        print("joint_limits:")
        # print(env.scene["steve"].data.default_joint_pos_limits)
        
        #loop through joint motion manager and print joint names and corresponding
        #joint name from config
        print("isaac nucleus dir:", ISAAC_NUCLEUS_DIR)
        print("isaaclab nucleus dir:", ISAACLAB_NUCLEUS_DIR)
        config_jn = steve_config["scene"]["joint_names"]
        for idx in range(len(env.motion_manager.motions["walk"]['joint_names_ordered'])):
            name = env.motion_manager.motions["walk"]['joint_names_ordered'][idx]
            print(f"mocap joint name: {name} -> robot joint name: {config_jn[idx]}")
        # Playback loop
        for step in tqdm(range(1000), desc="Simulation Steps"):  # Run for many steps
            
            
            # frame_idx = env.cmd["frame_idx"]
            # print(frame_idx)
            # Get current frame index from env commands


            # print("Current frame idx:", frame_idx)
            
            # Step simulation
            # root_pose =default_root_pose.clone()
            # #use root orientation from commands
            # root_pose[:, 3:] = env.cmd["root_orientation"]
            
            joint_pos = env.cmd["joint_position"]
            joint_vel = torch.zeros_like(env.cmd["joint_velocity"])
            default_root_pose[:, 2] = 1.5
            # # joint_vel = env.cmd["joint_velocity"]
            # print("joint pos shape:", joint_pos.shape)
            # print("pose values:", joint_pos[0] )
            # env.scene["steve"].write_joint_position_to_sim(joint_pos, joint_ids = env.motion_manager.motions["walk"]['joint_indices'])
            env.scene["steve"].set_joint_position_target(joint_pos, joint_ids = env.motion_manager.motions["walk"]['joint_indices'])
            # env.scene["steve"].write_joint_velocity_to_sim(joint_vel, joint_ids = env.motion_manager.motions["walk"]['joint_indices'])
            # env.scene["steve"].write_root_pose_to_sim(default_root_pose)
            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))
            
            # Capture frame
            # frame_img = rgb.get_data()
            # writer.append_data(frame_img)
            
            # Print progress
            if step % 30 == 0:
                actual_pos = env.scene["steve"].data.joint_pos[0]
                # print(f"Step {step}, Reward: {rewards.item():.4f}, Actual Pos[0]: {actual_pos[0]:.4f}")
    except Exception as e:
        import traceback
        print("An exception occurred:", e)
        traceback.print_exc()
    finally:
        # writer.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
