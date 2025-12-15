#! /usr/bin/env python3
if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, livestream=2).app
    

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
from utils import SkeletonVisualizer
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
from motion_manager import MotionManager
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

def map_range_torch(x, in_min, in_max, out_min, out_max):
    x_clipped = torch.clamp(x, in_min, in_max)
    return out_min + (x_clipped - in_min) * (out_max - out_min) / (in_max - in_min)


@configclass
class Steve_EnvCfg(ManagerBasedRLEnvCfg):
    scene = Steve_SceneCfg(num_envs=1, env_spacing=5.0)
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
        # for evaluation
        self.max_episode_length = 40 * 30  # 40 seconds
        self.episode_length_s = 40
        # to be used in training
        # self.max_episode_length = 3 * 30  # 3 seconds
        # self.episode_length_s = 3
        self.viewer.enable = True
        self.viewer.resolution = (1280, 720)
        self.viewer.eye = (4, 4, 2)
        self.viewer.lookat = (0.0, 0.0, 0.5)


def main():


    # # Attach rgb annotator
    import omni.replicator.core as rep

    # # Video writer
    writer = imageio.get_writer(str(ROOT / "videos" / "eval_run.mp4"), fps=30)

    try:
        cfg = Steve_EnvCfg()
        env = ManagerBasedRLEnv(cfg)
        default_root_pose = env.scene["steve"].data.root_link_pose_w
        obs = env.reset()
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        rep_path = viewport.render_product_path
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        print("RGB annotator path:", rep_path)
        # 
        # for i in range(10):
        #     env.step(torch.zeros_like(env.action_manager.action))
        # rgb.attach([rep_path])

        env.skeleton_viz = SkeletonVisualizer(env.motion_manager.motions[env.motion_name]['link_body_names'], device=env.device)

        config_jn = steve_config["joint_names"]
        for idx in range(len(env.motion_manager.motions[env.motion_name]['joint_names'])):
            name = env.motion_manager.motions[env.motion_name]['joint_names'][idx]
            # print(f"mocap joint name: {name} -> robot joint name: {config_jn[idx]}")

        for step in tqdm(range(1000), desc="Simulation Steps"):  # Run for many steps
            # Debug 2: Individual term values

            # terms = env.reward_manager.get_active_iterable_terms(0)
            # terms is a sequence of tuples (term_name, value)
            # for term_name, value in terms:
            #     print(f"  Term: {term_name}, Value: {value}")

            
            # Step simulation
            root_pose = default_root_pose.clone()
            
            # Use the current frame data that's already been updated by action manager
            current_joint_pos = env.cmd["joint_position"][0]
            current_root_orient = env.cmd["root_orientation"][0]
            
            root_pose = env.scene["steve"].data.root_link_pose_w.clone()
            # env.scene["steve"].set_joint_position_target(current_joint_pos, joint_ids = env.motion_manager.motions[env.motion_name]['joint_indices'])
            root_pose[:, 3:] = current_root_orient
            # env.scene["steve"].write_root_pose_to_sim(root_pose)
            body_pos_w = env.scene["steve"].data.body_pos_w.clone()
            local_body_pos_cmd = env.cmd["local_body_position"][0]

            
            # Synchronized skeleton visualization
            current_frame_idx = int(env.cmd["frame_idx"][0].item())
            current_pose = env.cmd["local_body_position"][0].clone()
            root_pos = env.cmd["root_position"][0].clone()
            
            # Add root position to current pose  
            current_pose[:, 0] += root_pos[0]
            current_pose[:, 1] += root_pos[1]
            current_pose[:, 2] += root_pos[2]

            env.skeleton_viz.draw(current_pose)
            # obs, rewards, dones, trunc, info = env.step(env.cmd["joint_position"])
            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))

            # # Capture frame
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
        writer.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
