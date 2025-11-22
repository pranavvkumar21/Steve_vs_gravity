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


        print("Environment reset successful!")
        joint_names = env.scene["steve"].data.joint_names
        print(f"Joint names: {joint_names}")
        
        # Load mocap data
        mocap_limits = torch.from_numpy(
            np.load("../data/mocap_data/01/01/01_01_mapped_limits.npy")
        ).float()  # (n_mocap_joints, 2)
        mocap_angles = torch.from_numpy(
            np.load("../data/mocap_data/01/01/01_01_mapped_joint_angles.npy")
        ).float()  # (n_frames, n_mocap_joints)
        mocap_orientations = torch.from_numpy(
            np.load("../data/mocap_data/01/01/01_01_root_orientations.npy")
        ).float()  # (n_frames, 4)
        
        # Move to device
        mocap_limits = mocap_limits.to("cuda")
        mocap_angles = mocap_angles.to("cuda")
        mocap_orientations = mocap_orientations.to("cuda")
        
        # Get simulation limits and root state
        limits = env.scene["steve"].data.default_joint_pos_limits[0]  # (n_joints, 2)
        root_state = env.scene["steve"].data.default_root_state  # (n_env, 13)
        
        # Create reorder map from mocap joints to sim joints
        mocap_joint_order = list(JOINT_MAPPING.keys())
        n_frames = mocap_angles.shape[0]
        n_sim_joints = len(joint_names)
        
        # Create mapping indices
        reorder_map = []
        for name in joint_names:
            if name in mocap_joint_order:
                reorder_map.append(mocap_joint_order.index(name))
            else:
                reorder_map.append(None)
        
        print(f"\nReorder map: {reorder_map}")
        print(f"Sim joints: {n_sim_joints}, Mocap joints: {len(mocap_joint_order)}")
        
        # Reorder mocap angles to match sim joint order
        ordered_mocap_angles = torch.zeros((n_frames, n_sim_joints), device="cuda")
        
        # Joints to set to zero (abdomen/neck - you may want to enable these)
        joints_to_set_zero = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z']
        joints_to_set_zero_indices = [
            joint_names.index(joint) for joint in joints_to_set_zero if joint in joint_names
        ]
        
        for idx in joints_to_set_zero_indices:
            print(f"Setting joint {joint_names[idx]} to zero")
            ordered_mocap_angles[:, idx] = 0.0
        
        # Map mocap data to sim joints
        for i, mocap_idx in enumerate(reorder_map):
            if mocap_idx is not None and joint_names[i] not in joints_to_set_zero:
                ordered_mocap_angles[:, i] = mocap_angles[:, mocap_idx]
            elif joint_names[i] not in joints_to_set_zero:
                ordered_mocap_angles[:, i] = 0.0
        
        print("\nJoint Limits Comparison:")
        print(f"{'Joint Name':<25} {'Sim Min':>10} {'Sim Max':>10} {'Mocap Min':>12} {'Mocap Max':>12}")
        print("-" * 79)
        
        for i, joint_name in enumerate(joint_names):
            sim_min = limits[i, 0].item()
            sim_max = limits[i, 1].item()
            
            if joint_name in mocap_joint_order:
                mocap_joint_idx = mocap_joint_order.index(joint_name)
                mocap_min = mocap_limits[mocap_joint_idx, 0].item()
                mocap_max = mocap_limits[mocap_joint_idx, 1].item()
                print(f"{joint_name:<25} {sim_min:>10.2f} {sim_max:>10.2f} {mocap_min:>12.2f} {mocap_max:>12.2f}")
            else:
                print(f"{joint_name:<25} {sim_min:>10.2f} {sim_max:>10.2f} {'N/A':>12} {'N/A':>12}")
        
        print(f"\nStarting playback of {n_frames} frames...")
        
        # Playback loop
        for step in range(10000):  # Run for many steps
            # Get frame index (every 4th mocap frame for 30Hz)
            frame_idx = (step * 4) % n_frames
            
            # Get mocap frame and clamp to sim limits
            frame = ordered_mocap_angles[frame_idx, :]
            frame_clamped = torch.clamp(frame, limits[:, 0], limits[:, 1])
            
            # Repeat for all environments
            frame_batch = frame_clamped.unsqueeze(0).repeat(env.num_envs, 1)
            
            # Set joint positions
            stiffness = 200.0
            damping = 20.0
            current_positions = env.scene["steve"].data.joint_pos
            position_errors = frame_batch - current_positions
            torque_commands = stiffness * position_errors - damping * env.scene["steve"].data.joint_vel
            env.scene["steve"].set_joint_effort(torque_commands)
            env.scene["steve"].set_joint_position_target(frame_batch)
            
            # Set root orientation from mocap
            root_state_copy = root_state.clone()
            root_state_copy[:, 3:7] = mocap_orientations[frame_idx, :].unsqueeze(0)
            env.scene["steve"].write_root_state_to_sim(root_state_copy)
            
            # Step simulation
            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))
            
            # Capture frame
            frame_img = rgb.get_data()
            writer.append_data(frame_img)
            
            # Print progress
            if step % 30 == 0:
                actual_pos = env.scene["steve"].data.joint_pos[0]
                max_error = torch.max(torch.abs(frame_clamped - actual_pos)).item()
                print(f"Step {step:4d}, Frame {frame_idx:4d}/{n_frames}, Max error: {max_error:.4f}")

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
