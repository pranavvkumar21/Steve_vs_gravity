#! /usr/bin/env python3
if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True,livestream=2).app
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

joint_mapping = {
    # Torso
    'lower_waist:0': ('lowerback', 0),
    'lower_waist:1': ('lowerback', 1),
    'pelvis': ('root', 0),

    # Right arm
    'right_upper_arm:0': ('rhumerus', 0),
    'right_upper_arm:2': ('rhumerus', 2),
    'right_lower_arm': ('rradius', 0),

    # Left arm
    'left_upper_arm:0': ('lhumerus', 0),
    'left_upper_arm:2': ('lhumerus', 2),
    'left_lower_arm': ('lradius', 0),

    # Right leg (hip)
    'right_thigh:0': ('rfemur', 0),
    'right_thigh:1': ('rfemur', 1),
    'right_thigh:2': ('rfemur', 2),

    # Left leg (hip)
    'left_thigh:0': ('lfemur', 0),
    'left_thigh:1': ('lfemur', 1),
    'left_thigh:2': ('lfemur', 2),

    # Knee joints
    'right_shin': ('rtibia', 0),
    'left_shin': ('ltibia', 0),

    # Ankle joints
    'right_foot:0': ('rfoot', 0),
    'right_foot:1': ('rfoot', 2),
    'left_foot:0': ('lfoot', 0),
    'left_foot:1': ('lfoot', 2),
}

def map_range_torch(x, in_min, in_max, out_min, out_max):
    x_clipped = torch.clamp(x, in_min, in_max)
    return out_min + (x_clipped - in_min) * (out_max - out_min) / (in_max - in_min)

@configclass
class Steve_EnvCfg(ManagerBasedRLEnvCfg):
    scene = Steve_SceneCfg(num_envs=4,env_spacing=5.0)
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    actions = ActionsCfg()
    events = EventsCfg()
    commands = CommandsCfg()

    def __post_init__(self):
        #30 hz
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

    # 3 attach rgb annotator
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp_path])

    # 4 write frames during eval loop
    writer = imageio.get_writer(str(ROOT / "videos" / "eval_run.mp4"), fps=30)

    try:
        cfg = Steve_EnvCfg()
        env = ManagerBasedRLEnv(cfg)
        obs = env.reset()

        print("joint names:", env.scene["steve"].data.joint_names)
        print(env.scene["steve"].is_fixed_base)
        print("Environment reset successful, starting stepping...")
        print("body names")
        joint_names = env.scene["steve"].data.joint_names
        limits = env.scene["steve"].data.default_joint_pos_limits  # (n_env, n_joints, 2)
        mocap_limits = torch.from_numpy(np.load("../data/mocap_data/01/01/01_01_mapped_limits.npy")).float()  # (n_joints, 2)
        mocap_angles = torch.from_numpy(np.load("../data/mocap_data/01/01/01_01_mapped_joint_angles.npy")).float()  # (n_frames, n_joints)
        mocap_orientations = torch.from_numpy(np.load("../data/mocap_data/01/01/01_01_root_orientations.npy")).float()  # (n_frames, 4)
        # ordered_mocap_angles = ordered_mocap_angles.to(limits.device)
        mocap_limits = mocap_limits.to(limits.device)
        mocap_limits_joint_order = list(joint_mapping.keys())
        root_state = env.scene["steve"].data.default_root_state # (n_env, 13)
        reorder_map = []
        for name in joint_names:
            if name in mocap_limits_joint_order:
                reorder_map.append(mocap_limits_joint_order.index(name))
            else:
                reorder_map.append(None) # Or use -1 or another placeholder
        print("Reorder map:", reorder_map)
        n_frames = mocap_angles.shape[0]
        n_sim_joints = len(joint_names)
        ordered_mocap_angles = torch.zeros((n_frames, n_sim_joints), device="cuda" if torch.cuda.is_available() else "cpu")
        joints_to_set_zero = ['lower_waist:0', 'lower_waist:1', 'pelvis']
        joints_to_set_zero_indices = [joint_names.index(joint) for joint in joints_to_set_zero if joint in joint_names]
        for idx in joints_to_set_zero_indices:
            print(f"Setting mocap angles for joint {joint_names[idx]} at index {idx} to zero.")
            ordered_mocap_angles[:, idx] = 0.0
        print("Reordering mocap angles to match simulation joint order...")
        for i, mocap_idx in enumerate(reorder_map):
            if mocap_idx is not None:
                ordered_mocap_angles[:, i] = mocap_angles[:, mocap_idx]
            else:
                ordered_mocap_angles[:, i] = 0.0  # Or another default, as needed
        print("Reordering complete.")
        print(f"Sim joint order: {joint_names}")
        print(f"Mocap joint order: {mocap_limits_joint_order}")
        print(f"Sim limits shape: {limits.shape}")  # (n_env, n_joints, 2)
        print(f"Mocap limits shape: {mocap_limits.shape}")  # (n_joints, 2)
        

        # Use first environment's limits for comparison
        limits_env0 = limits[0]  # (n_joints, 2)

        print("\nJoint Limits Comparison:")
        print(f"{'Joint Name':<30} {'Sim Min':>10} {'Sim Max':>10} {'Mocap Min':>12} {'Mocap Max':>12}")
        print("-" * 76)
        #create joint posiotion zero tensor
        joint_positions_zero = torch.zeros_like(limits[:,:,0])
        joint_velocities_zero = torch.zeros_like(limits[:,:,0])
        # env.scene["steve"].write_joint_state_to_sim(joint_positions_zero, joint_velocities_zero)

        # env.scene["steve"].write_root_state_to_sim(root_state)
        for i, joint_name in enumerate(joint_names):
            sim_min = limits_env0[i, 0].item()
            sim_max = limits_env0[i, 1].item()
            
            if joint_name in joint_mapping:
                mocap_joint_idx = mocap_limits_joint_order.index(joint_name)
                mocap_min = mocap_limits[mocap_joint_idx, 0].item()
                mocap_max = mocap_limits[mocap_joint_idx, 1].item()
                print(f"{joint_name:<30} {sim_min:>10.2f} {sim_max:>10.2f} {mocap_min:>12.2f} {mocap_max:>12.2f}")
            else:
                print(f"{joint_name:<30} {sim_min:>10.2f} {sim_max:>10.2f} {'N/A':>12} {'N/A':>12}")
        print(f"has debug vis implement: {env.scene['steve'].has_debug_vis_implementation}")
        #tessting joint
        joint_to_test = ["left_upper_arm:2"]  # Change to desired joint name


        joint_indices = [joint_names.index(name) for name in joint_to_test]
        print(f"Testing joint: {joint_to_test} at index {joint_indices}")

        def sine_waves(limits, joint_indices, t, frequency=0.7 ):
            """ for the given joint indices, return list of sine wave positions """
            sines = []
            for joint_index in joint_indices:
                min_limit = limits[joint_index][0].item()
                max_limit = limits[joint_index][1].item()
                amplitude = (max_limit - min_limit) / 2.0
                center = (max_limit + min_limit) / 2.0
                sines.append(center + amplitude * math.sin(2 * 3.14159 * frequency * t))
            return sines
        # ordered_mocap_angles = map_range_torch(ordered_mocap_angles, mocap_limits[:,0], mocap_limits[:,1], limits_env0[:,0], limits_env0[:,1])
        for step in range(100000):
            with torch.inference_mode():
                #get every 4th frame to match 30Hz
                # print("Step:", step)
                # print(ordered_mocap_angles.device)
                frame_ = ordered_mocap_angles[(step * 4) % n_frames, :]

                frame = torch.clamp(frame_, limits_env0[:,0], limits_env0[:,1])
                #repeat for all envs
                frame = frame.unsqueeze(0).repeat(env.num_envs,1)
                # print("Final frame shape to apply:", frame.shape)
                #create torch sine wave for testing for a single joint oscillate between sim limits of that joint

                # print(frame.shape)
                # print(frame[0])
                # env.scene["steve"].set_joint_position_target(frame, joint_ids=[env.scene["steve"].data.joint_names.index(name) for name in joint_names])
                # get testing joint position
                test_joint_pos = sine_waves(limits_env0, joint_indices, step * cfg.sim.dt)
                joint_angles = torch.zeros_like(frame)
                for idx, joint_index in enumerate(joint_indices):
                    joint_angles[:, joint_index] = test_joint_pos[idx]
                    # print(f"Step {step}: Setting joint {joint_names[joint_index]} to position {test_joint_pos[idx]:.4f}")
                env.scene["steve"].set_joint_position_target(joint_angles, joint_ids=[env.scene["steve"].data.joint_names.index(name) for name in joint_names])
                #ignore frame just set testing joint
                # root_state[ :,3:7] = mocap_orientations[(step * 4) % n_frames, :]
                env.scene["steve"].write_root_state_to_sim(root_state)
                obs, rewards, dones, trunc, info = env.step(torch.randn_like(env.action_manager.action))
                actual_pos = env.scene["steve"].data.joint_pos[0, joint_indices[0]].item()
                target_pos = test_joint_pos[0]
                print(f"Step {step}: Target={target_pos:.4f}, Actual={actual_pos:.4f}, Error={abs(target_pos - actual_pos):.4f}")
                frame = rgb.get_data()
                writer.append_data(frame)

                # l_foot_data = env.scene.sensors["l_foot_contact"].data
                # r_foot_data = env.scene.sensors["r_foot_contact"].data
                # l_forces = torch.linalg.norm(l_foot_data.net_forces_w, dim=-1).squeeze(-1)
                # r_forces = torch.linalg.norm(r_foot_data.net_forces_w, dim=-1).squeeze(-1)
                # print("Step:", step, "L foot forces:", l_forces.shape, "R foot forces:", r_forces.shape)
                # print("lfoot data shape:", l_foot_data.net_forces_w.shape)
                # print("rfoot data shape:", r_foot_data.net_forces_w.shape)
                # print("Step:", step, "Rewards:", rewards, "Dones:", dones)
                # print(step)

    except Exception as e:
        print("An exception occurred:", e)
    finally:
        writer.close()
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()