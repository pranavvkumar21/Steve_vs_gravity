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

        # After env.reset()
        # After env.reset()
        import math

        joint_names = env.scene["steve"].data.joint_names
        print(joint_names)
        
        limits = env.scene["steve"].data.default_joint_pos_limits[0]
        root_state  = env.scene["steve"].data.default_root_state
        shoulder_joints = [
            'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'
        ]
        shoulder_indices = [joint_names.index(name) for name in shoulder_joints]

        print("\n" + "="*80)
        print("SHOULDER JOINT LIMITS")
        print("="*80)
        for i, name in enumerate(shoulder_joints):
            idx = shoulder_indices[i]
            print(f"{name:<20} limits=[{limits[idx, 0]:7.4f}, {limits[idx, 1]:7.4f}]")

        for step in range(400):
            joint_angles = torch.zeros((env.num_envs, len(shoulder_joints)), device=limits.device)
            
            for i, idx in enumerate(shoulder_indices):
                min_limit = limits[idx, 0].item()
                max_limit = limits[idx, 1].item()
                amplitude = (max_limit - min_limit) / 2.0
                center = (max_limit + min_limit) / 2.0
                
                freq = 0.1  # Different frequency for each joint
                target = center + amplitude * math.sin(2 * 3.14159 * freq * step * cfg.sim.dt)
                
                # IMPORTANT: Clamp to limits!
                target = max(min_limit, min(max_limit, target))
                joint_angles[:, i] = target
            
            env.scene["steve"].set_joint_position_target(joint_angles, joint_ids=shoulder_indices)
            env.scene["steve"].write_root_state_to_sim(root_state)
            obs, _, _, _, _ = env.step(torch.zeros_like(env.action_manager.action))
            
            if step % 30 == 0:
                actual_pos = env.scene["steve"].data.joint_pos[0, shoulder_indices]
                print(f"\nStep {step:3d}:")
                for i, name in enumerate(shoulder_joints):
                    target = joint_angles[0, i].item()
                    actual = actual_pos[i].item()
                    error = abs(target - actual)
                    status = "✓" if error < 0.1 else "✗"
                    print(f"  {status} {name:<20} Target={target:7.4f}, Actual={actual:7.4f}, Error={error:.4f}")

    except Exception as e:
        print("An exception occurred:", e)
    finally:
        writer.close()
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()