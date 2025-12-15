import argparse
import sys
import os
import time
from pathlib import Path

# --- Argument Parsing ---
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["train", "eval"], default="eval", help="Mode: train or eval")
p.add_argument("--load", action="store_true", help="Load model if flag is set")
p.add_argument("--checkpoint", type=str, default=None, help="Path to specific .pt file to load")
args = p.parse_args()

print(f"Running in {args.mode} mode. Load flag is set to {args.load}")
time.sleep(1)

# --- Isaac Sim App Launch ---
# Must be done before importing torch/gym/isaaclab modules
if args.mode == "train":
    kit_args = "--/log/level=error"
    livestream = 0
    enable_cameras = False
else:
    livestream = 2
    kit_args = "--/log/level=warning"
    enable_cameras = True

from isaaclab.app import AppLauncher
# Launch the simulator
simulation_app = AppLauncher(headless=True, livestream=livestream, enable_cameras=enable_cameras, kit_args=kit_args).app

# --- Imports ---
import gymnasium as gym
import torch
import yaml
import numpy as np

# Isaac Lab Imports
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from Registration import register_envs
from steve_env import Steve_EnvCfg
import imageio.v2 as imageio
# RSL_RL Import (The correct location for the runner)
from rsl_rl.runners import OnPolicyRunner

# Import your config generator
from runner import create_runner_cfg

ROOT = Path(__file__).resolve().parent.parent

# Load environment configs
with open(ROOT / "config" / "env_config.yaml", "r") as f:
    env_config = yaml.safe_load(f)

with open(ROOT / "config" / "steve_config.yaml", "r") as f:
    config = yaml.safe_load(f)

def config_env(cfg):
    """Configures the environment based on mode (train/eval)."""
    cfg.scene.num_envs = env_config[args.mode]["num_envs"]

    if hasattr(cfg.scene, "terrain_importer") and hasattr(cfg.scene.terrain_importer, "terrain_generator"):
        cfg.scene.terrain_importer.terrain_generator.rows = config[args.mode]["rows"]
        cfg.scene.terrain_importer.terrain_generator.cols = config[args.mode]["cols"]
        cfg.scene.terrain_importer.terrain_generator.size = (config["scene"]["env_spacing"], config["scene"]["env_spacing"])

    if args.mode == "eval":
        pass
        # cfg.commands.velocity_command.ranges.lin_vel_x = (0.5, 0.5) # Fixed speed for eval
        # cfg.commands.velocity_command.ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands.velocity_command.ranges.ang_vel_z = (0.0, 0.0)

    cfg.scene.env_spacing = env_config["scene"]["env_spacing"]
    cfg.seed = env_config["env_config"]["seed"]
    return cfg

def main():
    # 1. Register and Create Environment
    register_envs()
    cfg = Steve_EnvCfg()
    cfg = config_env(cfg)

    print("Creating environment...")
    # Render mode is handled by Isaac Sim, usually None or 'rgb_array' for wrappers
    env = gym.make("Steve-v0", cfg=cfg, render_mode="rgb_array" if args.mode == "eval" else None)
    
    # 2. Wrap Environment for RSL_RL
    # This wrapper handles the API translation between Isaac Lab and RSL_RL
    env = RslRlVecEnvWrapper(env)

    # 3. Prepare Configuration
    # Get the config object from your runner.py
    runner_cfg_obj = create_runner_cfg()
    
    # Convert config object to dictionary (required by rsl_rl)
    # Isaac Lab config objects typically have a to_dict() method
    if hasattr(runner_cfg_obj, "to_dict"):
        agent_cfg = runner_cfg_obj.to_dict()
    else:
        # Fallback if it's already a dict or simple object
        agent_cfg = vars(runner_cfg_obj)

    # Override log directory to ensure it goes where you expect
    
    log_dir = Path(ROOT / "logs" / "steve_kick")
    agent_cfg["log_root_path"] = str(log_dir)

    # 4. Initialize RSL_RL Runner
    print(f"Initializing RSL_RL Runner with device: {env.device}")
    runner = OnPolicyRunner(
        env,
        agent_cfg,
        log_dir=log_dir,
        device=env.device
    )

    # 5. Load Model (if needed)
    if args.load or args.mode == "eval":
        resume_path = None
        if args.checkpoint:
            resume_path = args.checkpoint
        else:
            # Try to find the latest model in the experiment folder
            experiment_dir = log_dir 
            if experiment_dir.exists():
                # get latest .pt file its named  with model_<iteration>.pt
                pt_files = list(experiment_dir.glob("model_*.pt"))
                if pt_files:
                    latest_model = max(pt_files, key=os.path.getctime)
                    resume_path = str(latest_model)
                    print(f"Found latest model: {resume_path}")
                else:
                    print(f"Warning: No .pt files found in {experiment_dir}.")

                # resume_path = str(experiment_dir)
            else:
                print(f"Warning: Experiment directory {experiment_dir} not found.")
        
        if resume_path:
            print(f"Loading model from: {resume_path}")
            # load() signature: load(path, load_optimizer=True)
            runner.load(resume_path)
            runner.alg.value_loss_coef = 1.0
            print("Model loaded successfully.")
            print(f"Set value_loss_coef to {runner.alg.value_loss_coef}")

    # 6. Execution Loop
    if args.mode == "train":
        print("Starting training...")
        runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
        print("Training finished.")

    elif args.mode == "eval":
        print("Starting evaluation...")
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = env.reset()

        # Import the viewport utility (Standard Kit extension, always there)
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer
        # from omni.isaac.core.utils.viewports import set_camera_view
        from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController
        # import ctypes

        # Setup Video Writer
        video_path = str(ROOT / "videos" / f"steve_run_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(f"Recording to {video_path}...")
        writer = imageio.get_writer(video_path, fps=30)

        # Helper to store the captured frame
        # We use a list to make it mutable in the callback
        last_frame = [None] 

        import ctypes
        
        # Helper to get pointer from capsule
        # We use python's C-API via ctypes to call PyCapsule_GetPointer
        pyapi = ctypes.pythonapi
        pyapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        pyapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        def on_capture(buffer, size, width, height, format):
            try:
                # Extract pointer from PyCapsule
                # The name arg (2nd) must match what C++ set, or be None. 
                # usually None works for generic buffers.
                ptr = pyapi.PyCapsule_GetPointer(buffer, None)
                
                # Now read
                img_bytes = ctypes.string_at(ptr, size)
                
                img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
                last_frame[0] = img[:, :, :3]
            except Exception as e:
                print(f"Capture failed: {e}")



        # Get the viewport
        viewport_api = get_active_viewport()
        if not viewport_api:
            print("ERROR: No active viewport! Cannot record.")
            return

        base_env = env.unwrapped
        # camera_controller = ViewportCameraController(base_env, viewport_api)
        with torch.no_grad():
            for i in range(500):
                # 1. Step Physics
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                
                try:
                    # Access the underlying base environment
                    # RslRlVecEnvWrapper -> DirectRLEnv/ManagerBasedRLEnv

                    
                    # Get Robot Root Position
                    # In ManagerBasedRLEnv, the robot is usually at scene["robot"] or similar.
                    # Adjust 'steve' to match your asset name in SceneCfg
                    # root_state is usually (num_envs, 13) -> [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                    robot_pos = base_env.scene["steve"].data.root_pos_w[0, :3].cpu().numpy()
                    
                    # Calculate Camera Position
                    # Follow from 5m back, 2m side, 2m up
                    camera_offset = np.array([5.0, 1.0, 2.0]) 
                    eye = robot_pos + camera_offset
                    
                    # Look at the robot
                    cam_target = robot_pos # Look at torso/head height
                    
                    base_env.viewport_camera_controller.update_view_location(eye=eye, lookat=cam_target)
                    
                except Exception as e:
                    # Print once if it fails so we don't spam
                    if i == 0: print(f"Camera tracking failed: {e}")

                
                # 3. Request Capture
                # This queues a capture request for the *current* frame
                capture_viewport_to_buffer(viewport_api, on_capture)
                
                # 4. Wait/Sync (Hack for synchronous recording)
                # Since capture is async, we ideally wait, but usually update() triggers it.
                # If last_frame is None, we skip writing (or duplicate prev frame)
                if last_frame[0] is not None:
                    writer.append_data(last_frame[0])
                    last_frame[0] = None # Clear it
                
                if i % 50 == 0:
                    print(f"Step {i}")

        writer.close()
        print("Done.")


    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
