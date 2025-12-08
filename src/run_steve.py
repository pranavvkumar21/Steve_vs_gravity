
import argparse
import time

p = argparse.ArgumentParser()

# define command line arguments
p.add_argument("--mode", choices=["train", "eval"], default="eval", help="Mode: train or eval")
p.add_argument("--load", action="store_true", help="Load model if flag is set")
p.add_argument("--bc", action="store_true", help="Use behavior cloning if flag is set")
args = p.parse_args()

print(f"Running in {args.mode} mode. Load flag is set to {args.load}")
time.sleep(2)

import os
from pathlib import Path
import yaml
from tabulate import tabulate
import pyfiglet


if args.mode == "train":
    kit_args="--/log/level=error"
    livestream=0
    enable_cameras = False
else:
    livestream=2
    kit_args="--/log/level=warning"
    enable_cameras = True

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True, livestream=livestream, enable_cameras=enable_cameras, kit_args=kit_args).app


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import gymnasium as gym
from Registration import register_envs
from steve_env import Steve_EnvCfg
import imageio.v2 as imageio



from callbacks import TensorboardCallback
from callbacks import create_checkpoint_callback, get_latest_model_path

import numpy as np
import yaml
from pathlib import Path
import os
from tqdm import tqdm
from natsort import natsorted
import torch

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "env_config.yaml", "r") as f:
    env_config = yaml.safe_load(f)
with open(ROOT / "config" / "steve_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = ROOT / "models"


def config_env(cfg):
    cfg.scene.num_envs = env_config[args.mode]["num_envs"]

    #if terrain generator exists, set its rows and cols from env_config
    if hasattr(cfg.scene, "terrain_importer") and hasattr(cfg.scene.terrain_importer, "terrain_generator"):
        cfg.scene.terrain_importer.terrain_generator.rows = config[args.mode]["rows"]
        cfg.scene.terrain_importer.terrain_generator.cols = config[args.mode]["cols"]
        cfg.scene.terrain_importer.terrain_generator.size = (config["scene"]["env_spacing"], config["scene"]["env_spacing"])

    if args.mode == "eval":
        cfg.commands.velocity_command.ranges.lin_vel_x = (0.3, 1.0)
        cfg.commands.velocity_command.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.velocity_command.ranges.ang_vel_z = (0.2, 0.5)

    cfg.scene.env_spacing = env_config["scene"]["env_spacing"]
    cfg.seed = env_config["env_config"]["seed"]
    return cfg

def config_train(model):
    model.learning_rate = get_linear_fn(env_config["train"]["learning_rate"]["start"], 
        env_config["train"]["learning_rate"]["end"], 
        env_config["train"]["learning_rate"]["end_fraction"]
    )
    model.batch_size = env_config["train"]["batch_size"]
    model.n_epochs = env_config["train"]["n_epochs"]
    model.gamma = env_config["train"]["gamma"]
    model.ent_coef = env_config["train"]["ent_coef"]
    model.clip_range = lambda _: env_config["train"]["clip_range"] 
    model.target_kl = env_config["train"]["target_kl"]
    model.n_steps = env_config["train"]["n_steps"]

    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=model.learning_rate(1), 
        **model.policy.optimizer_kwargs
    )
    return model

def print_model_info(model):
    data = [
        ["Algorithm", type(model).__name__],
        ["Policy", model.policy.__class__.__name__],
        ["Num timesteps", model.num_timesteps],
        ["Learning rate", model.learning_rate],
        ["Gamma", model.gamma],
        ["Clip range", getattr(model, "clip_range", "N/A")],
        ["Ent coef", getattr(model, "ent_coef", "N/A")],
        ["VF coef", getattr(model, "vf_coef", "N/A")],
        ["Batch size", getattr(model, "batch_size", "N/A")],
        ["N steps", getattr(model, "n_steps", "N/A")],
    ]
    print(tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

def main():

    # Register environments
    register_envs()
    cfg = Steve_EnvCfg()
    cfg = config_env(cfg)
    
    # Create the environment
    env = gym.make("Steve-v0", cfg=cfg, render_mode="rgb_array" if args.mode == "eval" else None)
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    obs = env.reset()

    # If training with behavior cloning, wrap with VecNormalize and fit running mean/std
    if not args.load and args.bc:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print("Fitting vec env running mean/std...")
        for _ in tqdm(range(1000), desc="Fitting running mean/std"):  # Collect ~1000 steps to fit running mean/std
            actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            obs, _, dones, _, = env.step(actions)
        print("Done fitting vec env running mean/std.")

    if args.bc:
        from bc import generate_expert_trajectories_wrapped as generate_expert_trajectories
        from imitation.algorithms.bc import BC

        print(pyfiglet.figlet_format("---Behavior Cloning Mode---", font="slant"))
        time.sleep(2)

        # Generate expert trajectories for behavior cloning
        print("Generating expert trajectories...")
        expert_demos = generate_expert_trajectories(env, num_trajectories=400, action_scale=20*torch.pi/180)

        #print some stats about the expert demos
        print(f"Generated {len(expert_demos)} expert trajectories.")
        traj = expert_demos[0]
        print(f"Trajectory length: {len(traj.obs)}")
        print(f"Action shape: {traj.acts[0].shape}")
        print(f"Action range: [{traj.acts.min():.3f}, {traj.acts.max():.3f}]")
        print(f"Action mean: {traj.acts.mean():.3f}")

        # Define BC model
        rng = np.random.default_rng(42)
        policy_kwargs = dict(net_arch=[1024, 512, 256], activation_fn=torch.nn.ReLU)
        lr_schedule = lambda _: 1e-3
        policy = ActorCriticPolicy(env.observation_space, env.action_space, net_arch=policy_kwargs['net_arch'], lr_schedule=lr_schedule)
        bc_model = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_demos,
            batch_size=64,
            rng = rng,
            policy=policy,
        )

            # Train BC model
        print("Starting behavior cloning training...")
        bc_model.train(n_epochs=200)

        # Save BC model
        torch.save(bc_model.policy.state_dict(), str(MODEL_PATH / "bc_policy_state_dict.pt"))
        print("Saved BC policy to 'models/bc_policy.pt'")
        
        # env.close()
        # simulation_app.close()


    if args.mode == "eval":
        import omni.replicator.core as rep
        rp_path = Steve_EnvCfg.viewer.cam_prim_path
        # 1 set up video writer
        writer = imageio.get_writer(str(ROOT / "videos" / "eval_run.mp4"), fps=30)
        # 2 attach rgb annotator
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp_path])

    if not args.load :

        # Define policy architecture
        policy_kwargs = dict(
            net_arch=dict(
                pi=[1024, 512, 256],  # Policy (Actor) network layers
                vf=[1024, 512, 256]   # Value Function (Critic) network layers
            ),
            log_std_init=-2.0,
            full_std=True
        )

        # Create PPO model
        model = PPO(
            "MlpPolicy", 
            env, 
            n_steps=env_config["train"]["n_steps"],
            batch_size=env_config["train"]["batch_size"],
            learning_rate=get_linear_fn(env_config["train"]["learning_rate"]["start"], 
                env_config["train"]["learning_rate"]["end"], 
                env_config["train"]["learning_rate"]["end_fraction"]
            ),
            clip_range=env_config["train"]["clip_range"],
            gamma=env_config["train"]["gamma"],
            ent_coef=env_config["train"]["ent_coef"],
            n_epochs=env_config["train"]["n_epochs"],
            verbose=1,
            target_kl=env_config["train"]["target_kl"],
            use_sde=False,
            sde_sample_freq=4,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            vf_coef=env_config["train"]["vf_coef"],  # Adjust the policy architecture if needed
            tensorboard_log=str(ROOT / "logs/ppo_pybullet_tensorboard/"),
        )

        # Load BC policy weights into PPO model if bc flag is set
        if args.bc:
            print("Loading BC policy weights into PPO model...")
            if os.path.exists(MODEL_PATH / "bc_policy_state_dict.pt"):
                bc_policy_state_dict = torch.load(str(MODEL_PATH / "bc_policy_state_dict.pt"))
                model.policy.load_state_dict(bc_policy_state_dict, strict=False)
                print("Loaded BC policy weights into PPO model.")
                #set log std to -2 for all action dimensions
                with torch.no_grad():
                    if hasattr(model.policy, "log_std"):
                        model.policy.log_std[:] = -2.0
            else:
                print("BC policy state dict not found, proceeding without loading BC weights.")
        

    else:
        # Load the latest model
        latest_model_path = get_latest_model_path(str(MODEL_PATH), "steve_ppo_model")
        if latest_model_path is None:
            print("No saved model found. run with --load False to train a new model")
            print("Exiting...")
            exit(0)

        # Load VecNormalize statistics if available
        vec_path = str(Path(latest_model_path).parent / "steve_ppo_model_vecnormalize.pkl")
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            print(f"Loaded VecNormalize stats from {vec_path}")
        else:
            print(f"No VecNormalize stats found at {vec_path}, proceeding without loading them.")
        print("Loading model from:", latest_model_path)

        # Load the PPO model
        model = PPO.load(latest_model_path, env=env)
        model.set_env(env)

        #configure model for training if in train mode
        model = config_train(model)

    print_model_info(model)
    time.sleep(2)


    if args.mode == "train":

        print(pyfiglet.figlet_format("---Training Mode---", font="slant"))
        time.sleep(2)

        # Set up Callbacks
        callbacks = CallbackList([TensorboardCallback(), create_checkpoint_callback(args.load)])
        total_timesteps = env_config["train"]["num_iterations"] * env_config["train"]["n_steps"] * env_config["train"]["num_envs"]

        #train the model
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=not args.load, callback=callbacks)

        # Save the final model
        save_path = env_config["save_config"]["save_path"] + env_config["save_config"]["save_prefix"] + time.strftime("%Y-%m-%d_%H-%M-%S")
        model.save(save_path)
    else:

        print(pyfiglet.figlet_format("---Evaluation Mode---", font="slant"))
        time.sleep(2)

        # reset all envs
        obs = env.reset()
        print(obs.shape)
        start_time = time.time()
        for i in range(1024):

            action, _states = model.predict(obs,deterministic=True)
            obs, rewards, dones, info = env.step(action)
            print(f"Step {i+1} completed.")


            frame = rgb.get_data()
            writer.append_data(frame)


        end_time = time.time()
        elapsed_time = end_time - start_time

        writer.close()

        print(f"Elapsed time for 1024 steps: {elapsed_time:.2f}")

        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()