#!/usr/bin/env python3
from isaaclab_rl.rsl_rl import ( RslRlOnPolicyRunnerCfg, 
    RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg)
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
with open(ROOT/"config"/"runner_config.yaml", 'r') as f:
    runner_cfg_dict = yaml.safe_load(f)["runner"]



def create_runner_cfg():
    algorithm_cfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        learning_rate=3.5e-4,
        num_learning_epochs=4,
        num_mini_batches=4,
        gamma=0.992,
        lam=0.95,
        entropy_coef=0.005,
        desired_kl=0.02,
        clip_param=0.2,
        normalize_advantage_per_mini_batch=True,
        value_loss_coef=2.8,
        max_grad_norm=1.0,
        
    )
    policy_cfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.26,
        # noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    runner_cfg = RslRlOnPolicyRunnerCfg(
        experiment_name = "make_steve_walk",
        run_name = "steve_godspeed_1",
        # log_root_path = str(ROOT/"logs"),
        seed = 42,
        num_steps_per_env = 64,
        max_iterations = 8000,
        logger = "tensorboard",
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        clip_actions=1.0,
        save_interval = 50,
        policy = policy_cfg,
        algorithm = algorithm_cfg,
    )
    return runner_cfg
