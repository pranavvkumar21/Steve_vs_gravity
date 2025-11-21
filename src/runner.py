#!/usr/bin/env python3
from isaaclab_rl.rsl_rl import ( RslRlOnPolicyRunner, RslRlOnPolicyRunnerCfg, 
    RslRlPpoAlgorithmCfg, RslRlActorCriticCfg)
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
with open(ROOT/"config"/"runner_config.yaml", 'r') as f:
    runner_cfg_dict = yaml.safe_load(f)["runner"]



def create_runner_cfg():
    algorithm_cfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        learning_rate=3.5e-4,
        num_learning_epochs=10,
        num_mini_batches=10,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        desired_kl=0.015,
        clip_param=0.2,
        normalize_advantage_per_mini_batch=True,
        value_loss_coef=1.0,
        
    )
    policy_cfg = RslRlActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )
    runner_cfg = RslRlOnPolicyRunnerCfg(
        experiment_name = "make_steve_walk",
        run_name = "steve_godspeed_1",
        log_root_path = str(ROOT/"logs"),
        seed = 42,
        num_steps_per_env = 128,
        max_iterations = 500,
        logger = "tensorboard",
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        clip_actions=1.0,
        save_interval = 50,
        policy = policy_cfg,
        algorithm = algorithm_cfg,
    )
    return runner_cfg
