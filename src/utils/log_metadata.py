import json
import os
from pathlib import Path
from datetime import datetime
import torch
ROOT = Path(__file__).resolve().parent.parent.parent

def log_metadata(runner, log_dir=ROOT/"logs"):
    #check if file exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_learing_iteration = runner.current_learing_iteration
    runner_cfg = runner.cfg
    experiment_name = runner_cfg.experiment_name
    run_name = runner_cfg.run_name
    seed = runner_cfg.seed
    #timestamp needs to be pretty format readable with YY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    alg_cfg = runner_cfg["algorithm"]
    policy_cfg = runner_cfg["policy"]

    metadata = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "seed": seed,
        "timestamp": timestamp,
        "algorithm_cfg": alg_cfg.to_dict(),
        "policy_cfg": policy_cfg.to_dict()
    }
    log_path = Path(log_dir) / f"{experiment_name}_metadata.json"
    #append new entry to json file
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            data = json.load(f)
        data.append(metadata)
    else:
        data = [metadata]

    with open(log_path, 'w') as f:
        json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    pass