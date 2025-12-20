import json
import os
from pathlib import Path
from datetime import datetime
import torch
ROOT = Path(__file__).resolve().parent.parent.parent

def log_metadata(runner_cfg log_dir=ROOT/"logs"):
    #check if file exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    experiment_name = runner_cfg.experiment_name
    run_name = runner_cfg.run_name
    seed = runner_cfg.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_cfg = runner_cfg.alg_cfg
    policy_cfg = runner_cfg.policy
    #all runs of the same experiment go into same file
    #if some config is changed mid run then loaded and restarted then we append timestamp to filename

    cfg_data = {
        "algorithm_cfg": alg_cfg.to_dict(),
    }
    metadata = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "seed": seed,
        "timestamp": timestamp,
        "configurations": cfg_data

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