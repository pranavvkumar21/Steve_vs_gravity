import yaml
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent

def get_normalised_joint_weights():
    with open(ROOT / "config" / "steve_config.yaml", "r") as f:
        joint_weight_config = yaml.safe_load(f)
    
    
    joint_weights = joint_weight_config["joint_weights"]
    robot_joint_names = joint_weight_config["joint_names"]
    total_weight = sum(joint_weights.values())
    normalised_weights = {k: v / total_weight for k, v in joint_weights.items()}

    #return torch array of normalised weights in the order of joint names in steve_config.yaml
    ordered_weights = np.zeros(len(robot_joint_names), dtype=np.float32)
    for k,v in normalised_weights.items():
        for i, joint_name in enumerate(robot_joint_names):
            if k in joint_name:
                ordered_weights[i] = v

    return normalised_weights["root"], ordered_weights
