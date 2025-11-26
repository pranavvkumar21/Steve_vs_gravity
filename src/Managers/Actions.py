import isaaclab.envs.mdp as mdp   
from isaaclab.utils import configclass
from isaaclab.managers import (ActionTerm,
    ActionTermCfg ,
    SceneEntityCfg)
import yaml
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

class PhaseAction(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self.asset_name = cfg.asset_name  # e.g., "steve"
        self._raw = None
        self._proc = None
        self.env = env
        self.sim_step = 0
        # No joint_ids needed since we're not controlling joints

        # Ensure per-env buffer exists
        if not hasattr(env, "cmd"):
            self.env.cmd = {}
        if "phase" not in self.env.cmd:
            E, d = env.num_envs, env.device
            self.env.cmd["phase"] = torch.zeros(E, 1, device=d)  # Shape [num_envs, 1]

    @property
    def action_dim(self) -> int:
        return 0  # No action input needed
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc

    def process_actions(self, actions: torch.Tensor) -> None:
        """No-op since action_dim is 0"""
        # When action_dim=0, this receives empty tensor or isn't called
        if actions.numel() > 0:
            self._raw = actions.clone()
            self._proc = actions
        else:
            # Create dummy tensors to avoid None issues
            self._raw = torch.empty(self.num_envs, 0, device=self.device)
            self._proc = torch.empty(self.num_envs, 0, device=self.device)

    def apply_actions(self) -> None:
        """Increment phase and update joint targets from motion data"""
        if self.sim_step == 0:
            motion_name = self.env.motion_name  # You can make this configurable or track per-env
            motion = self.env.motion_manager.motions[motion_name]
            
            # Increment frame index (vectorized for all envs)
            frame_count = motion['frame_count']
            is_cyclic = motion['is_cyclic']
            
            # Current frame indices for all envs
            current_frame_idx = self.env.cmd["frame_idx"][:, 0]
            
            # Increment all frame indices at once
            next_frame_idx = current_frame_idx + 1
            
            # Handle wrapping for cyclic motion (vectorized)
            if is_cyclic:
                next_frame_idx = torch.remainder(next_frame_idx, frame_count)
            else:
                # Clamp to last frame for non-cyclic motions
                next_frame_idx = torch.clamp(next_frame_idx, 0, frame_count - 1)
                # Optionally, set a done flag if reached the end will do this later
            
            # Update frame index
            self.env.cmd["frame_idx"][:, 0] = next_frame_idx
            
            # Get joint positions and velocities for all envs at once (vectorized indexing)
            next_frame_idx_int = next_frame_idx.long()
            self.env.cmd["joint_position"][:] = motion['joint_positions'][next_frame_idx_int]
            self.env.cmd["joint_velocity"][:] = motion['joint_velocities'][next_frame_idx_int]
            self.env.cmd["root_orientation"][:] = motion['root_orientations'][next_frame_idx_int]
            self.env.cmd["phase"][:, 0] = motion['phase'][next_frame_idx_int]
        self.sim_step = (self.sim_step + 1) % self.env.cfg.decimation


@configclass
class ActionsCfg:
    pass
    joint_pos = mdp.JointPositionToLimitsActionCfg(
        asset_name = "steve",
        joint_names = steve_config["scene"]["joint_names"],
        scale=1.0,
        rescale_to_limits=True
    )
    phase = ActionTermCfg(class_type=PhaseAction, asset_name="steve", clip={".*":(-1.0, 1.0)})
