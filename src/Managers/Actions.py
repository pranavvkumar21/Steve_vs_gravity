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
        
        # Track root position offset for cyclic motion
        self.root_offset = None

        # Ensure per-env buffer exists
        if not hasattr(env, "cmd"):
            self.env.cmd = {}
        if "phase" not in self.env.cmd:
            E, d = env.num_envs, env.device
            self.env.cmd["phase"] = torch.zeros(E, 1, device=d)  # Shape [num_envs, 1]
            # Initialize root offset tracking
            self.root_offset = torch.zeros(E, 3, device=d)

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
                # Check for frames that are wrapping back to 0
                wrapping_envs = next_frame_idx >= frame_count
                
                # For envs that are wrapping, calculate root displacement
                if torch.any(wrapping_envs):
                    # Get current root position (last frame of cycle)
                    last_frame_root = motion['root_positions'][frame_count - 1]
                    # Get first frame root position
                    first_frame_root = motion['root_positions'][0]
                    # Calculate displacement to maintain continuity
                    displacement = last_frame_root - first_frame_root
                    
                    # print(f"Cycle wrap detected! Displacement: {displacement}")
                    # print(f"Root offset before: {self.root_offset[wrapping_envs]}")
                    
                    # Add displacement to root offset for wrapping environments
                    self.root_offset[wrapping_envs] += displacement
                    
                    # print(f"Root offset after: {self.root_offset[wrapping_envs]}")
                
                next_frame_idx = torch.remainder(next_frame_idx, frame_count)
            else:
                # Clamp to last frame for non-cyclic motions
                next_frame_idx = torch.clamp(next_frame_idx, 0, frame_count - 1)
                # Optionally, set a done flag if reached the end will do this later
            
            # Update frame index
            self.env.cmd["frame_idx"][:, 0] = next_frame_idx
            
            # Get joint positions and velocities for all envs at once (vectorized indexing)
            next_frame_idx_int = next_frame_idx.long()
            self.env.cmd["joint_position"][:] = motion['joint_positions'][next_frame_idx_int].clone()
            self.env.cmd["joint_velocity"][:] = motion['joint_velocities'][next_frame_idx_int].clone()
            self.env.cmd["root_orientation"][:] = motion['root_orientations'][next_frame_idx_int].clone()
            self.env.cmd["local_body_position"][:] = motion['local_body_positions'][next_frame_idx_int].clone()
            
            # Apply root offset for continuous motion
            root_pos = motion['root_positions'][next_frame_idx_int].clone()
            root_pos += self.root_offset
            self.env.cmd["root_position"][:] = root_pos
            
            self.env.cmd["phase"][:, 0] = motion['phase'][next_frame_idx_int].clone()
        self.sim_step = (self.sim_step + 1) % self.env.cfg.decimation

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the root offset for specified environments"""
        if self.root_offset is not None and len(env_ids) > 0:
            # print(f"Resetting root offset for envs {env_ids}")
            self.root_offset[env_ids] = 0.0


@configclass
class ActionsCfg:
    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name = "steve",
        joint_names = steve_config["joint_names"],
        scale=18*torch.pi/180,
    )
    phase = ActionTermCfg(class_type=PhaseAction, asset_name="steve", clip={".*":(-1.0, 1.0)})
