import torch 
import numpy as np
import random

class MotionManager:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.motions = {}

    def load_motion(self, motion_name, file_path, is_cyclic=True):
        # load npz file
        data = np.load(file_path)
        self.motions[motion_name] = {
            'joint_positions': torch.tensor(data['joint_positions'], device=self.device, dtype=torch.float32),
            'joint_velocities': torch.tensor(data['joint_velocities'], device=self.device, dtype=torch.float32),
            'root_orientations': torch.tensor(data['root_orientations'], device=self.device, dtype=torch.float32),
            
        }
        self.motions[motion_name]['frame_count'] = self.motions[motion_name]['joint_positions'].shape[0]
        self.motions[motion_name]['is_cyclic'] = is_cyclic
        self.create_phase(motion_name)
    def create_phase(self, motion_name):
        #create tensor of shape (T,) with values linearly spaced between 0 and 1 using frame count
        self.motions[motion_name]['phase'] = torch.linspace(0, 1, steps=self.motions[motion_name]['frame_count'], device=self.device)
    def get(self, motion_name, frame_idx):
        #return joint positions, velocity and phase at frame_idx
        motion = self.motions[motion_name]
        joint_positions = motion['joint_positions'][frame_idx]
        joint_velocities = motion['joint_velocities'][frame_idx]
        root_orientations = motion['root_orientations'][frame_idx]
        phase = motion['phase'][frame_idx]
        return joint_positions, joint_velocities, root_orientations, phase
    def sample(self, motion_name, batch_size=1):
        motion = self.motions[motion_name]
        
        # Generate random frame indices for all samples at once
        frame_indices = torch.randint(0, motion['frame_count'], (batch_size,), device=self.device)
        
        if batch_size == 1:
            # Return single samples (no batch dimension)
            idx = frame_indices[0]
            return (
                motion['joint_positions'][idx],
                motion['joint_velocities'][idx],
                motion['root_orientations'][idx],
                motion['phase'][idx],
                idx
            )
        else:
            # Return batched samples
            return (
                motion['joint_positions'][frame_indices],
                motion['joint_velocities'][frame_indices],
                motion['root_orientations'][frame_indices],
                motion['phase'][frame_indices],
                frame_indices
            )
    def reorder_joints(self, motion_name, robot_joint_names, mapped_joint_names):
        """
        Reorders joint_positions and joint_velocities in motions[motion_name]
        according to the order in robot_joint_names.
        
        Arguments:
            motion_name (str): name of the loaded motion
            robot_joint_names (list): list of joint names in desired/robot order
            mapped_joint_names (list): list of joint names (keys from JOINT_MAPPING), matches data order
        """
        motion = self.motions[motion_name]
        
        # Create index mapping from robot_joint_names to mapped_joint_names
        index_map = [mapped_joint_names.index(name) for name in robot_joint_names]

        # Reorder joint_positions and joint_velocities (torch tensors)
        motion['joint_positions'] = motion['joint_positions'][:, index_map]
        motion['joint_velocities'] = motion['joint_velocities'][:, index_map]
    def clamp_to_joint_limits(self, motion_name, joint_limits):
        joint_positions = self.motions[motion_name]['joint_positions']  # [num_frames, num_joints]
      # Should be [1, num_joints]

        min_limits = joint_limits[0,:, 0]  # [num_joints]
        max_limits = joint_limits[0,:, 1]  # [num_joints]
        print(joint_positions.shape)  # Should be [num_frames, num_joints]
        # print(min_limits.shape)       # Should be [1, num_joints]
        # print(max_limits.shape) 
        self.motions[motion_name]['joint_positions'] = torch.clamp(
            joint_positions,
            min_limits,
            max_limits
        )