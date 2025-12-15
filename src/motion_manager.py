import torch 
import numpy as np
import random
import pickle
from isaaclab.utils.math import quat_mul, quat_apply, quat_box_minus
import matplotlib.pyplot as plt
class MotionManager:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.motions = {}

    def load_motion(self, motion_name, file_path, is_cyclic=True, frame_range=(0, -1)):
        # load npz file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Loaded motion data keys:", data.keys())
        self.motions[motion_name] = {
            'joint_positions': torch.tensor(data['dof_pos'], device=self.device, dtype=torch.float32),
            'root_orientations': torch.tensor(data['root_rot'], device=self.device, dtype=torch.float32),
            'root_positions': torch.tensor(data['root_pos'], device=self.device, dtype=torch.float32),
            'local_body_positions': torch.tensor(data['local_body_pos'], device=self.device, dtype=torch.float32),
        }
        for key in self.motions[motion_name]:
            #for walk frame range is 135 to 180
            self.motions[motion_name][key] = self.motions[motion_name][key][frame_range[0]:frame_range[1], :]

        self.motions[motion_name]["link_body_names"] = data["link_body_list"]
        num_frames = self.motions[motion_name]['root_orientations'].shape[0]

        #change root orientations from xyzw to wxyz
        self.motions[motion_name]['root_orientations'] = self.motions[motion_name]['root_orientations'][:, [3,0,1,2]]

        #rotate root orientations and positions by 90 degrees about z axis to align with isaac sim coordinate system
        z_rot90 = torch.tensor([0.7071, 0, 0, 0.7071], device=self.device).repeat(num_frames, 1)

        self.motions[motion_name]['root_orientations'] = quat_mul(
            self.motions[motion_name]['root_orientations'],
            z_rot90  
        )

        self.motions[motion_name]['root_positions'] = quat_apply(
            z_rot90,
            self.motions[motion_name]['root_positions']
        )
        # remove "pelvis" from joint positions and joint names
        self.motions[motion_name]['joint_names'] = [name for name in data['joint_names_ordered'] if name != "pelvis"]

        #we remove wrist joints from motion data to make action space smaller
        num_joints = self.motions[motion_name]['joint_positions'].shape[1]

        #find indices of wrist joints in joint names ordered
        wrist_indices = [i for i, name in enumerate(self.motions[motion_name]['joint_names']) if "wrist" in name.lower()]
        keep_indices = [i for i in range(num_joints) if i not in wrist_indices]

        # remove wrist joints from joint positions
        self.motions[motion_name]['joint_positions'] = self.motions[motion_name]['joint_positions'][:, keep_indices]

        self.motions[motion_name]['joint_names'] = [self.motions[motion_name]['joint_names'][i] for i in keep_indices]

        self.motions[motion_name]['joint_velocities'] = torch.zeros_like(self.motions[motion_name]['joint_positions'])
        self.motions[motion_name]['root_lin_velocities'] = torch.zeros_like(self.motions[motion_name]['root_positions'])
        self.motions[motion_name]['root_ang_velocities'] = torch.zeros_like(self.motions[motion_name]['root_positions'])

        fps  = data['fps']
        for i in range(0, self.motions[motion_name]['joint_positions'].shape[0]-1):
            self.motions[motion_name]['joint_velocities'][i] = (self.motions[motion_name]['joint_positions'][i+1] - self.motions[motion_name]['joint_positions'][i]) * fps
            self.motions[motion_name]['root_lin_velocities'][i] = (self.motions[motion_name]['root_positions'][i+1] - self.motions[motion_name]['root_positions'][i]) * fps

            # #compute angular velocity from quaternion difference
            q_current = self.motions[motion_name]['root_orientations'][i]
            q_next = self.motions[motion_name]['root_orientations'][i+1]

            delta_rot = quat_box_minus(q_next.unsqueeze(0), q_current.unsqueeze(0))
            self.motions[motion_name]['root_ang_velocities'][i] = delta_rot.squeeze(0) * fps
        #set last velocity to be same as second last
        self.motions[motion_name]['joint_velocities'][-1] = self.motions[motion_name]['joint_velocities'][-2]
        self.motions[motion_name]['root_lin_velocities'][-1] = self.motions[motion_name]['root_lin_velocities'][-2]
        self.motions[motion_name]['root_ang_velocities'][-1] = self.motions[motion_name]['root_ang_velocities'][-2]
        
        self.motions[motion_name]['frame_count'] = self.motions[motion_name]['joint_positions'].shape[0]
        self.motions[motion_name]['is_cyclic'] = is_cyclic

        #remove "pelvis" from joint names ordered
        

        print(f"Loaded motion '{motion_name}' with {self.motions[motion_name]['frame_count']} frames.")
        print("joint pos shape:", self.motions[motion_name]['joint_positions'].shape)
        #prinit first few root positions
        print("root positions:", self.motions[motion_name]['root_positions'][:5])
        # print local body positions shape
        # print("local body pos shape:", self.motions[motion_name]['local_body_positions'].shape)


        self.create_phase(motion_name)

    def create_phase(self, motion_name):
        #create tensor of shape (T,) with values linearly spaced between 0 and 1 using frame count
        self.motions[motion_name]['phase'] = torch.linspace(0, 1, steps=self.motions[motion_name]['frame_count'], device=self.device)

    def sample(self, motion_name, batch_size=1, till_percent=1.0):
        motion = self.motions[motion_name]

        max_frame = int(motion['frame_count'] * till_percent)
        
        # Generate random frame indices for all samples at once
        frame_indices = torch.randint(0, max_frame, (batch_size,), device=self.device)
        
        if batch_size == 1:
            # Return single samples (no batch dimension)
            idx = frame_indices[0]
            return (
                motion['joint_positions'][idx],
                motion['joint_velocities'][idx],
                motion['root_positions'][idx],
                motion['root_orientations'][idx],
                motion['root_lin_velocities'][idx],
                motion['root_ang_velocities'][idx],
                motion['local_body_positions'][idx],
                motion['phase'][idx],
                idx
            )
        else:
            # Return batched samples
            return (
                motion['joint_positions'][frame_indices],
                motion['joint_velocities'][frame_indices],
                motion['root_positions'][frame_indices],
                motion['root_orientations'][frame_indices],
                motion['root_lin_velocities'][frame_indices],
                motion['root_ang_velocities'][frame_indices],
                motion['local_body_positions'][frame_indices],
                motion['phase'][frame_indices],
                frame_indices
            )
    def reorder_joints(self, motion_name, robot_joint_names, mapped_joint_names):

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
        # print(joint_positions.shape)  # Should be [num_frames, num_joints]
        # print(min_limits.shape)       # Should be [1, num_joints]
        # print(max_limits.shape) 
        self.motions[motion_name]['joint_positions'] = torch.clamp(
            joint_positions,
            min_limits,
            max_limits
        )
    def get_joint_indices(self,motion_name, robot_joint_names):
        """
        Returns a list of indices to map mocap_joint_names to robot_joint_names.
        
        Arguments:
            mocap_joint_names (list): list of joint names in mocap order
            robot_joint_names (list): list of joint names in robot order
            
        Returns:
            list: list of indices such that mocap_joint_names[indices] gives robot_joint_names
        """
        mocap_joint_names = self.motions[motion_name]['joint_names']
        joint_indices = []
        for name in mocap_joint_names:
            if name not in robot_joint_names:
                # continue
                # print(f"Warning: joint name {name} not found in robot joint names.")
                continue
            
            idx = robot_joint_names.index(name)
            # print(f"Mapping mocap joint {name} to robot joint index {idx}")
            joint_indices.append(idx)
        self.motions[motion_name]['joint_indices'] = joint_indices
        return joint_indices
    def get_body_link_indices(self, motion_name, robot_body_names):
        mocap_body_names = self.motions[motion_name]['link_body_names']
        body_link_indices = []
        for name in mocap_body_names:
            if name not in robot_body_names:
                # continue
                print(f"Warning: body link name {name} not found in robot body names.")
                continue
            
            idx = robot_body_names.index(name)
            # print(f"Mapping mocap body link {name} to robot body index {idx}")
            body_link_indices.append(idx)
        self.motions[motion_name]['body_link_indices'] = body_link_indices
        return body_link_indices
    def move_reference_root_to_origin(self, motion_name,default_root_pos, height_offset=0.84):

        root_positions = self.motions[motion_name]['root_positions']
        # offset = root_positions[0] - default_root_pos
        self.motions[motion_name]['root_positions'] = root_positions - root_positions[0]
        #set z height to height_offset
        self.motions[motion_name]['root_positions'][:,2] += height_offset
        print(f"Moved root positions of motion '{motion_name}' ")
        #print first few root positions
        print("New root positions:", self.motions[motion_name]['root_positions'][:5])
        # print min and max root height
        print("Root height range:", torch.min(self.motions[motion_name]['root_positions'][:,2]), torch.max(self.motions[motion_name]['root_positions'][:,2]))
        # self.plot_root_trajectory(motion_name)
    def plot_root_trajectory(self, motion_name):
        #3d plot of root positions with color gradient showing direction of travel
        root_positions = self.motions[motion_name]['root_positions'].cpu().numpy()
        fig = plt.figure(figsize=(12, 8))
        
        # Create 3D subplot
        ax = fig.add_subplot(121, projection='3d')
        
        #set equal aspect ratio
        max_range = np.array([root_positions[:,0].max()-root_positions[:,0].min(), 
                             root_positions[:,1].max()-root_positions[:,1].min(), 
                             root_positions[:,2].max()-root_positions[:,2].min()]).max() / 2.0
        mid_x = (root_positions[:,0].max()+root_positions[:,0].min()) * 0.5
        mid_y = (root_positions[:,1].max()+root_positions[:,1].min()) * 0.5
        mid_z = (root_positions[:,2].max()+root_positions[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Create color gradient from start (blue) to end (red)
        num_frames = root_positions.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, num_frames))  # Blue to yellow gradient
        
        # Plot trajectory with color gradient
        for i in range(num_frames - 1):
            ax.plot3D(root_positions[i:i+2, 0], 
                     root_positions[i:i+2, 1], 
                     root_positions[i:i+2, 2], 
                     color=colors[i], linewidth=2)
        
        # Mark start and end points
        ax.scatter(root_positions[0, 0], root_positions[0, 1], root_positions[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(root_positions[-1, 0], root_positions[-1, 1], root_positions[-1, 2], 
                  c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Root Trajectory of {motion_name}')
        ax.legend()
        
        # Create 2D top-down view subplot
        ax2 = fig.add_subplot(122)
        
        # Plot 2D trajectory with same color gradient
        for i in range(num_frames - 1):
            ax2.plot(root_positions[i:i+2, 0], 
                    root_positions[i:i+2, 1], 
                    color=colors[i], linewidth=2)
        
        # Mark start and end points in 2D
        ax2.scatter(root_positions[0, 0], root_positions[0, 1], 
                   c='green', s=100, marker='o', label='Start')
        ax2.scatter(root_positions[-1, 0], root_positions[-1, 1], 
                   c='red', s=100, marker='s', label='End')
        
        # Add arrows to show direction
        mid_idx = num_frames // 2
        dx = root_positions[mid_idx+1, 0] - root_positions[mid_idx, 0]
        dy = root_positions[mid_idx+1, 1] - root_positions[mid_idx, 1]
        ax2.arrow(root_positions[mid_idx, 0], root_positions[mid_idx, 1], 
                 dx*5, dy*5, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Top-down Root Trajectory of {motion_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=0, vmax=num_frames/30.0))  # Assuming 30 fps
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=[ax, ax2], shrink=0.8, aspect=20)
        cbar.set_label('Time (seconds)')
        
        plt.tight_layout()
        #save plot
        plt.savefig(f'{motion_name}_root_trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Root trajectory plot saved as '{motion_name}_root_trajectory.png'")
        print(f"Start position: [{root_positions[0, 0]:.3f}, {root_positions[0, 1]:.3f}, {root_positions[0, 2]:.3f}]")
        print(f"End position: [{root_positions[-1, 0]:.3f}, {root_positions[-1, 1]:.3f}, {root_positions[-1, 2]:.3f}]")
        
        # Calculate total distance traveled
        distances = np.linalg.norm(np.diff(root_positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        print(f"Total distance traveled: {total_distance:.3f} m")
        
        # Calculate net displacement
        net_displacement = np.linalg.norm(root_positions[-1] - root_positions[0])
        print(f"Net displacement: {net_displacement:.3f} m")