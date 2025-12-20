

def hex_to_rgba(hex_str, alpha=1.0):
    """Convert hex color string to RGBA tuple (0-1 range)."""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b, alpha)

# --- User's Data Structures (Copied & Adapted) ---

BODY_CONNECTIONS = [
    # Spine/Torso chain
    ('pelvis', 'waist_yaw_link'), ('waist_yaw_link', 'waist_roll_link'),
    ('waist_roll_link', 'torso_link'), ('torso_link', 'head_link'),
    # Left leg chain
    ('pelvis', 'left_hip_pitch_link'), ('left_hip_pitch_link', 'left_hip_roll_link'),
    ('left_hip_roll_link', 'left_hip_yaw_link'), ('left_hip_yaw_link', 'left_knee_link'),
    ('left_knee_link', 'left_ankle_pitch_link'), ('left_ankle_pitch_link', 'left_ankle_roll_link'),
    ('left_ankle_roll_link', 'left_toe_link'),
    # Right leg chain
    ('pelvis', 'right_hip_pitch_link'), ('right_hip_pitch_link', 'right_hip_roll_link'),
    ('right_hip_roll_link', 'right_hip_yaw_link'), ('right_hip_yaw_link', 'right_knee_link'),
    ('right_knee_link', 'right_ankle_pitch_link'), ('right_ankle_pitch_link', 'right_ankle_roll_link'),
    ('right_ankle_roll_link', 'right_toe_link'),
    # Left arm chain
    ('torso_link', 'left_shoulder_pitch_link'), ('left_shoulder_pitch_link', 'left_shoulder_roll_link'),
    ('left_shoulder_roll_link', 'left_shoulder_yaw_link'), ('left_shoulder_yaw_link', 'left_elbow_link'),
    ('left_elbow_link', 'left_wrist_roll_link'), ('left_wrist_roll_link', 'left_wrist_pitch_link'),
    ('left_wrist_pitch_link', 'left_wrist_yaw_link'), ('left_wrist_yaw_link', 'left_rubber_hand'),
    # Right arm chain
    ('torso_link', 'right_shoulder_pitch_link'), ('right_shoulder_pitch_link', 'right_shoulder_roll_link'),
    ('right_shoulder_roll_link', 'right_shoulder_yaw_link'), ('right_shoulder_yaw_link', 'right_elbow_link'),
    ('right_elbow_link', 'right_wrist_roll_link'), ('right_wrist_roll_link', 'right_wrist_pitch_link'),
    ('right_wrist_pitch_link', 'right_wrist_yaw_link'), ('right_wrist_yaw_link', 'right_rubber_hand'),
]

def get_link_colors():
    """Same color definition as provided."""
    return {
        'pelvis': '#1f77b4', 'pelvis_contour_link': '#aec7e8',
        'waist_yaw_link': '#1f77b4', 'waist_roll_link': '#417dc1',
        'torso_link': '#2e8b57', 'head_link': '#4b0082', 'head_mocap': '#8a2be2',
        'imu_in_torso': '#696969',
        'left_hip_pitch_link': '#d62728', 'left_hip_roll_link': '#ff7f0e',
        'left_hip_yaw_link': '#d62728', 'left_knee_link': '#ff7f0e',
        'left_ankle_pitch_link': '#d62728', 'left_ankle_roll_link': '#ff7f0e', 'left_toe_link': '#8b0000',
        'right_hip_pitch_link': '#2ca02c', 'right_hip_roll_link': '#98df8a',
        'right_hip_yaw_link': '#2ca02c', 'right_knee_link': '#98df8a',
        'right_ankle_pitch_link': '#2ca02c', 'right_ankle_roll_link': '#98df8a', 'right_toe_link': '#006400',
        'left_shoulder_pitch_link': '#9467bd', 'left_shoulder_roll_link': '#c5b0d5',
        'left_shoulder_yaw_link': '#9467bd', 'left_elbow_link': '#c5b0d5',
        'left_wrist_roll_link': '#9467bd', 'left_wrist_pitch_link': '#c5b0d5',
        'left_wrist_yaw_link': '#9467bd', 'left_rubber_hand': '#4b0082',
        'right_shoulder_pitch_link': '#ff7f0e', 'right_shoulder_roll_link': '#ffbb78',
        'right_shoulder_yaw_link': '#ff7f0e', 'right_elbow_link': '#ffbb78',
        'right_wrist_roll_link': '#ff7f0e', 'right_wrist_pitch_link': '#ffbb78',
        'right_wrist_yaw_link': '#ff7f0e', 'right_rubber_hand': '#ff8c00',
    }

# --- Helper: Color Converter ---
def hex_to_rgba(hex_str, alpha=1.0):
    """Convert hex color string to RGBA tuple (0-1 range)."""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b, alpha)

import torch
import numpy as np
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis

class SkeletonVisualizer:
    def __init__(self, link_names, device="cuda:0"):
        """
        Visualizes a humanoid skeleton using Isaac Lab markers (Spheres for joints, Cylinders for bones).
        
        Args:
            link_names (list[str]): List of link names matching the order of input data.
            device (str): Device to store tensors on.
        """
        self.device = device
        self.link_names = link_names
        self.name_to_idx = {name: i for i, name in enumerate(link_names)}
        
        # --- 1. Setup Joint Markers (Spheres) ---
        # We create a unique prototype for each unique color found in your map
        colors_map = get_link_colors() # Using your global function
        unique_colors = sorted(list(set(colors_map.values())))
        
        # Map hex strings to prototype indices
        self.color_to_proto_idx = {color: i for i, color in enumerate(unique_colors)}
        
        # Create config with one sphere per color
        joint_markers_dict = {}
        for i, color in enumerate(unique_colors):
            # Convert hex #RRGGBB to (R, G, B) float tuple
            c_hex = color.lstrip('#')
            rgb = tuple(int(c_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            
            # Use safe marker names instead of color strings with special chars
            joint_markers_dict[f"joint_color_{i}"] = sim_utils.SphereCfg(
                radius=0.03, # Adjust joint size here
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
            )

        joint_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Skeleton/Joints",
            markers=joint_markers_dict
        )
        self.joint_viz = VisualizationMarkers(joint_cfg)

        # Pre-compute the prototype indices for the joints based on link names
        joint_indices = []
        for name in link_names:
            c = colors_map.get(name, '#808080') # Default grey
            # Use the index of this color's prototype, or 0 if not found
            idx = self.color_to_proto_idx.get(c, 0) 
            joint_indices.append(idx)
        
        self.joint_indices_tensor = torch.tensor(joint_indices, device=self.device, dtype=torch.int)


        # --- 2. Setup Bone Markers (Cylinders) ---
        # Single prototype for all bones (Grey/Black)
        bone_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Skeleton/Bones",
            markers={
                "bone": sim_utils.CylinderCfg(
                    radius=0.015, # Bone thickness
                    height=1.0,   # Base height, will be scaled
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2))
                )
            }
        )
        self.bone_viz = VisualizationMarkers(bone_cfg)

        # Pre-compute bone connection indices
        self.bone_pairs = []
        for link1, link2 in BODY_CONNECTIONS:
            if link1 in self.name_to_idx and link2 in self.name_to_idx:
                self.bone_pairs.append([self.name_to_idx[link1], self.name_to_idx[link2]])
        
        self.bone_pairs = torch.tensor(self.bone_pairs, device=self.device, dtype=torch.long)


    def draw(self, body_positions):
        """
        Draws the skeleton for a single frame.
        
        Args:
            body_positions (np.ndarray or torch.Tensor): Shape (num_links, 3). World frame.
        """
        # Ensure input is a tensor on the correct device
        if isinstance(body_positions, np.ndarray):
            pos_tensor = torch.from_numpy(body_positions).float().to(self.device)
        else:
            pos_tensor = body_positions.to(self.device).float()

        # --- 1. Draw Joints ---
        # Simply place markers at the positions, using pre-calculated color indices
        self.joint_viz.visualize(
            translations=pos_tensor,
            marker_indices=self.joint_indices_tensor
        )

        # --- 2. Draw Bones ---
        if len(self.bone_pairs) == 0:
            return

        # Get start (p1) and end (p2) positions for all bones
        p1 = pos_tensor[self.bone_pairs[:, 0]] # Shape (Num_Bones, 3)
        p2 = pos_tensor[self.bone_pairs[:, 1]] # Shape (Num_Bones, 3)

        # A. Calculate Position (Midpoint)
        midpoints = (p1 + p2) / 2.0

        # B. Calculate Scale (Z = Length)
        # Default cylinder is height 1.0, so scaling Z by distance gives correct length
        diff = p2 - p1
        lengths = torch.norm(diff, dim=1, keepdim=True)
        # Scale x,y (thickness) is 1.0 (relative to radius in config), z is length
        scales = torch.cat([torch.ones_like(lengths), torch.ones_like(lengths), lengths], dim=1)

        # C. Calculate Orientation
        # We need to rotate the default Z-up cylinder to align with the vector (p2 - p1)
        
        # Direction vector of the bone
        directions = diff / (lengths + 1e-6) # Normalize
        
        # Reference vector (Up)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(directions)
        
        # Rotation Axis = Z cross Direction
        rot_axes = torch.cross(z_axis, directions, dim=1)
        
        # Rotation Angle = acos(Z dot Direction)
        dot_products = torch.sum(z_axis * directions, dim=1)
        angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        # Handle singularity: if direction is parallel to Z, cross product is 0. 
        # We can just use identity or 180 flip, but for skeleton visualization usually fine.
        # (Robust solution would check for zero norm of rot_axes)
        
        # Create quaternions
        quats = quat_from_angle_axis(angles, rot_axes)

        self.bone_viz.visualize(
            translations=midpoints,
            orientations=quats,
            scales=scales
        )
