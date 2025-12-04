import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, livestream=2).app
    
from motion_manager import MotionManager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

# Define body link connections for skeleton visualization
BODY_CONNECTIONS = [
    # Spine/Torso chain
    ('pelvis', 'waist_yaw_link'),
    ('waist_yaw_link', 'waist_roll_link'),
    ('waist_roll_link', 'torso_link'),
    ('torso_link', 'head_link'),
    
    # Left leg chain
    ('pelvis', 'left_hip_pitch_link'),
    ('left_hip_pitch_link', 'left_hip_roll_link'),
    ('left_hip_roll_link', 'left_hip_yaw_link'),
    ('left_hip_yaw_link', 'left_knee_link'),
    ('left_knee_link', 'left_ankle_pitch_link'),
    ('left_ankle_pitch_link', 'left_ankle_roll_link'),
    ('left_ankle_roll_link', 'left_toe_link'),
    
    # Right leg chain
    ('pelvis', 'right_hip_pitch_link'),
    ('right_hip_pitch_link', 'right_hip_roll_link'),
    ('right_hip_roll_link', 'right_hip_yaw_link'),
    ('right_hip_yaw_link', 'right_knee_link'),
    ('right_knee_link', 'right_ankle_pitch_link'),
    ('right_ankle_pitch_link', 'right_ankle_roll_link'),
    ('right_ankle_roll_link', 'right_toe_link'),
    
    # Left arm chain
    ('torso_link', 'left_shoulder_pitch_link'),
    ('left_shoulder_pitch_link', 'left_shoulder_roll_link'),
    ('left_shoulder_roll_link', 'left_shoulder_yaw_link'),
    ('left_shoulder_yaw_link', 'left_elbow_link'),
    ('left_elbow_link', 'left_wrist_roll_link'),
    ('left_wrist_roll_link', 'left_wrist_pitch_link'),
    ('left_wrist_pitch_link', 'left_wrist_yaw_link'),
    ('left_wrist_yaw_link', 'left_rubber_hand'),
    
    # Right arm chain
    ('torso_link', 'right_shoulder_pitch_link'),
    ('right_shoulder_pitch_link', 'right_shoulder_roll_link'),
    ('right_shoulder_roll_link', 'right_shoulder_yaw_link'),
    ('right_shoulder_yaw_link', 'right_elbow_link'),
    ('right_elbow_link', 'right_wrist_roll_link'),
    ('right_wrist_roll_link', 'right_wrist_pitch_link'),
    ('right_wrist_pitch_link', 'right_wrist_yaw_link'),
    ('right_wrist_yaw_link', 'right_rubber_hand'),
]

def get_link_colors():
    """Define color scheme for different body parts"""
    colors = {
        # Core/Spine - Blue tones
        'pelvis': '#1f77b4',
        'pelvis_contour_link': '#aec7e8',
        'waist_yaw_link': '#1f77b4',
        'waist_roll_link': '#417dc1',
        'torso_link': '#2e8b57',
        'head_link': '#4b0082',
        'head_mocap': '#8a2be2',
        'imu_in_torso': '#696969',
        
        # Left leg - Red tones
        'left_hip_pitch_link': '#d62728',
        'left_hip_roll_link': '#ff7f0e',
        'left_hip_yaw_link': '#d62728',
        'left_knee_link': '#ff7f0e',
        'left_ankle_pitch_link': '#d62728',
        'left_ankle_roll_link': '#ff7f0e',
        'left_toe_link': '#8b0000',
        
        # Right leg - Green tones
        'right_hip_pitch_link': '#2ca02c',
        'right_hip_roll_link': '#98df8a',
        'right_hip_yaw_link': '#2ca02c',
        'right_knee_link': '#98df8a',
        'right_ankle_pitch_link': '#2ca02c',
        'right_ankle_roll_link': '#98df8a',
        'right_toe_link': '#006400',
        
        # Left arm - Purple tones
        'left_shoulder_pitch_link': '#9467bd',
        'left_shoulder_roll_link': '#c5b0d5',
        'left_shoulder_yaw_link': '#9467bd',
        'left_elbow_link': '#c5b0d5',
        'left_wrist_roll_link': '#9467bd',
        'left_wrist_pitch_link': '#c5b0d5',
        'left_wrist_yaw_link': '#9467bd',
        'left_rubber_hand': '#4b0082',
        
        # Right arm - Orange tones
        'right_shoulder_pitch_link': '#ff7f0e',
        'right_shoulder_roll_link': '#ffbb78',
        'right_shoulder_yaw_link': '#ff7f0e',
        'right_elbow_link': '#ffbb78',
        'right_wrist_roll_link': '#ff7f0e',
        'right_wrist_pitch_link': '#ffbb78',
        'right_wrist_yaw_link': '#ff7f0e',
        'right_rubber_hand': '#ff8c00',
    }
    return colors

def create_humanoid_animation(motion_manager, motion_name="walk"):
    """Create animated 3D visualization of humanoid motion"""
    
    # Get motion data
    body_positions = motion_manager.motions[motion_name]["local_body_positions"].cpu().numpy()
    root_positions = motion_manager.motions[motion_name]["root_positions"].cpu().numpy()
    root_orientations = motion_manager.motions[motion_name]["root_orientations"].cpu().numpy()
    # body_positions += root_positions[:, np.newaxis, :]  # Adjust body positions by root
    link_names = motion_manager.motions[motion_name]["link_body_names"]
    num_frames, num_links, _ = body_positions.shape
    
    print(f"Creating animation for {num_frames} frames, {num_links} body links")
    print(f"Link names: {link_names}")
    print(f"dof names: {motion_manager.motions[motion_name]['joint_names']}")
    
    # Create name to index mapping
    name_to_idx = {name: i for i, name in enumerate(link_names)}
    colors = get_link_colors()
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds for consistent view
    x_min, x_max = np.min(body_positions[:, :, 0]), np.max(body_positions[:, :, 0])
    y_min, y_max = np.min(body_positions[:, :, 1]), np.max(body_positions[:, :, 1])
    z_min, z_max = np.min(body_positions[:, :, 2]), np.max(body_positions[:, :, 2])
    
    # Add some padding
    padding = 0.2
    x_range = max(x_max - x_min, 0.5)
    y_range = max(y_max - y_min, 0.5)
    z_range = max(z_max - z_min, 0.5)
    max_range = max(x_range, y_range, z_range)
    
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    center_z = (z_max + z_min) / 2
    
    ax.set_xlim(center_x - max_range/2 - padding, center_x + max_range/2 + padding)
    ax.set_ylim(center_y - max_range/2 - padding, center_y + max_range/2 + padding)
    ax.set_zlim(center_z - max_range/2 - padding, center_z + max_range/2 + padding)
    
    # Initialize empty plot elements
    joint_plots = {}
    bone_plots = []
    
    # Create plots for each joint
    for i, link_name in enumerate(link_names):
        color = colors.get(link_name, '#808080')  # Default gray if not defined
        joint_plots[link_name] = ax.scatter([], [], [], 
                                          c=color, s=100, alpha=0.8, 
                                          label=link_name if i < 10 else "")  # Limit legend entries
    
    # Create plots for skeleton connections
    for connection in BODY_CONNECTIONS:
        link1, link2 = connection
        if link1 in name_to_idx and link2 in name_to_idx:
            line, = ax.plot([], [], [], 'k-', linewidth=2, alpha=0.6)
            bone_plots.append((line, link1, link2))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Humanoid Body Position Animation')
    
    # Add legend (limited to avoid clutter)
    if len(joint_plots) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Animation function
    def animate(frame):
        # Update joint positions
        for link_name, plot in joint_plots.items():
            if link_name in name_to_idx:
                idx = name_to_idx[link_name]
                x, y, z = body_positions[frame, idx, :]
                plot._offsets3d = ([x], [y], [z])
        
        # Update skeleton connections
        for line, link1, link2 in bone_plots:
            if link1 in name_to_idx and link2 in name_to_idx:
                idx1, idx2 = name_to_idx[link1], name_to_idx[link2]
                x1, y1, z1 = body_positions[frame, idx1, :]
                x2, y2, z2 = body_positions[frame, idx2, :]
                line.set_data([x1, x2], [y1, y2])
                line.set_3d_properties([z1, z2])
        
        # Update title with frame info
        ax.set_title(f'Humanoid Body Position Animation - Frame {frame+1}/{num_frames}')
        return list(joint_plots.values()) + [line for line, _, _ in bone_plots]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=100, blit=False, repeat=True)
    
    # Save as GIF
    print("Saving animation as GIF...")
    gif_path = ROOT / 'videos' / 'humanoid_body_animation.gif'
    gif_path.parent.mkdir(exist_ok=True)
    
    # Use pillow writer for better compression
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Motion Visualization'))
    anim.save(str(gif_path), writer=writer, dpi=100)
    
    print(f"Animation saved as {gif_path}")
    
    # Also create a static plot showing all frames as a trajectory
    create_trajectory_plot(body_positions, link_names, colors)
    
    plt.close(fig)  # Close the animation figure
    return str(gif_path)

def create_trajectory_plot(body_positions, link_names, colors):
    """Create a static plot showing trajectory paths of key body parts"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot trajectories for key body parts
    key_links = ['pelvis', 'head_link', 'left_rubber_hand', 'right_rubber_hand', 
                'left_toe_link', 'right_toe_link']
    
    name_to_idx = {name: i for i, name in enumerate(link_names)}
    num_frames = body_positions.shape[0]
    
    for link_name in key_links:
        if link_name in name_to_idx:
            idx = name_to_idx[link_name]
            trajectory = body_positions[:, idx, :]
            color = colors.get(link_name, '#808080')
            
            # Plot 3D trajectory
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    color=color, linewidth=2, label=link_name, alpha=0.8)
            
            # Mark start and end points
            ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                       c='green', s=50, marker='o')
            ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                       c='red', s=50, marker='s')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Body Part Trajectories')
    ax1.legend(fontsize=8)
    
    # Top-down view (XY plane)
    ax2 = fig.add_subplot(222)
    for link_name in key_links:
        if link_name in name_to_idx:
            idx = name_to_idx[link_name]
            trajectory = body_positions[:, idx, :]
            color = colors.get(link_name, '#808080')
            
            ax2.plot(trajectory[:, 0], trajectory[:, 1], 
                    color=color, linewidth=2, label=link_name, alpha=0.8)
            ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=30)
            ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=30)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-down View (XY)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend(fontsize=8)
    
    # Side view (XZ plane)
    ax3 = fig.add_subplot(223)
    for link_name in key_links:
        if link_name in name_to_idx:
            idx = name_to_idx[link_name]
            trajectory = body_positions[:, idx, :]
            color = colors.get(link_name, '#808080')
            
            ax3.plot(trajectory[:, 0], trajectory[:, 2], 
                    color=color, linewidth=2, label=link_name, alpha=0.8)
            ax3.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=30)
            ax3.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=30)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # Front view (YZ plane)
    ax4 = fig.add_subplot(224)
    for link_name in key_links:
        if link_name in name_to_idx:
            idx = name_to_idx[link_name]
            trajectory = body_positions[:, idx, :]
            color = colors.get(link_name, '#808080')
            
            ax4.plot(trajectory[:, 1], trajectory[:, 2], 
                    color=color, linewidth=2, label=link_name, alpha=0.8)
            ax4.scatter(trajectory[0, 1], trajectory[0, 2], c='green', s=30)
            ax4.scatter(trajectory[-1, 1], trajectory[-1, 2], c='red', s=30)
    
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (YZ)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save trajectory plot
    trajectory_path = ROOT / 'videos' / 'body_trajectories.png'
    plt.savefig(str(trajectory_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Trajectory plot saved as {trajectory_path}")

def main():
    # Load motion data
    motion_manager = MotionManager(device='cpu')  # Use CPU to avoid CUDA issues in plotting
    motion_name = "walk"
    motion_manager.load_motion("walk", str(ROOT / "data/retargeted_lafan_h1/walk1_sub1.pkl"), is_cyclic=True)
    
    # Print information about the data
    print(f"Motion data shape: {motion_manager.motions['walk']['local_body_positions'].shape}")
    print(f"Link body names: {motion_manager.motions['walk']['link_body_names']}")
    
    # Create the animation
    gif_path = create_humanoid_animation(motion_manager, motion_name)
    print(f"\nVisualization complete! Check these files:")
    print(f"  - Animation GIF: {gif_path}")
    print(f"  - Trajectory plot: {ROOT / 'videos' / 'body_trajectories.png'}")
    simulation_app.close()

if __name__ == "__main__":
    main()
