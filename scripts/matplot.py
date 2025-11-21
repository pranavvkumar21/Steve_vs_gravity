import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mocap_data import load_motion
# Assuming your load_motion function and AMCParser imports exist per your snippet.

def forward_kinematics(joints, motion, root_name='root'):
    joints[root_name].set_motion(motion)

def plot_frame(joints, ax):
    xs, ys, zs = [], [], []
    for joint in joints.values():
        print(joint.name)
        print(joint.coordinate)
        coord = joint.coordinate
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])
    ax.scatter(xs, ys, zs, color='r')

    # Draw bones as lines between joints and parents
    for joint in joints.values():
        if joint.parent is not None:
            c1 = joint.coordinate.flatten()
            c2 = joint.parent.coordinate.flatten()
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 'b')

def main():
    skeleton, motion_data = load_motion(subject="01", sequence="01")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Choose a frame to visualize
    frame_idx = 0
    frame_motion = motion_data[frame_idx]

    forward_kinematics(skeleton, frame_motion)

    plot_frame(skeleton, ax)

    # Setting labels and aspect
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'CMU Mocap Frame {frame_idx}')
    ax.view_init(elev=15, azim=-70)  # set viewpoint
    
    # plt.show()
    plt.savefig('output_plot.png')  

if __name__ == "__main__":
    main()
