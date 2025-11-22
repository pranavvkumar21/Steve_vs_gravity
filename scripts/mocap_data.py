import sys
import numpy as np
from AMCParser.amc_parser import parse_amc, parse_asf
import urllib.request
import os
import numpy as np
import transforms3d.euler as euler
from transforms3d import quaternions
from joint_mappings import JOINT_MAPPING


def download_motion(subject="01", sequence="01", download_dir="./mocap_data"):
    download_path = os.path.join(download_dir,subject)
    os.makedirs(download_path,exist_ok=True)
    base_url = 'http://mocap.cs.cmu.edu/subjects/'
    base_download_url = os.path.join
    asf_url = f"{base_url}/{subject}/{subject}.asf"
    amc_url = f"{base_url}/{subject}/{subject}_{sequence}.amc"
    asf_download_path = os.path.join(download_path,f"{subject}.asf")
    amc_download_path = os.path.join(download_path,f"{subject}_{sequence}.amc")
    if not os.path.isfile(asf_download_path):
        urllib.request.urlretrieve(asf_url,asf_download_path)
    if not os.path.isfile(amc_download_path):
        urllib.request.urlretrieve(amc_url,amc_download_path)
    return asf_download_path, amc_download_path

def load_motion(subject="01", sequence="01", download_dir="./mocap_data"):
    download_path = os.path.join(download_dir,subject)
    asf_path = os.path.join(download_path,f"{subject}.asf")
    amc_path = os.path.join(download_path,f"{subject}_{sequence}.amc")
    if not os.path.isfile(asf_path) or not os.path.isfile(amc_path):
        download_motion(subject, sequence, download_dir)
    skeleton = parse_asf(asf_path)
    motion_data = parse_amc(amc_path)
    return skeleton, motion_data

def asf_limits_as_array(joints, motion):
    n_joints = len(joints)
    n_frames = len(motion)
    #n_joints x 3 (rx, ry, rz) x 2 (min, max)
    limits_array = []
    for j, joint in enumerate(joints):
        limits_array.append(np.array(joint.limits))
    limits_array = np.array(limits_array)
    #convert dgrees to radians
    limits_array = np.deg2rad(limits_array)
    return limits_array

def get_joint_angles_as_array(joints, motion):
    n_joints = len(joints)
    n_frames = len(motion)
    angles_array = np.zeros((n_frames, n_joints, 3))  # nframes x njoints x 3 (rx, ry, rz)
    for f in range(n_frames):
        frame_data = motion[f]
        for joint in joints:
            if joint.name in frame_data:
                joint_data = frame_data[joint.name]
                for d, dof in enumerate(joint.dof):
                    if dof == 'rx':
                        angles_array[f, joint.id, 0] = joint_data[d]
                    elif dof == 'ry':
                        angles_array[f, joint.id, 1] = joint_data[d]
                    elif dof == 'rz':
                        angles_array[f, joint.id, 2] = joint_data[d]
    # Convert degrees to radians
    angles_array = np.deg2rad(angles_array)
    return angles_array


def get_points3d(joints, motions, z_is_up=True, z_rotation_angle_deg=-90, scale=0.056444):
    n_frames = len(motions)
    n_joints = len(joints)
    points3d = np.empty((n_frames, n_joints, 3), dtype=np.float32)

    for frame_idx, motion in enumerate(motions):
        joints['root'].set_motion(motion)
        for j in joints.values():
            jid = j.id
            points3d[frame_idx, jid] = np.squeeze(j.coordinate)

    # Define z_is_up transformation matrix
    R_zup = np.eye(3, dtype=np.float32)
    if z_is_up:
        R_zup = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

    # Define Z-axis rotation matrix (z_rotation_angle in degrees)
    angle_rad = np.deg2rad(z_rotation_angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R_zrot = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    # Combined transformation
    R_combined = R_zup @ R_zrot

    # Apply combined rotation to all points (batch matrix multiply)
    points3d = points3d @ R_combined.T

    # Apply scaling (e.g., inches to mm)
    points3d = points3d * scale

    return points3d

def get_relative_joint_positions(points3d):
    root_joint_id = 0
    root_positions = points3d[:, root_joint_id, :]     
    relative_positions = points3d - root_positions[:, np.newaxis, :]
    return relative_positions
def transform_quaternion_to_zup_and_rotate(quat_yup):
    """
    Convert quaternion from Y-up to Z-up and apply additional -90° Z-axis rotation.
    quat_yup: (w, x, y, z) in Y-up coords.
    Returns: (w, x, y, z) in IsaacLab Z-up with -90 deg rotation around Z applied.
    """
    # Rotation matrix for Y-up to Z-up
    R_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # -90° rotation around Z axis
    angle = np.deg2rad(-90)
    c, s = np.cos(angle), np.sin(angle)
    R_neg90z = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    
    # Convert quat to matrix (Y-up)
    R_mocap = quaternions.quat2mat(quat_yup)
    
    # Apply transforms: first Y->Z, then rotate -90deg about Z
    R_transformed = R_neg90z @ R_zup @ R_mocap @ R_zup.T @ R_neg90z.T
    
    quat_transformed = quaternions.mat2quat(R_transformed)
    return quat_transformed

def get_root_orientation(motion, as_quart=False):
    frames = len(motion)
    orientations = []
    for f in range(frames):
        root_data = motion[f]['root']
        rotation_deg = root_data[3:]
        rotation_rad = np.deg2rad(rotation_deg)
        R = euler.euler2mat(rotation_rad[0], rotation_rad[1], rotation_rad[2], axes='sxyz')

        if as_quart:
            q = quaternions.mat2quat(R)
            q_zup_rotated = transform_quaternion_to_zup_and_rotate(q)
            orientations.append(q_zup_rotated)
        else:
            R_zup = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ], dtype=np.float32)
            R_neg90z = np.array([
                [np.cos(np.deg2rad(-90)), -np.sin(np.deg2rad(-90)), 0],
                [np.sin(np.deg2rad(-90)),  np.cos(np.deg2rad(-90)), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            R_transformed = R_neg90z @ R_zup @ R @ R_zup.T @ R_neg90z.T
            orientations.append(R_transformed)

    return np.array(orientations)

def get_root_linear_velocity(points3d, fps):
    """
    Compute linear velocity of the root joint from global 3D positions.

    Parameters:
    - points3d: np.array of shape (n_frames, n_joints, 3)
    - fps: frame rate (frames per second)

    Returns:
    - velocities: np.array of shape (n_frames, 3), linear velocity per frame
    """
    root_positions = points3d[:, 0, :]  # Assuming root has joint id 0
    dt = 1.0 / fps

    velocities = np.zeros_like(root_positions)
    velocities[1:] = (root_positions[1:] - root_positions[:-1]) / dt
    velocities[0] = np.zeros(3)  # zero velocity for first frame

    return velocities


def joint_name_to_id(joints):
    name_to_id = {}
    for j in joints.values():
        name_to_id[j.name] = j.id
    return name_to_id

def get_mapped_joint_angles_as_array(joint_angles: np.ndarray, name_to_id: dict, joint_mapping=JOINT_MAPPING):
    n_joints = len(joint_mapping)
    n_frames = len(joint_angles)
    angles_array = np.zeros((n_frames, n_joints))  # nframes x njoints
    
    for f in range(n_frames):
        for target_joint_name, mapping in joint_mapping.items():
            # Handle None mappings (ankle_y)
            if mapping is None:
                continue  # Leave as zero
            
            source_joint_name, dof = mapping
            if source_joint_name in name_to_id:
                source_joint_id = name_to_id[source_joint_name]
                target_joint_id = list(joint_mapping.keys()).index(target_joint_name)
                if dof == "rx":
                    angles_array[f, target_joint_id] = joint_angles[f, source_joint_id, 0]
                elif dof == "ry":
                    angles_array[f, target_joint_id] = joint_angles[f, source_joint_id, 1]
                elif dof == "rz":
                    angles_array[f, target_joint_id] = joint_angles[f, source_joint_id, 2]
    
    return angles_array


def get_mapped_limits_as_array(joints, joint_mapping=JOINT_MAPPING):
    n_joints = len(joint_mapping)
    limits_array = np.zeros((n_joints, 2))  # njoints x 2 (min, max)
    
    for target_joint_name, mapping in joint_mapping.items():
        # Handle None mappings (ankle_y joints that don't exist in mocap)
        if mapping is None:
            target_joint_id = list(joint_mapping.keys()).index(target_joint_name)
            limits_array[target_joint_id] = [0.0, 0.0]  # Set to zero limits
            continue
        
        source_joint_name, dof = mapping
        if source_joint_name in joints:
            joint = joints[source_joint_name]
            target_joint_id = list(joint_mapping.keys()).index(target_joint_name)
            if dof == "rx":
                limits_array[target_joint_id] = joint.limits[0]
            elif dof == "ry":
                limits_array[target_joint_id] = joint.limits[1]
            elif dof == "rz":
                limits_array[target_joint_id] = joint.limits[2]
    
    # Convert degrees to radians
    limits_array = np.deg2rad(limits_array)
    return limits_array



def get_joint_velocities(mapped_joint_angles: np.ndarray, fps: int):
    n_frames = len(mapped_joint_angles)
    n_joints = mapped_joint_angles.shape[1]
    velocities = np.zeros((n_frames, n_joints))
    dt = 1.0 / fps
    for f in range(1, n_frames):
        velocities[f] = (mapped_joint_angles[f] - mapped_joint_angles[f - 1]) / dt
    velocities[0] = np.zeros(n_joints)  # zero velocity for first frame
    return velocities


            
def invert_joint_angles(mapped_joint_angles, mapped_limits, inverted_joint_names):

    inverted_angles = mapped_joint_angles.copy()
    inverted_limits = mapped_limits.copy()
    joint_map = list(JOINT_MAPPING.keys())
    for joint_name in inverted_joint_names:
        joint_idx = joint_map.index(joint_name)
        inverted_angles[:, joint_idx] = -inverted_angles[:, joint_idx]
        inverted_limits[joint_idx] = [-mapped_limits[joint_idx, 1], -mapped_limits[joint_idx, 0]]
    
    return inverted_angles, inverted_limits

def main():
    subject = "01"
    sequence = "01"
    download_dir = "../data/mocap_data"
    asf_path, amc_path = download_motion(subject, sequence, download_dir)
    skeleton, motion_data = load_motion(subject, sequence, download_dir)
    print(f"Loaded motion data for subject {subject}, sequence {sequence}")
    print(f"Number of frames: {len(motion_data[0])}")
    limits_array = asf_limits_as_array(list(skeleton.values()), motion_data)
    angles_array = get_joint_angles_as_array(list(skeleton.values()), motion_data)
    #print limits and angle of femur(id:7) for frame 0
    #print limits for all jin
    femur_id = skeleton['rfemur'].id
    print(f"Limits for femur (id:{femur_id}) at frame 0: {limits_array[femur_id]}")
    print(f"Angles for femur (id:{femur_id}) at frame 0: {angles_array[0, femur_id]}")
    points3d = get_points3d(skeleton, motion_data)
    print(f"3D points shape: {points3d.shape}")
    print(f"3D points at frame 0 for root (id:{0}): {points3d[0, 0]}")
    relative_positions = get_relative_joint_positions(points3d)
    print(f"Relative positions at frame 0 for root (id:{0}): {relative_positions[0, 0]}")
    root_orientations = get_root_orientation(motion_data, as_quart=True)
    print(f"Root orientation (quaternion) at frame 0: {root_orientations[0]}")
    # print(f"Joints in skeleton: {list(skeleton.joints.keys())}")
    fps = 120  # CMU Mocap default fps
    root_velocities = get_root_linear_velocity(points3d, fps)
    print(f"Root linear velocity at frame 1: {root_velocities[1]}")
    name_to_id = joint_name_to_id(skeleton)
    mapped_joint_angles = get_mapped_joint_angles_as_array(angles_array, name_to_id, JOINT_MAPPING)
    print(f"Mapped joint angles shape: {mapped_joint_angles.shape}")
    print(f"Mapped joint angles at frame 0: {mapped_joint_angles[0]}")
    joint_velocities = get_joint_velocities(mapped_joint_angles, fps)
    print(f"Joint velocities shape: {joint_velocities.shape}")
    print(f"Joint velocities at frame 1: {joint_velocities[1]}")
    mapped_limits_array = get_mapped_limits_as_array(skeleton, JOINT_MAPPING)
    #print mapped limits
    print("Mapped joint limits:")
    for i, joint_name in enumerate(JOINT_MAPPING.keys()):
        print(f"\t {joint_name}: {mapped_limits_array[i]}")
    print(f"Mapped limits shape: {mapped_limits_array.shape}")
    print(f"Mapped limits for first joint: {mapped_limits_array[0]}")
    # Example: Invert joint angles for left and right shin and lower arm
    inverted_joint_names = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow']
    # inverted_joint_names = JOINT_MAPPING.keys()  # Invert all joints
    mapped_joint_angles, mapped_limits_array = invert_joint_angles(mapped_joint_angles, mapped_limits_array, inverted_joint_names)
    print(f"Inverted joint angles for joints: {inverted_joint_names}")
    # save numpy arrays to disk
    os.makedirs(os.path.join(download_dir, subject, sequence), exist_ok=True)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_limits.npy"), limits_array)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_angles.npy"), angles_array)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_points3d.npy"), points3d)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_relative_positions.npy"), relative_positions)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_root_orientations.npy"), root_orientations)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_root_velocities.npy"), root_velocities)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_mapped_joint_angles.npy"), mapped_joint_angles)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_joint_velocities.npy"), joint_velocities)
    np.save(os.path.join(download_dir, subject, sequence, f"{subject}_{sequence}_mapped_limits.npy"), mapped_limits_array)
    print(f"Saved processed data to {os.path.join(download_dir, subject, sequence)}")
    # print shapes all arrays saved
    print(f"Limits array shape: {limits_array.shape}")
    print(f"Angles array shape: {angles_array.shape}")
    print(f"Points3D array shape: {points3d.shape}")
    print(f"Relative positions array shape: {relative_positions.shape}")
    print(f"Root orientations array shape: {root_orientations.shape}")
    print(f"Root velocities array shape: {root_velocities.shape}")
    print(f"Mapped joint angles array shape: {mapped_joint_angles.shape}")
    print(f"Joint velocities array shape: {joint_velocities.shape}")
    print(f"Mapped limits array shape: {mapped_limits_array.shape}")


if __name__ == "__main__":
    main()