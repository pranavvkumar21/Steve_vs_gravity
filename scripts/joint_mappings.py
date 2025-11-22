JOINT_MAPPING= {
    # Torso
    'torso': ('lowerback', "rx"),  # rx
    # 'torso_y': ('lowerback', "ry"),  # ry
    # 'torso_z': ('lowerback', "rz"),  # rz

    # Left Hip
    'left_hip_pitch': ('lfemur', "rx"),
    'left_hip_roll': ('lfemur', "ry"),
    'left_hip_yaw': ('lfemur', "rz"),

    # Right Hip
    'right_hip_pitch': ('rfemur', "rx"),
    'right_hip_roll': ('rfemur', "ry"),
    'right_hip_yaw': ('rfemur', "rz"),

    # Left Shoulder
    'left_shoulder_pitch': ('lhumerus', "rx"),
    'left_shoulder_roll': ('lhumerus', "ry"),
    'left_shoulder_yaw': ('lhumerus', "rz"),

    # Right Shoulder
    'right_shoulder_pitch': ('rhumerus', "rx"),
    'right_shoulder_roll': ('rhumerus', "ry"),
    'right_shoulder_yaw': ('rhumerus', "rz"),

    # Left Elbow
    'left_elbow': ('lradius', "rx"),

    # Right Elbow
    'right_elbow': ('rradius', "rx"),

    # Left Knee
    'left_knee': ('ltibia', "rx"),

    # Right Knee
    'right_knee': ('rtibia', "rx"),

    # Left Ankle (rx only per request)
    'left_ankle': ('lfoot', "rx"),

    # Right Ankle (rx only per request)
    'right_ankle': ('rfoot', "rx"),
}
