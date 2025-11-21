JOINT_MAPPING = {
    # Torso
    'lower_waist:0': ('lowerback', "ry"),
    'lower_waist:1': ('lowerback', "rx"),
    'pelvis': ('lowerback', "rz"),

    # Right arm
    'right_upper_arm:0': ('rhumerus', "ry"),
    'right_upper_arm:2': ('rhumerus', "rz"),
    'right_lower_arm': ('rradius', "rx"),

    # Left arm
    'left_upper_arm:0': ('lhumerus', "ry"),
    'left_upper_arm:2': ('lhumerus', "rz"),
    'left_lower_arm': ('lradius', "rx"),

    # Right leg (hip)
    'right_thigh:2': ('rfemur', "rz"),
    'right_thigh:1': ('rfemur', "rx"),
    'right_thigh:0': ('rfemur', "ry"),

    # Left leg (hip)
    'left_thigh:2': ('lfemur', "rz"),
    'left_thigh:1': ('lfemur', "rx"),
    'left_thigh:0': ('lfemur', "ry"),

    # Knee joints
    'right_shin': ('rtibia', "rx"),
    'left_shin': ('ltibia', "rx"),
    # Ankle joints
    'right_foot:0': ('rfoot', "rx"),
    'right_foot:1': ('rfoot', "rz"),
    'left_foot:0': ('lfoot', "rx"),
    'left_foot:1': ('lfoot', "rz"),
}