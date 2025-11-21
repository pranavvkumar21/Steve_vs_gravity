JOINT_MAPPING = {
    # Abdomen/Torso (maps to lowerback in ASF)
    'abdomen_x': ('lowerback', 'rx'),
    'abdomen_y': ('lowerback', 'ry'),
    'abdomen_z': ('lowerback', 'rz'),
    
    # Neck (maps to lowerneck or upperneck in ASF - choosing lowerneck)
    'neck_x': ('lowerneck', 'rx'),
    'neck_y': ('lowerneck', 'ry'),
    'neck_z': ('lowerneck', 'rz'),
    
    # Right shoulder (maps to rhumerus in ASF)
    'right_shoulder_x': ('rhumerus', 'rx'),
    'right_shoulder_y': ('rhumerus', 'ry'),
    'right_shoulder_z': ('rhumerus', 'rz'),
    
    # Left shoulder (maps to lhumerus in ASF)
    'left_shoulder_x': ('lhumerus', 'rx'),
    'left_shoulder_y': ('lhumerus', 'ry'),
    'left_shoulder_z': ('lhumerus', 'rz'),
    
    # Right elbow (maps to rradius in ASF)
    'right_elbow': ('rradius', 'rx'),
    
    # Left elbow (maps to lradius in ASF)
    'left_elbow': ('lradius', 'rx'),
    
    # Right hip (maps to rfemur in ASF)
    'right_hip_x': ('rfemur', 'rx'),
    'right_hip_y': ('rfemur', 'ry'),
    'right_hip_z': ('rfemur', 'rz'),
    
    # Left hip (maps to lfemur in ASF)
    'left_hip_x': ('lfemur', 'rx'),
    'left_hip_y': ('lfemur', 'ry'),
    'left_hip_z': ('lfemur', 'rz'),
    
    # Right knee (maps to rtibia in ASF)
    'right_knee': ('rtibia', 'rx'),
    
    # Left knee (maps to ltibia in ASF)
    'left_knee': ('ltibia', 'rx'),
    
    # Right ankle (maps to rfoot in ASF - only has rx and rz in ASF)
    'right_ankle_x': ('rfoot', 'rx'),
    'right_ankle_y': None,  # Not in ASF, may need special handling
    'right_ankle_z': ('rfoot', 'rz'),
    
    # Left ankle (maps to lfoot in ASF - only has rx and rz in ASF)
    'left_ankle_x': ('lfoot', 'rx'),
    'left_ankle_y': None,  # Not in ASF, may need special handling
    'left_ankle_z': ('lfoot', 'rz'),
}
