import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,  DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from pathlib import Path
import yaml
from isaaclab_assets import G1_MINIMAL_CFG  # Changed from HUMANOID_CFG

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        # rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 5000.0,
                "waist_roll_joint": 5000.0,
                "waist_pitch_joint": 5000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "steve_config.yaml", 'r') as f:
    steve_config = yaml.safe_load(f)

pattern = "(" + "|".join(steve_config["body_contact_links"]) + ")"


class Steve_SceneCfg(InteractiveSceneCfg):
    """Configuration for the Steve scene."""
    
    # Use HUMANOID_28_CFG instead
    steve = G1_29DOF_CFG.replace( prim_path="{ENV_REGEX_NS}/Steve")

    
    ground = AssetBaseCfg(prim_path="/World/terrain", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Steve/.*" + pattern,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )

    # In Steve_EnvCfg class
    # tiled_camera = CameraCfg(
    #     "{ENV_REGEX_NS}/RecorderCamera", 
    #     offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 2.0), rot=(0.9239, 0.0, 0.3827, 0.0), convention="world"), # LookAt (0,0,0) roughly
    #     update_period=0.0,
    #     height=720,
    #     width=1280,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0)
    # )


    # l_foot_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Steve/left_foot",
    #     update_period=0.0,
    #     history_length=1,
    #     filter_prim_paths_expr=["/World/terrain"],
    #     force_threshold=1.0,      
    #     debug_vis=False,
    # )

    # r_foot_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Steve/right_foot",
    #     update_period=0.0,
    #     history_length=1,
    #     filter_prim_paths_expr=["/World/terrain"],
    #     force_threshold=1.0,      
    #     debug_vis=False,
    # )
