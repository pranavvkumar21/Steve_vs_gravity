import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

def body_touch(env):
    data = env.scene.sensors["body_contact"].data
    F = data.net_forces_w          # [env, bodies, 3]
    in_contact = (F.norm(dim=-1) > 1.0).any(dim=-1)
    # if in_contact.any():
    #     print("Body contact time_out:")
    return in_contact


def motion_end_time_out(env):
    #if not is_cyclic
    motion_name = env.motion_name
    if not env.motion_manager.motions[motion_name]['is_cyclic']:
        # get frame idx from env.cmd
        frame_idx = env.cmd["frame_idx"]  # [E, 1]
        mocap_length = env.motion_manager.motions["walk"]['joint_positions'].shape[0]
        # check if any env has reached the end of the motion
        time_out = (frame_idx[:, 0] >= mocap_length -1)
        return time_out
    else:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Body touch
    body_touch = DoneTerm(func=body_touch, time_out=True)

    # (3) Motion end time out
    # motion_end_time_out = DoneTerm(func=motion_end_time_out, time_out=True)