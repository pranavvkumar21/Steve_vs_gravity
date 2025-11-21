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

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Body touch
    body_touch = DoneTerm(func=body_touch, time_out=True)