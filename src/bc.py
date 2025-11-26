def generate_expert_trajectories_wrapped(env_wrapped, num_trajectories=100, action_scale=0.25):
    """Generate trajectories with wrapped environment (handles batched obs)"""
    from imitation.data.types import Trajectory
    import numpy as np
    
    # Get the unwrapped base env for accessing scene
    base_env = env_wrapped
    while hasattr(base_env, 'venv'):
        base_env = base_env.venv
    if hasattr(base_env, 'unwrapped'):
        base_env = base_env.unwrapped
    
    # Get number of parallel environments
    num_envs = env_wrapped.num_envs
    print(f"Collecting from {num_envs} parallel environments")
    
    trajectories = []

    obs = env_wrapped.reset()
    joint_ids = base_env.motion_manager.motions["walk"]['joint_indices']
    # for i in range(num_envs):
    #     init_error = base_env.cmd["joint_position"][i] - base_env.scene["steve"].data.joint_pos[i][joint_ids]
    #     print(f"Env {i} init joint error:", init_error.cpu().numpy())

    
    # Track trajectories for each parallel env
    active_trajs = [{'obs': [], 'acts': []} for _ in range(num_envs)]
    
    obs = env_wrapped.reset()  # Shape: (num_envs, obs_dim)
    
    collected_count = 0
    joint_limits = base_env.scene["steve"].data.default_joint_pos_limits[0][joint_ids]
    while collected_count < num_trajectories:
        for env_idx in range(num_envs):
            current_pos = base_env.scene["steve"].data.joint_pos[env_idx][joint_ids]
            target_pos = base_env.cmd["joint_position"][env_idx]
            #map to -1,1 
            # print("target pos:", target_pos.shape)
            # print("joint limits:", joint_limits.shape)
            mapped_target = 2 * (target_pos - joint_limits[:, 0]) / (joint_limits[:, 1] - joint_limits[:, 0]) - 1
            delta = target_pos - current_pos
            expert_action = delta / action_scale
            
            # Store observation
            active_trajs[env_idx]['obs'].append(obs[env_idx])
            active_trajs[env_idx]['acts'].append(mapped_target.cpu().numpy())
        
        actions_batch = np.array([
            (base_env.cmd["joint_position"][i] - base_env.scene["steve"].data.joint_pos[i][joint_ids]).cpu().numpy() / action_scale
            for i in range(num_envs)
        ])
        
        obs, rewards, dones, infos = env_wrapped.step(actions_batch)
        
        for env_idx in range(num_envs):
            if dones[env_idx] and len(active_trajs[env_idx]['obs']) > 0:
                # Append the final observation after done
                active_trajs[env_idx]['obs'].append(obs[env_idx])
                
                # Now obs length is acts length + 1
                trajectory = Trajectory(
                    obs=np.array(active_trajs[env_idx]['obs']),
                    acts=np.array(active_trajs[env_idx]['acts']),
                    infos=None,
                    terminal=True
                )
                trajectories.append(trajectory)
                collected_count += 1
                
                active_trajs[env_idx] = {'obs': [], 'acts': []}
                
                if collected_count % 10 == 0:
                    print(f"Collected {collected_count}/{num_trajectories} trajectories")
                
                if collected_count >= num_trajectories:
                    break

    
    return trajectories[:num_trajectories]  # Return exactly num_trajectories
