"""Create a StarCraftC2Env environment.

Parameters:
map_name : str, optional
   The name of the SC2 map to play (default is "8m"). The full list
   can be found by running bin/map_list.
step_mul : int, optional
   How many game steps per agent step (default is 8). None
   indicates to use the default map step_mul.
move_amount : float, optional
   How far away units are ordered to move per step (default is 2).
difficulty : str, optional
   The difficulty of built-in computer AI bot (default is "7").
game_version : str, optional
   StarCraft II game version (default is None). None indicates the
   latest version.
seed : int, optional
   Random seed used during game initialisation. This allows to
continuing_episode : bool, optional
   Whether to consider episodes continuing or finished after time
   limit is reached (default is False).
obs_all_health : bool, optional
   Agents receive the health of all units (in the sight range) as part
   of observations (default is True).
obs_own_health : bool, optional
   Agents receive their own health as a part of observations (default
   is False). This flag is ignored when obs_all_health == True.
obs_last_action : bool, optional
   Agents receive the last actions of all units (in the sight range)
   as part of observations (default is False).
obs_pathing_grid : bool, optional
   Whether observations include pathing values surrounding the agent
   (default is False).
obs_terrain_height : bool, optional
   Whether observations include terrain height values surrounding the
   agent (default is False).
obs_instead_of_state : bool, optional
   Use combination of all agents' observations as the global state
   (default is False).
obs_timestep_number : bool, optional
   Whether observations include the current timestep of the episode
   (default is False).
state_last_action : bool, optional
   Include the last actions of all agents as part of the global state
   (default is True).
state_timestep_number : bool, optional
   Whether the state include the current timestep of the episode
   (default is False).
reward_sparse : bool, optional
   Receive 1/-1 reward for winning/loosing an episode (default is
   False). Whe rest of reward parameters are ignored if True.
reward_only_positive : bool, optional
   Reward is always positive (default is True).
reward_death_value : float, optional
   The amount of reward received for killing an enemy unit (default
   is 10). This is also the negative penalty for having an allied unit
   killed if reward_only_positive == False.
reward_win : float, optional
   The reward for winning in an episode (default is 200).
reward_defeat : float, optional
   The reward for loosing in an episode (default is 0). This value
   should be nonpositive.
reward_negative_scale : float, optional
   Scaling factor for negative rewards (default is 0.5). This
   parameter is ignored when reward_only_positive == True.
reward_scale : bool, optional
   Whether or not to scale the reward (default is True).
reward_scale_rate : float, optional
   Reward scale rate (default is 20). When reward_scale == True, the
   reward received by the agents is divided by (max_reward /
   reward_scale_rate), where max_reward is the maximum possible
   reward per episode without considering the shield regeneration
   of Protoss units.
replay_dir : str, optional
   The directory to save replays (default is None). If None, the
   replay will be saved in Replays directory where StarCraft II is
   installed.
replay_prefix : str, optional
   The prefix of the replay to be saved (default is None). If None,
   the name of the map will be used.
window_size_x : int, optional
   The length of StarCraft II window size (default is 1920).
window_size_y: int, optional
   The height of StarCraft II window size (default is 1200).
heuristic_ai: bool, optional
   Whether or not to use a non-learning heuristic AI (default False).
heuristic_rest: bool, optional
   At any moment, restrict the actions of the heuristic AI to be
   chosen from actions available to RL agents (default is False).
   Ignored if heuristic_ai == False.
debug: bool, optional
   Log messages about observations, state, actions and rewards for
   debugging purposes (default is False).
       """

map_name = "5m_vs_6m"
step_mul = 8
move_amount = 2
difficulty = "7"
game_version = None
seed = 1
continuing_episode = False
obs_all_health = True
obs_own_health = True
obs_last_action = False
obs_pathing_grid = False
obs_terrain_height = False
obs_instead_of_state = False
obs_timestep_number = False
state_last_action = True
state_timestep_number = False
reward_sparse = False
reward_only_positive = True
reward_death_value = 10
reward_win = 200
reward_defeat = 0
reward_negative_scale = 0.5
reward_scale = True
reward_scale_rate = 20
replay_dir = ""
replay_prefix = ""
window_size_x = 1920
window_size_y = 1200
heuristic_ai = False
heuristic_rest = False
debug = False

env_args = {'map_name': map_name, 'step_mul': step_mul, 'move_amount': move_amount, 'difficulty': difficulty,
            'game_version': game_version,
            'seed': seed, 'continuing_episode': continuing_episode, 'obs_all_health': obs_all_health,
            'obs_own_health': obs_own_health,
            'obs_last_action': obs_last_action, 'obs_pathing_grid': obs_pathing_grid,
            'obs_terrain_height': obs_terrain_height,
            'obs_instead_of_state': obs_instead_of_state, 'obs_timestep_number': obs_timestep_number,
            'state_last_action': state_last_action,
            'state_timestep_number': state_timestep_number, 'reward_sparse': reward_sparse,
            'reward_only_positive': reward_only_positive,
            'reward_death_value': reward_death_value, 'reward_win': reward_win, 'reward_defeat': reward_defeat,
            'reward_negative_scale': reward_negative_scale,
            'reward_scale': reward_scale, 'reward_scale_rate': reward_scale_rate, 'replay_dir': replay_dir,
            'replay_prefix': replay_prefix,
            'window_size_x': window_size_x, 'window_size_y': window_size_y, 'heuristic_ai': heuristic_ai,
            'heuristic_rest': heuristic_rest,
            'debug': debug}
