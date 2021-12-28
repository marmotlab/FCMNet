import numpy as np

from environments.core import World, Agent, Landmark

NUM_AGENTS = 5
FINISHED_THRESHOLD = 0.01
PENALTY_FOR_EXISTING = 0.2
LAMBDA = 0.8
COLORS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0.9, 0.9]]


class Scenario(object):
    def __init__(self):
        self.num_agents = NUM_AGENTS

    def make_world(self):
        world = World()
        # Set world properties first
        world.dim_c = 10
        world.collaborative = True  # Whether agents share rewards
        # Add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
        # Add landmarks
        world.landmarks = [Landmark() for _ in range(self.num_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # Make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Assign goals to agents
        for i, landmark in enumerate(world.landmarks):
            landmark.color = COLORS[i]

        landmarks_without_replacement = np.random.choice(world.landmarks, self.num_agents, replace=False)

        for i, ag in enumerate(world.agents):
            ag.goal = landmarks_without_replacement[i]
            ag.color = ag.goal.color
            ag.previous_distance = 0
            ag.current_distance = 0
            ag.rewarded = False

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for ag in world.agents:
            ag.previous_distance = np.sum(np.square(ag.state.p_pos - ag.goal.state.p_pos))

    @staticmethod
    def reward(agent, world, distance_reward=lambda x: x):
        if agent is None or agent.goal is None:
            print("\n\n\nAGENT INIT PROBLEM!!\n\n\n")
            return 0.0
        own_reward = 0
        others_reward = 0
        sum_distances = 0
        for ag in world.agents:
            sum_distances += np.sum(np.square(ag.goal.state.p_pos - ag.state.p_pos))
        mean_distances = sum_distances / NUM_AGENTS
        dist = np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
        agent.current_distance = dist
        own_reward -= distance_reward(agent.current_distance - agent.previous_distance)
        own_reward -= PENALTY_FOR_EXISTING
        agent.previous_distance = agent.current_distance
        others_reward -= distance_reward(mean_distances)
        return LAMBDA * own_reward + (1 - LAMBDA) * others_reward

    def reward_exponential(self, agent, world):
        return self.reward(agent, world, lambda x: min(2000, np.exp(x)))

    @staticmethod
    def move_goals_randomly(world, prob_change_goal_locations=0.005):
        """Goals teleport randomly around the grid with some probability"""
        for landmark in world.landmarks:
            if np.random.rand() < prob_change_goal_locations:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    @staticmethod
    def move_goals_brownian(world, prob_change_goal_locations=0.05, sigma=0.1):
        """The goals move to a random point whose coordinates differ from the earlier point's ~ N(0,sigma)"""
        for landmark in world.landmarks:
            if np.random.rand() < prob_change_goal_locations:
                landmark.state.p_pos += np.random.normal(0, sigma, world.dim_p)
                landmark.state.p_pos = np.clip(landmark.state.p_pos, -1, +1)
                landmark.state.p_vel = np.zeros(world.dim_p)

    @staticmethod
    def is_finished(world):
        distances_to_goals = []
        for ag in world.agents:
            distances_to_goals.append(np.sum(np.square(ag.goal.state.p_pos - ag.state.p_pos)))
        dist_less_than_epsilon = [d < FINISHED_THRESHOLD for d in distances_to_goals]
        return all(dist_less_than_epsilon)

    @staticmethod
    def is_finished_sum(world):
        sum_distances = 0
        for ag in world.agents:
            sum_distances += np.sum(np.square(ag.goal.state.p_pos - ag.state.p_pos))
        return sum_distances < 0.01

    @staticmethod
    def observation(agent, world):
        own_pos = agent.state.p_pos
        own_vel = agent.state.p_vel
        own_goal_pos = agent.goal.state.p_pos

        all_pos, other_goal_pos, all_vel, all_goal_pos = [], [], [], []

        for ag in world.agents:
            if ag is not agent:
                other_goal_pos.append(np.array([ag.goal.state.p_pos]))
            all_goal_pos.append(np.array([ag.goal.state.p_pos]))
            all_pos.append(np.array([ag.state.p_pos]))
            all_vel.append(np.array([ag.state.p_vel]))

        own_pos = np.array(own_pos)
        own_vel = np.array(own_vel)
        all_pos = np.array(all_pos).reshape(-1)
        own_goal_pos = np.array(own_goal_pos)
        other_goal_pos = np.array(other_goal_pos).reshape(-1)
        all_goal_pos = np.array(all_goal_pos).reshape(-1)
        all_vel = np.array(all_vel).reshape(-1)

        actor_obs = np.concatenate((own_pos, own_vel, other_goal_pos))
        critic_obs = np.concatenate((own_pos, own_goal_pos, own_vel, all_pos, all_goal_pos, all_vel))

        return actor_obs, critic_obs
