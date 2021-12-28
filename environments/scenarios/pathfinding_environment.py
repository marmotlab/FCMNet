import math

import gym
import numpy as np
from gym import spaces

from environments.scenarios.scenarios import Scenario
from environments import rendering


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, shared_viewer=True):
        scenario = Scenario()
        self.world = scenario.make_world()
        self.agents = self.world.policy_agents
        self.num_agents = len(self.world.policy_agents)
        self.episode_reward = 0
        self.episode_len = 0

        # Scenario callbacks
        self.reset_callback = scenario.reset_world
        self.reward_callback = scenario.reward
        self.observation_callback = scenario.observation
        self.info_callback = None
        self.done_callback = scenario.is_finished
        self.modify_env_callback = scenario.move_goals_randomly

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        self.state_space = []
        for _ in self.agents:
            u_action_space = spaces.Discrete(self.world.dim_p * 2 + 1)
            self.action_space.append(u_action_space)
            obs_dim = 12
            state_dim = 36
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            self.state_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,), dtype=np.float32))

        # Rendering
        self.graph_links = []
        self.old_links = []
        self.drop_next = []
        self.toggle = True
        self.show_links = False
        self.render_geoms = []
        self.render_geoms_xform = []
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.num_agents
        self._reset_render()

    def step(self, action_n, modify_env=True):
        reward_n = []
        done_n = []
        info = {}
        # Set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        self.world.step()
        if modify_env:
            self._modify_env()

        # Record reward and done for each agent
        for i, agent in enumerate(self.agents):
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done())

        self.episode_len += 1
        self.episode_reward += np.mean(reward_n)
        info['episode_reward'] = self.episode_reward
        info['episode_len'] = self.episode_len
        return np.array(reward_n), all(done_n), info

    def get_obs_state(self):
        obs_n = []
        state_n = []
        for i, agent in enumerate(self.agents):
            obs, state = self._get_obs(agent)
            obs_n.append(obs)
            state_n.append(state)
        return np.array(obs_n), np.array(state_n)

    def _modify_env(self):
        if self.modify_env_callback is None:
            return
        self.modify_env_callback(self.world)

    def _reset(self):
        return self.reset()

    def reset(self):
        self.episode_reward = 0
        self.episode_len = 0
        self.drop_next = []
        # Reset world
        self.reset_callback(self.world)
        # Reset renderer
        self._reset_render()
        # Record observation and state for each agent
        obs_n = []
        state_n = []
        for agent in self.agents:
            obs, state = self._get_obs(agent)
            obs_n.append(obs)
            state_n.append(state)
        obs_n = np.array(obs_n)
        state_n = np.array(state_n)
        return obs_n, state_n

    def get_info(self):
        return self.num_agents, self.action_space[0].n, self.observation_space[0].shape[0], self.state_space[0].shape[0]

    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        # Get an observation for a particular agent
        return self.observation_callback(agent, self.world)

    def _get_done(self):
        if self.done_callback is None:
            return False
        # Get a done for a particular agent
        return self.done_callback(self.world)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        # Get a reward for a particular agent
        return self.reward_callback(agent, self.world)

    def _set_action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        # Set an action for a particular agent
        if agent.movable:
            if action == 1:
                agent.action.u[0] = -1.0
            if action == 2:
                agent.action.u[0] = +1.0
            if action == 3:
                agent.action.u[1] = -1.0
            if action == 4:
                agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, links, mode='human', screen_width=800, screen_height=800):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            if self.viewers[i] is None:
                self.viewers[i] = rendering.Viewer(screen_height, screen_width)

        agent_list = self.world.agents
        if not self.graph_links:
            self.graph_links = links

        self.render_geoms = []
        self.render_geoms_xform = []
        for entity in self.world.entities:
            if entity.shape == 0:
                geom = rendering.make_circle(entity.size)
            else:
                n = entity.shape + 3
                v = []
                r = 0.07
                for i in range(n):
                    v.append((math.cos(2 * math.pi * i * r / n), math.sin(2 * math.pi * i * r / n)))
                geom = rendering.make_polygon(v)

            xform = rendering.Transform()
            if 'agent' in entity.name:
                geom.set_color(*entity.color, alpha=0.5)
            else:
                geom.set_color(*entity.color)
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

        if self.drop_next:
            for i in range(len(self.drop_next[0])):
                start = int(self.drop_next[0][i])
                end = int(self.drop_next[1][i])
                if self.show_links:
                    tem_pgeom = self.viewers[0].draw_line(agent_list[start].state.p_pos, agent_list[end].state.p_pos)
                    tem_pgeom.set_color(1, 0, 0)

        if self.show_links:
            for i in range(len(self.graph_links[0].tolist())):
                start = self.graph_links[0][i].item()
                end = self.graph_links[1][i].item()
                if self.drop_next and (start, end) in list(zip(list(self.drop_next[0]), list(self.drop_next[1]))):
                    continue
                if self.toggle:
                    geom2 = self.viewers[0].draw_line(agent_list[start].state.p_pos, agent_list[end].state.p_pos)
                    if self.old_links and (start, end) in list(zip(self.old_links[0], self.old_links[1])):
                        geom2.set_color(0.3, 0.3, 0.3, 0.3)
                        geom2.line_width = 0.5
                    else:
                        geom2.set_color(0, 2, 1)
                        geom2.line_width = 10

                    self.render_geoms.append(geom2)

        for viewer in self.viewers:
            viewer.geoms = []
            for geom in self.render_geoms:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))
        return np.array(results)

    def _render(self, links, mode='rgb_array', screen_width=800, screen_height=800):
        return self.render(links, mode, screen_height, screen_width)

    # create receptor field locations in local coordinate frame
    @staticmethod
    def _make_receptor_locations():
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx
