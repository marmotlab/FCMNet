import numpy as np

from alg_parameters import *
from util import swap_flat


class Runner(object):
    """Run multiple episode in multiprocessing environments."""

    def __init__(self, env, model):
        """Initialize environments"""
        self.env = env
        self.model = model
        self.episode_rewards = np.zeros((N_ENVS,))
        self.env.reset()
        self.obs = env.get_obs()
        self.state = env.get_state()
        self.state = np.repeat(self.state, N_AGENTS, axis=1)
        self.cent_state = np.concatenate((self.obs, self.state), axis=-1)
        self.dones = [False for _ in range(N_ENVS)]
        self.actor_hidden_state_c = np.zeros((N_ENVS * N_AGENTS, ACTOR_LAYER2))
        self.actor_hidden_state_h = np.zeros((N_ENVS * N_AGENTS, ACTOR_LAYER2))
        self.critic_hidden_state_c = np.zeros((N_ENVS * N_AGENTS, CRITIC_LAYER2))
        self.critic_hidden_state_h = np.zeros((N_ENVS * N_AGENTS, CRITIC_LAYER2))

    def run(self):
        # Use to store experiences
        mb_obs, mb_cent_state, mb_rewards, mb_values, mb_dones, \
            ep_infos, mb_actions, mb_ps = [], [], [], [], [], [], [], []
        mb_actor_hidden_state_c, mb_actor_hidden_state_h, \
            mb_critic_hidden_state_c, mb_critic_hidden_state_h = [], [], [], []
        episode_rewrads_info = []

        for _ in range(N_STEPS):
            mb_obs.append(self.obs)
            mb_cent_state.append(self.cent_state)
            mb_actor_hidden_state_c.append(self.actor_hidden_state_c)
            mb_critic_hidden_state_c.append(self.critic_hidden_state_c)
            mb_actor_hidden_state_h.append(self.actor_hidden_state_h)
            mb_critic_hidden_state_h.append(self.critic_hidden_state_h)

            valid_action = self.env.get_avail_actions()
            actions, values, ps, critic_hidden_state, actor_hidden_state = self.model.step(self.obs, self.cent_state,
                                                                                           valid_action,
                                                                                           self.critic_hidden_state_c,
                                                                                           self.critic_hidden_state_h,
                                                                                           self.actor_hidden_state_c,
                                                                                           self.actor_hidden_state_h)
            self.critic_hidden_state_c, self.critic_hidden_state_h = critic_hidden_state
            self.actor_hidden_state_c, self.actor_hidden_state_h = actor_hidden_state

            mb_values.append(values)
            mb_ps.append(ps)
            mb_actions.append(actions)
            mb_dones.append(self.dones)

            rewards, self.dones, infos = self.env.step(actions)

            self.obs = self.env.get_obs()
            self.state = self.env.get_state()
            self.state = np.repeat(self.state, N_AGENTS, axis=1)
            self.cent_state = np.concatenate((self.obs, self.state), axis=-1)
            self.episode_rewards += rewards

            true_index = np.argwhere(self.dones)
            if len(true_index) != 0:
                # Initialize memory
                true_index = np.squeeze(true_index)
                self.actor_hidden_state_c = np.reshape(self.actor_hidden_state_c, (-1, N_AGENTS, ACTOR_LAYER2))
                self.actor_hidden_state_h = np.reshape(self.actor_hidden_state_h, (-1, N_AGENTS, ACTOR_LAYER2))
                self.critic_hidden_state_c = np.reshape(self.critic_hidden_state_c, (-1, N_AGENTS, CRITIC_LAYER2))
                self.critic_hidden_state_h = np.reshape(self.critic_hidden_state_h, (-1, N_AGENTS, CRITIC_LAYER2))

                self.actor_hidden_state_c[true_index] = np.zeros(self.actor_hidden_state_c[true_index].shape)
                self.actor_hidden_state_h[true_index] = np.zeros(self.actor_hidden_state_h[true_index].shape)
                self.critic_hidden_state_c[true_index] = np.zeros(self.critic_hidden_state_c[true_index].shape)
                self.critic_hidden_state_h[true_index] = np.zeros(self.critic_hidden_state_h[true_index].shape)

                self.actor_hidden_state_c = np.reshape(self.actor_hidden_state_c, (-1, ACTOR_LAYER2))
                self.actor_hidden_state_h = np.reshape(self.actor_hidden_state_h, (-1, ACTOR_LAYER2))
                self.critic_hidden_state_c = np.reshape(self.critic_hidden_state_c,
                                                        (-1, CRITIC_LAYER2))
                self.critic_hidden_state_h = np.reshape(self.critic_hidden_state_h,
                                                        (-1, CRITIC_LAYER2))

                # Record information of the episode when it ends
                episode_rewrads_info.append(np.nanmean(self.episode_rewards[true_index]))
                self.episode_rewards[true_index] = np.zeros(self.episode_rewards[true_index].shape)
                if true_index.shape == ():
                    ep_infos.append(infos[true_index])
                else:
                    for item in true_index:
                        ep_infos.append(infos[item])
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_cent_state = np.asarray(mb_cent_state, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.expand_dims(mb_rewards, axis=-1)
        mb_rewards = np.repeat(mb_rewards, N_AGENTS, axis=-1)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_ps = np.asarray(mb_ps, dtype=np.float32)
        mb_actor_hidden_state_c = np.asarray(mb_actor_hidden_state_c, dtype=np.float32)
        mb_actor_hidden_state_h = np.asarray(mb_actor_hidden_state_h, dtype=np.float32)
        mb_critic_hidden_state_c = np.asarray(mb_critic_hidden_state_c, dtype=np.float32)
        mb_critic_hidden_state_h = np.asarray(mb_critic_hidden_state_h, dtype=np.float32)
        mb_actor_hidden_state_c = np.reshape(mb_actor_hidden_state_c, (N_STEPS, -1, N_AGENTS, ACTOR_LAYER2))
        mb_actor_hidden_state_h = np.reshape(mb_actor_hidden_state_h, (N_STEPS, -1, N_AGENTS, ACTOR_LAYER2))
        mb_critic_hidden_state_c = np.reshape(mb_critic_hidden_state_c, (N_STEPS, -1, N_AGENTS, CRITIC_LAYER2))
        mb_critic_hidden_state_h = np.reshape(mb_critic_hidden_state_h, (N_STEPS, -1, N_AGENTS, CRITIC_LAYER2))
        # Calculate advantages
        last_values = self.model.value(self.cent_state, self.critic_hidden_state_c,
                                       self.critic_hidden_state_h)
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        for t in reversed(range(N_STEPS)):
            if t == N_STEPS - 1:
                next_nonterminal = 1.0 - self.dones
                next_nonterminal = np.expand_dims(next_nonterminal, axis=1)
                next_nonterminal = np.repeat(next_nonterminal, N_AGENTS, axis=1)
                next_values = last_values
            else:
                next_nonterminal = 1.0 - mb_dones[t + 1]
                next_nonterminal = np.expand_dims(next_nonterminal, axis=1)
                next_nonterminal = np.repeat(next_nonterminal, N_AGENTS, axis=1)
                next_values = mb_values[t + 1]
            delta = mb_rewards[t] + GAMMA * next_values * next_nonterminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + GAMMA * LAM * next_nonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values
        return (*map(swap_flat, (mb_obs, mb_cent_state, mb_returns, mb_values, mb_actions, mb_ps,
                                 mb_critic_hidden_state_c, mb_critic_hidden_state_h, mb_actor_hidden_state_c,
                                 mb_actor_hidden_state_h)), ep_infos, np.nanmean(episode_rewrads_info))
