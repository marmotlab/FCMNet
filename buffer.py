import random
from collections import deque

import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, obs, action, reward, done, next_obs, state, next_state, actor_hidden_state_c, actor_hidden_state_h):
        experience = (obs, action, reward, done, next_obs, state, next_state,
                      actor_hidden_state_c, actor_hidden_state_h)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        o_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        d_batch = np.array([_[3] for _ in batch])
        next_o_batch = np.array([_[4] for _ in batch])
        s_batch = np.array([_[5] for _ in batch])
        next_s_batch = np.array([_[6] for _ in batch])
        state_c_batch = np.array([_[7] for _ in batch])
        state_h_batch = np.array([_[8] for _ in batch])

        return o_batch, a_batch, r_batch, d_batch, next_o_batch, s_batch, next_s_batch, state_c_batch, state_h_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
