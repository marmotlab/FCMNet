import multiprocessing as mp
import random

import gym
import numpy as np

from alg_parameters import *
import environments


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def build_multiprocessing_env(nenv):
    def get_env_fn(rank):
        def init_env():
            env = gym.make('MultiagentCatch-v0')
            np.random.seed(ENV_SEED + rank * 100)
            random.seed(ENV_SEED + rank * 100)
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(nenv)])


class SubprocVecEnv(object):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_info', None))
        info_dic = self.remotes[0].recv()

        self.num_agents = info_dic[0]
        self.obs_shape = info_dic[2]
        self.state_shape = info_dic[3]
        self.n_actions = info_dic[1]
        self.episode_len = EPISODE_LEN

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.num_envs)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        reward, dones, infos = zip(*results)
        return np.stack(reward), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        _ = [remote.recv() for remote in self.remotes]

    def get_obs_state(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_obs_state', None))
        results = [remote.recv() for remote in self.remotes]
        obs, state = zip(*results)
        return np.stack(obs), np.stack(state)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        reward, done, info = env.step(action[0])
        if done:
            env.reset()
        return reward, done, info

    parent_remote.close()
    envs = env_fn_wrappers.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(step_env(envs, data))
            elif cmd == 'reset':
                remote.send(envs.reset())
            elif cmd == 'close':
                envs.close()
                remote.close()
                break
            elif cmd == 'get_info':
                remote.send(envs.get_info())
            elif cmd == 'get_obs_state':
                remote.send(envs.get_obs_state())
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
