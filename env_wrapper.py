import multiprocessing as mp

import numpy as np
from smac.env import StarCraft2Env

from env_parameters import env_args


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
            env = StarCraft2Env(**env_args)
            env.seed(env_args['seed'] + rank * 100)
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

        self.remotes[0].send(('get_env_info', None))
        info_dic = self.remotes[0].recv()

        self.num_agents = info_dic['n_agents']
        self.obs_shape = info_dic['obs_shape']
        self.state_shape = info_dic['state_shape']
        self.cent_state_shape = self.obs_shape + self.state_shape
        self.n_actions = info_dic['n_actions']
        self.episode_len = info_dic['episode_limit']

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

    def get_state(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_state', None))
        state = [remote.recv() for remote in self.remotes]
        state = np.stack(state)
        return np.expand_dims(state, axis=1)

    def get_avail_actions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_avail_actions', None))
        avail_actions = [remote.recv() for remote in self.remotes]
        return np.stack(avail_actions)

    def get_obs(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_obs', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

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
            elif cmd == 'get_env_info':
                remote.send(envs.get_env_info())
            elif cmd == 'get_state':
                remote.send(envs.get_state())
            elif cmd == 'get_obs':
                remote.send(envs.get_obs())
            elif cmd == 'get_avail_actions':
                remote.send(envs.get_avail_actions())
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
