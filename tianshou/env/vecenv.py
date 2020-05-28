import ctypes
import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import IntEnum
from multiprocessing import Process, Pipe

import gym
import numpy as np

try:
    import ray
except ImportError:
    pass


class BaseVectorEnv(ABC, gym.Wrapper):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    """

    def __init__(self, env_fns):
        self._env_fns = env_fns
        self.env_num = len(env_fns)

    def __len__(self):
        """Return len(self), which is the number of environments."""
        return self.env_num

    @abstractmethod
    def reset(self, id=None):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        pass

    @abstractmethod
    def step(self, action):
        """Run one timestep of all the environments’ dynamics. When the end of
        episode is reached, you are responsible for calling reset(id) to reset
        this environment’s state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(self, seed=None):
        """Set the seed for all environments. Accept ``None``, an int (which
        will extend ``i`` to ``[i, i + 1, i + 2, ...]``) or a list.
        """
        pass

    @abstractmethod
    def sample(self):
        """
        sample action from env
        :return: action sampled from env
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """Render all of the environments."""
        pass

    @abstractmethod
    def close(self):
        """Close all of the environments."""
        pass


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env):
        super().__init__(env)
        if isinstance(env, list):
            if callable(env[0]):
                self.envs = [_() for _ in env]
            else:
                self.envs = env
        else:
            self.envs = [env]

    def reset(self, id=None):
        if id is None:
            self._obs = np.stack([e.reset() for e in self.envs])
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self._obs[i] = self.envs[i].reset()
        return self._obs

    def step(self, action):
        assert len(action) == self.env_num
        result = [e.step(a) for e, a in zip(self.envs, action)]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def sample(self):
        act = [e.action_space.sample() for e in self.envs]
        return np.stack(act)

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            result.append(e.seed(s))
        return result

    def render(self, **kwargs):
        result = []
        for e in self.envs:
            result.append(e.render(**kwargs))
        return result

    def close(self):
        return [e.close() for e in self.envs]


class Cmd(IntEnum):
    step = 1
    reset = 2
    render = 3
    sample = 4
    close = 5
    seed = 6


class ProcWorker(Process):
    def __init__(self, parent, p, env_fn):
        super(ProcWorker, self).__init__()
        self.parent = parent
        self.env = env_fn()
        self.p = p

    def run(self):
        self.parent.close()
        try:
            while True:
                cmd, data = self.p.recv()
                if cmd == Cmd.step:
                    self.p.send(self.env.step(data))
                elif cmd == Cmd.sample:
                    self.p.send(self.env.action_space.sample())
                elif cmd == Cmd.reset:
                    self.p.send(self.env.reset())
                elif cmd == Cmd.close:
                    self.p.send(self.env.close())
                    self.p.close()
                    break
                elif cmd == Cmd.render:
                    self.p.send(self.env.render(**data))
                elif cmd == Cmd.seed:
                    self.p.send(self.env.seed(data))
                else:
                    self.p.close()
                    raise NotImplementedError
        except KeyboardInterrupt:
            self.p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.closed = False
        self.parent_remote, self.child_remote = zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            ProcWorker(parent, child, env_fn)
            for (parent, child, env_fn) in zip(self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def step(self, action):
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send([Cmd.step, a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def sample(self):
        for p in self.parent_remote:
            p.send([Cmd.sample, None])
        act = np.stack([p.recv() for p in self.parent_remote])
        return act

    def reset(self, id=None):
        if id is None:
            for p in self.parent_remote:
                p.send([Cmd.reset, None])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self.parent_remote[i].send([Cmd.reset, None])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send([Cmd.seed, s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs):
        for p in self.parent_remote:
            p.send([Cmd.render, kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self):
        if self.closed:
            return
        for p in self.parent_remote:
            p.send([Cmd.close, None])
        result = [p.recv() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()
        return result


class ShmWorker(Process):
    def __init__(self, idx, env_num, env_fn, run_condition, worker_done_cnt, worker_ready_cnt, after_cmd,
                 action_arr, obs_arr, reward_arr, done_arr, cmd, cmd_mask):
        Process.__init__(self)
        self.idx = idx
        self.env_num = env_num
        self.env = env_fn()
        obs_shape = self.env.observation_space.shape or ()
        action_shape = self.env.action_space.shape or ()
        obs_type = ctypes.c_float if isinstance(self.env.observation_space, gym.spaces.Box) else ctypes.c_int
        action_type = ctypes.c_float if isinstance(self.env.action_space, gym.spaces.Box) else ctypes.c_int
        self._obs = np.frombuffer(obs_arr, dtype=obs_type).reshape((self.env_num,) + obs_shape)
        self._action = np.frombuffer(action_arr, dtype=action_type).reshape((self.env_num,) + action_shape)
        self._reward = np.frombuffer(reward_arr, dtype=ctypes.c_float).reshape((self.env_num,))
        self._done = np.frombuffer(done_arr, dtype=ctypes.c_bool).reshape((self.env_num,))

        self.run_condition = run_condition
        self.worker_done_cnt = worker_done_cnt
        self.worker_ready_cnt = worker_ready_cnt
        self._cmd = cmd
        self._cmd_mask = cmd_mask
        self._after_cmd = after_cmd
        self.close = 0

    def run(self):
        # with self.worker_ready_cnt.get_lock():
        #     self.worker_ready_cnt.value += 1
        self._after_cmd.wait()

        while not self.close:
            try:
                # print(self.idx, 'started')
                with self.run_condition:
                    # print(self.idx, 'waiting for cmd')
                    self.run_condition.wait()
                    # self.run_condition.notify_all()
                # with self.worker_ready_cnt.get_lock():
                #     self.worker_ready_cnt.value -= 1
                # print(self.idx, self._cmd)
                if self._cmd_mask[self.idx]:
                    if self._cmd.value == Cmd.step:
                        obs, reward, done, info = self.env.step(self._action[self.idx])
                        self._obs[self.idx], self._reward[self.idx], self._done[self.idx] = obs, reward, done
                    elif self._cmd.value == Cmd.reset:
                        self._obs[self.idx] = self.env.reset()
                    elif self._cmd.value == Cmd.sample:
                        self._action[self.idx] = self.env.action_space.sample()
                    elif self._cmd.value == Cmd.close:
                        self.env.close()
                        self.close = 1
                    elif self._cmd.value == Cmd.render:
                        self.env.render()
                    elif self._cmd.value == Cmd.seed:
                        self._reward[self.idx] = self.env.seed(int(self._reward[self.idx]))[0]
                    else:
                        raise NotImplementedError()
                with self.worker_done_cnt.get_lock():
                    self.worker_done_cnt.value += 1
                self._after_cmd.wait()
            except KeyboardInterrupt:
                self.close = 1
        # while self.worker_done_cnt.value != 0:
        #     ...


class ShmVecEnv(BaseVectorEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.worker_done_cnt = mp.Value('i', 0)
        self.worker_ready_cnt = mp.Value('i', 0)
        self.run_condition = mp.Condition()
        tmp_env = env_fns[0]()
        obs_shape = tmp_env.observation_space.shape or ()
        action_shape = tmp_env.action_space.shape or ()
        obs_type = ctypes.c_float if isinstance(tmp_env.observation_space, gym.spaces.Box) else ctypes.c_int
        action_type = ctypes.c_float if isinstance(tmp_env.action_space, gym.spaces.Box) else ctypes.c_int
        obs_array = mp.Array(obs_type, np.prod((self.env_num,) + obs_shape).item(), lock=False)
        action_array = mp.Array(action_type, np.prod((self.env_num,) + action_shape).item(), lock=False)
        reward_array = mp.Array(ctypes.c_float, self.env_num, lock=False)
        done_array = mp.Array(ctypes.c_bool, self.env_num, lock=False)
        self._obs = np.frombuffer(obs_array, dtype=obs_type).reshape((self.env_num,) + obs_shape)
        self._action = np.frombuffer(action_array, dtype=action_type).reshape((self.env_num,) + action_shape)
        self._reward = np.frombuffer(reward_array, dtype=ctypes.c_float).reshape((self.env_num,))
        self._done = np.frombuffer(done_array, dtype=ctypes.c_bool).reshape((self.env_num,))
        self._cmd_mask = mp.Array(ctypes.c_bool, self.env_num, lock=False)
        self._cmd = mp.Value(ctypes.c_int, lock=False)
        self._after_cmd = mp.Barrier(self.env_num + 1)
        self.processes = [
            ShmWorker(i, self.env_num, env_fns[i], self.run_condition, self.worker_done_cnt, self.worker_ready_cnt,
                      self._after_cmd,
                      action_array, obs_array, reward_array, done_array, self._cmd, self._cmd_mask)
            for i in range(len(env_fns))
        ]
        for p in self.processes: p.start()
        self.closed = False
        self._info = {}
        self._after_cmd.wait()

    def fire_cmd(self):
        with self.run_condition:
            self.run_condition.notify_all()
        while self.worker_done_cnt.value != self.env_num:
            with self.run_condition:
                self.run_condition.notify_all()
        self._after_cmd.wait()
        with self.worker_done_cnt.get_lock():
            self.worker_done_cnt.value = 0
        # while self.worker_ready_cnt.value != self.env_num:
        #     ...

    def step(self, action):
        assert len(action) == self.env_num
        self._cmd.value = Cmd.step
        for i in range(self.env_num): self._cmd_mask[i] = True
        for i in range(len(self._action)):
            self._action[i] = action[i]
        self.fire_cmd()
        return self._obs.copy(), self._reward.copy(), self._done.copy(), self._info

    def sample(self):
        self._cmd.value = Cmd.sample
        for i in range(self.env_num): self._cmd_mask[i] = True
        self.fire_cmd()
        return self._action.copy()

    def reset(self, id=None):
        self._cmd.value = Cmd.reset
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        for i in range(self.env_num):
            if i in id:
                self._cmd_mask[i] = True
            else:
                self._cmd_mask[i] = False
        self.fire_cmd()
        return self._obs.copy()

    def seed(self, seed=0):
        self._cmd.value = Cmd.seed
        for i in range(self.env_num): self._cmd_mask[i] = True
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        assert len(seed) == self.env_num
        for i in range(self.env_num):
            self._reward[i] = seed[i]
        self.fire_cmd()
        return self._reward.copy()

    def render(self, **kwargs):
        self._cmd.value = Cmd.render
        for i in range(self.env_num): self._cmd_mask[i] = True
        self.fire_cmd()

    def close(self):
        self._cmd.value = Cmd.close
        if self.closed:
            return
        for i in range(self.env_num): self._cmd_mask[i] = True
        self.fire_cmd()
        self.closed = True
        for p in self.processes:
            p.join()


###### pipe version
class ShmPipeWorker(mp.Process):
    def __init__(self, idx, env_num, env_fn, action_arr, obs_arr, reward_arr, done_arr, parent, p):
        super(ShmPipeWorker, self).__init__()
        self.idx = idx
        self.env_num = env_num
        self.env = env_fn()
        obs_shape = self.env.observation_space.shape or ()
        action_shape = self.env.action_space.shape or ()
        obs_type = ctypes.c_float if isinstance(self.env.observation_space, gym.spaces.Box) else ctypes.c_int
        action_type = ctypes.c_float if isinstance(self.env.action_space, gym.spaces.Box) else ctypes.c_int
        self._obs = np.frombuffer(obs_arr, dtype=obs_type).reshape((self.env_num,) + obs_shape)
        self._action = np.frombuffer(action_arr, dtype=action_type).reshape((self.env_num,) + action_shape)
        self._reward = np.frombuffer(reward_arr, dtype=ctypes.c_float).reshape((self.env_num,))
        self._done = np.frombuffer(done_arr, dtype=ctypes.c_bool).reshape((self.env_num,))
        self.parent = parent
        self.p = p
        self.close = 0

    def run(self):
        self.parent.close()
        response = bytes([1])
        # with self.worker_ready_cnt.get_lock():
        #     self.worker_ready_cnt.value += 1
        while not self.close:
            try:
                cmd = int.from_bytes(self.p.recv_bytes(), 'little')
                if cmd == Cmd.step:
                    obs, reward, done, info = self.env.step(self._action[self.idx])
                    self._obs[self.idx], self._reward[self.idx], self._done[self.idx] = obs, reward, done
                    self.p.send_bytes(response)
                elif cmd == Cmd.reset:
                    self._obs[self.idx] = self.env.reset()
                    self.p.send_bytes(response)
                elif cmd == Cmd.sample:
                    self._action[self.idx] = self.env.action_space.sample()
                    self.p.send_bytes(response)
                elif cmd == Cmd.close:
                    self.env.close()
                    self.close = 1
                    self.p.send_bytes(response)
                elif cmd == Cmd.render:
                    self.env.render()
                    self.p.send_bytes(response)
                elif cmd == Cmd.seed:
                    self._reward[self.idx] = self.env.seed(int(self._reward[self.idx]))[0]
                    self.p.send_bytes(response)
                else:
                    self.close = 1
                    raise NotImplementedError()
            except KeyboardInterrupt:
                self.close = 1


class ShmPipeVecEnv(BaseVectorEnv):
    def __init__(self, env_fns):
        super(ShmPipeVecEnv, self).__init__(env_fns)
        tmp_env = env_fns[0]()
        obs_shape = tmp_env.observation_space.shape or ()
        action_shape = tmp_env.action_space.shape or ()
        obs_type = ctypes.c_float if isinstance(tmp_env.observation_space, gym.spaces.Box) else ctypes.c_int
        action_type = ctypes.c_float if isinstance(tmp_env.action_space, gym.spaces.Box) else ctypes.c_int
        obs_array = mp.Array(obs_type, np.prod((self.env_num,) + obs_shape).item(), lock=False)
        action_array = mp.Array(action_type, np.prod((self.env_num,) + action_shape).item(), lock=False)
        reward_array = mp.Array(ctypes.c_float, self.env_num, lock=False)
        done_array = mp.Array(ctypes.c_bool, self.env_num, lock=False)
        self._obs = np.frombuffer(obs_array, dtype=obs_type).reshape((self.env_num,) + obs_shape)
        self._action = np.frombuffer(action_array, dtype=action_type).reshape((self.env_num,) + action_shape)
        self._reward = np.frombuffer(reward_array, dtype=ctypes.c_float).reshape((self.env_num,))
        self._done = np.frombuffer(done_array, dtype=ctypes.c_bool).reshape((self.env_num,))
        self.parent_remote, self.child_remote = zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            ShmPipeWorker(i, self.env_num, env_fns[i], action_array, obs_array, reward_array, done_array,
                          self.parent_remote[i], self.child_remote[i])
            for i in range(self.env_num)
        ]
        for p in self.processes: p.start()
        self.closed = False
        self._info = {}
        # while self.worker_ready_cnt.value != self.env_num:
        #     ...

    def step(self, action):
        assert len(action) == self.env_num
        cmd = Cmd.step.to_bytes(1, 'little')
        for i in range(len(self._action)):
            self._action[i] = action[i]
        fire = [p.send_bytes(cmd) for p in self.parent_remote]
        result = [p.recv_bytes() for p in self.parent_remote]
        return self._obs.copy(), self._reward.copy(), self._done.copy(), self._info

    def sample(self):
        cmd = Cmd.sample.to_bytes(1, 'little')
        for p in self.parent_remote:
            p.send_bytes(cmd)
        result = [p.recv_bytes() for p in self.parent_remote]
        return self._action.copy()

    def reset(self, id=None):
        cmd = Cmd.reset.to_bytes(1, 'little')
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        for i in id:
            self.parent_remote[i].send_bytes(cmd)
        result = [self.parent_remote[i].recv_bytes() for i in id]
        return self._obs.copy()

    def seed(self, seed=0):
        cmd = Cmd.seed.to_bytes(1, 'little')
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        assert len(seed) == self.env_num
        for p in self.parent_remote:
            p.send_bytes(cmd)
        result = [p.recv_bytes() for p in self.parent_remote]
        return self._reward.copy()

    def render(self, **kwargs):
        cmd = Cmd.render.to_bytes(1, 'little')
        fire = [p.send_bytes(cmd) for p in self.parent_remote]
        result = [p.recv_bytes() for p in self.parent_remote]

    def close(self):
        cmd = Cmd.close.to_bytes(1, 'little')
        if self.closed:
            return
        fire = [p.send_bytes(cmd) for p in self.parent_remote]
        result = [p.recv_bytes() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. However, according to our
    test, it is about two times slower than
    :class:`~tianshou.env.SubprocVectorEnv`.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip3 install ray')
        self.envs = [
            ray.remote(gym.Wrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def step(self, action):
        assert len(action) == self.env_num
        result = ray.get([e.step.remote(a) for e, a in zip(self.envs, action)])
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id=None):
        if id is None:
            result_obj = [e.reset.remote() for e in self.envs]
            self._obs = np.stack(ray.get(result_obj))
        else:
            result_obj = []
            if np.isscalar(id):
                id = [id]
            for i in id:
                result_obj.append(self.envs[i].reset.remote())
            for _, i in enumerate(id):
                self._obs[i] = ray.get(result_obj[_])
        return self._obs

    def seed(self, seed=None):
        if not hasattr(self.envs[0], 'seed'):
            return
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        return ray.get([e.seed.remote(s) for e, s in zip(self.envs, seed)])

    def render(self, **kwargs):
        if not hasattr(self.envs[0], 'render'):
            return
        return ray.get([e.render.remote(**kwargs) for e in self.envs])

    def close(self):
        return ray.get([e.close.remote() for e in self.envs])
