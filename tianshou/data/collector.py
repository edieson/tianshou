import collections
import time
import warnings
from collections import namedtuple

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.env import BaseVectorEnv, VectorEnv

Experience = namedtuple('Exp', ['hidden', 'obs', 'act', 'reward', 'obs_next', 'done'])

HIDDEN_SIZE = 256


class Collector(object):
    """The :class:`~tianshou.data.Collector` enables the policy to interact
    with different types of environments conveniently.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: an environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
        ``None``, it will automatically assign a small-size
        :class:`~tianshou.data.ReplayBuffer`.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.

    Example:
    ::

        policy = PGPolicy(...)  # or other policies if you wish
        env = gym.make('CartPole-v0')
        replay_buffer = ReplayBuffer(size=10000)
        # here we set up a collector with a single environment
        collector = Collector(policy, env, buffer=replay_buffer)

        # the collector supports vectorized environments as well
        envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(3)])
        buffers = [ReplayBuffer(size=5000) for _ in range(3)]
        # you can also pass a list of replay buffer to collector, for multi-env
        # collector = Collector(policy, envs, buffer=buffers)
        collector = Collector(policy, envs, buffer=replay_buffer)

        # collect at least 3 episodes
        collector.collect(n_episode=3)
        # collect 1 episode for the first env, 3 for the third env
        collector.collect(n_episode=[1, 0, 3])
        # collect at least 2 steps
        collector.collect(n_step=2)
        # collect episodes with visual rendering (the render argument is the
        #   sleep time between rendering consecutive frames)
        collector.collect(n_episode=1, render=0.03)

        # sample data with a given number of batch-size:
        batch_data = collector.sample(batch_size=64)
        # policy.learn(batch_data)  # btw, vanilla policy gradient only
        #   supports on-policy training, so here we pick all data in the buffer
        batch_data = collector.sample(batch_size=0)
        policy.learn(batch_data)
        # on-policy algorithms use the collected data only once, so here we
        #   clear the buffer
        collector.reset_buffer()

    For the scenario of collecting data from multiple environments to a single
    buffer, the cache buffers will turn on automatically. It may return the
    data more than the given limitation.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(self, policy, env, buffer=None, episodic=False, stat_size=5, **kwargs):
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            self.env = VectorEnv([env])
        else:
            self.env = env
        self._collect_step = 0
        self._collect_episode = 0
        self._collect_time = 0
        self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        self._episodic = episodic
        if self._episodic and buffer is not None:
            self._cached_buf = [ReplayBuffer(buffer._maxsize // self.env.env_num) for _ in range(self.env.env_num)]
        self.stat_size = stat_size
        self._step_speed = collections.deque([], self.stat_size)
        self._episode_speed = collections.deque([], self.stat_size)
        self._episode_length = collections.deque([], self.stat_size)
        self._episode_reward = collections.deque([], self.stat_size)
        self.reset()

    def reset(self):
        """Reset all related variables in the collector."""
        self.reset_env()
        self.reset_buffer()
        # state over batch is either a list, an np.ndarray, or a torch.Tensor
        self._step_speed.clear()
        self._episode_speed.clear()
        self._episode_length.clear()
        self._episode_reward.clear()
        self._collect_step = 0
        self._collect_episode = 0
        self._collect_time = 0

    def reset_buffer(self):
        """Reset the main data buffer."""
        if self._episodic:
            [b.reset() for b in self._cached_buf]
        if self.buffer is not None:
            self.buffer.reset()

    def get_env_num(self):
        """Return the number of environments the collector has."""
        return self.env.env_num

    def reset_env(self):
        """Reset all of the environment(s)' states and reset all of the cache
        buffers (if need).
        """
        self._obs = self.env.reset()
        self._act = self._rew = self._done = None

        self._hidden_next = self._hidden = np.zeros((self.get_env_num(), HIDDEN_SIZE))

        self.reward = np.zeros(self.env.env_num)
        self.length = np.zeros(self.env.env_num)

    def seed(self, seed=None):
        """Reset all the seed(s) of the given environment(s)."""
        return self.env.seed(seed)

    def render(self, **kwargs):
        """Render all the environment(s)."""
        return self.env.render(**kwargs)

    def close(self):
        """Close the environment(s)."""
        self.env.close()


    def _to_numpy(self, x):
        """Return an object without torch.Tensor."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, dict):
            for k in x:
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].cpu().numpy()
            return x
        elif isinstance(x, Batch):
            x.to_numpy()
            return x
        return x

    def collect(self, n_step=0, n_episode=0, sampling=False, render=None):
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :type n_episode: int or list
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).

        .. note::

            One and only one collection number specification is permitted,
            either ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``v/st`` the speed of steps per second.
            * ``v/ep`` the speed of episode per second.
            * ``rew`` the mean reward over collected episodes.
            * ``len`` the mean length over collected episodes.
        """
        warning_count = 0
        start_time = time.time()
        assert not (n_step and n_episode), "One and only one collection number specification is permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env.env_num)
        while True:
            if warning_count >= 100000:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            batch_data = Batch(obs=self._obs, act=self._act, rew=self._rew, done=self._done)
            if sampling == True:
                self._act = self.env.sample()
            else:
                with torch.no_grad():
                    result = self.policy(batch_data, self._hidden)
                if hasattr(result, 'hidden') and result.hidden is not None:
                    self._hidden_next = result.hidden
                if isinstance(result.act, torch.Tensor):
                    self._act = self._to_numpy(result.act)
                elif not isinstance(self._act, np.ndarray):
                    self._act = np.array(result.act)
                else:
                    self._act = result.act
            obs_next, self._rew, self._done, _ = self.env.step(self._act)
            if render is not None:
                self.env.render()
                if render > 0:
                    time.sleep(render)
            self.length += 1
            self.reward += self._rew

            for i in range(self.env.env_num):
                warning_count += 1
                collection = Experience(
                    self._hidden[i], self._obs[i], self._act[i], self._rew[i], obs_next[i], self._done[i]
                )
                if not self._episodic:
                    cur_step += 1
                    if self.buffer is not None:
                        self.buffer.add(collection)
                else:
                    self._cached_buf[i].add(collection)
                if self._done[i]:
                    if self._episodic:
                        cur_step += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.extend(self._cached_buf[i])
                    cur_episode[i] += 1
                    self._episode_reward.append(self.reward[i])
                    self._episode_length.append(self.length[i])
                    self.reward[i], self.length[i] = 0, 0

            if sum(self._done):
                ids = np.where(self._done)[0]
                obs_next = self.env.reset(ids)
                self._hidden_next[self._done] = 0.
            self._obs = obs_next
            self._hidden = self._hidden_next

            if n_episode and np.sum(cur_episode) >= n_episode:
                break
            if n_step != 0 and cur_step >= n_step:
                break
        cur_episode = sum(cur_episode)
        duration = time.time() - start_time
        self._step_speed.append(cur_step / duration)
        self._episode_speed.append(cur_episode / duration)
        self._collect_step += cur_step
        self._collect_episode += cur_episode
        self._collect_time += duration
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'n/buffer': len(self.buffer) if self.buffer else 0,
            'v/st': np.nanmean(self._step_speed),
            'v/ep': np.nanmean(self._episode_speed) if self._collect_episode else 0,
            'ep/reward': np.nanmean(self._episode_reward) if self._collect_episode else 0,
            'ep/len': np.nanmean(self._episode_length) if self._collect_episode else 0,
        }

    def sample(self, batch_size):
        """Sample a data batch from the internal replay buffer. It will call
        :meth:`~tianshou.policy.BasePolicy.process_fn` before returning
        the final batch data.

        :param int batch_size: ``0`` means it will extract all the data from
            the buffer, otherwise it will extract the data with the given
            batch_size.
        """
        batch_data, indice = self.buffer.sample(batch_size)
        batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data
