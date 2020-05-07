import time
import torch
import warnings
import numpy as np

from tianshou.utils import MovAvg
from tianshou.env import BaseVectorEnv, VectorEnv
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer


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

    def __init__(self, policy, env, buffer=None, stat_size=100, **kwargs):
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            self.env = VectorEnv([env])
        else:
            self.env = env
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0
        self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        # need multiple cache buffers only if storing in one buffer
        self._cached_buf = []
        if isinstance(self.buffer, ReplayBuffer):
            self._cached_buf = [ListReplayBuffer() for _ in range(self.env.env_num)]
        elif not self.buffer is None:
            raise TypeError('The buffer in data collector is invalid!')
        self.stat_size = stat_size
        self.reset()

    def reset(self):
        """Reset all related variables in the collector."""
        self.reset_env()
        self.reset_buffer()
        # state over batch is either a list, an np.ndarray, or a torch.Tensor
        self.state = None
        self.step_speed = MovAvg(self.stat_size)
        self.episode_speed = MovAvg(self.stat_size)
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0

    def reset_buffer(self):
        """Reset the main data buffer."""
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
        self._act = self._rew = self._done = self._info = None

        self.reward = np.zeros(self.env.env_num)
        self.length = np.zeros(self.env.env_num)

        for b in self._cached_buf:
            b.reset()

    def seed(self, seed=None):
        """Reset all the seed(s) of the given environment(s)."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        """Render all the environment(s)."""
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        """Close the environment(s)."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def _reset_state(self, id):
        """Reset self.state[id]."""
        if self.state is None:
            return
        if isinstance(self.state, list):
            self.state[id] = None
        elif isinstance(self.state, dict):
            for k in self.state:
                if isinstance(self.state[k], list):
                    self.state[k][id] = None
                elif isinstance(self.state[k], torch.Tensor) or \
                        isinstance(self.state[k], np.ndarray):
                    self.state[k][id] = 0
        elif isinstance(self.state, torch.Tensor) or \
                isinstance(self.state, np.ndarray):
            self.state[id] = 0

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

    def collect(self, n_step=0, n_episode=0, sampling=False, render=None, log_fn=None):
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :type n_episode: int or list
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param function log_fn: a function which receives env info, typically
            for tensorboard logging.

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
        reward_sum = 0
        length_sum = 0
        while True:
            if warning_count >= 100000:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            batch_data = Batch(
                obs=self._obs, act=self._act, rew=self._rew,
                done=self._done, obs_next=None, info=self._info
            )
            if sampling:
                self._act = self.env.sample()
                self.state = None
                self._policy = [{}] * self.env.env_num
            else:
                with torch.no_grad():
                    result = self.policy(batch_data, self.state)
                self.state = result.state if hasattr(result, 'state') else None
                self._policy = self._to_numpy(result.policy) \
                    if hasattr(result, 'policy') \
                    else [{}] * self.env.env_num
                if isinstance(result.act, torch.Tensor):
                    self._act = self._to_numpy(result.act)
                elif not isinstance(self._act, np.ndarray):
                    self._act = np.array(result.act)
                else:
                    self._act = result.act
            obs_next, self._rew, self._done, self._info = self.env.step(self._act)
            if log_fn is not None:
                log_fn(self._info)
            if render is not None:
                self.env.render()
                if render > 0:
                    time.sleep(render)
            self.length += 1
            self.reward += self._rew

            for i in range(self.env.env_num):
                data = {
                    'obs': self._obs[i], 'act': self._act[i],
                    'rew': self._rew[i], 'done': self._done[i],
                    'obs_next': obs_next[i], 'info': self._info[i],
                    'policy': self._policy[i]}
                if self._cached_buf:
                    warning_count += 1
                    self._cached_buf[i].add(**data)
                if self._done[i]:
                    cur_episode[i] += 1
                    reward_sum += self.reward[i]
                    length_sum += self.length[i]
                    if self._cached_buf:
                        cur_step += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.update(self._cached_buf[i])
                            self._cached_buf[i].reset()
                    self.reward[i], self.length[i] = 0, 0
                    self._reset_state(i)
            if sum(self._done):
                obs_next = self.env.reset(np.where(self._done)[0])
            self._obs = obs_next
            if n_episode:
                if np.sum(cur_episode) >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
        cur_episode = sum(cur_episode)
        duration = max(time.time() - start_time, 1e-9)
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'ep/reward': reward_sum / cur_episode,
            'ep/len': length_sum / cur_episode,
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
