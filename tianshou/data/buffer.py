import numpy as np

from tianshou.data.batch import Batch


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, index):
        return self._storage[index]

    def reset(self):
        self._storage.clear()
        self._next_idx = 0

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, oth_buffer):
        i = begin = oth_buffer._next_idx % len(oth_buffer)
        while True:
            self.add(oth_buffer[i])
            i = (i + 1) % len(oth_buffer)
            if i == begin:
                break

    def _encode_sample(self, idxes):
        hidden, obs, act, rew, obs_next, done = map(np.array, zip(*[self._storage[i] for i in idxes]))
        return Batch(hidden=hidden,
                     obs=obs,
                     act=act,
                     rew=np.expand_dims(rew, axis=1),
                     done=np.expand_dims(done, axis=1),
                     obs_next=obs_next,
                     )
        # hiddens, obs_ts, actions, rewards, obs_tp1s, dones = [], [], [], [], [], []
        # for i in idxes:
        #     data = self._storage[i]
        #     hidden, obs_t, action, reward, obs_tp1, done = data
        #     hiddens.append(hidden)
        #     obs_ts.append(obs_t)
        #     actions.append(action)
        #     rewards.append([reward])
        #     obs_tp1s.append(obs_tp1)
        #     dones.append([done])
        # return Batch(
        #     obs=np.array(obs_ts),
        #     act=np.array(actions),
        #     rew=np.array(rewards),
        #     done=np.array(dones),
        #     obs_next=np.array(obs_tp1s),
        #     hidden=np.array(hiddens)
        # )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = np.arange(len(self._storage)) if batch_size <= 0 else np.random.choice(len(self._storage), batch_size)
        return self._encode_sample(idxes), idxes


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, beta):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha
        self._beta = beta
        self._weight_sum = 0.0
        self._weights = np.zeros(size, dtype=np.float64)
        self._amortization_freq = 50
        self._amortization_counter = 0
        # self._it_capacity = 1
        # while self._it_capacity < size:
        #     self._it_capacity *= 2

        # self._it_sum = SumSegmentTree(self._it_capacity)
        # self._it_min = MinSegmentTree(self._it_capacity)

    def add(self, data, weight=1.0):
        """See ReplayBuffer.store_effect"""
        self._weight_sum += np.abs(weight) ** self._alpha - self._weights[self._next_idx]
        self._weights[self._next_idx] = np.abs(weight) ** self._alpha
        super().add(data)
        self._check_weight_sum()
        # self._it_sum[idx] = self._max_priority ** self._alpha
        # self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        indice = np.random.choice(
            len(self._storage), batch_size,
            p=(self._weights / self._weights.sum())[:len(self._storage)],
            replace=False
        )
        return indice

    def sample(self, batch_size):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = np.arange(len(self._storage)) if batch_size == 0 else self._sample_proportional(batch_size)
        impt_weight = np.power(len(self._storage) * (self._weights[idxes] / self._weight_sum), -self._beta)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample.impt_weight = impt_weight
        self._check_weight_sum()
        return encoded_sample, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        self._weight_sum += np.power(np.abs(priorities), self._alpha).sum() - self._weights[idxes].sum()
        self._weights[idxes] = np.power(np.abs(priorities), self._alpha)

    def _check_weight_sum(self) -> None:
        # keep an accurate _weight_sum
        self._amortization_counter += 1
        if self._amortization_counter % self._amortization_freq == 0:
            self._weight_sum = np.sum(self._weights)
            self._amortization_counter = 0
