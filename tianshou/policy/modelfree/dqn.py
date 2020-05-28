from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, PrioritizedReplayBuffer
from tianshou.policy import BasePolicy


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float gamma: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, model, optim, gamma=0.99,
                 estimation_step=1, target_update_freq=0, epsilon_decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 1
        self.eps_decay = epsilon_decay
        self.min_eps = 0.1
        assert 0 <= gamma <= 1, 'discount_factor should in [0, 1]'
        self._gamma = gamma
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._cnt = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()

    def set_eps(self, eps):
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self):
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.model.train()

    def eval(self):
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.model.eval()

    def sync_weight(self):
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def process_fn(self, batch, buffer, indice):
        r"""Compute the n-step return for Q-learning targets:

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) \max_a Q_{old}(s_{t + n}, \arg\max_a
            (Q_{new}(s_{t + n}, a)))

        , where :math:`\gamma` is the discount factor,
        :math:`\gamma \in [0, 1]`, :math:`d_t` is the done flag of step
        :math:`t`. If there is no target network, the :math:`Q_{old}` is equal
        to :math:`Q_{new}`.
        """
        returns = np.zeros_like(indice)
        gammas = np.zeros_like(indice) + self._n_step
        for n in range(self._n_step - 1, -1, -1):
            now = (indice + n) % len(buffer)
            gammas[[buffer[i].done > 0 for i in now]] = n
            returns[[buffer[i].done > 0 for i in now]] = 0
            returns = np.array([buffer[i].reward for i in now]) + self._gamma * returns
        terminal = (indice + self._n_step - 1) % len(buffer)
        terminal_batch = buffer._encode_sample(terminal)

        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(terminal_batch, input='obs_next', eps=0).act
            target_q = self(terminal_batch, model='model_old', input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q[np.arange(len(a)), a]
        else:
            target_q = self(terminal_batch, input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q.max(axis=1)
        target_q[gammas != self._n_step] = 0
        returns += (self._gamma ** gammas) * target_q
        batch.returns = returns
        if isinstance(buffer, PrioritizedReplayBuffer):
            q = self(batch).logits
            q = q[np.arange(len(q)), batch.act]
            r = batch.returns
            if isinstance(r, np.ndarray):
                r = torch.tensor(r, device=q.device, dtype=q.dtype)
            td = r - q
            buffer.update_priorities(indice, td.detach().cpu().numpy())
            impt_weight = torch.tensor(batch.impt_weight, device=q.device, dtype=torch.float)
            loss = (td.pow(2) * impt_weight).mean()
            if not hasattr(batch, 'loss'):
                batch.loss = loss
            else:
                batch.loss += loss
        return batch

    def forward(self, batch, state=None, model='model', input='obs', eps=None, **kwargs):
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        q, h = model(obs, state=state)
        act = q.max(dim=1)[1].detach().cpu().numpy()
        # add eps to act
        if eps is None:
            self.eps *= self.eps_decay
            eps = max(self.eps, self.min_eps)
        for i in range(len(q)):
            if np.random.rand() < eps:
                act[i] = np.random.randint(q.shape[1])
        return Batch(logits=q, act=act, state=h)

    def learn(self, batch, **kwargs):
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        if hasattr(batch, 'loss'):
            loss = batch.loss
        else:
            q = self(batch).logits
            q = q[np.arange(len(q)), batch.act]
            r = batch.returns
            if isinstance(r, np.ndarray):
                r = torch.tensor(r, device=q.device, dtype=q.dtype)
            loss = F.mse_loss(q, r)
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.item()}
