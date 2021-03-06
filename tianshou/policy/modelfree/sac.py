import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from tianshou.data import Batch
from tianshou.policy import DDPGPolicy
from tianshou.utils import grad_statistics

GRAD_L2_CLIP = 0.1


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param float alpha: entropy regularization coefficient, default to 0.2.
    :param action_range: the action range (minimum, maximum).
    :type action_range: [float, float]
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor, actor_optim, critic, critic_optim,
                 critic_target, action_space, device, tau=0.005, gamma=0.99,
                 alpha=0.2, **kwargs):
        super().__init__(None, None, None, None, tau, gamma, 0,
                         [action_space.low[0], action_space.high[0]],
                         **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic, self.critic_target = critic, critic_target
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = critic_optim
        self.critic_target.eval()
        self._alpha = alpha
        self.device = device
        # self._epsilon = np.finfo(np.float32).eps.item()
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(actor.device))
        self.log_alpha = torch.zeros(1, device=actor.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self._epsilon = 1e-6

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self):
        for o, n in zip(self.critic_target.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch, state=None, input='obs', **kwargs):
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state)
        assert isinstance(logits, tuple)
        dist = torch.distributions.Normal(*logits)
        x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        log_prob = dist.log_prob(x) - torch.log(self._action_scale * (1 - y.pow(2)) + self._epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, log_prob=log_prob)

    def learn(self, batch, **kwargs):
        rew = torch.tensor(batch.rew, dtype=torch.float, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            q1_next, q2_next = self.critic_target(batch.obs_next, obs_next_result.act)
            target_q = torch.min(q1_next, q2_next) - self._alpha * obs_next_result.log_prob
            target_q = (rew + (1. - done) * self._gamma * target_q)
        # critic
        current_q1, current_q2 = self.critic(batch.obs, batch.act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        # actor
        obs_result = self(batch)
        current_q1a, current_q2a = self.critic(batch.obs, obs_result.act)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(current_q1a, current_q2a)).mean()

        self.critic_optim.zero_grad()
        critic1_loss.backward()
        nn_utils.clip_grad_norm_(self.critic.parameters(), GRAD_L2_CLIP)
        critic1_grad_max, critic1_grad_l2 = grad_statistics(self.critic)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        critic2_loss.backward()
        nn_utils.clip_grad_norm_(self.critic.parameters(), GRAD_L2_CLIP)
        critic2_grad_max, critic2_grad_l2 = grad_statistics(self.critic)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn_utils.clip_grad_norm_(self.actor.parameters(), GRAD_L2_CLIP)
        actor_grad_max, actor_grad_l2 = grad_statistics(self.actor)
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (obs_result.log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self._alpha = self.log_alpha.exp()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
            'loss/alpha_loss': alpha_loss.item(),
            'entropy/temprature': self._alpha.clone().item(),
            'g/c1_max': critic1_grad_max,
            'g/c1_l2': critic1_grad_l2,
            'g/c2_max': critic2_grad_max,
            'g/c2_l2': critic2_grad_l2,
            'g/actor_max': actor_grad_max,
            'g/actor_l2': actor_grad_l2
        }
