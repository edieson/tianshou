import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import ShmPipeVecEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer

if __name__ == '__main__':
    from net import ActorProb, DQCritic
else:  # pytest
    from test.continuous.net import ActorProb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--il-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--step-per-epoch', type=int, default=2000)
    parser.add_argument('--collect-per-step', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--test-episode', type=int, default=30)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--comment', type=str, default='sac')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--rew-norm', type=bool, default=False)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    log_path = os.path.join(args.logdir, args.task, args.comment)
    writer = SummaryWriter(log_path)
    return args, log_path, writer


from gym.wrappers import TransformReward


class BipedalWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=3):
        super(BipedalWrapper, self).__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        act_noise = 0.3 * (np.random.random(action.shape) * 2 - 1)
        action += act_noise
        r = 0.0
        obs_, reward_, done_, info_ = self.env.step(action)
        for i in range(self.action_repeat - 1):
            obs_, reward_, done_, info_ = self.env.step(action)
            r = r + reward_
            if done_:
                return obs_, 0.0, done_, info_
        return obs_, r, done_, info_


def test_sac():
    args, log_path, writer = get_args()
    env = gym.make(args.task)
    if args.task == 'Pendulum-v0':
        env.spec.reward_threshold = -250
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = ShmPipeVecEnv([
        lambda: TransformReward(BipedalWrapper(gym.make(args.task)), lambda reward: 5 * reward)
        for _ in range(args.training_num)
    ])
    # test_envs = gym.make(args.task)
    test_envs = ShmPipeVecEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed + 1)
    # model
    actor = ActorProb(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic = DQCritic(args.layer_num, args.state_shape, args.action_shape, args.device).to(args.device)
    critic_target = DQCritic(args.layer_num, args.state_shape, args.action_shape, args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = SACPolicy(
        actor, actor_optim, critic, critic_optim, critic_target, env.action_space,
        args.device, args.tau, args.gamma, args.alpha,
        reward_normalization=args.rew_norm, ignore_done=False)

    if args.mode == 'test':
        policy.load_state_dict(
            torch.load("{}/{}/{}/policy.pth".format(args.logdir, args.task, args.comment), map_location=args.device)
        )
        env = gym.make(args.task)
        collector = Collector(
            policy,
            env
            # Monitor(env, 'video', force=True)
        )
        result = collector.collect(n_episode=10, render=args.render)
        print(f'Final reward: {result["ep/reward"]}, length: {result["ep/len"]}')
        collector.close()
        exit()
    # collector
    train_collector = Collector(policy, train_envs, ReplayBuffer(args.buffer_size))
    train_collector.collect(10000, sampling=True)
    test_collector = Collector(policy, test_envs)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold + 5

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_episode,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    assert stop_fn(result['best_reward'])

    pprint.pprint(result)


if __name__ == '__main__':
    test_sac()
