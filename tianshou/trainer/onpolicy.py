import collections
import time

import numpy as np
import tqdm

from tianshou.trainer import test_episode, gather_info
from tianshou.utils import tqdm_config


def onpolicy_trainer(policy,
                     train_collector,
                     test_collector,
                     max_epoch,
                     step_per_epoch,
                     collect_per_step,
                     repeat_per_collect,
                     episode_per_test,
                     batch_size,
                     train_fn=None,
                     stop_fn=None,
                     save_fn=None,
                     test_in_train=False,
                     writer=None,
                     log_interval=10,
                     verbose=True,
                     **kwargs):
    """A wrapper for on-policy trainer procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of frames the collector would
        collect before the network update. In other words, collect some frames
        and do one policy network update.
    :param int repeat_per_collect: the number of repeat time for policy
        learning, for example, set it to 2 means the policy needs to learn each
        given batch data twice.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :type episode_per_test: int or list of ints
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param function log_fn: a function receives env info for logging.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = collections.defaultdict(lambda: collections.deque([], 30))
    start_time = time.time()
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}', **tqdm_config) as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=collect_per_step)
                data = {}
                if test_in_train and stop_fn and stop_fn(result['ep/reward']):
                    test_result = test_episode(policy, test_collector, episode_per_test)
                    if stop_fn and stop_fn(test_result['ep/reward']):
                        if save_fn:
                            save_fn(policy)
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['ep/reward'])
                    else:
                        policy.train()
                        if train_fn:
                            train_fn(epoch)
                # for i in range(repeat_per_collect):
                losses = policy.learn(train_collector.sample(0), batch_size, repeat_per_collect)
                train_collector.reset_buffer()
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                global_step += step
                for k in result.keys():
                    if not k[0] in ['v', 'n']:
                        data[k] = f'{result[k]:.2f}'
                for k in losses.keys():
                    stat[k].extend(losses[k])
                    if not k[0] in ['g']:
                        data[k] = f'{np.nanmean(stat[k]):.1f}'
                if writer and global_step % log_interval == 0:
                    for k in result.keys():
                        writer.add_scalar(k, result[k], global_step=global_step)
                    for k in losses.keys():
                        writer.add_scalar(k, np.nanmean(stat[k]), global_step=global_step)
                t.update(step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(policy, test_collector, episode_per_test)
        if best_epoch == -1 or best_reward < result['ep/reward']:
            best_reward = result['ep/reward']
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["ep/reward"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)
