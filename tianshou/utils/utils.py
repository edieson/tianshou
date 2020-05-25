import torch
import torch.nn as nn


def grad_statistics(net):
    grad_max = 0.0
    grad_means = 0.0
    grad_count = 0
    for p in net.parameters():
        if p.grad is None:
            continue
        grad_max = max(grad_max, p.grad.abs().max().item())
        grad_means += (p.grad ** 2).mean().sqrt().item()
        grad_count += 1
    return grad_max, grad_means / grad_count


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
