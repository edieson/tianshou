import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu',
                 softmax=False):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s, **kwargs):
        logits, h = self.preprocess(s, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


class Recurrent(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.nn = nn.LSTM(input_size=128, hidden_size=128, num_layers=layer_num)
        self.fc2 = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if state is not None and not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s)
        s = s.view(length, bsz, -1)
        hidden = None
        if state is not None:
            state = state.view(length, bsz, -1)
            hn, cn = torch.chunk(state, 2, dim=-1)
            hidden = (hn.contiguous(), cn.contiguous())
        self.nn.flatten_parameters()
        # we store the stack data in [bsz, len, ...] format
        s, (h, c) = self.nn(s, hidden)
        s = self.fc2(s[-1, :])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, torch.cat((h, c), -1).detach()
