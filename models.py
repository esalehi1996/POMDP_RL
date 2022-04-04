from torch.nn import ELU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


class rho_net(nn.Module):
    def __init__(self, num_obs, num_actions, AIS_state_size=5):
        super(rho_net, self).__init__()
        input_ndims = num_obs + num_actions
        # input_ndims = num_obs
        self.AIS_state_size = AIS_state_size
        self.fc1 = nn.Linear(input_ndims, AIS_state_size)
        self.lstm1 = nn.LSTM(AIS_state_size, AIS_state_size, batch_first=True)

        self.apply(weights_init_)

    def forward(self, x, batch_size, hidden, device):
        if hidden == None:
            hidden = (torch.zeros(1, batch_size, self.AIS_state_size).to(device),
                      torch.zeros(1, batch_size, self.AIS_state_size).to(device))
        x = F.elu(self.fc1(x))
        x, hidden = self.lstm1(x, hidden)
        return x, hidden


# psi is the same as \hat P^y in the paper
class psi_net(nn.Module):
    def __init__(self, num_obs, num_actions, AIS_state_size=5):
        super(psi_net, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.softmax = nn.Softmax(dim=2)
        self.fc1_r = nn.Linear(input_ndims, int(AIS_state_size / 2))
        self.fc1_d = nn.Linear(input_ndims, int(AIS_state_size / 2))
        self.fc2_r = nn.Linear(int(AIS_state_size / 2), 1)
        self.fc2_d = nn.Linear(int(AIS_state_size / 2), num_obs)

    def forward(self, x):
        x_r = F.elu(self.fc1_r(x))
        x_d = F.elu(self.fc1_d(x))
        reward = self.fc2_r(x_r)
        obs_probs = self.softmax(self.fc2_d(x_d))
        return reward, obs_probs


class policy_net(nn.Module):
    def __init__(self, num_actions, AIS_state_size=5, exploration_temp=1.):
        super(policy_net, self).__init__()
        self.exploration_temp = exploration_temp
        input_ndims = AIS_state_size
        self.fc1 = nn.Linear(input_ndims, AIS_state_size)
        self.fc2 = nn.Linear(AIS_state_size, num_actions)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        action_logits = self.fc2(x)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, x):
        x = F.elu(self.fc1(x))
        action_logits = self.fc2(x)
        # print(action_logits.shape)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class policy_net_true_state(nn.Module):
    def __init__(self, num_actions, num_states, AIS_state_size=5):
        super(policy_net_true_state, self).__init__()
        input_ndims = num_states
        self.fc1 = nn.Linear(input_ndims, AIS_state_size)
        self.fc2 = nn.Linear(AIS_state_size, num_actions)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        action_logits = self.fc2(x)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, x):
        x = F.elu(self.fc1(x))
        action_logits = self.fc2(x)
        # print(action_logits.shape)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class QNetwork_discrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=5):
        super(QNetwork_discrete, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        # print(xu.shape)

        x1 = F.elu(self.linear1(state))
        # x1 = F.tanh(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.elu(self.linear4(state))
        # x2 = F.tanh(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


def convert_int_to_onehot(value, num_values):
    onehot = torch.zeros(num_values)
    onehot[int(value)] = 1.
    return onehot


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)