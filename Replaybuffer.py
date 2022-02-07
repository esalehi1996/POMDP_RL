from copy import deepcopy
import random
import torch
import numpy as np
import torch.nn.functional as F


class Rec_ReplayMemory:
    def __init__(self, capacity, obs_dim, act_dim, max_sequence_length, seed):
        random.seed(seed)
        self.capacity = capacity
        self.max_seq_len = max_sequence_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.buffer_states = np.zeros([self.capacity, self.max_seq_len], dtype=np.int32)
        self.buffer_actions = np.zeros([self.capacity, self.max_seq_len], dtype=np.int32)
        self.buffer_rewards = np.zeros([self.capacity, self.max_seq_len], dtype=np.float32)
        self.buffer_ep_len = np.zeros([self.capacity], dtype=np.int32)

        self.position = 0

        self.full = False

    def push(self, ep_states, ep_actions, ep_rewards):
        # if len(self.buffer_states) < self.capacity:
        #     self.buffer_states.append(None)
        #     self.buffer_actions.append(None)
        #     self.buffer_rewards.append(None)
        #     # self.buffer_true_states.append(None)
        # self.buffer_states[self.position] = deepcopy(ep_states)
        # self.buffer_actions[self.position] = deepcopy(ep_actions)
        # self.buffer_rewards[self.position] = deepcopy(ep_rewards)
        # self.buffer_true_states[self.position] = deepcopy(ep_true_states)
        np_states = np.array(ep_states)
        np_actions = np.array(ep_actions)
        np_rewards = np.array(ep_rewards)
        self.buffer_ep_len[self.position] = len(ep_states)
        self.buffer_states[self.position, :] = np.zeros([self.max_seq_len], dtype=np.int32)
        self.buffer_actions[self.position, :] = np.zeros([self.max_seq_len], dtype=np.int32)
        self.buffer_rewards[self.position, :] = np.zeros([self.max_seq_len], dtype=np.float32)
        self.buffer_states[self.position, :len(ep_states)] = np_states
        self.buffer_actions[self.position, :len(ep_states)] = np_actions
        self.buffer_rewards[self.position, :len(ep_states)] = np_rewards

        if self.full == False and self.position + 1 == self.capacity:
            self.full = True

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        tmp = self.position
        if self.full:
            tmp = self.capacity
        idx = np.random.choice(tmp, batch_size, replace=False)

        batch_lengths = self.buffer_ep_len[idx]

        batch_idx = np.random.randint(0, batch_lengths)

        # print('batch_lens',batch_lengths)
        # print('batch_idx',batch_idx)
        # assert False

        max_len = np.amax(batch_lengths)

        batch_mask = torch.ones((batch_size))

        for i in range(batch_size):
            if batch_idx[i] == batch_lengths[i] - 1:
                batch_mask[i] = 0

        batch_idx = torch.from_numpy(batch_idx)

        # print('batch_mask', batch_mask)

        batch_obs = torch.from_numpy(self.buffer_states[idx, :max_len + 1])

        batch_acts = torch.from_numpy(self.buffer_actions[idx, :max_len + 1])
        batch_rewards = torch.from_numpy(self.buffer_rewards[idx, :max_len + 1])

        # print('batch_observations',batch_obs)
        # print('batch_actions',batch_acts)
        # print('batch_rewards',batch_rewards)

        return batch_obs, batch_acts, batch_rewards, batch_idx, batch_lengths, batch_mask

    def sample_full_ep(self, batch_size):
        tmp = self.position
        if self.full:
            tmp = self.capacity
        idx = np.random.choice(tmp, batch_size, replace=False)

        batch_lengths = self.buffer_ep_len[idx]
        # print(batch_lengths)

        max_len = np.amax(batch_lengths)

        # print(np.amax(batch_lengths))

        batch_obs = torch.from_numpy(self.buffer_states[idx, :max_len])

        # print(batch_obs.shape)
        # print(batch_obs)

        # # batch_obs = F.one_hot(batch_obs.to((torch.int64)) , num_classes=self.obs_dim)

        # print(batch_obs.shape)
        # print(batch_obs)

        batch_acts = torch.from_numpy(self.buffer_actions[idx, :max_len])
        # batch_acts = torch.from_numpy(self.buffer_actions[idx,1:max_len])
        batch_rewards = torch.from_numpy(self.buffer_rewards[idx, :max_len])
        return batch_obs, batch_acts, batch_rewards, batch_lengths, max_len

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.position + 1