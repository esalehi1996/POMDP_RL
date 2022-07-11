from copy import deepcopy
import random
import torch
import numpy as np
import torch.nn.functional as F


class r2d2_ReplayMemory:
    def __init__(self, capacity, obs_dim, act_dim, max_sequence_length, args):
        self.capacity = capacity
        self.max_seq_len = max_sequence_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = args['gamma']
        self.burn_in_len = args['burn_in_len']
        self.learning_obs_len = args['learning_obs_len']
        self.forward_len = args['forward_len']
        self.AIS_state_size = args['AIS_state_size']
        self.batch_size = args['batch_size']
        self.highdim = False
        if args['env_name'][:8] == 'MiniGrid':
            self.highdim = True
        self.buffer_burn_in_history = np.zeros([self.capacity, self.burn_in_len , self.obs_dim + self.act_dim], dtype=np.float32)
        self.buffer_learning_history = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.obs_dim + self.act_dim], dtype=np.float32)
        self.buffer_current_act = np.zeros([self.capacity, self.learning_obs_len], dtype=np.int32)
        if self.highdim:
            self.buffer_next_obs = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.obs_dim], dtype=np.float32)
        else:
            self.buffer_next_obs = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.obs_dim + 1],
                                            dtype=np.float32)
        self.buffer_model_input_act = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.act_dim], dtype=np.float32)
        # self.buffer_burn_in_actions = np.zeros([self.capacity, self.burn_in_len , self.act_dim], dtype=np.float32)
        # self.buffer_learning_actions = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.act_dim], dtype=np.float32)
        self.buffer_rewards = np.zeros([self.capacity, self.learning_obs_len], dtype=np.float32)
        self.buffer_model_target_rewards = np.zeros([self.capacity, self.learning_obs_len + self.forward_len], dtype=np.float32)
        self.buffer_burn_in_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_forward_idx = np.zeros([self.capacity, self.learning_obs_len], dtype=np.int32)
        self.buffer_final_flag = np.zeros([self.capacity, self.learning_obs_len], dtype=np.int32)
        self.buffer_learning_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_learn_forward_len = np.zeros([self.capacity], dtype=np.int32)
        self.buffer_hidden = (torch.zeros(self.capacity , self.AIS_state_size) , torch.zeros(self.capacity , self.AIS_state_size))
        # self.buffer_full_ep_states = np.zeros([self.capacity, self.max_seq_len , self.obs_dim], dtype=np.float32)
        # self.buffer_full_ep_actions = np.zeros([self.capacity, self.max_seq_len], dtype=np.int32)
        # self.buffer_full_ep_rewards = np.zeros([self.capacity, self.max_seq_len], dtype=np.float32)
        # self.buffer_full_ep_len = np.zeros([self.capacity], dtype=np.int32)

        self.position_r2d2 = 0
        # self.position_full_ep = 0
        # self.max_full_ep_size = 0

        self.full = False

    def reset(self, seed):
        random.seed(seed)

        self.position_r2d2 = 0
        # self.position_full_ep = 0
        # self.max_full_ep_size = 0

        self.full = False

    def push(self, ep_states, ep_actions, ep_rewards , ep_hiddens):


        # print(len(ep_states))
        # print(len(ep_actions))
        # print(len(ep_rewards))
        # print(len(ep_hiddens))
        #
        # print(ep_states[0].shape)
        # print(len(ep_states), ep_states[0].shape)
        # # print(len(ep_actions), ep_actions)
        # # print(len(ep_rewards), ep_rewards)
        # #
        # assert False
        # for i in range(len(ep_states)):
        #     print(i,ep_states[i],ep_actions[i],ep_rewards[i])

        ls_actions = [np.zeros(self.act_dim) for i in range(len(ep_actions))]
        ls_actions_ = [np.zeros(self.act_dim) for i in range(len(ep_actions))]
        # current_act_ls = [np.zeros(self.act_dim) for i in range(len(ep_actions))]
        for i in range(len(ep_actions)-1):
            ls_actions[i+1][ep_actions[i]] = 1
        for i in range(len(ep_actions)):
            ls_actions_[i][ep_actions[i]] = 1

        if self.highdim:
            ls_next_obs = [np.zeros(self.obs_dim) for i in range(len(ep_states))]
            for i in range(len(ep_states) - 1):
                ls_next_obs[i] = ep_states[i + 1]
        else:
            ls_next_obs = [np.zeros(self.obs_dim+1) for i in range(len(ep_states))]
            for i in range(len(ep_states)-1):
                ls_next_obs[i][:self.obs_dim] = ep_states[i+1]
            ls_next_obs[len(ep_states)-1][self.obs_dim] = 1
        # for i in range(len(ep_states)):
        #     print(i,ep_states[i],ls_next_obs[i],ep_actions[i],ep_rewards[i])
        # for i in range(len(ep_actions)):
        #     current_act_ls[i][ep_actions[i]] = 1







        burn_in_act_list = [ls_actions[max(0,x-self.burn_in_len):x] for x in range(0, len(ep_states), self.learning_obs_len)]
        learning_act_list = [ls_actions[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]

        # print(burn_in_act_list)
        # print(learning_act_list)
        # forward_act_list = [ls_actions[x + self.forward_len:x + self.forward_len + self.learning_obs_len] for x in range(0, len(ep_states), self.learning_obs_len)]

        # for i, hidden in enumerate(burn_in_act_list):
        #     print(i,len(hidden),hidden)

        current_act_list = [ep_actions[x:x + self.learning_obs_len] for x in range(0, len(ep_states), self.learning_obs_len)]
        next_obs_list = [ls_next_obs[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]
        model_input_act_list = [ls_actions_[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]




        burn_in_obs_list = [ep_states[max(0,x-self.burn_in_len):x] for x in range(0, len(ep_states), self.learning_obs_len)]
        learning_obs_list = [ep_states[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]
        ep_rewards_list = [ep_rewards[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_rewards), self.learning_obs_len)]
        # forward_obs_list = [ep_states[x + self.forward_len:x + self.forward_len + self.learning_obs_len] for x in range(0, len(ep_states), self.learning_obs_len)]
        hidden_list = [ep_hiddens[max(0,x-self.burn_in_len)] for x in range(0, len(ep_states), self.learning_obs_len)]

        # print(len(hidden_list))
        # for i, hidden in enumerate(hidden_list):
        #     print(i,hidden)

        # print(len(burn_in_act_list))
        # print(len(learning_act_list))
        # print(len(burn_in_obs_list))
        # print(len(learning_obs_list))
        # print(len(hidden_list))

        for i, hidden in enumerate(hidden_list):
            if hidden is None:
                hidden_list[i] = (torch.zeros( self.AIS_state_size),torch.zeros(self.AIS_state_size))
            else:
                hidden_list[i] = (hidden[0].view(-1),hidden[1].view(-1))

        # for i, hidden in enumerate(learning_obs_list):
        #     print(i,hidden)


        discounted_sum = [[sum_rewards(ep_rewards[x+y:x+y+self.forward_len],self.gamma) for y in range(0,min(self.learning_obs_len, len(ep_states)-x))] for x in range(0, len(ep_states) , self.learning_obs_len)]
        # print(discounted_sum)
        # for i in range(len(hidden_list)):
        #     print(i,len(learning_act_list[i]))
        for i in range(len(hidden_list)):
            # print('-------------',i,'---------------')
            # print(self.position_r2d2)
            # print(np.array(burn_in_act_list[i]).shape)
            # print(np.array(burn_in_obs_list[i]).shape)
            if np.array(burn_in_obs_list[i]).shape[0] != 0:
                self.buffer_burn_in_history[self.position_r2d2, :len(burn_in_act_list[i]), :] = np.concatenate((np.array(burn_in_obs_list[i]), np.array(burn_in_act_list[i])), axis=1)
            self.buffer_learning_history[self.position_r2d2, :len(learning_act_list[i]), :] = np.concatenate((np.array(learning_obs_list[i]), np.array(learning_act_list[i])), axis=1)
            # print(self.buffer_learning_history[self.position_r2d2, :len(learning_act_list[i]), :])
            # print(self.buffer_current_act[self.position_r2d2,:,:])
            self.buffer_hidden[0][self.position_r2d2, :] = hidden_list[i][0]
            self.buffer_hidden[1][self.position_r2d2, :] = hidden_list[i][1]
            # print(self.buffer_hidden[0][self.position_r2d2, :],self.buffer_hidden[1][self.position_r2d2, :])
            self.buffer_rewards[self.position_r2d2,:len(discounted_sum[i])] = np.array(discounted_sum[i])
            self.buffer_model_target_rewards[self.position_r2d2, :len(ep_rewards_list[i])] = np.array(ep_rewards_list[i])
            self.buffer_burn_in_len[self.position_r2d2] = len(burn_in_obs_list[i])
            # print(self.buffer_burn_in_len[self.position_r2d2])
            self.buffer_learning_len[self.position_r2d2] = len(discounted_sum[i])
            self.buffer_learn_forward_len[self.position_r2d2] = len(learning_act_list[i])
            if np.array(next_obs_list[i]).shape[0] != 0:
                self.buffer_next_obs[self.position_r2d2,:len(next_obs_list[i]) , :] = np.array(next_obs_list[i])
            self.buffer_current_act[self.position_r2d2, :len(current_act_list[i])] = np.array(current_act_list[i])
            self.buffer_model_input_act[self.position_r2d2,:len(model_input_act_list[i]),:] = np.array(model_input_act_list[i])
            self.buffer_forward_idx[self.position_r2d2,:len(discounted_sum[i])] = np.array([min(j+self.forward_len,len(learning_obs_list[i])-1) for j in range(len(discounted_sum[i]))])
            self.buffer_final_flag[self.position_r2d2,:len(discounted_sum[i])] = np.array([int(i*self.learning_obs_len + self.forward_len + j < len(ep_states)) for j in range(len(discounted_sum[i]))])
            # print(self.buffer_forward_idx[self.position_r2d2])
            # print(self.buffer_final_flag[self.position_r2d2])
            # print(self.buffer_burn_in_history[self.position_r2d2, :len(burn_in_act_list[i]), :])
            # print('----learning')
            # print(np.concatenate((np.array(learning_obs_list[i]), np.array(learning_act_list[i])), axis=1))
            # print(self.buffer_learning_history[self.position_r2d2, :len(learning_act_list[i]), :])
            # print('hidden')
            # print(hidden_list[i])
            # print(self.buffer_hidden[0][self.position_r2d2,:], self.buffer_hidden[1][self.position_r2d2,:])
            # print('-----reward')
            # print(discounted_sum[i])
            # print(self.buffer_rewards[self.position_r2d2,:len(learning_act_list[i])])
            if self.full is False and self.position_r2d2 + 1 == self.capacity:
                self.full = True

            if self.position_r2d2 + 1 == self.capacity:
                self.position_full_ep = 0

            self.position_r2d2 = (self.position_r2d2 + 1) % self.capacity


        # for i, hidden in enumerate(hidden_list):
        #     print(i,hidden)
        # print(ep_rewards)
        # # print(len(discounted_sum[2]),discounted_sum[2])
        # for i in range(0,len(discounted_sum)):
        #     print("------------------",i,"-------------")
        #     print(discounted_sum[i])
        #     print(ep_rewards[i*self.learning_obs_len:(i+1)*self.learning_obs_len])

        # print('learning_history',self.buffer_learning_history[:2, :, :])
        # print('burn_in_history',self.buffer_burn_in_history[:2, :, :])
        # print('rewards',self.buffer_rewards[:self.position_r2d2,:])
        # print('learning_length',self.buffer_learning_len[:self.position_r2d2])
        # print('final_flag',self.buffer_final_flag[:self.position_r2d2,:])
        # print('forward idx',self.buffer_forward_idx[:self.position_r2d2,:])
        # print('current action',self.buffer_current_act[:self.position_r2d2,:])






        # print(sum_rewards(ls,gamma))

        # if len(ep_states) > 1:
        #     np_states = np.array(ep_states)
        #     np_actions = np.array(ep_actions)
        #     np_rewards = np.array(ep_rewards)
        #     self.buffer_full_ep_len[self.position_full_ep] = len(ep_states)
        #     self.buffer_full_ep_states[self.position_full_ep, :, :] = np.zeros([self.max_seq_len, self.obs_dim], dtype=np.float32)
        #     self.buffer_full_ep_actions[self.position_full_ep, :] = np.zeros([self.max_seq_len], dtype=np.int32)
        #     self.buffer_full_ep_rewards[self.position_full_ep, :] = np.zeros([self.max_seq_len], dtype=np.float32)
        #     self.buffer_full_ep_states[self.position_full_ep, :len(ep_states)] = np_states
        #     self.buffer_full_ep_actions[self.position_full_ep, :len(ep_states)] = np_actions
        #     self.buffer_full_ep_rewards[self.position_full_ep, :len(ep_states)] = np_rewards
        #
        #     self.position_full_ep = self.position_full_ep + 1
        #
        #     if self.max_full_ep_size < self.position_full_ep:
        #         self.max_full_ep_size = self.position_full_ep

        # assert False

        # print('b',self.buffer_states[self.position,:,:].shape,self.buffer_states[self.position,:,:])

        # if self.full == False and self.position + 1 == self.capacity:
        #     self.full = True

        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        tmp = self.position_r2d2
        if self.full:
            tmp = self.capacity
        # print(tmp,batch_size)
        idx = np.random.choice(tmp, batch_size, replace=False)
        torch_idx = torch.from_numpy(idx)

        # print('starting sampling')
        #
        # print(idx , torch_idx)

        # self.buffer_burn_in_history = np.zeros([self.capacity, self.burn_in_len, self.obs_dim + self.act_dim],
        #                                        dtype=np.float32)
        # self.buffer_learning_history = np.zeros(
        #     [self.capacity, self.learning_obs_len + self.forward_len, self.obs_dim + self.act_dim], dtype=np.float32)
        # # self.buffer_burn_in_actions = np.zeros([self.capacity, self.burn_in_len , self.act_dim], dtype=np.float32)
        # # self.buffer_learning_actions = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.act_dim], dtype=np.float32)
        # self.buffer_rewards = np.zeros([self.capacity, self.learning_obs_len], dtype=np.float32)
        # self.buffer_burn_in_len = np.zeros([self.capacity], dtype=np.int32)
        # self.buffer_forward_idx = np.zeros([self.capacity, self.learning_obs_len], dtype=np.int32)
        # self.buffer_final_flag = np.zeros([self.capacity, self.learning_obs_len], dtype=np.int32)
        # self.buffer_learning_len = np.zeros([self.capacity], dtype=np.int32)
        # self.buffer_hidden = (torch.zeros(self.capacity, self.AIS_state_size), torch.zeros(self.capacity, self.AIS_state_size))

        batch_burn_in_hist = self.buffer_burn_in_history[idx,:,:]
        batch_learn_hist = self.buffer_learning_history[idx,:,:]
        batch_rewards = self.buffer_rewards[idx,:]
        batch_burn_in_len = self.buffer_burn_in_len[idx]
        batch_forward_idx = self.buffer_forward_idx[idx,:]
        batch_final_flag = self.buffer_final_flag[idx,:]
        batch_learn_len = self.buffer_learning_len[idx]
        batch_hidden = (self.buffer_hidden[0][torch_idx] , self.buffer_hidden[1][torch_idx])
        batch_current_act = self.buffer_current_act[idx,:]
        batch_learn_forward_len = self.buffer_learn_forward_len[idx]
        batch_next_obs = self.buffer_next_obs[idx]
        batch_model_input_act = self.buffer_model_input_act[idx]
        batch_model_target_reward = self.buffer_model_target_rewards[idx]

        # print(batch_hidden)

        # assert False

        return batch_burn_in_hist, batch_learn_hist, batch_rewards, batch_learn_len, batch_forward_idx, batch_final_flag, batch_current_act , batch_hidden , batch_burn_in_len , batch_learn_forward_len , batch_next_obs , batch_model_input_act , batch_model_target_reward

    # def sample_full_ep(self, batch_size):
    #     idx = np.random.choice(self.max_full_ep_size, batch_size, replace=False)
    #
    #     batch_lengths = self.buffer_full_ep_len[idx]
    #
    #     max_len = np.amax(batch_lengths)
    #
    #     # print(np.amax(batch_lengths))
    #
    #     batch_obs = torch.from_numpy(self.buffer_full_ep_states[idx, :max_len])
    #
    #     # print(batch_obs.shape)
    #     # print(batch_obs)
    #
    #     # # batch_obs = F.one_hot(batch_obs.to((torch.int64)) , num_classes=self.obs_dim)
    #
    #     # print(batch_obs.shape)
    #     # print(batch_obs)
    #
    #     batch_acts = torch.from_numpy(self.buffer_full_ep_actions[idx, :max_len])
    #     # batch_acts = torch.from_numpy(self.buffer_actions[idx,1:max_len])
    #     batch_rewards = torch.from_numpy(self.buffer_full_ep_rewards[idx, :max_len])
    #     return batch_obs, batch_acts, batch_rewards, batch_lengths

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.position_r2d2 + 1

    # def len_fullep(self):
    #     return self.max_full_ep_size



def sum_rewards(reward_list, gamma):
    ls = [reward_list[i] * gamma ** i for i in range(0,len(reward_list))]
    return sum(ls)