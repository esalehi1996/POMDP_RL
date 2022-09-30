from copy import deepcopy
import random
import torch
import numpy as np
from SumTree import SumTree,MinTree
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
        if args['PER']:
            self.PER = True
            self.SumTree = SumTree(capacity)
            self.MinTree = MinTree(capacity)
            self.PER_e = 1e-6
            self.PER_a = 0.6
            self.PER_beta = 0.4
            total_updates = args['num_steps'] / args['rl_update_every_n_steps']
            # print(total_updates)
            self.PER_beta_increment_per_sampling = (1 - self.PER_beta)/total_updates
            # print(self.PER_beta_increment_per_sampling)
        else:
            self.PER = False
        self.buffer_burn_in_history = np.zeros([self.capacity, self.burn_in_len , self.obs_dim + self.act_dim + 1], dtype=np.float32)
        self.buffer_learning_history = np.zeros([self.capacity, self.learning_obs_len + self.forward_len, self.obs_dim + self.act_dim + 1], dtype=np.float32)
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
        self.buffer_gammas = np.zeros([self.capacity, self.learning_obs_len], dtype=np.float32)
        self.buffer_final_flag_for_model = np.zeros([self.capacity, self.learning_obs_len + self.forward_len], dtype=np.int32)
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
        if self.PER is True:
            self.SumTree.reset()

    def push(self, ep_states, ep_actions, ep_rewards , ep_hiddens , sac):


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
        ls_rewards = [0.0] + ep_rewards[:len(ep_rewards)-1]
        # for i in range(len(ep_states)):
        #     print(i,ep_states[i],ep_actions[i],ep_rewards[i],ls_rewards[i])
        #
        # assert False

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
        # print(ls_next_obs)
        # assert False
        # for i in range(len(ep_states)):
        #     print(i,ep_states[i],ls_next_obs[i],ep_actions[i],ep_rewards[i])
        # for i in range(len(ep_actions)):
        #     current_act_ls[i][ep_actions[i]] = 1







        burn_in_act_list = [ls_actions[max(0,x-self.burn_in_len):x] for x in range(0, len(ep_states), self.learning_obs_len)]
        learning_act_list = [ls_actions[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]

        burn_in_r_list = [ls_rewards[max(0, x - self.burn_in_len):x] for x in range(0, len(ep_states), self.learning_obs_len)]
        learning_r_list = [ls_rewards[x:x + self.learning_obs_len + self.forward_len] for x in range(0, len(ep_states), self.learning_obs_len)]

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
                hidden_list[i] = (hidden[0].cpu().view(-1),hidden[1].cpu().view(-1))

        # for i, hidden in enumerate(learning_obs_list):
        #     print(i,hidden)
        ep_rewards_ = ep_rewards[:len(ep_rewards)-1]
        # print(ep_rewards_)


        discounted_sum = [[sum_rewards(ep_rewards_[x+y:x+y+self.forward_len],self.gamma) if x+y != len(ep_rewards_) else ep_rewards[x+y] for y in range(0,min(self.learning_obs_len, len(ep_states)-x))] for x in range(0, len(ep_states) , self.learning_obs_len)]
        # print(discounted_sum)
        # if len(ep_rewards_)
        # for i in range(len(hidden_list)):
        #     print(i,len(learning_act_list[i]))
        start_index = self.position_r2d2
        buffer_fill = False
        for i in range(len(hidden_list)):
            # print('-------------',i,'---------------')
            # print(discounted_sum[i])
            # print(self.position_r2d2)
            # print(np.array(burn_in_act_list[i]).shape)
            # print(np.array(burn_in_obs_list[i]).shape)
            # print(np.array(burn_in_r_list[i]).reshape(-1,1).shape,np.array(learning_r_list[i]).reshape(-1,1).shape)
            if np.array(burn_in_obs_list[i]).shape[0] != 0:
                self.buffer_burn_in_history[self.position_r2d2, :len(burn_in_act_list[i]), :] = np.concatenate((np.array(burn_in_obs_list[i]), np.array(burn_in_act_list[i]) , np.array(burn_in_r_list[i]).reshape(-1,1)), axis=1)
            self.buffer_learning_history[self.position_r2d2, :len(learning_act_list[i]), :] = np.concatenate((np.array(learning_obs_list[i]), np.array(learning_act_list[i]), np.array(learning_r_list[i]).reshape(-1,1)), axis=1)
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
            self.buffer_final_flag[self.position_r2d2,:len(discounted_sum[i])] = np.array([int(i*self.learning_obs_len +  j < len(ep_states) -1 ) for j in range(len(discounted_sum[i]))])
            self.buffer_final_flag_for_model[self.position_r2d2,:len(model_input_act_list[i])] = np.array([int(i*self.learning_obs_len +  j < len(ep_states) -1 ) for j in range(len(model_input_act_list[i]))])
            self.buffer_gammas[self.position_r2d2,:len(discounted_sum[i])] = np.array([self.gamma ** (min(j+self.forward_len,len(learning_obs_list[i])-1) - j) for j in range(len(discounted_sum[i]))])
            if self.full is False and self.position_r2d2 + 1 == self.capacity:
                self.full = True
            if self.position_r2d2 + 1 == self.capacity:
                buffer_fill = True

            if self.position_r2d2 + 1 == self.capacity:
                self.position_full_ep = 0

            self.position_r2d2 = (self.position_r2d2 + 1) % self.capacity

        # print(start_index , self.position_r2d2 , len(hidden_list))


        if self.PER is True:
            if buffer_fill is False:
                input_vals = {
                    "batch_burn_in_hist" : self.buffer_burn_in_history[start_index:self.position_r2d2, :, :],
                    "batch_learn_hist" : self.buffer_learning_history[start_index:self.position_r2d2, :, :] ,
                    "batch_rewards" : self.buffer_rewards[start_index:self.position_r2d2, :] ,
                    "batch_burn_in_len": self.buffer_burn_in_len[start_index:self.position_r2d2],
                    "batch_forward_idx": self.buffer_forward_idx[start_index:self.position_r2d2, :],
                    "batch_final_flag": self.buffer_final_flag[start_index:self.position_r2d2, :],
                    "batch_learn_len": self.buffer_learning_len[start_index:self.position_r2d2],
                    "batch_hidden": (self.buffer_hidden[0][start_index:self.position_r2d2], self.buffer_hidden[1][start_index:self.position_r2d2]),
                    "batch_current_act": self.buffer_current_act[start_index:self.position_r2d2, :],
                    "batch_learn_forward_len": self.buffer_learn_forward_len[start_index:self.position_r2d2],
                    "batch_next_obs": self.buffer_next_obs[start_index:self.position_r2d2],
                    "batch_model_input_act": self.buffer_model_input_act[start_index:self.position_r2d2],
                    "batch_model_target_reward": self.buffer_model_target_rewards[start_index:self.position_r2d2],
                    "batch_gammas": self.buffer_gammas[start_index:self.position_r2d2],
                    "batch_final_flag_for_model": self.buffer_final_flag_for_model[start_index:self.position_r2d2]
                }
                # print(input_vals['batch_learn_hist'].shape)
            else:
                # print('filled')
                input_vals = {
                    "batch_burn_in_hist": np.concatenate((self.buffer_burn_in_history[start_index:, :, :],self.buffer_burn_in_history[:self.position_r2d2, :, :])),
                    "batch_learn_hist": np.concatenate((self.buffer_learning_history[start_index:, :, :],self.buffer_learning_history[:self.position_r2d2, :, :])),
                    "batch_rewards": np.concatenate((self.buffer_rewards[start_index:, :],self.buffer_rewards[:self.position_r2d2, :])),
                    "batch_burn_in_len": np.concatenate((self.buffer_burn_in_len[start_index:],self.buffer_burn_in_len[:self.position_r2d2])),
                    "batch_forward_idx": np.concatenate((self.buffer_forward_idx[start_index:, :],self.buffer_forward_idx[:self.position_r2d2, :])),
                    "batch_final_flag": np.concatenate((self.buffer_final_flag[start_index:, :],self.buffer_final_flag[:self.position_r2d2, :])),
                    "batch_learn_len": np.concatenate((self.buffer_learning_len[start_index:],self.buffer_learning_len[:self.position_r2d2])),
                    "batch_hidden": (torch.cat((self.buffer_hidden[0][start_index:],self.buffer_hidden[0][:self.position_r2d2])),
                                     torch.cat((self.buffer_hidden[1][start_index:],self.buffer_hidden[1][:self.position_r2d2]))),
                    "batch_current_act": np.concatenate((self.buffer_current_act[start_index:, :],self.buffer_current_act[:self.position_r2d2, :])),
                    "batch_learn_forward_len": np.concatenate((self.buffer_learn_forward_len[start_index:],self.buffer_learn_forward_len[:self.position_r2d2])),
                    "batch_next_obs": np.concatenate((self.buffer_next_obs[start_index:],self.buffer_next_obs[:self.position_r2d2])),
                    "batch_model_input_act": np.concatenate((self.buffer_model_input_act[start_index:],self.buffer_model_input_act[:self.position_r2d2])),
                    "batch_model_target_reward": np.concatenate((self.buffer_model_target_rewards[start_index:],self.buffer_model_target_rewards[:self.position_r2d2])),
                    "batch_gammas": np.concatenate((self.buffer_gammas[start_index:],self.buffer_gammas[:self.position_r2d2])),
                    "batch_final_flag_for_model": np.concatenate((self.buffer_final_flag_for_model[start_index:],self.buffer_final_flag_for_model[:self.position_r2d2]))
                }
                # print(input_vals['batch_learn_hist'].shape)
                # print(self.buffer_learning_history[start_index:, :, :].shape,self.buffer_learning_history[:self.position_r2d2, :, :].shape )



            # print(input_vals)

            priorities = sac.compute_priorities(len(hidden_list) , input_vals)

            # print(priorities)

            priorities = (priorities + self.PER_e) ** self.PER_a

            # print(priorities)

            for p in priorities:
                # print(p)
                self.SumTree.add(p)
                self.MinTree.add(p)









    def sample(self, batch_size):
        # print(self.__len__())

        if self.PER is False:
            tmp = self.position_r2d2
            if self.full:
                tmp = self.capacity
            idx = np.random.choice(tmp, batch_size, replace=False)
            torch_idx = torch.from_numpy(idx)
        else:
            idx = np.zeros(batch_size, dtype=int)
            tree_idx = np.zeros(batch_size, dtype=int)
            priorities = np.zeros(batch_size)

            # print(self.SumTree.tree[99:99+self.__len__()])
            # print(self.MinTree.tree[99:99+self.__len__()])
            # print(self.MinTree.min())
            # print(np.amin(self.SumTree.tree[99:99+self.__len__()]))

            segment = self.SumTree.total() / batch_size

            # print(segment)

            self.PER_beta = np.min([1., self.PER_beta + self.PER_beta_increment_per_sampling])

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)


                s = random.uniform(a, b)

                # print(i, a, b , s)
                (id, p , didx) = self.SumTree.get(s)
                priorities[i] = p
                idx[i] = didx
                tree_idx[i] = id





            sampling_probabilities = priorities / self.SumTree.total()
            p_min = self.MinTree.min() / self.SumTree.total()
            is_max = np.power(self.SumTree.n_entries * p_min, -self.PER_beta)
            is_weight = np.power(self.SumTree.n_entries * sampling_probabilities, -self.PER_beta)
            is_weight /= is_max
            torch_idx = torch.from_numpy(idx)
            is_weight = np.reshape(is_weight, (batch_size, 1))
            is_weight_td = np.tile(is_weight, (1, self.learning_obs_len))
            is_weight_model = np.tile(is_weight, (1, self.learning_obs_len + self.forward_len))

            # print(idx)
            # print(torch_idx)
            # print(tree_idx)
            # print(is_weight)
            # print(is_weight_td)
            # print(is_weight_model)


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
        batch_gammas = self.buffer_gammas[idx]
        batch_final_flag_for_model = self.buffer_final_flag_for_model[idx]

        # print(batch_hidden)

        if self.PER is True:
            return batch_burn_in_hist, batch_learn_hist, batch_rewards, batch_learn_len, batch_forward_idx, batch_final_flag, batch_current_act , batch_hidden , batch_burn_in_len , batch_learn_forward_len , batch_next_obs , batch_model_input_act , batch_model_target_reward , batch_gammas , batch_final_flag_for_model , tree_idx , is_weight_td , is_weight_model
        else:
            return batch_burn_in_hist, batch_learn_hist, batch_rewards, batch_learn_len, batch_forward_idx, batch_final_flag, batch_current_act , batch_hidden , batch_burn_in_len , batch_learn_forward_len , batch_next_obs , batch_model_input_act , batch_model_target_reward , batch_gammas , batch_final_flag_for_model

    def update_priorities(self,tree_idx,priorities):
        # print(priorities)
        priorities = (priorities + self.PER_e) ** self.PER_a


        for p,idx in zip(priorities,tree_idx):
            # print(p)
            self.SumTree.update(idx,p)
            self.MinTree.update(idx,p)



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
            return self.position_r2d2

    # def len_fullep(self):
    #     return self.max_full_ep_size



def sum_rewards(reward_list, gamma):
    ls = [reward_list[i] * gamma ** i for i in range(0,len(reward_list))]
    return sum(ls)
