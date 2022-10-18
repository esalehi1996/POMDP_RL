import os

from models import *
from autoencoder.simple_autoencoder import autoencoder
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import numpy as np
import math
import random

class SAC(object):
    def __init__(self, env, args):


        self.args = args

        if args['env_name'][:8] == 'MiniGrid':
            action_space = env.action_space

        else:
            num_inputs = env.observation_space.n
            action_space = env.action_space
            self.obs_dim = num_inputs
            self.state_size = num_inputs

        self.act_dim = action_space.n
        self.Lambda = args['AIS_lambda']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['SAC_alpha']
        self.AIS_state_size = args['AIS_state_size']
        self.target_update_interval = args['target_update_interval']
        self.action_space = action_space
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        self.rl_alg = args['rl_alg']
        print(self.rl_alg)
        self.model_alg = args['model_alg']
        # self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args['cuda'] else "cpu")

        highdim = False
        if args['env_name'][:8] == 'MiniGrid':
            highdim = True
            autoencoder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'autoencoder', args['env_name'])
            # print(autoencoder_path)
            # print(os.path.join(autoencoder_path, 'autoencoder_final.pth'))
            # print('gggg')
            self.autoencoder_model = autoencoder(True).to(self.device)
            self.autoencoder_model.load_state_dict(
                torch.load(os.path.join(autoencoder_path, 'autoencoder_final.pth'), map_location=self.device))
            self.observation_mean = torch.load(os.path.join(autoencoder_path, 'mean.pt'))
            self.observation_scaler = torch.load(os.path.join(autoencoder_path, 'max_vals.pt')) * 1.2
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.observation_mean, self.observation_scaler)])
            num_inputs = self.autoencoder_model.latent_space_size
            if args['QL_VAE_disable']:
                num_inputs = env.reset()['image'].reshape(-1).shape[0]
            self.obs_dim = num_inputs




        self.policy = policy_net(action_space.n, self.AIS_state_size).to(self.device)
        double = False
        if self.args['replay_type'] == 'vanilla':
            double = True
        noisy_net = False
        if self.args['noisy_net'] is True:
            noisy_net = True
        self.critic = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size'] ,noisy_net, double).to(device=self.device)
        self.critic_target = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size'] ,noisy_net,  double).to(self.device)
        self.critic_target.train()
        self.critic.train()

        # self.policy_cpu = policy_net(action_space.n, self.AIS_state_size)
        # self.q_cpu = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size'] , double)

        # if self.alg == 'SAC+AIS':
        self.rho = rho_net(num_inputs, action_space.n, self.AIS_state_size , args['QL_VAE_disable']).to(self.device)
        # self.rho_cpu = rho_net(num_inputs, action_space.n, self.AIS_state_size)
        if args['AIS_loss'] == 'MMD':
            self.psi = psi_net(num_inputs, action_space.n, self.AIS_state_size , highdim).to(self.device)
        elif args['AIS_loss'] == 'KL' and args['env_name'][:8] == 'MiniGrid':
            self.psi = psi_net_highdim_KL(num_inputs,action_space.n,self.AIS_state_size).to(self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args['rl_lr'])
        self.policy_optim = Adam(self.policy.parameters(), lr=args['rl_lr'])

        if self.model_alg == 'AIS':
            self.AIS_optimizer = Adam(list(self.rho.parameters()) + list(self.psi.parameters()), lr=args['AIS_lr'])
        if self.model_alg == 'None':
            self.AIS_optimizer = Adam(self.rho.parameters(), lr=args['AIS_lr'])

        # if self.alg == 'SAC':
        #     self.rho_policy = rho_net_lowdim(num_inputs, action_space.n, self.AIS_state_size).to(self.device)
        #     self.rho_q = rho_net_lowdim(num_inputs, action_space.n, self.AIS_state_size).to(self.device)
        #     self.critic_rho_target = rho_net_lowdim(num_inputs, action_space.n, self.AIS_state_size).to(self.device)
        #
        #     self.critic_optim = Adam(list(self.critic.parameters()) + list(self.rho_q.parameters()), lr=args.lr)
        #     self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        #     hard_update(self.critic_rho_target, self.rho_q)

        # self.critic_optim = Adam(self.critic.parameters() , lr=args.lr)
        # self.AIS_q_optim = Adam(self.rho_q.parameters() , lr=args.rnn_lr)
        # self.AIS_p_optim = Adam(self.rho_policy.parameters() , lr=args.rnn_lr)
        hard_update(self.critic_target, self.critic)
        # hard_update(self.q_cpu, self.critic)
        # hard_update(self.rho_cpu, self.rho)
        # hard_update(self.policy_cpu, self.policy)
        self.update_to_q = 0
        if self.rl_alg == 'QL':
            self.eps_greedy_parameters = {
                "EPS_START" : args['EPS_start'],
                "EPS_END" : args['EPS_end'] ,
                "EPS_DECAY" : args['EPS_decay']
            }
            self.env_steps = 0
        if self.automatic_entropy_tuning is True:
            # self.target_entropy = -torch.prod(torch.Tensor(action_space.n).to(self.device)).item()
            # print(torch.Tensor(action_space.n))
            # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            # self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.target_entropy = \
                -np.log(1.0 / self.action_space.n) * self.alpha

            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=args['rl_lr'])

        # self.policy_optim = Adam(list(self.policy.parameters()) + list(self.rho_policy.parameters()), lr=args.lr)

    def get_encoded_obs(self, obs):
        obs = Image.fromarray(obs)
        obs = self.transform(obs)
        obs = obs.reshape(-1)
        # print(obs)
        # print(self.autoencoder_model
        if self.args['QL_VAE_disable'] == False:
            obs = obs.to(self.device)
            encoded_obs = self.autoencoder_model(obs, getLatent=True).cpu().detach()
        else:
            encoded_obs = obs


        return encoded_obs



    def select_action(self, state, action, reward, hidden_p, start , EPS_up , evaluate):
        # print('**************start************')
        # print(start)
        # print('state',state)
        # print('action',action)
        # print('reward',reward)
        # print('hidden',hidden_p)

        with torch.no_grad():
            if start:
                action = torch.zeros(self.action_space.n)
                reward = torch.Tensor([0])
            else:
                action = convert_int_to_onehot(action, self.action_space.n)
                reward = torch.Tensor([reward])
            # state = convert_int_to_onehot(state, self.state_size) this is what was commented ***
            rho_input = torch.cat((state, action , reward)).reshape(1, 1, -1).to(self.device)
            # if self.alg == 'SAC+AIS':
            ais_z, hidden_p = self.rho(rho_input, 1, hidden_p, self.device , [] , self.args['replay_type'] , pack_sequence = False)
            # if self.alg == 'SAC':
            #     ais_z, hidden_p = self.rho_policy(rho_input, 1, hidden_p)
            # print('ais',ais_z)
            ais_z = ais_z[ais_z.shape[0] - 1]
            # print('ais',ais_z)
            # print('hidden',hidden_p)
            if evaluate is False and EPS_up:
                self.env_steps += 1
            if self.rl_alg == 'QL' and self.args['noisy_net'] is False:
                eps_threshold = self.eps_greedy_parameters['EPS_END'] + (
                            self.eps_greedy_parameters['EPS_START'] - self.eps_greedy_parameters['EPS_END']) * \
                                math.exp(-1. * self.env_steps / self.eps_greedy_parameters['EPS_DECAY'])
                sample = random.random()
                if sample < eps_threshold and evaluate is False:
                    return torch.tensor([[random.randrange(self.act_dim)]],dtype=torch.long).cpu().numpy()[0][0] , hidden_p
                if sample < self.args['test_epsilon'] and evaluate is True:
                    return torch.tensor([[random.randrange(self.act_dim)]], dtype=torch.long).cpu().numpy()[0][
                               0], hidden_p
            if self.rl_alg == 'SAC':
                action, pi, _ = self.policy.sample(ais_z.detach())
                # print(action)
                # raise "Error"
            if self.rl_alg == 'QL':
                # print(qf.shape,qf)
                if self.args['replay_type'] == 'vanilla':
                    qf1 , qf2 = self.critic(ais_z.detach())
                    qf = (torch.min(qf1, qf2))
                else:
                    # if self.args['noisy_net'] is True:
                    #     if evaluate is False:
                    #         self.critic.train()
                    #     else:
                    #         self.critic.eval()
                    qf = self.critic(ais_z.detach())
                # print(qf.max(1)[1])
                max_ac = qf.max(1)[1]

                # print(min_q.max(1))
                # raise "Error"


            # print('action',action)
            # print('pi',pi)

        if self.rl_alg == 'SAC':
            return action.detach().cpu().numpy()[0][0], hidden_p
        if self.rl_alg == 'QL':
            return max_ac.detach().cpu().numpy()[0] , hidden_p

    def update_parameters(self, memory, batch_size, updates):
        if self.args['replay_type'] == 'vanilla':
            model_losses = 0
            qf_losses, policy_losses = self.update_parameters_vanilla(memory,batch_size,updates)
        if self.args['replay_type'] == 'r2d2':
            qf_losses, policy_losses , model_losses = self.update_parameters_r2d2(memory, batch_size, updates)

        return qf_losses, policy_losses , model_losses

    def compute_priorities(self, batch_size , input_vals):
        # print(input_vals)

        with torch.no_grad():
            batch_hidden = (input_vals['batch_hidden'][0].view(1, batch_size, -1).to(self.device),
                            input_vals['batch_hidden'][1].view(1, batch_size, -1).to(self.device))

            batch_burn_in_hist = torch.from_numpy(input_vals['batch_burn_in_hist']).to(self.device)

            zero_idx = []
            batch_burn_in_len_ = input_vals['batch_burn_in_len']
            for i, len in enumerate(batch_burn_in_len_):
                if len == 0:
                    zero_idx.append(i)
                    batch_burn_in_len_[i] = 1


            batch_burn_in_len_[zero_idx] = 1

            _, hidden_burn_in = self.rho(batch_burn_in_hist, batch_size, batch_hidden, self.device,
                                         list(batch_burn_in_len_), self.args['replay_type'])

            hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]

            batch_learn_hist = torch.from_numpy(input_vals['batch_learn_hist']).to(self.device)

            ais_z, hidden = self.rho(batch_learn_hist, batch_size, hidden_burn_in, self.device,
                                     list(input_vals['batch_learn_forward_len']), self.args['replay_type'])




            if self.args['PER_type'] == 'TD':

                unpacked_ais_z, lens_unpacked = pad_packed_sequence(ais_z, batch_first=True)

                q_z = pack_padded_sequence(unpacked_ais_z, list(input_vals['batch_learn_len']), batch_first=True,
                                           enforce_sorted=False)


                qf = self.critic(q_z.data)

                batch_current_act = torch.from_numpy(input_vals['batch_current_act']).to(self.device)

                # print(batch_current_act)

                packed_current_act = pack_padded_sequence(batch_current_act, list(input_vals['batch_learn_len']), batch_first=True,
                                                          enforce_sorted=False)

                # print(packed_current_act.data)

                qf = qf.gather(1, packed_current_act.data.view(-1, 1).long())

                batch_forward_idx = torch.from_numpy(input_vals['batch_forward_idx']).to(self.device)
                # print('forward')
                # print(batch_forward_idx)
                packed_batch_forward_idx = pack_padded_sequence(batch_forward_idx, list(input_vals['batch_learn_len']),
                                                                batch_first=True, enforce_sorted=False)
                # print(packed_batch_forward_idx)
                unpacked_batch_forward_idx, _ = pad_packed_sequence(packed_batch_forward_idx, batch_first=True)


                ais_z_target = unpacked_ais_z.gather(1,
                                                     unpacked_batch_forward_idx.view(batch_size, -1, 1).expand(-1, -1,
                                                                                                               self.AIS_state_size).long()).detach()
                # print(ais_z_target.shape,ais_z_target)
                # print(ais_z_target.shape,ais_z_target)
                # print(list(batch_learn_len))
                packed_target_ais = pack_padded_sequence(ais_z_target, list(input_vals['batch_learn_len']), batch_first=True,
                                                         enforce_sorted=False)

                qf_target = self.critic_target(packed_target_ais.data)

                max_idx = self.critic(packed_target_ais.data).max(1)[1]

                # print(packed_target_ais.data.shape)
                # print(qf_target.shape)
                # print(max_idx.shape)

                qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())

                # print(qf_target.shape)

                batch_rewards = torch.from_numpy(input_vals['batch_rewards']).to(self.device)
                batch_gammas = torch.from_numpy(input_vals['batch_gammas']).to(self.device)

                packed_reward = pack_padded_sequence(batch_rewards, list(input_vals['batch_learn_len']), batch_first=True,
                                                     enforce_sorted=False)
                packed_gammas = pack_padded_sequence(batch_gammas, list(input_vals['batch_learn_len']), batch_first=True,
                                                     enforce_sorted=False)

                # print(packed_reward.data.shape)
                # print(input_vals['batch_learn_len'])

                batch_final_flag = torch.from_numpy(input_vals['batch_final_flag']).to(self.device)

                # print(batch_final_flag)

                packed_final = pack_padded_sequence(batch_final_flag, list(input_vals['batch_learn_len']), batch_first=True,
                                                    enforce_sorted=False)

                # print(packed_final.data.shape)

                next_q_value = packed_reward.data + packed_gammas.data * packed_final.data * qf_target.view(-1)

                # print(packed_reward)

                # if self.args['TD_loss'] == 'mse':
                #     qf_loss = F.mse_loss(qf.view(-1), next_q_value, reduce=False)
                # elif self.args['TD_loss'] == 'smooth_l1':
                #     qf_loss = F.smooth_l1_loss(qf.view(-1), next_q_value, reduce=False)
                # qf_loss = F.smooth_l1_loss(qf.view(-1), next_q_value , reduce = False)
                qf_loss = torch.abs(qf.view(-1) - next_q_value)

                # print(qf_loss)

                packed_loss = rnn_utils.PackedSequence(qf_loss,packed_reward.batch_sizes,packed_reward.sorted_indices,packed_reward.unsorted_indices)

                # print(packed_loss)

                unpacked_loss, loss_batch = pad_packed_sequence(packed_loss, batch_first=True)

                # print(loss_batch)
                #
                # print(unpacked_loss)

                priorities = torch.sum(unpacked_loss,1)

                # print(priorities)

                priorities = torch.div(priorities,loss_batch.to(self.device))

                # print(priorities)

                return priorities.cpu().numpy()
                # print('TD')



            elif self.args['PER_type'] == 'Model':

                torch_model_input_act = torch.from_numpy(input_vals['batch_model_input_act']).to(self.device)

                input_psi_acts_packed = pack_padded_sequence(torch_model_input_act, list(input_vals['batch_learn_forward_len']),
                                                             batch_first=True,
                                                             enforce_sorted=False)

                psi_input = torch.cat((ais_z.data, input_psi_acts_packed.data), 1).to(self.device)

                next_obs = torch.from_numpy(input_vals['batch_next_obs']).to(self.device)

                next_obs_packed = pack_padded_sequence(next_obs, list(input_vals['batch_learn_forward_len']), batch_first=True,
                                                       enforce_sorted=False)

                batch_final_flag_for_model = torch.from_numpy(input_vals['batch_final_flag_for_model']).to(self.device)

                batch_final_flag_for_model_packed = pack_padded_sequence(batch_final_flag_for_model,
                                                                         list(input_vals['batch_learn_forward_len']),
                                                                         batch_first=True,
                                                                         enforce_sorted=False)

                predicted_obs = self.psi.predict_obs(psi_input)

                pow = torch.pow(torch.norm(predicted_obs, dim=1), 2)

                if self.args['env_name'][:8] == 'MiniGrid':
                    dot = torch.matmul(next_obs_packed.data.view(pow.shape[0], 1, self.obs_dim),
                                       predicted_obs.view(pow.shape[0], self.obs_dim, 1))
                    next_obs_loss = ((pow - 2 * dot.view(-1)) * batch_final_flag_for_model_packed.data)


                    # print(next_obs_loss.shape)


                else:
                    dot = torch.matmul(next_obs_packed.data.view(pow.shape[0], 1, self.obs_dim + 1),
                                       predicted_obs.view(pow.shape[0], self.obs_dim + 1, 1))

                    next_obs_loss = (pow - 2 * dot.view(-1)).mean()
                    assert False

                predicted_reward = self.psi.predict_reward(psi_input)


                target_reward = torch.from_numpy(input_vals['batch_model_target_reward']).to(self.device)


                packed_target_reward = pack_padded_sequence(target_reward, list(input_vals['batch_learn_forward_len']),
                                                            batch_first=True,
                                                            enforce_sorted=False)

                reward_loss = F.mse_loss(predicted_reward.view(-1), packed_target_reward.data , reduce = False)

                # print(reward_loss.shape)

                total_model_loss = next_obs_loss * self.Lambda + reward_loss * (1 - self.Lambda)

                packed_loss = rnn_utils.PackedSequence(total_model_loss, packed_target_reward.batch_sizes, packed_target_reward.sorted_indices,
                                                       packed_target_reward.unsorted_indices)


                unpacked_loss, loss_batch = pad_packed_sequence(packed_loss, batch_first=True)

                # print(unpacked_loss)
                # print(loss_batch)

                priorities = torch.sum(unpacked_loss, 1)

                # print(priorities)

                priorities = torch.div(priorities, loss_batch.to(self.device))

                # print(priorities , loss_batch)

                # priorities = (priorities - self.args['MMD_min']) / np.abs(self.args['MMD_min'])
                priorities = torch.pow(self.args['Model_PER_exponent'],priorities)
                # print('pppp_gooo',priorities)


                return priorities.cpu().numpy()

    def update_parameters_r2d2(self, memory, batch_size, updates):

        # Sample a batch from memory
        # print('*********start_update**********')
        qf_losses = torch.zeros(updates)
        model_losses = torch.zeros(updates)
        policy_losses = torch.zeros(updates)
        # print(list(range(updates)))
        for i_updates in range(updates):
            self.update_to_q += 1
            if self.args['PER'] is True:
                batch_burn_in_hist, batch_learn_hist, batch_rewards, batch_learn_len, batch_forward_idx, batch_final_flag, batch_current_act, batch_hidden, batch_burn_in_len, batch_learn_forward_len, batch_next_obs, batch_model_input_act, batch_model_target_reward, batch_gammas, batch_final_flag_for_model , tree_idx , is_weight_td , is_weight_model  = memory.sample(
                    batch_size)
            else:
                batch_burn_in_hist, batch_learn_hist, batch_rewards, batch_learn_len, batch_forward_idx, batch_final_flag, batch_current_act, batch_hidden, batch_burn_in_len, batch_learn_forward_len, batch_next_obs, batch_model_input_act, batch_model_target_reward, batch_gammas, batch_final_flag_for_model = memory.sample(
                batch_size)

            # print(batch_rewards.shape)
            # print(batch_learn_len)
            # print(batch_burn_in_len)
            # print(batch_learn_forward_len)
            # print(batch_forward_idx.shape)
            # print(batch_final_flag.shape)
            # print(batch_current_act.shape)
            # print(batch_hidden[0].shape,batch_hidden[1].shape)
            # print(batch_hidden)
            batch_hidden = (batch_hidden[0].view(1, batch_size, -1).to(self.device),
                            batch_hidden[1].view(1, batch_size, -1).to(self.device))
            # print(batch_hidden)
            batch_burn_in_hist = torch.from_numpy(batch_burn_in_hist).to(self.device)
            # print(batch_burn_in_hist.shape)
            #
            #
            # print(list(batch_burn_in_len))


            zero_idx = []
            for i, len in enumerate(list(batch_burn_in_len)):
                if len == 0:
                    zero_idx.append(i)

            # print(zero_idx)
            # print(type(batch_burn_in_len[zero_idx]))
            batch_burn_in_len[zero_idx] = 1
            # print(list(batch_burn_in_len))
            # if self.model_alg == 'AIS':
            #     with torch.no_grad():
            # if self.model_alg == 'AIS':
            #     _, hidden_burn_in = self.rho(batch_burn_in_hist, batch_size, batch_hidden, self.device , list(batch_burn_in_len), self.args['replay_type'])
            # else:
            # with torch.no_grad():
            _, hidden_burn_in = self.rho(batch_burn_in_hist, batch_size, batch_hidden, self.device,
                                         list(batch_burn_in_len), self.args['replay_type'])
            # else:
            # with torch.no_grad():
            # _, hidden_burn_in = self.rho(batch_burn_in_hist, batch_size, batch_hidden, self.device,
            #                                  list(batch_burn_in_len))

            # print(hidden_burn_in[0].shape)
            #
            # print(hidden_burn_in[0][:,zero_idx,:],hidden_burn_in[1][:,zero_idx,:])
            #
            hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]
            # print(hidden_burn_in)
            # print(list(batch_learn_forward_len))

            #
            # print(hidden_burn_in[0][:, zero_idx, :], hidden_burn_in[1][:, zero_idx, :])
            batch_learn_hist = torch.from_numpy(batch_learn_hist).to(self.device)
            # self.critic.train()
            # if self.model_alg == 'AIS':
            # with torch.no_grad():
            ais_z, hidden = self.rho(batch_learn_hist, batch_size, hidden_burn_in, self.device,
                                     list(batch_learn_forward_len), self.args['replay_type'])
            if self.model_alg == 'AIS':
                torch_model_input_act = torch.from_numpy(batch_model_input_act).to(self.device)
                # print(torch_model_input_act.shape)
                input_psi_acts_packed = pack_padded_sequence(torch_model_input_act, list(batch_learn_forward_len),
                                                             batch_first=True,
                                                             enforce_sorted=False)


                psi_input = torch.cat((ais_z.data, input_psi_acts_packed.data), 1).to(self.device)

                next_obs = torch.from_numpy(batch_next_obs).to(self.device)

                next_obs_packed = pack_padded_sequence(next_obs, list(batch_learn_forward_len), batch_first=True,
                                                       enforce_sorted=False)

                batch_final_flag_for_model = torch.from_numpy(batch_final_flag_for_model).to(self.device)

                batch_final_flag_for_model_packed = pack_padded_sequence(batch_final_flag_for_model,
                                                                         list(batch_learn_forward_len),
                                                                         batch_first=True,
                                                                         enforce_sorted=False)

                if self.args['AIS_loss'] == 'MMD':
                    predicted_obs = self.psi.predict_obs(psi_input)

                    pow = torch.pow(torch.norm(predicted_obs, dim=1), 2)

                    if self.args['env_name'][:8] == 'MiniGrid':
                        dot = torch.matmul(next_obs_packed.data.view(pow.shape[0], 1, self.obs_dim),
                                           predicted_obs.view(pow.shape[0], self.obs_dim, 1))
                        next_obs_loss = ((pow - 2 * dot.view(-1)) * batch_final_flag_for_model_packed.data)




                    else:
                        dot = torch.matmul(next_obs_packed.data.view(pow.shape[0], 1, self.obs_dim + 1),
                                           predicted_obs.view(pow.shape[0], self.obs_dim + 1, 1))

                        next_obs_loss = (pow - 2 * dot.view(-1)).mean()





                elif self.args['AIS_loss'] == 'KL' and self.args['env_name'][:8] == 'MiniGrid':

                    # print(batch_final_flag_for_model_packed.data.shape,batch_final_flag_for_model_packed.data)

                    mvg_dist_mean, mvg_dist_std, mvg_dist_mix = self.psi.predict_obs(psi_input)

                    # m = MultivariateNormal(mvg_dist_mean[j - 1, :, d],
                    #                        torch.diag(bc.mvg_dist_std_estimates[j - 1, :, d]))

                    m = Normal(mvg_dist_mean, mvg_dist_std)

                    # print(m.sample().shape)
                    target = next_obs_packed.data.unsqueeze(1).expand(-1, mvg_dist_mean.shape[1], -1)

                    # print(m.log_prob(target).shape)

                    mixture_probs = torch.sum(m.log_prob(target), 2) + torch.log(mvg_dist_mix)

                    # print(mvg_dist_mix.shape,mvg_dist_mix)
                    #

                    # entropy = torch.sum(mvg_dist_mix*torch.log(mvg_dist_mix),1)
                    #
                    # print(torch.max(mixture_probs, dim = -1 , keepdim=True)[0].shape)

                    g_log_probs = mixture_probs - torch.max(mixture_probs, dim=-1, keepdim=True)[0]

                    max_probs = torch.max(mixture_probs, dim=-1, keepdim=True)[0].squeeze()

                    # print(mixture_probs - torch.max(mixture_probs, dim = -1 , keepdim=True)[0])

                    next_obs_loss = - ((torch.logsumexp(g_log_probs,
                                                        dim=1) + max_probs) * batch_final_flag_for_model_packed.data).mean()



                predicted_reward = self.psi.predict_reward(psi_input)
                # print(predicted_reward.shape)
                # print(psi_input.shape)

                target_reward = torch.from_numpy(batch_model_target_reward).to(self.device)
                # print(batch_model_target_reward.shape)

                packed_target_reward = pack_padded_sequence(target_reward, list(batch_learn_forward_len),
                                                            batch_first=True,
                                                            enforce_sorted=False)

                reward_loss = F.mse_loss(predicted_reward.view(-1), packed_target_reward.data,reduce=False)

                if self.args['PER'] is True:
                    is_weight_model = torch.from_numpy(is_weight_model).to(self.device)
                    is_weight_model_packed = pack_padded_sequence(is_weight_model,
                                                                  list(batch_learn_forward_len),
                                                                  batch_first=True,
                                                                  enforce_sorted=False)

                    total_model_loss = next_obs_loss * self.Lambda + reward_loss * (1 - self.Lambda)

                    if self.args['PER_type'] == 'Model':
                        packed_loss = rnn_utils.PackedSequence(total_model_loss.detach(), packed_target_reward.batch_sizes,
                                                               packed_target_reward.sorted_indices,
                                                               packed_target_reward.unsorted_indices)

                        unpacked_loss, loss_batch = pad_packed_sequence(packed_loss, batch_first=True)



                        priorities = torch.sum(unpacked_loss, 1)

                        # print(priorities)

                        priorities = torch.div(priorities, loss_batch.to(self.device))

                        # priorities = (priorities - self.args['MMD_min']) / np.abs(self.args['MMD_min'])
                        priorities = torch.pow(self.args['Model_PER_exponent'],priorities)
                        # print('ppppppppp',priorities)

                    total_model_loss = total_model_loss * is_weight_model_packed.data
                    total_model_loss = total_model_loss.mean()





                else:

                    total_model_loss = (next_obs_loss * self.Lambda + reward_loss * (1 - self.Lambda)).mean()

                # ais_z = ais_z.detach
                model_losses[i_updates] = total_model_loss

                # assert False

            # if self.model_alg == 'None':
            #     ais_z, hidden = self.rho(batch_learn_hist, batch_size, hidden_burn_in, self.device,
            #                                      list(batch_learn_forward_len))

            # print(ais_z)
            # print(ais_z.data.shape)

            self.critic.train()

            unpacked_ais_z, lens_unpacked = pad_packed_sequence(ais_z, batch_first=True)
            if self.model_alg == 'AIS':
                unpacked_ais_z = unpacked_ais_z.detach()

            # print(unpacked_ais_z.shape)
            # print(lens_unpacked)
            # print(batch_learn_len)

            q_z = pack_padded_sequence(unpacked_ais_z, list(batch_learn_len), batch_first=True, enforce_sorted=False)

            # print('qzdata',q_z.data.requires_grad)

            qf = self.critic(q_z.data)

            # print(qf.shape)
            # print(qf)

            batch_current_act = torch.from_numpy(batch_current_act).to(self.device)

            # print(batch_current_act)

            packed_current_act = pack_padded_sequence(batch_current_act, list(batch_learn_len), batch_first=True,
                                                      enforce_sorted=False)

            # print(packed_current_act.data)

            qf = qf.gather(1, packed_current_act.data.view(-1, 1).long())

            # print(qf)

            # print(qf.shape)
            # print(lens_unpacked)
            # print('gggggggggggg')
            # print(list(batch_learn_len))
            # print(unpacked_ais_z.shape , unpacked_ais_z)
            # print(batch_forward_idx.shape , batch_forward_idx)

            # print(batch_forward_idx.view(batch_size,-1, 1).expand(-1,-1,self.AIS_state_size))

            with torch.no_grad():
                batch_forward_idx = torch.from_numpy(batch_forward_idx).to(self.device)
                packed_batch_forward_idx = pack_padded_sequence(batch_forward_idx, list(batch_learn_len),
                                                                batch_first=True, enforce_sorted=False)
                unpacked_batch_forward_idx, _ = pad_packed_sequence(packed_batch_forward_idx, batch_first=True)
                # print(batch_forward_idx)
                # print(unpacked_batch_forward_idx)
                # print(unpacked_ais_z.shape,unpacked_ais_z)

                ais_z_target = unpacked_ais_z.gather(1,
                                                     unpacked_batch_forward_idx.view(batch_size, -1, 1).expand(-1, -1,
                                                                                                               self.AIS_state_size).long()).detach()
                # print(ais_z_target.shape,ais_z_target)
                # print(ais_z_target.shape,ais_z_target)
                # print(list(batch_learn_len))
                packed_target_ais = pack_padded_sequence(ais_z_target, list(batch_learn_len), batch_first=True,
                                                         enforce_sorted=False)

                qf_target = self.critic_target(packed_target_ais.data)
                # print(qf_target.shape)
                # qf_target = self.critic_target(packed_target_ais.data).max(1)[0]
                # print(qf_target.shape)
                # assert False
                max_idx = self.critic(packed_target_ais.data).max(1)[1]

                # print(packed_target_ais.data.shape)
                # print(qf_target.shape)
                # print(max_idx.shape)

                qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())

                # print(qf_target.shape)

                batch_rewards = torch.from_numpy(batch_rewards).to(self.device)
                batch_gammas = torch.from_numpy(batch_gammas).to(self.device)

                packed_reward = pack_padded_sequence(batch_rewards, list(batch_learn_len), batch_first=True,
                                                     enforce_sorted=False)
                packed_gammas = pack_padded_sequence(batch_gammas, list(batch_learn_len), batch_first=True,
                                                     enforce_sorted=False)

                # print(packed_reward.data.shape)

                batch_final_flag = torch.from_numpy(batch_final_flag).to(self.device)

                # print(batch_final_flag)

                packed_final = pack_padded_sequence(batch_final_flag, list(batch_learn_len), batch_first=True,
                                                    enforce_sorted=False)

                # print(packed_final.data.shape)

                next_q_value = packed_reward.data + packed_gammas.data * packed_final.data * qf_target.view(-1)

                # print(next_q_value.shape)
                # assert False

            # print(qf.shape)
            # print(qf)
            if self.args['TD_loss'] == 'mse':
                qf_loss = F.mse_loss(qf.view(-1), next_q_value, reduce=False)
            elif self.args['TD_loss'] == 'smooth_l1':
                qf_loss = F.smooth_l1_loss(qf.view(-1), next_q_value, reduce=False)
            if self.args['PER'] is True:
                is_weight_td = torch.from_numpy(is_weight_td).to(self.device)
                is_weight_td_packed = pack_padded_sequence(is_weight_td,
                                                          list(batch_learn_len),
                                                          batch_first=True,
                                                          enforce_sorted=False)

                qf_loss = qf_loss * is_weight_td_packed.data
                if self.args['PER_type'] == 'TD':
                    diff = torch.abs(qf.view(-1) - next_q_value)
                    packed_diff = rnn_utils.PackedSequence(diff, packed_reward.batch_sizes, packed_reward.sorted_indices,
                                                       packed_reward.unsorted_indices)

                    # print(packed_loss.data.shape,packed_loss.batch_sizes)

                    unpacked_diff, diff_batch = pad_packed_sequence(packed_diff, batch_first=True)

                    priorities = torch.sum(unpacked_diff, 1)

                    priorities = torch.div(priorities, diff_batch.to(self.device))



            # qf_loss = qf1_loss + qf2_loss
            qf_loss = qf_loss.mean()
            # print(qf_loss)
            qf_losses[i_updates] = qf_loss

            self.critic_optim.zero_grad()
            # if self.model_alg == 'None':
            self.AIS_optimizer.zero_grad()
            if self.model_alg == 'AIS':
                total_model_loss.backward()
                if self.args['AIS_loss'] == 'KL':
                    torch.nn.utils.clip_grad_norm_(list(self.rho.parameters()) + list(self.psi.parameters()),
                                                   max_norm=1.0, error_if_nonfinite=True)
                self.AIS_optimizer.step()
            qf_loss.backward()
            self.critic_optim.step()
            if self.model_alg == 'None':
                self.AIS_optimizer.step()

            memory.update_priorities(tree_idx, priorities.detach().cpu().numpy())

        if self.update_to_q % self.target_update_interval == 0:
            # hard_update(self.critic_target, self.critic)
            # print('hard update')
            soft_update(self.critic_target, self.critic, self.tau)
            # if self.alg == 'SAC':
            #     soft_update(self.critic_rho_target, self.rho_q, self.tau)

        # assert False
        qf_losses = qf_losses.mean()
        model_losses = model_losses.mean()
        # if self.rl_alg == 'SAC':
        #     policy_losses = policy_losses.mean()
        if self.rl_alg == 'QL':
            policy_losses = torch.zeros(1)
        # hard_update(self.policy_cpu, self.policy)
        # hard_update(self.q_cpu, self.critic)
        # # if self.model_alg == 'None':
        # hard_update(self.rho_cpu, self.rho)

        return qf_losses.item(), policy_losses.item(), model_losses

    def save_model(self, dir, seed):
        import os
        path = os.path.join(dir, 'Seed_' + str(seed) + '_models.pt')

        torch.save({
            'AIS_rho': self.rho.state_dict(),
            'AIS_psi': self.psi.state_dict(),
            'Q_target': self.critic_target.state_dict(),
            'Q': self.critic.state_dict(),
            'policy': self.policy.state_dict(),
        }, path)

    def load_model(self, path):

        checkpoint = torch.load(path)

        self.policy.load_state_dict(checkpoint['policy'])
        # self.policy_cpu.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['Q'])
        self.critic_target.load_state_dict(checkpoint['Q_target'])
        # self.q_cpu.load_state_dict(checkpoint['Q'])
        self.rho.load_state_dict(checkpoint['AIS_rho'])
        # self.rho_cpu.load_state_dict(checkpoint['AIS_rho'])
        self.psi.load_state_dict(checkpoint['AIS_psi'])

    def get_obs_dim(self):
        return self.obs_dim














    # def update_model(self, memory, batch_size, updates):
    #     # print('start training')
    #     # model_loss = torch.zeros(1)
    #     losses = torch.zeros(updates)
    #     # print(range(updates))
    #     for i_update in range(updates):
    #         # print('model_update')
    #         batch_input_obs, batch_target_obs , batch_acts, batch_rewards, batch_lengths , batch_final_flag , _ = memory.sample_full_ep(batch_size)
    #
    #         #
    #         # print('list',list(batch_lengths))
    #
    #         batch_rewards = batch_rewards.to(self.device)
    #
    #         # input_obs = F.one_hot(batch_obs.to((torch.int64)), num_classes=self.obs_dim).to(self.device) ** this is what was commented
    #         input_obs = batch_input_obs[:,:,:].to(self.device)
    #         # print('one_hot_obs',input_obs.shape , input_obs)
    #
    #         input_acts = batch_acts[:, :batch_acts.shape[1] - 1]
    #         input_acts = F.one_hot(input_acts.to((torch.int64)), num_classes=self.act_dim)
    #
    #         input_acts = torch.cat((torch.zeros(batch_size, 1, self.act_dim), input_acts), 1).to(self.device)
    #
    #         # print('one_hot_acts',input_acts.shape , input_acts)
    #         input_rewards = torch.cat((torch.zeros(batch_size, 1).to(self.device), batch_rewards[:, :batch_rewards.shape[1] - 1]),
    #                                   1).view(batch_size, -1, 1).to(self.device)
    #
    #         rho_input = torch.cat((input_obs, input_acts, input_rewards), 2).to(self.device)
    #
    #
    #
    #         # rho_input = torch.cat((input_obs, input_acts), 2).to(self.device)
    #
    #         # print(rho_input)
    #
    #         # assert False
    #
    #         # print('rho_input',rho_input.shape , rho_input)
    #
    #
    #
    #         # packed_rho_input = pack_padded_sequence(rho_input, list(batch_lengths), batch_first=True, enforce_sorted=False)
    #         #
    #         # print('rho_input_packed',packed_rho_input )
    #         #
    #         # print(packed_rho_input.batch_sizes.shape)
    #
    #         hidden = None
    #         ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device , list(batch_lengths) , 'r2d2')
    #
    #         # print(ais_z)
    #         # print(ais_z.data.shape)
    #         # print(ais_z.data)
    #         #
    #         # print(hidden[0].shape,hidden[1].shape)
    #
    #
    #
    #
    #         # print('hidden',hidden.shape,hidden)
    #         # print('ais_z',ais_z.data.shape)
    #
    #         input_psi_acts = F.one_hot(batch_acts.to((torch.int64)), num_classes=self.act_dim).to(self.device)
    #
    #         # print(batch_lengths)
    #         # print('input_psi_acts',input_psi_acts.shape,input_psi_acts)
    #
    #         input_psi_acts_packed = pack_padded_sequence(input_psi_acts, list(batch_lengths), batch_first=True,enforce_sorted=False)
    #
    #         # print(input_psi_acts_packed)
    #         # print('input_psi_acts',input_psi_acts_packed.data,input_psi_acts_packed.data.shape)
    #         # print(ais_z.data.shape)
    #
    #
    #         true_obs = batch_target_obs.to(self.device)
    #
    #         # print('true obs',true_obs,true_obs.shape)
    #
    #         psi_input = torch.cat((ais_z.data, input_psi_acts_packed.data), 1).to(self.device)
    #
    #         # print('psi_input',psi_input.shape,psi_input)
    #
    #         if self.args['AIS_loss'] == 'MMD':
    #             obs_probs = self.psi.predict_obs(psi_input)
    #
    #             # print(obs_probs.shape, obs_probs)
    #             # #
    #             # # print(true_obs.shape)
    #             # print(batch_lengths)
    #             # print(batch_target_obs)
    #
    #             true_obs_packed = pack_padded_sequence(true_obs, batch_lengths, batch_first=True,
    #                                                          enforce_sorted=False)
    #
    #
    #             pow = torch.pow(torch.norm(obs_probs, dim=1), 2)
    #             # print(pow)
    #
    #             # print(true_obs_packed.data,obs_probs)
    #             # dot = torch.matmul(true_obs_packed.data.view(pow.shape[0],1,self.obs_dim+1), obs_probs.view(pow.shape[0],self.obs_dim+1,1))
    #             # print(dot.view(-1))
    #             #
    #             # next_obs_loss = (pow - 2 * dot.view(-1)).mean()
    #
    #             # print(next_obs_loss)
    #             # print('model_loss' , model_loss)
    #
    #             if self.args['env_name'][:8] == 'MiniGrid':
    #                 # print(batch_final_flag)
    #                 batch_final_flag_packed = pack_padded_sequence(batch_final_flag,batch_lengths,batch_first=True,enforce_sorted=False).to(self.device)
    #                 # print(batch_final_flag_packed.data)
    #                 dot = torch.matmul(true_obs_packed.data.view(pow.shape[0], 1, self.obs_dim),
    #                                    obs_probs.view(pow.shape[0], self.obs_dim, 1))
    #                 next_obs_loss = ((pow - 2 * dot.view(-1)) * batch_final_flag_packed.data).mean()
    #
    #             else:
    #                 dot = torch.matmul(true_obs_packed.data.view(pow.shape[0], 1, self.obs_dim + 1),
    #                                    obs_probs.view(pow.shape[0], self.obs_dim + 1, 1))
    #
    #                 next_obs_loss = (pow - 2 * dot.view(-1)).mean()
    #         elif self.args['AIS_loss'] == 'KL' and self.args['env_name'][:8] == 'MiniGrid':
    #
    #             mvg_dist_mean, mvg_dist_std, mvg_dist_mix = self.psi.predict_obs(psi_input)
    #
    #             # m = MultivariateNormal(mvg_dist_mean[j - 1, :, d],
    #             #                        torch.diag(bc.mvg_dist_std_estimates[j - 1, :, d]))
    #
    #             m = Normal(mvg_dist_mean, mvg_dist_std)
    #
    #             # print(m.sample().shape)
    #             next_obs_packed = pack_padded_sequence(true_obs, batch_lengths, batch_first=True,
    #                                                    enforce_sorted=False)
    #             target = next_obs_packed.data.unsqueeze(1).expand(-1, mvg_dist_mean.shape[1], -1)
    #
    #             # print(m.log_prob(target).shape)
    #
    #             mixture_probs = torch.sum(m.log_prob(target), 2) + torch.log(mvg_dist_mix)
    #
    #             # print(mvg_dist_mix.shape,mvg_dist_mix)
    #             #
    #
    #             # entropy = torch.sum(mvg_dist_mix*torch.log(mvg_dist_mix),1)
    #             #
    #             # print(torch.max(mixture_probs, dim = -1 , keepdim=True)[0].shape)
    #
    #             g_log_probs = mixture_probs - torch.max(mixture_probs, dim=-1, keepdim=True)[0]
    #
    #             max_probs = torch.max(mixture_probs, dim=-1, keepdim=True)[0].squeeze()
    #
    #             # print(mixture_probs - torch.max(mixture_probs, dim = -1 , keepdim=True)[0])
    #             batch_final_flag_packed = pack_padded_sequence(batch_final_flag, batch_lengths, batch_first=True,
    #                                                            enforce_sorted=False).to(self.device)
    #
    #             next_obs_loss = - ((torch.logsumexp(g_log_probs,
    #                                                 dim=1) + max_probs) * batch_final_flag_packed.data).mean()
    #
    #
    #
    #
    #
    #         reward_est = self.psi.predict_reward(psi_input)
    #
    #         # print(reward_est.shape,batch_rewards.shape)
    #
    #         packed_batch_rewards = pack_padded_sequence(batch_rewards, list(batch_lengths), batch_first=True,
    #                                                     enforce_sorted=False)
    #
    #         # print(packed_batch_rewards)
    #         # print(packed_batch_rewards.data.shape)
    #         # print(torch.sign(packed_batch_rewards.data)*torch.pow(torch.abs(packed_batch_rewards.data),0.5)/5)
    #
    #         reward_loss = F.mse_loss(reward_est.view(-1), packed_batch_rewards.data)
    #
    #         model_loss = next_obs_loss * self.Lambda + reward_loss * (1 - self.Lambda)
    #
    #         self.AIS_optimizer.zero_grad()
    #         model_loss.backward()
    #         # print(model_loss.grad)
    #         # for param in self.rho.parameters():
    #         #   print(param.grad)
    #         # for param in self.psi.parameters():
    #         #   print(param.grad)
    #         self.AIS_optimizer.step()
    #         #  print(model_loss)
    #         losses[i_update] = model_loss
    #
    #     losses = losses.mean()
    #     # hard_update(self.rho_cpu, self.rho)
    #     return losses.detach().cpu().item()
    #
    # def update_parameters_vanilla(self, memory, batch_size, updates):
    #     # Sample a batch from memory
    #     # print('*********start_update**********')
    #     qf_losses = torch.zeros(updates)
    #     policy_losses = torch.zeros(updates)
    #     # print(list(range(updates)))
    #     for i_updates in range(updates):
    #         batch_obs, batch_acts, batch_rewards, batch_idx, batch_lengths, batch_mask = memory.sample(batch_size)
    #
    #         # print('q update')
    #         # print(batch_obs.shape)
    #         # print(batch_acts.shape)
    #         # print(batch_rewards.shape)
    #         # print(batch_idx)
    #         # print(batch_lengths)
    #
    #         batch_idx_ = torch.unsqueeze(batch_idx, 1)
    #         # print(batch_idx_)
    #
    #         temp_ones = torch.ones((batch_size, 1, self.AIS_state_size))
    #
    #         batch_idx_ = (temp_ones * batch_idx_[:, :, None]).to(self.device)
    #
    #         # print(batch_idx_.shape, batch_idx_)
    #
    #         # input_obs = F.one_hot(batch_obs.to((torch.int64)), num_classes=self.obs_dim).to(self.device) this is what was commented
    #
    #         input_obs = batch_obs.to(self.device)
    #
    #         # print('batch_obs',batch_obs.shape,batch_obs)
    #
    #         input_acts = batch_acts[:, :batch_acts.shape[1] - 1]
    #         input_acts = F.one_hot(input_acts.to((torch.int64)), num_classes=self.act_dim)
    #
    #         input_acts = torch.cat((torch.zeros(batch_size, 1, self.act_dim), input_acts), 1).to(self.device)
    #
    #         # print('input_acts',input_acts.shape,input_acts)
    #         input_rewards = torch.cat((torch.zeros(batch_size, 1), batch_rewards[:,:batch_rewards.shape[1]-1]), 1).view(batch_size,-1,1).to(self.device)
    #
    #         rho_input = torch.cat((input_obs, input_acts , input_rewards ), 2).to(self.device)
    #
    #         # print(batch_rewards.shape, batch_rewards)
    #         # print(batch_rewards[:,:batch_rewards.shape[1]-1].shape, batch_rewards[:,:batch_rewards.shape[1]-1])
    #         # print(torch.cat((torch.zeros(batch_size, 1), batch_rewards[:,:batch_rewards.shape[1]-1]), 1).to(self.device).shape,torch.cat((torch.zeros(batch_size, 1), batch_rewards[:,:batch_rewards.shape[1]-1]), 1).to(self.device))
    #         #
    #         # print(rho_input.shape,rho_input)
    #
    #         # print('rho_input',rho_input.shape,rho_input)
    #         hidden = None
    #
    #         if self.model_alg == 'AIS':
    #             with torch.no_grad():
    #                 ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device , [] , self.args['replay_type'])
    #         if self.model_alg == 'None':
    #             ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device , [] , self.args['replay_type'])
    #         # print('gggggg',batch_idx_.shape, batch_idx_)
    #         # print('ais_z',ais_z.shape,ais_z)
    #         h_input = ais_z.gather(1, batch_idx_.long())
    #         # print('h_input' , h_input.shape , h_input)
    #         h_next_input = ais_z.gather(1,  torch.add(batch_idx_.long(), 1))
    #         # print('gggggg', batch_idx_.shape, batch_idx_)
    #         qf1, qf2 = self.critic(h_input.squeeze())
    #
    #         # print('qf',qf1.shape,qf1)
    #         # print(batch_idx_.shape,batch_idx_)
    #         # print(batch_acts.shape,batch_acts)
    #
    #         batch_idx_ = torch.unsqueeze(batch_idx, 1).to(self.device)
    #         # print('idx',batch_idx_.shape, batch_idx_)
    #         q_a = batch_acts.to(self.device).gather(1, batch_idx_.long())
    #
    #         # print(q_a.shape,q_a)
    #
    #
    #
    #         # print('batch_acts',batch_acts.shape,batch_acts)
    #         # print('q_a',q_a.shape,q_a)
    #
    #         qf1 = qf1.gather(1, q_a.long())
    #         qf2 = qf2.gather(1, q_a.long())
    #
    #         # print('qf1',qf1.shape,qf1)
    #
    #
    #         with torch.no_grad():
    #
    #             # print('next_state_pi', next_state_pi.shape,next_state_pi)
    #
    #             qf1_next_target, qf2_next_target = self.critic_target(h_next_input.squeeze())
    #
    #             if self.rl_alg == 'QL':
    #                 min_qf_next_target =  (torch.min(qf1_next_target, qf2_next_target)).max(1)[0].unsqueeze(1)
    #                 # print('min_qf',min_qf_next_target.shape, min_qf_next_target)
    #
    #             if self.rl_alg == 'SAC':
    #                 _, next_state_pi, next_state_log_pi = self.policy.sample(h_next_input.squeeze())
    #                 min_qf_next_target = (next_state_pi * (
    #                         torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi)).sum(dim=1,
    #                                                                                                            keepdim=True)
    #
    #             q_r = batch_rewards.to(self.device).gather(1, batch_idx_.long())
    #             # print('q_r',q_r.shape,q_r)
    #             # print('batch_rewards',batch_rewards.shape,batch_rewards)
    #             # print('batch_idx_',batch_idx_.shape,batch_idx_)
    #             # print('batch_mask',batch_mask.shape,batch_mask)
    #
    #             # print('min_qf_next_target',min_qf_next_target.shape,min_qf_next_target)
    #             next_q_value = q_r + self.gamma * batch_mask.unsqueeze(1).to(self.device) * min_qf_next_target
    #         # print('next_q', next_q_value.shape , next_q_value)
    #
    #         qf1_loss = F.mse_loss(qf1,
    #                               next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    #         qf2_loss = F.mse_loss(qf2,
    #                               next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    #         # qf_loss = qf1_loss + qf2_loss
    #         qf_loss = qf1_loss + qf2_loss
    #         # print(qf_loss)
    #         qf_losses[i_updates] = qf_loss
    #         # raise 'Error'
    #
    #         # print('qf_lossssssss', qf_loss.shape, qf_loss)
    #
    #         if self.rl_alg == 'SAC':
    #
    #             _, pi, log_pi = self.policy.sample(h_input.detach().squeeze())
    #             # print('pi', pi.shape , pi)
    #             with torch.no_grad():
    #                 qf1_pi, qf2_pi = self.critic(h_input.squeeze())
    #                 # print('qf1 and qf2 pi' , qf1_pi , qf2_pi)
    #                 min_qf_pi = torch.min(qf1_pi, qf2_pi)
    #                 # print('min_qf_pi',min_qf_pi)
    #             entropies = -torch.sum(pi * log_pi, dim=1, keepdim=True)
    #             # print('entropy before sum',pi * log_pi)
    #             # print('entropies',entropies)
    #             q = torch.sum(min_qf_pi * pi, dim=1, keepdim=True)
    #             # print('expected q before sum', min_qf_pi * pi)
    #             # print('q policy',q)
    #             policy_loss = (-(self.alpha * entropies) - q).mean()
    #
    #             # print('policy',policy_loss.shape,policy_loss)
    #             policy_losses[i_updates] = policy_loss
    #
    #         if self.automatic_entropy_tuning:
    #             alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    #
    #             self.alpha_optim.zero_grad()
    #             alpha_loss.backward()
    #             self.alpha_optim.step()
    #
    #             self.alpha = self.log_alpha.exp()
    #
    #         self.critic_optim.zero_grad()
    #         if self.model_alg == 'None':
    #             self.AIS_optimizer.zero_grad()
    #         qf_loss.backward()
    #         self.critic_optim.step()
    #         if self.model_alg == 'None':
    #             self.AIS_optimizer.step()
    #         if self.rl_alg == 'SAC':
    #             self.policy_optim.zero_grad()
    #             policy_loss.backward()
    #             self.policy_optim.step()
    #
    #     if updates % self.target_update_interval == 0:
    #         # hard_update(self.critic_target, self.critic)
    #         soft_update(self.critic_target, self.critic, self.tau)
    #         # if self.alg == 'SAC':
    #         #     soft_update(self.critic_rho_target, self.rho_q, self.tau)
    #
    #     # assert False
    #     qf_losses = qf_losses.mean()
    #     if self.rl_alg == 'SAC':
    #         policy_losses = policy_losses.mean()
    #     if self.rl_alg == 'QL':
    #         policy_losses = torch.zeros(1)
    #     # # hard_update(self.policy_cpu, self.policy)
    #     # # hard_update(self.q_cpu, self.critic)
    #     # if self.model_alg == 'None':
    #     #     hard_update(self.rho_cpu, self.rho)
    #
    #
    #     return qf_losses.item(), policy_losses.item()



