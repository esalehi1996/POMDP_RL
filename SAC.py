import os

from models import *
# from autoencoder.simple_autoencoder import autoencoder
import torch
import torch.nn.functional as F
from torch.optim import Adam


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.act_dim = action_space.n
        self.obs_dim = num_inputs
        self.Lambda = args['AIS_lambda']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['SAC_alpha']
        self.AIS_state_size = args['AIS_state_size']
        self.target_update_interval = args['target_update_interval']
        self.action_space = action_space
        self.state_size = num_inputs
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        self.rl_alg = args['rl_alg']
        print(self.rl_alg)
        self.model_alg = args['model_alg']
        # self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        self.policy = policy_net_lowdim(action_space.n, self.AIS_state_size).to(self.device)

        self.critic = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size']).to(device=self.device)
        self.critic_target = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size']).to(self.device)

        self.policy_cpu = policy_net_lowdim(action_space.n, self.AIS_state_size)
        self.q_cpu = QNetwork_discrete(self.AIS_state_size, action_space.n, args['hidden_size'])

        # if self.alg == 'SAC+AIS':
        self.rho = rho_net_lowdim(num_inputs, action_space.n, self.AIS_state_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args['rl_lr'])
        self.policy_optim = Adam(self.policy.parameters(), lr=args['rl_lr'])
        self.rho_cpu = rho_net_lowdim(num_inputs, action_space.n, self.AIS_state_size)
        self.psi = psi_net_lowdim(num_inputs, action_space.n, self.AIS_state_size).to(self.device)

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
        hard_update(self.q_cpu, self.critic)
        hard_update(self.rho_cpu, self.rho)
        hard_update(self.policy_cpu, self.policy)
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

    def select_action(self, state, action, reward, hidden_p, start , evaluate):
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
            state = convert_int_to_onehot(state, self.state_size)
            rho_input = torch.cat((state, action)).reshape(1, 1, -1)
            # print('rho_in',rho_input)
            # if self.alg == 'SAC+AIS':
            ais_z, hidden_p = self.rho_cpu(rho_input, 1, hidden_p, 'cpu')
            # if self.alg == 'SAC':
            #     ais_z, hidden_p = self.rho_policy(rho_input, 1, hidden_p)
            # print('ais',ais_z)
            ais_z = ais_z[ais_z.shape[0] - 1]
            # print('ais',ais_z)
            # print('hidden',hidden_p)
            if self.rl_alg == 'SAC':
                action, pi, _ = self.policy_cpu.sample(ais_z.detach())
                # print(action)
                # raise "Error"
            if self.rl_alg == 'QL':
                qf1, qf2 = self.q_cpu(ais_z.detach())
                # print(qf1 , qf2)
                min_q = (torch.min(qf1, qf2))
                # print(min_q.shape,min_q)
                max_ac = min_q.max(1)[1]
                # print(min_q.max(1))
                # raise "Error"


            # print('action',action)
            # print('pi',pi)

        if self.rl_alg == 'SAC':
            return action.detach().cpu().numpy()[0][0], hidden_p
        if self.rl_alg == 'QL':
            import math
            import random
            eps_threshold = self.eps_greedy_parameters['EPS_END'] + (self.eps_greedy_parameters['EPS_START'] - self.eps_greedy_parameters['EPS_END']) * \
                            math.exp(-1. * self.env_steps / self.eps_greedy_parameters['EPS_DECAY'])
            self.env_steps += 1
            sample = random.random()
            if sample > eps_threshold:
                return max_ac.detach().cpu().numpy()[0] , hidden_p
            else:
                # print('gggggggggggg',torch.tensor([[random.randrange(self.act_dim)]],dtype=torch.long).cpu().numpy())
                return torch.tensor([[random.randrange(self.act_dim)]],dtype=torch.long).cpu().numpy()[0][0] , hidden_p

    def update_model(self, memory, batch_size, updates):
        # print('start training')
        # model_loss = torch.zeros(1)
        losses = torch.zeros(updates)
        # print(range(updates))
        for i_update in range(updates):
            batch_obs, batch_acts, batch_rewards, batch_lengths, max_len = memory.sample_full_ep(batch_size)

            # print('obs',batch_obs.shape,batch_obs)
            # print('acts',batch_acts.shape,batch_acts)
            # print('rewards',batch_rewards.shape,batch_rewards)
            # print('lengths',batch_lengths.shape,batch_lengths)
            batch_rewards = batch_rewards.to(self.device)

            input_obs = F.one_hot(batch_obs.to((torch.int64)), num_classes=self.obs_dim).to(self.device)
            # print('one_hot_obs',input_obs.shape , input_obs)

            input_acts = batch_acts[:, :batch_acts.shape[1] - 1]
            input_acts = F.one_hot(input_acts.to((torch.int64)), num_classes=self.act_dim)

            input_acts = torch.cat((torch.zeros(batch_size, 1, self.act_dim), input_acts), 1).to(self.device)

            # print('one_hot_acts',input_acts.shape , input_acts)

            rho_input = torch.cat((input_obs, input_acts), 2).to(self.device)

            # print('rho_input',rho_input.shape , rho_input)

            hidden = None
            ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device)

            # print('hidden',hidden.shape,hidden)
            # print('ais_z',ais_z.shape, ais_z)

            input_psi_acts = F.one_hot(batch_acts.to((torch.int64)), num_classes=self.act_dim).to(self.device)

            true_obs = input_obs[:, 1:, :]

            psi_input = torch.cat((ais_z, input_psi_acts), 2).to(self.device)

            reward_est, obs_probs = self.psi(psi_input)

            # print('reward_estimate',reward_est.shape,reward_est)
            # print('true_rewards',batch_rewards.shape,batch_rewards)

            # print('predicted obs probs',obs_probs.shape,obs_probs)
            # print('true_obs',true_obs.shape,true_obs)
            # print('true_obs reshaped',true_obs.reshape(-1,1,self.obs_dim).float().shape,true_obs.reshape(-1,1,self.obs_dim).float())
            obs_probs = obs_probs[:, :obs_probs.shape[1] - 1, :]

            # print('predicted obs probs minus the last one',obs_probs.shape,obs_probs)

            # print(torch.pow(torch.norm(obs_probs,dim = 2),2).shape)
            pow = torch.pow(torch.norm(obs_probs, dim=2), 2)
            dot = torch.matmul(true_obs.view(batch_size, max_len - 1, 1, self.obs_dim).float(),
                               obs_probs.view(batch_size, max_len - 1, self.obs_dim, 1))
            dot = dot.view(batch_size, max_len - 1)

            # next_obs_loss = torch.pow(torch.norm(obs_probs,dim = 2),2) - 2*torch.bmm(true_obs.view(-1,1,self.obs_dim).float(), obs_probs.view(-1,self.obs_dim,1)).view(-1)
            mask = torch.ones((batch_size, max_len)).to(self.device)
            for i in range(batch_size):
                mask[i, batch_lengths[i]:] = 0
            # print(mask.shape,mask)
            # print(mask[:,1:].shape,mask[:,1:])
            # print(mask[:,batch_lengths])

            next_obs_loss = pow - 2 * dot
            #  print(next_obs_loss.shape,next_obs_loss)
            next_obs_loss = next_obs_loss * mask[:, 1:]
            #  print(next_obs_loss.shape,next_obs_loss)
            #  print(batch_lengths)
            #  print(batch_rewards.shape,batch_rewards)
            next_obs_loss = next_obs_loss.mean()
            # print(next_obs_loss.shape,next_obs_loss)
            reward_loss = F.mse_loss(reward_est.view(batch_size, max_len) * mask, batch_rewards * mask)
            # print(reward_loss.shape,reward_loss)
            model_loss = next_obs_loss * self.Lambda + reward_loss * (1 - self.Lambda)

            self.AIS_optimizer.zero_grad()
            model_loss.backward()
            # print(model_loss.grad)
            # for param in self.rho.parameters():
            #   print(param.grad)
            # for param in self.psi.parameters():
            #   print(param.grad)
            self.AIS_optimizer.step()
            #  print(model_loss)
            losses[i_update] = model_loss

        losses = losses.mean()
        hard_update(self.rho_cpu, self.rho)
        return losses.detach().cpu().item()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # print('*********start_update**********')
        qf_losses = torch.zeros(updates)
        policy_losses = torch.zeros(updates)
        # print(list(range(updates)))
        for i_updates in range(updates):
            batch_obs, batch_acts, batch_rewards, batch_idx, batch_lengths, batch_mask = memory.sample(batch_size)

            batch_idx_ = torch.unsqueeze(batch_idx, 1)

            temp_ones = torch.ones((batch_size, 1, self.AIS_state_size))

            batch_idx_ = (temp_ones * batch_idx_[:, :, None]).to(self.device)

            input_obs = F.one_hot(batch_obs.to((torch.int64)), num_classes=self.obs_dim).to(self.device)

            input_acts = batch_acts[:, :batch_acts.shape[1] - 1]
            input_acts = F.one_hot(input_acts.to((torch.int64)), num_classes=self.act_dim)

            input_acts = torch.cat((torch.zeros(batch_size, 1, self.act_dim), input_acts), 1).to(self.device)

            rho_input = torch.cat((input_obs, input_acts), 2)

            # print('rho_input',rho_input.shape,rho_input)
            hidden = None

            if self.model_alg == 'AIS':
                with torch.no_grad():
                    ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device)
            if self.model_alg == 'None':
                ais_z, hidden = self.rho(rho_input, batch_size, hidden, self.device)

            # print('ais_z',ais_z.shape,ais_z)
            h_input = ais_z.gather(1, batch_idx_.long())
            # print('h_input' , h_input.shape , h_input)
            h_next_input = ais_z.gather(1, batch_idx_.long() + 1)

            qf1, qf2 = self.critic(h_input.squeeze())

            # print('qf',qf1.shape,qf1)

            batch_idx_ = torch.unsqueeze(batch_idx, 1).to(self.device)
            q_a = batch_acts.to(self.device).gather(1, batch_idx_.long())

            # print('batch_acts',batch_acts.shape,batch_acts)
            # print('q_a',q_a.shape,q_a)

            qf1 = qf1.gather(1, q_a.long())
            qf2 = qf2.gather(1, q_a.long())

            # print('qf1',qf1.shape,qf1)

            with torch.no_grad():

                # print('next_state_pi', next_state_pi.shape,next_state_pi)

                qf1_next_target, qf2_next_target = self.critic_target(h_next_input.squeeze())

                if self.rl_alg == 'QL':
                    min_qf_next_target =  (torch.min(qf1_next_target, qf2_next_target)).max(1)[0].unsqueeze(1)
                    # print('min_qf',min_qf_next_target.shape, min_qf_next_target)

                if self.rl_alg == 'SAC':
                    _, next_state_pi, next_state_log_pi = self.policy.sample(h_next_input.squeeze())
                    min_qf_next_target = (next_state_pi * (
                            torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi)).sum(dim=1,
                                                                                                               keepdim=True)

                q_r = batch_rewards.to(self.device).gather(1, batch_idx_.long())
                # print('q_r',q_r.shape,q_r)
                # print('batch_rewards',batch_rewards.shape,batch_rewards)
                # print('batch_idx_',batch_idx_.shape,batch_idx_)
                # print('batch_mask',batch_mask.shape,batch_mask)

                # print('min_qf_next_target',min_qf_next_target.shape,min_qf_next_target)
                next_q_value = q_r + self.gamma * batch_mask.unsqueeze(1).to(self.device) * min_qf_next_target
            # print('next_q', next_q_value.shape , next_q_value)

            qf1_loss = F.mse_loss(qf1,
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2,
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf_loss = qf1_loss + qf2_loss
            qf_loss = qf1_loss + qf2_loss
            # print(qf_loss)
            qf_losses[i_updates] = qf_loss
            # raise 'Error'

            # print('qf_lossssssss', qf_loss.shape, qf_loss)

            if self.rl_alg == 'SAC':

                _, pi, log_pi = self.policy.sample(h_input.detach().squeeze())
                # print('pi', pi.shape , pi)
                with torch.no_grad():
                    qf1_pi, qf2_pi = self.critic(h_input.squeeze())
                    # print('qf1 and qf2 pi' , qf1_pi , qf2_pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    # print('min_qf_pi',min_qf_pi)
                entropies = -torch.sum(pi * log_pi, dim=1, keepdim=True)
                # print('entropy before sum',pi * log_pi)
                # print('entropies',entropies)
                q = torch.sum(min_qf_pi * pi, dim=1, keepdim=True)
                # print('expected q before sum', min_qf_pi * pi)
                # print('q policy',q)
                policy_loss = (-(self.alpha * entropies) - q).mean()

                # print('policy',policy_loss.shape,policy_loss)
                policy_losses[i_updates] = policy_loss

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()

            self.critic_optim.zero_grad()
            if self.model_alg == 'None':
                self.AIS_optimizer.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
            if self.model_alg == 'None':
                self.AIS_optimizer.step()
            if self.rl_alg == 'SAC':
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            # hard_update(self.critic_target, self.critic)
            soft_update(self.critic_target, self.critic, self.tau)
            # if self.alg == 'SAC':
            #     soft_update(self.critic_rho_target, self.rho_q, self.tau)

        # assert False
        qf_losses = qf_losses.mean()
        if self.rl_alg == 'SAC':
            policy_losses = policy_losses.mean()
        if self.rl_alg == 'QL':
            policy_losses = torch.zeros(1)
        hard_update(self.policy_cpu, self.policy)
        hard_update(self.q_cpu, self.critic)
        if self.model_alg == 'None':
            hard_update(self.rho_cpu, self.rho)


        return qf_losses.item(), policy_losses.item()

    def save_model(self , dir):
        import os
        path = os.path.join(dir, 'models.pt')

        torch.save({
            'AIS_rho' : self.rho.state_dict(),
            'AIS_psi' : self.psi.state_dict(),
            'Q_target' : self.critic_target.state_dict(),
            'Q': self.critic.state_dict(),
            'policy': self.policy.state_dict() ,
        } , path)

    def load_model(self , path):

        checkpoint = torch.load(path)

        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_cpu.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['Q'])
        self.critic_target.load_state_dict(checkpoint['Q_target'])
        self.q_cpu.load_state_dict(checkpoint['Q'])
        self.rho.load_state_dict(checkpoint['AIS_rho'])
        self.rho_cpu.load_state_dict(checkpoint['AIS_rho'])
        self.psi.load_state_dict(checkpoint['AIS_psi'])