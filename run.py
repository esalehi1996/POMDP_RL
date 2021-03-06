import itertools

import numpy as np

from env.Tiger import TigerEnv
from env.RockSampling import RockSamplingEnv
from env.DroneSurveillance import DroneSurveillanceEnv
from env.cheesemaze import CheeseMazeEnv
from env.voicemail import VoicemailEnv
from SAC import SAC
from Replaybuffer import Rec_ReplayMemory
from r2d2replaybuffer import r2d2_ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from models import convert_int_to_onehot
from args import Args
import pickle
# env = gym.make("Battleship-v0")
# env = TigerEnv()
def run_exp(args):
    writer = SummaryWriter(args['logdir'])
    list_of_test_rewards_allseeds = []
    list_of_discount_test_rewards_allseeds = []
    if args['env_name'] == 'Tiger':
        env = TigerEnv()
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'RockSampling':
        env = RockSamplingEnv()
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    elif args['env_name'] == 'Cheesemaze':
        env = CheeseMazeEnv()
        max_env_steps = 100
    elif args['env_name'] == 'Voicemail':
        env = VoicemailEnv()
        max_env_steps = 100
    elif args['env_name'] == 'DroneSurveillance':
        env = DroneSurveillanceEnv()
        max_env_steps = 200
    elif args['env_name'][:8] == 'MiniGrid':
        env = gym.make(args['env_name'])
        max_env_steps = 400
    args['max_env_steps'] = max_env_steps
    sac = SAC(env, args)
    if args['env_name'][:8] == 'MiniGrid':
        state_size = sac.get_obs_dim()
        print(env.action_space, state_size)
    else:
        state_size = env.observation_space.n
        print(env.action_space, state_size)
    if args['replay_type'] == 'vanilla':
        memory = Rec_ReplayMemory(args['replay_size'], state_size, env.action_space.n, max_env_steps)
        print('vanilla')
    if args['replay_type'] == 'r2d2':
        memory = r2d2_ReplayMemory(args['replay_size'], state_size, env.action_space.n, max_env_steps, args)
        print('r2d2')
    for seed in range(args['num_seeds']):
        list_of_test_rewards = []
        list_of_discount_test_rewards = []
        print('-------------------------------------')
        print('seed number '+str(seed)+' running')
        print('-------------------------------------')
        total_numsteps = 0
        k_steps = 0
        updates = 1
        sac = SAC(env, args)

        if args['load_from_path'] != 'None':
            print('---------------  Loading Model from path:' , args['load_from_path'] , '-----------------')
            sac.load_model(args['load_from_path'])
        memory.reset(seed)

        ls_running_rewards = []
        avg_reward = 0
        avg_episode_steps = 0
        avg_q_loss = 0
        avg_p_loss = 0
        avg_model_loss = 0
        model_updates = 0
        k_episode = 0
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.reset()
            ls_states = []
            ls_actions = []
            ls_rewards = []
            ls_hiddens = []
            start = True
            hidden_p = None
            action = 0
            reward = 0
            # ls_rewards.append(0.)
            # ls_actions.append(0.)

            while not done:
                # print('staaart')
                if args['env_name'][:8] == 'MiniGrid':
                    state = state['image']
                    state = sac.get_encoded_obs(state)
                else:
                    state = convert_int_to_onehot(state, state_size)
                # print('printing state',state.shape, state)
                # raise "Error"
                ls_states.append(state.numpy())
                ls_hiddens.append(hidden_p)
                if i_episode <= args['random_actions_until'] or args['only_train_model']:
                    # print('random action mode')
                    # action = 0
                    # # action = env.action_space.sample()  # Sample random action
                    # if episode_steps > 10:
                    action = env.action_space.sample()
                else:
                    EPS_up = False
                    if i_episode >= args['start_updates_to_p_q_after']:
                        EPS_up = True
                    action, hidden_p = sac.select_action(state, action, reward, hidden_p, start , EPS_up , evaluate = False)  # Sample action from policy
                    # print(action)
                if start == True:
                    start = False

                next_state, reward, done, _ = env.step(action)  # Step
                ls_actions.append(action)
                ls_rewards.append(reward)
                if args['model_alg'] == 'AIS' and args['replay_type'] == 'vanilla':
                    if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_model_after'] and total_numsteps % args['update_model_every_n_steps'] == 0:
                        model_loss = sac.update_model(memory, args['batch_size'], args['model_updates_per_step'])
                        avg_model_loss += model_loss
                        model_updates += 1
                if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_p_q_after'] and total_numsteps % args['rl_update_every_n_steps'] == 0 and not args['only_train_model']:
                    critic_loss, policy_loss , model_loss = sac.update_parameters(memory, args['batch_size'], args['p_q_updates_per_step'])
                    updates += 1
                    avg_p_loss += policy_loss
                    avg_q_loss += critic_loss
                    if args['replay_type'] == 'r2d2':
                        avg_model_loss += model_loss
                        model_updates += 1
                    # avg_p_loss += 0
                    # avg_q_loss += 0
                episode_steps += 1
                total_numsteps += 1
                k_steps += 1
                episode_reward = reward + episode_reward

                state = next_state
                if total_numsteps % args['logging_freq'] == args['logging_freq']-1:
                    avg_reward , avg_discount_adj_reward = log_test_and_save(env, sac, writer, args, avg_reward, avg_q_loss, avg_p_loss, avg_model_loss, updates,
                                      model_updates, k_episode, i_episode, total_numsteps, avg_episode_steps, state_size , seed  )
                    list_of_test_rewards.append(avg_reward)
                    list_of_discount_test_rewards.append(avg_discount_adj_reward)
                    avg_reward = 0
                    avg_episode_steps = 0
                    avg_q_loss = 0
                    avg_p_loss = 0
                    avg_model_loss = 0
                    model_updates = 0
                    updates = 0
                    k_episode = 0
                if episode_steps >= max_env_steps:
                    break
            # print('gg')
            memory.push(ls_states, ls_actions, ls_rewards , ls_hiddens)  # Append transition to memory
            k_episode += 1
            # print(ls_states,ls_actions,ls_rewards)
            ls_running_rewards.append(episode_reward)
            avg_reward = avg_reward + episode_reward
            avg_episode_steps = episode_steps + avg_episode_steps

            if total_numsteps > args['num_steps']:
                break

        list_of_test_rewards_allseeds.append(list_of_test_rewards)
        list_of_discount_test_rewards_allseeds.append(list_of_discount_test_rewards)



    env.close()
    writer.close()
    # print('---------------------')
    # print(list_of_test_rewards_allseeds)
    # print(list_of_discount_test_rewards_allseeds)
    # print(len(list_of_test_rewards_allseeds))
    # print(len(list_of_discount_test_rewards_allseeds))
    # print(len(list_of_test_rewards_allseeds[0]))
    # print(len(list_of_discount_test_rewards_allseeds[0]))
    arr_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_d_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    for i in range(args['num_seeds']):
        arr_r[i,:] = np.array(list_of_test_rewards_allseeds[i])
        arr_d_r[i,:] = np.array(list_of_discount_test_rewards_allseeds[i])

    np.save(args['logdir']+'/'+args['exp_name']+'_arr_r',arr_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_d_r', arr_d_r)
    # print(arr_r)
    # print(arr_d_r)




def log_test_and_save(env , sac , writer , args , avg_reward , avg_q_loss , avg_p_loss , avg_model_loss , updates , model_updates , k_episode , i_episode , total_numsteps , avg_episode_steps , state_size , seed ):
    sac.save_model(args['logdir'],seed)
    avg_running_reward = avg_reward / k_episode
    avg_reward = 0.
    avg_discount_adj_reward = 0.
    episodes = 10


    if not args['only_train_model']:
        for _ in range(episodes):
            start = True
            hidden_p = None
            action = 0
            reward = 0
            state = env.reset()
            episode_reward = 0
            episode_rewards = []
            done = False
            steps = 0
            while not done:
                if args['env_name'][:8] == 'MiniGrid':
                    state = state['image']
                    state = sac.get_encoded_obs(state)
                else:
                    state = convert_int_to_onehot(state, state_size)
                steps += 1
                action, hidden_p = sac.select_action(state, action, reward, hidden_p, start,False, evaluate=True)
                # action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                episode_rewards.append(reward)
                episode_reward += reward

                state = next_state
                if start == True:
                    start = False
                if steps >= args['max_env_steps']:
                    break
            avg_reward += episode_reward
            rets = []
            R = 0
            for i, r in enumerate(episode_rewards[::-1]):
                R = r + args['gamma'] * R
                rets.insert(0, R)
            avg_discount_adj_reward += rets[0]
    avg_reward /= episodes
    avg_discount_adj_reward /= episodes


    # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    if args['model_alg'] == 'AIS':
        if updates == 0:
            avgql = 0
            avgpl = 0
        else:
            avgql = avg_q_loss / (updates)
            avgpl = avg_p_loss / (updates)
        if model_updates == 0:
            avgml = 0
        else:
            avgml = avg_model_loss / (model_updates)
        print(
            "Seed: {}, Episode: {}, Total_num_steps: {},  episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}, avg_q_loss: {}, avg_p_loss: {} , avg_model_loss: {}".format(
                seed,i_episode, total_numsteps, avg_episode_steps / k_episode, avg_running_reward, avg_reward,
                avg_discount_adj_reward, avgql, avgpl, avgml))
    if args['model_alg'] == 'None':
        if updates == 0:
            avgql = 0
            avgpl = 0
        else:
            avgql = avg_q_loss / updates
            avgpl = avg_p_loss / updates
        print(
            "Seed: {}, Episode: {}, Total_num_steps: {}, episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}, avg_q_loss: {}, avg_p_loss: {}".format(
                seed,i_episode, total_numsteps,
                avg_episode_steps / k_episode, avg_running_reward,
                avg_reward, avg_discount_adj_reward,
                avgql, avgpl))

    writer.add_scalar('Seed'+str(seed)+'Evaluation reward', avg_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Discount adjusted Evaluation reward', avg_discount_adj_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Training reward', avg_running_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Average Steps for each episode', avg_episode_steps, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Policy', avgpl, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Value', avgql, total_numsteps)
    if args['model_alg'] == 'AIS':
        writer.add_scalar('Seed'+str(seed)+'Loss/Model', avgml, total_numsteps)
    print("----------------------------------------")
    writer.flush()

    return avg_reward , avg_discount_adj_reward