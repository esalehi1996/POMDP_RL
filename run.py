import itertools

import numpy as np
import os
from env.Tiger import TigerEnv
from env.RockSampling import RockSamplingEnv
from env.DroneSurveillance import DroneSurveillanceEnv
from env.cheesemaze import CheeseMazeEnv
from env.voicemail import VoicemailEnv
from env.custom_minigrid import *
from SAC import SAC
from Replaybuffer import Rec_ReplayMemory
from r2d2replaybuffer import r2d2_ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import torch
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from PIL import Image
import matplotlib.pyplot as plt
from models import convert_int_to_onehot
import moviepy.editor as mpy
from args import Args
import pickle
# env = gym.make("Battleship-v0")
# env = TigerEnv()
def run_exp(args):
    writer = SummaryWriter(args['logdir'])
    list_of_test_rewards_allseeds = []
    list_of_discount_test_rewards_allseeds = []
    list_of_nonzero_reward_count_allseeds = []
    list_of_mmd_est_allseeds = []
    list_of_reward_loss_allseeds = []
    if args['env_name'] == 'Tiger':
        env = TigerEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'RockSampling':
        env = RockSamplingEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    elif args['env_name'] == 'Cheesemaze':
        env = CheeseMazeEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'Voicemail':
        env = VoicemailEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'DroneSurveillance':
        env = DroneSurveillanceEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    # elif args['env_name'][:14] == 'CustomMiniGrid':
    #     if args['env_name'][15:18] == 'Key':
    #         env = CustomKeyCorridor(max_steps=args['max_env_steps'] , room_size=int(args['env_name'][18:]))
    #     elif args['env_name'][15:20] == 'Multi':
    #         env = CustomMultiRoomEnv(max_steps=args['max_env_steps'] , minNumRooms=int(args['env_name'][20:]) , maxNumRooms=int(args['env_name'][20:]))
    #     max_env_steps = args['max_env_steps']
    elif args['env_name'][:8] == 'MiniGrid':
        env = gym.make(args['env_name'])
        test_env = gym.make(args['env_name'])
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
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
        memory = Rec_ReplayMemory(args['replay_size'], state_size, env.action_space.n, max_env_steps , args)
        print('vanilla')
    if args['replay_type'] == 'r2d2':
        memory = r2d2_ReplayMemory(args['replay_size'], state_size, env.action_space.n, max_env_steps, args)
        print('r2d2')
    for seed in range(args['num_seeds']):
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        test_env.seed(seed)
        list_of_test_rewards = []
        list_of_discount_test_rewards = []
        list_of_nonzero_reward_count = []
        list_of_mmd_est = []
        list_of_reward_loss = []
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
        avg_mmd_est = 0
        avg_obs_norm = 0
        avg_reward = 0
        avg_episode_steps = 0
        avg_q_loss = 0
        avg_p_loss = 0
        avg_model_loss = 0
        avg_reward_loss = 0
        model_updates = 0
        k_episode = 0
        num_nonzero_rewards = 0
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
            # evalll = False
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
                    if total_numsteps >= args['start_updates_to_p_q_after']:
                        EPS_up = True
                    action, hidden_p = sac.select_action(state, action, reward, hidden_p, start , EPS_up , evaluate = False)  # Sample action from policy
                    # print(action)
                if start == True:
                    start = False

                next_state, reward, done, _ = env.step(action)  # Step
                # if evalll is True:
                #     print(done, "done flag!!")
                ls_actions.append(action)
                ls_rewards.append(reward)
                # if args['model_alg'] == 'AIS' and args['replay_type'] == 'vanilla':
                #     if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_model_after'] and total_numsteps % args['update_model_every_n_steps'] == 0:
                #         model_loss = sac.update_model(memory, args['batch_size'], args['model_updates_per_step'])
                #         avg_model_loss += model_loss
                #         model_updates += 1
                if reward != 0:
                    num_nonzero_rewards += 1
                if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_p_q_after'] and total_numsteps % args['rl_update_every_n_steps'] == 0 and not args['only_train_model']:
                    critic_loss, policy_loss , model_loss ,reward_loss , mmd_est  = sac.update_parameters(memory, args['batch_size'], args['p_q_updates_per_step'])
                    updates += 1
                    avg_mmd_est += mmd_est
                    # avg_obs_norm += obs_norm
                    avg_p_loss += policy_loss
                    avg_q_loss += critic_loss
                    # if args['replay_type'] == 'r2d2':
                    avg_model_loss += model_loss
                    avg_reward_loss += reward_loss
                    model_updates += 1
                    # avg_p_loss += 0
                    # avg_q_loss += 0
                episode_steps += 1
                total_numsteps += 1
                k_steps += 1
                episode_reward = reward + episode_reward

                state = next_state
                if total_numsteps % args['logging_freq'] == args['logging_freq']-1:
                    # print(episode_steps)
                    # evalll = True
                    avg_reward , avg_discount_adj_reward = log_test_and_save(test_env, sac, writer, args, avg_reward, avg_q_loss, avg_p_loss, avg_mmd_est, avg_model_loss , avg_reward_loss, updates,
                                      model_updates, k_episode, i_episode, total_numsteps, avg_episode_steps, state_size , seed  , num_nonzero_rewards)
                    list_of_test_rewards.append(avg_reward)
                    list_of_discount_test_rewards.append(avg_discount_adj_reward)
                    list_of_nonzero_reward_count.append(num_nonzero_rewards/args['logging_freq'])

                    list_of_mmd_est.append(avg_mmd_est/model_updates)
                    list_of_reward_loss.append(avg_reward_loss/model_updates)

                    avg_mmd_est = 0
                    num_nonzero_rewards = 0
                    avg_reward = 0
                    avg_episode_steps = 0
                    avg_q_loss = 0
                    avg_p_loss = 0
                    avg_model_loss = 0
                    avg_reward_loss = 0
                    model_updates = 0
                    updates = 0
                    k_episode = 0
                if episode_steps >= max_env_steps:
                    break
            # print('gg')
            # if evalll is True:
            #     evalll = False
            #     print(episode_steps)
            memory.push(ls_states, ls_actions, ls_rewards , ls_hiddens , sac)  # Append transition to memory
            k_episode += 1
            # print(ls_states,ls_actions,ls_rewards)
            ls_running_rewards.append(episode_reward)
            avg_reward = avg_reward + episode_reward
            avg_episode_steps = episode_steps + avg_episode_steps

            if total_numsteps > args['num_steps']:
                break

        # print('making_videoo')
        if args['env_name'][:8] == 'MiniGrid':
            make_video(env,sac,args,seed , state_size)
        # memory.save_buffer(args['logdir'],seed)
        list_of_test_rewards_allseeds.append(list_of_test_rewards)
        list_of_discount_test_rewards_allseeds.append(list_of_discount_test_rewards)
        list_of_nonzero_reward_count_allseeds.append(list_of_nonzero_reward_count)
        list_of_mmd_est_allseeds.append(list_of_mmd_est)
        list_of_reward_loss_allseeds.append(list_of_reward_loss)



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
    arr_count_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_mmd_est = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_reward_loss = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    for i in range(args['num_seeds']):
        arr_r[i,:] = np.array(list_of_test_rewards_allseeds[i])
        arr_d_r[i,:] = np.array(list_of_discount_test_rewards_allseeds[i])
        arr_count_r[i,:] = np.array(list_of_nonzero_reward_count_allseeds[i])
        arr_mmd_est[i,:] = np.array(list_of_mmd_est_allseeds[i])
        arr_reward_loss[i,:] = np.array(list_of_reward_loss_allseeds[i])

    np.save(args['logdir']+'/'+args['exp_name']+'_arr_r',arr_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_d_r', arr_d_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_freq_nonzero_rewards', arr_count_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_mmd_est', arr_mmd_est)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_reward_loss', arr_reward_loss)
    # print(arr_r)
    # print(arr_d_r)


def make_video(env , sac , args , seed , state_size):
    num_episodes = 10
    l = 0
    env.reset()
    render = env.render()
    # full_img = full_img.reshape(1, full_img.shape[0], full_img.shape[1], full_img.shape[2])
    max_size = max(args['max_env_steps']//4 , 200)
    full_img = np.zeros([num_episodes * max_size +1 , render.shape[0], render.shape[1], render.shape[2]], dtype=np.uint8)
    # print(full_img.shape, full_img.dtype)
    #
    # assert False

    for ep_i in range(num_episodes):
        # print(ep_i)
        start = True
        hidden_p = None
        action = 0
        reward = 0
        state = env.reset()
        done = False
        steps = 0
        while not done:
            img = env.render()
            full_img[l,:,:,:] = img
            if args['env_name'][:8] == 'MiniGrid':
                state = state['image']
                state = sac.get_encoded_obs(state)
            else:
                state = convert_int_to_onehot(state, state_size)
            l += 1
            steps += 1
            action, hidden_p = sac.select_action(state, action, reward, hidden_p, start, False, evaluate=True)
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if start == True:
                start = False
            if steps >= args['max_env_steps']//4 and steps >= 200:
                break
    # print(l)
    # print(full_img.shape)
    # imgs = [Image.fromarray(img) for img in full_img]
    path = os.path.join(args['logdir'], 'Seed_' + str(seed) + '_video.mp4')
    # imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    clip = mpy.ImageSequenceClip(list(full_img[:l+1,:,:,:]), fps=5)
    clip.write_videofile(path)







def log_test_and_save(env , sac , writer , args , avg_reward , avg_q_loss , avg_p_loss , avg_mmd_est , avg_model_loss , avg_reward_loss , updates , model_updates , k_episode , i_episode , total_numsteps , avg_episode_steps , state_size , seed , num_nonzero_rewards ):
    if total_numsteps % int(args['num_steps']/10) == int(args['num_steps']/10)  - 1:
        sac.save_model(args['logdir'],seed , total_numsteps)
    else:
        sac.save_model(args['logdir'], seed, -1)
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
                    # print('max steps reached!!!')
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
    # if args['model_alg'] == 'AIS':
    if True:
        if updates == 0:
            avgql = 0
            avgpl = 0
        else:
            avgql = avg_q_loss / (updates)
            avgpl = avg_p_loss / (updates)
        if model_updates == 0:
            avgml = 0
            avgr = 0
            avg_mmd = 0
        else:
            avgml = avg_model_loss / (model_updates)
            avgr = avg_reward_loss / (model_updates)
            avg_mmd = avg_mmd_est / model_updates
        print(
            "Seed: {}, Episode: {}, Total_num_steps: {},  episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}, avg_q_loss: {}, avg_p_loss: {} , avg_model_loss: {} , avg_reward_loss: {}, avg_mmd_est: {}".format(
                seed,i_episode, total_numsteps, avg_episode_steps / k_episode, avg_running_reward, avg_reward,
                avg_discount_adj_reward, avgql, avgpl, avgml , avgr,avg_mmd))
    # if args['model_alg'] == 'None':
    #     if updates == 0:
    #         avgql = 0
    #         avgpl = 0
    #     else:
    #         avgql = avg_q_loss / updates
    #         avgpl = avg_p_loss / updates
    #     print(
    #         "Seed: {}, Episode: {}, Total_num_steps: {}, episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}, avg_q_loss: {}, avg_p_loss: {}".format(
    #             seed,i_episode, total_numsteps,
    #             avg_episode_steps / k_episode, avg_running_reward,
    #             avg_reward, avg_discount_adj_reward,
    #             avgql, avgpl))

    writer.add_scalar('Seed'+str(seed)+'Evaluation reward', avg_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Discount adjusted Evaluation reward', avg_discount_adj_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Training reward', avg_running_reward, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Average Steps for each episode', avg_episode_steps, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Policy', avgpl, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Value', avgql, total_numsteps)
    # if args['model_alg'] == 'AIS':
    writer.add_scalar('Seed'+str(seed)+'Loss/Model', avgml, total_numsteps)
    writer.add_scalar('Seed'+str(seed)+'Loss/Reward', avgr, total_numsteps)
    writer.add_scalar('Seed' + str(seed) + 'Loss/MMD_estimate', avg_mmd, total_numsteps)
    print("----------------------------------------")
    writer.flush()

    return avg_reward , avg_discount_adj_reward
