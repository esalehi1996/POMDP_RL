import itertools
from env.Tiger import TigerEnv
from env.RockSampling import RockSamplingEnv
from env.DroneSurveillance import DroneSurveillanceEnv
from env.cheesemaze import CheeseMazeEnv
from env.voicemail import VoicemailEnv
from SAC import SAC
from Replaybuffer import Rec_ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from args import Args
import pickle
# env = gym.make("Battleship-v0")
# env = TigerEnv()
def run_exp(args):
    writer = SummaryWriter(args['logdir'])
    if args['env_name'] == 'Tiger':
        env = TigerEnv()
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'RockSampling':
        env = RockSamplingEnv()
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    elif args['env_name'] == 'Cheesemaze':
        max_env_steps = 100
    elif args['env_name'] == 'Voicemail':
        max_env_steps = 100
    elif args['env_name'] == 'DroneSurveillance':
        max_env_steps = 200
    total_numsteps = 0
    k_steps = 0
    updates = 1
    print(env.action_space, env.observation_space)
    sac = SAC(env.observation_space.n, env.action_space, args)

    if args['load_from_path'] != 'None':
        print('---------------  Loading Model from path:' , args['load_from_path'] , '-----------------')
        sac.load_model(args['load_from_path'])

    memory = Rec_ReplayMemory(args['replay_size'], env.observation_space.n, env.action_space.n, 1000, args['seed'])

    ls_running_rewards = []
    avg_reward = 0
    avg_episode_steps = 0
    avg_q_loss = 0
    avg_p_loss = 0
    avg_model_loss = 0
    model_updates = 0
    k_episode = 0
    for i_episode in itertools.count(1):
        k_episode += 1
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        ls_states = []
        ls_actions = []
        ls_rewards = []
        start = True
        hidden_p = None
        action = 0
        reward = 0
        # ls_rewards.append(0.)
        # ls_actions.append(0.)

        while not done:
            ls_states.append(state)
            if i_episode <= args['random_actions_until'] or args['only_train_model']:
                # print('random action mode')
                # action = 0
                # # action = env.action_space.sample()  # Sample random action
                # if episode_steps > 10:
                action = env.action_space.sample()
            else:
                action, hidden_p = sac.select_action(state, action, reward, hidden_p, start , False)  # Sample action from policy
                # print(action)
            # print(action)
            if start == True:
                start = False

            next_state, reward, done, _ = env.step(action)  # Step
            ls_actions.append(action)
            ls_rewards.append(reward)
            if args['model_alg'] == 'AIS':
                if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_model_after'] and total_numsteps % args['update_model_every_n_steps'] == 0:
                    model_loss = sac.update_model(memory, args['batch_size'], args['model_updates_per_step'])
                    avg_model_loss += model_loss
                    model_updates += 1
            if len(memory) > args['batch_size'] and i_episode >= args['start_updates_to_p_q_after'] and total_numsteps % args['rl_update_every_n_steps'] == 0 and not args['only_train_model']:
                critic_loss, policy_loss = sac.update_parameters(memory, args['batch_size'], args['p_q_updates_per_step'])
                updates += 1
                avg_p_loss += policy_loss
                avg_q_loss += critic_loss
            episode_steps += 1
            total_numsteps += 1
            k_steps += 1
            episode_reward = reward + episode_reward

            state = next_state
            if episode_steps >= max_env_steps:
                break
        # print('gg')
        memory.push(ls_states, ls_actions, ls_rewards)  # Append transition to memory
        # print(ls_states,ls_actions,ls_rewards)
        ls_running_rewards.append(episode_reward)
        avg_reward = avg_reward + episode_reward
        avg_episode_steps = episode_steps + avg_episode_steps
        # running_reward =  0.9 * running_reward + 0.1 * round(episode_reward, 2)
        # if args.alg == 'SAC+AIS':
        #     if len(memory) > args.batch_size and i_episode >= args.start_updates_to_model_after:
        #         model_loss = sac.update_model(memory, args.batch_size, args.model_updates_per_step)
        #         avg_model_loss += model_loss
        #         model_updates += 1
        # if len(memory) > args.batch_size and i_episode >= args.start_updates_to_p_q_after:
        #     critic_loss, policy_loss = sac.update_parameters(memory, args.batch_size, args.p_q_updates_per_step)
        #     updates += 1
        #     avg_p_loss += policy_loss
        #     avg_q_loss += critic_loss
        if total_numsteps > args['num_steps']:
            break

        # writer.add_scalar('reward/train', episode_reward, i_episode)
        # if k_episode >= 50 :
        #     if args['model_alg'] == 'AIS' and args['rl_alg'] == 'SAC':
        #         if updates == 0:
        #             avgql = 0
        #             avgpl = 0
        #         else:
        #             avgql = avg_q_loss / (updates)
        #             avgpl = avg_p_loss / (updates)
        #         if model_updates == 0:
        #             avgml = 0
        #         else:
        #             avgml = avg_model_loss / (model_updates)
        #         print(
        #             "Episode: {}, episode steps: {}, avg_reward: {}, avg_q_loss: {}, avg_p_loss: {} , avg_model_loss: {}".format(
        #                 i_episode, avg_episode_steps / k_episode, avg_reward / k_episode, avgql, avgpl, avgml))
        #     if args['model_alg'] == 'none' and args['rl_alg'] == 'SAC':
        #         if updates == 0:
        #             avgql = 0
        #             avgpl = 0
        #         else:
        #             avgql = avg_q_loss / updates
        #             avgpl = avg_p_loss / updates
        #         print("Episode: {}, episode steps: {}, avg_reward: {}, avg_q_loss: {}, avg_p_loss: {}".format(i_episode,
        #                                                                                                       avg_episode_steps / k_episode,
        #                                                                                                       avg_reward / k_episode,
        #                                                                                                       avgql, avgpl))
        #     # print("Episode: {}, episode steps: {}, avg_reward: {}".format(i_episode,avg_episode_steps/k_episode, avg_reward/k_episode))
        #     k_episode = 0
        #     avg_reward = 0
        #     avg_episode_steps = 0
        #     avg_p_loss = 0
        #     avg_q_loss = 0
        #     model_updates = 0
        #     avg_model_loss = 0
        #     updates = 0



        if k_steps >= args['logging_freq']:
            sac.save_model(args['logdir'])
            avg_running_reward = avg_reward/k_episode
            k_steps = 0
            avg_reward = 0.
            episodes = 10

            if not args['only_train_model']:
                for _  in range(episodes):
                    start = True
                    hidden_p = None
                    action = 0
                    reward = 0
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    steps = 0
                    while not done:
                        steps += 1
                        action, hidden_p = sac.select_action(state, action, reward, hidden_p, start , evaluate=False)
                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward

                        state = next_state
                        if start == True:
                            start = False
                        if steps >= max_env_steps:
                            break
                    avg_reward += episode_reward
            avg_reward /= episodes

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
                print("Episode: {}, Total_num_steps: {},  episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_q_loss: {}, avg_p_loss: {} , avg_model_loss: {}".format(
                        i_episode,total_numsteps , avg_episode_steps / k_episode, avg_running_reward, avg_reward, avgql, avgpl, avgml))
            if args['model_alg'] == 'None':
                if updates == 0:
                    avgql = 0
                    avgpl = 0
                else:
                    avgql = avg_q_loss / updates
                    avgpl = avg_p_loss / updates
                print("Episode: {}, Total_num_steps: {}, episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_q_loss: {}, avg_p_loss: {}".format(i_episode, total_numsteps ,
                                                                                                              avg_episode_steps / k_episode, avg_running_reward ,
                                                                                                              avg_reward,
                                                                                                              avgql, avgpl))


            writer.add_scalar('Evaluation reward', avg_reward , total_numsteps)
            writer.add_scalar('Training reward', avg_running_reward, total_numsteps)
            writer.add_scalar('Average Steps for each episode', avg_episode_steps , total_numsteps)
            writer.add_scalar('Loss/Policy', avgpl, total_numsteps)
            writer.add_scalar('Loss/Value', avgql, total_numsteps)
            if args['model_alg'] == 'AIS':
                writer.add_scalar('Loss/Model', avgml, total_numsteps)
            k_episode = 0
            avg_reward = 0
            avg_episode_steps = 0
            avg_p_loss = 0
            avg_q_loss = 0
            model_updates = 0
            avg_model_loss = 0
            updates = 0
            print("----------------------------------------")
            writer.flush()


    env.close()
    writer.close()