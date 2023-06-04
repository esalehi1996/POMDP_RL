import os
import time

import argparse
import json
from run import *




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds' ,type=int, default=5)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--SAC_alpha', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--AIS_state_size', type=int, default=16)
    parser.add_argument('--rl_lr', type=float, default=5e-4)
    parser.add_argument('--AIS_lr', type=float, default=1e-3)
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--p_q_updates_per_step', type=int, default=1)
    parser.add_argument('--model_updates_per_step', type=int, default=1)

    parser.add_argument('--random_actions_until',type=int, default=0)
    parser.add_argument('--start_updates_to_model_after', type=int, default=0)
    parser.add_argument('--start_updates_to_p_q_after', type=int, default=0)
    parser.add_argument('--target_update_interval', type=int, default=100)
    parser.add_argument('--replay_size', type=int, default=50000)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--AIS_lambda', type=float, default=0.99)
    parser.add_argument('--rl_alg', type=str, default='QL')
    parser.add_argument('--model_alg', type=str, default='None')
    parser.add_argument('--max_env_steps', type=int, default=-1)
    parser.add_argument('--logging_freq' , type=int , default=1000)
    parser.add_argument('--load_from_path', type=str, default='None')
    parser.add_argument('--update_model_every_n_steps' , type=int , default=1)
    parser.add_argument('--rl_update_every_n_steps', type=int, default=5)
    parser.add_argument('--only_train_model' , type=bool, default=False)
    parser.add_argument('--EPS_start', type=float, default=1.0)
    parser.add_argument('--EPS_decay', type=int, default=50000)
    parser.add_argument('--EPS_end', type=float, default=0.05)
    parser.add_argument('--burn_in_len', type=int, default=50)
    parser.add_argument('--learning_obs_len', type=int, default=10)
    parser.add_argument('--forward_len', type=int, default=3)
    parser.add_argument('--replay_type', type=str, default='r2d2')
    parser.add_argument('--AIS_loss', type=str, default='MMD')
    parser.add_argument('--test_epsilon', type=float, default=0.0)
    parser.add_argument('--QL_VAE_disable' , type=bool , default=False )
    parser.add_argument('--noisy_net' , type=bool , default=False)
    parser.add_argument('--PER', type=bool, default=False)
    parser.add_argument('--PER_type', type=str, default='TD')
    parser.add_argument('--Model_PER_exponent', type=float, default=1.20)
    parser.add_argument('--PER_e', type=float, default=1e-7)
    parser.add_argument('--TD_loss', type=str, default='mse')
    parser.add_argument('--EPS_decay_type', type=str, default='exponential')
    parser.add_argument('--PER_eta', type=float, default=0.9)
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    print(args)
    print(params)

    print(args.env_name)

    # HARDCODE EPISODE LENGTHS FOR THE ENVS USED IN THIS MB ASSIGNMENT
    # if params['env_name']=='reacher-ift6163-v0':
    #     params['ep_len']=200
    # if params['env_name']=='cheetah-ift6163-v0':
    #     params['ep_len']=500
    # if params['env_name']=='obstacles-ift6163-v0':
    #     params['ep_len']=100
    #
    # ##################################
    # ### CREATE DIRECTORY FOR LOGGING
    # ##################################
    #
    # logdir_prefix = 'hw4_'  # keep for autograder
    #
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    print(data_path)
    #
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    #
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    #
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    print(params)
    config_path = os.path.join(logdir,'config.json')
    print(config_path)
    with open(config_path, 'w') as fp:
        json.dump(params, fp , indent=4)
    #
    # ###################
    # ### RUN TRAINING
    # ###################
    #
    # trainer = MB_Trainer(params)
    # trainer.run_training_loop()
    run_exp(params)


if __name__ == "__main__":
    main()
