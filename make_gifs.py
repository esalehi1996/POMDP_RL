import os
import json
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from PIL import Image
# Get the list of all files and directories
# in the root directory
from SAC import SAC
from models import convert_int_to_onehot


def make_video(env , sac , args , seed , state_size , path):
    num_episodes = 10
    l = 0
    env.reset()
    full_img = env.render('rgb_array')
    full_img = full_img.reshape(1, full_img.shape[0], full_img.shape[1], full_img.shape[2])

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
            full_img = np.concatenate([full_img, img.reshape(1, img.shape[0], img.shape[1], img.shape[2])], 0)
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
            if steps >= args['max_env_steps']:
                break
    # print(l)
    # print(full_img.shape)
    imgs = [Image.fromarray(img) for img in full_img]
    vpath = os.path.join(path, 'Seed_' + str(seed) + '_video.gif')
    imgs[0].save(vpath, save_all=True, append_images=imgs[1:], duration=200, loop=0)




cwd = os.getcwd()

data_dir = cwd + '/data/'
dir_list = os.listdir(data_dir)

# print the list
for file in dir_list:
    d = os.path.join(data_dir, file)
    if os.path.isdir(d) and 'MiniGrid' in d:
        os.chdir(d)
        if 'config.json' not in os.listdir(d):
            continue
        f = open(os.path.join(d,'config.json'))
        args = json.load(f)
        # print(os.getcwd())
        f.close()

        ls_model_files = []
        num_seeds = 0
        for f_name in os.listdir(d):
            if 'models.pt' in f_name:
                ls_model_files.append(f_name)
                num_seeds += 1

        if num_seeds == 0:
            continue

        if args['env_name'][:8] != 'MiniGrid':
            continue

        print('making video for :' , d , 'num seeds:' ,num_seeds)
        print(args)
        print(args['QL_VAE_disable'])

        # print(ls_model_files,num_seeds)
        env = gym.make(args['env_name'])

        for i in range(num_seeds):
            print('seed number :',i)
            sac = SAC(env, args)
            state_size = sac.get_obs_dim()
            print(os.path.join(d,ls_model_files[i]))
            sac.load_model(os.path.join(d,ls_model_files[i]))
            make_video(env , sac , args , i , state_size , d)






