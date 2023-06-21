import gym
import gym_minigrid
from gym_minigrid.wrappers import *

# env_list = [ 'MiniGrid-SimpleCrossingS9N1-v0' , 'MiniGrid-SimpleCrossingS9N2-v0' , 'MiniGrid-SimpleCrossingS9N3-v0' , 'MiniGrid-SimpleCrossingS11N5-v0' ,  'MiniGrid-LavaCrossingS9N1-v0' , 'MiniGrid-LavaCrossingS9N2-v0' ]
# env_list = ['MiniGrid-LavaCrossingS9N3-v0' , 'MiniGrid-Unlock-v0' , 'MiniGrid-UnlockPickup-v0' , 'MiniGrid-DoorKey-5x5-v0' , 'MiniGrid-DoorKey-6x6-v0' , 'MiniGrid-DoorKey-8x8-v0' ]
# env_list = ['MiniGrid-LavaCrossingS11N5-v0' , 'MiniGrid-KeyCorridorS3R1-v0' , 'MiniGrid-KeyCorridorS3R2-v0' , 'MiniGrid-KeyCorridorS3R3-v0' , 'MiniGrid-ObstructedMaze-1Dl-v0']
# env_list = [ 'MiniGrid-ObstructedMaze-1Dlh-v0'  , 'MiniGrid-MultiRoom-N2-S4-v0' , 'MiniGrid-MultiRoom-N4-S5-v0' , 'MiniGrid-RedBlueDoors-6x6-v0' , 'MiniGrid-RedBlueDoors-8x8-v0']
env_list = [ 'MiniGrid-SimpleCrossingS9N1-v0' , 'MiniGrid-SimpleCrossingS9N2-v0' , 'MiniGrid-SimpleCrossingS9N3-v0' , 'MiniGrid-SimpleCrossingS11N5-v0' ,  'MiniGrid-LavaCrossingS9N1-v0' , 'MiniGrid-LavaCrossingS9N2-v0' , 'MiniGrid-LavaCrossingS9N3-v0' , 'MiniGrid-LavaCrossingS11N5-v0'   , 'MiniGrid-Unlock-v0' , 'MiniGrid-UnlockPickup-v0' , 'MiniGrid-DoorKey-5x5-v0' , 'MiniGrid-DoorKey-6x6-v0' , 'MiniGrid-DoorKey-8x8-v0' , 'MiniGrid-KeyCorridorS3R1-v0' , 'MiniGrid-KeyCorridorS3R2-v0' , 'MiniGrid-KeyCorridorS3R3-v0' , 'MiniGrid-ObstructedMaze-1Dl-v0' , 'MiniGrid-ObstructedMaze-1Dlh-v0'  , 'MiniGrid-MultiRoom-N2-S4-v0' , 'MiniGrid-MultiRoom-N4-S5-v0' , 'MiniGrid-RedBlueDoors-6x6-v0' , 'MiniGrid-RedBlueDoors-8x8-v0']

num_eps = 5000

for env_name in env_list:

    env = gym.make(env_name)
    avg_reward = 0
    for _ in range(num_eps):
        episode_reward = 0


        state = env.reset()
        done = False
        while not done:
            next_state, reward, done, _ = env.step(env.action_space.sample())
            episode_reward += reward

        avg_reward += episode_reward

    avg_reward /= num_eps




    print( env_name , "&  {:.4f} \\\\ ".format(avg_reward) )












