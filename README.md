# POMDP_RL

This repository contains code for solving Partially observable environments using off-policy model-free reinforcement learning algorithms such as Q-learning and Soft Actor-Critic.




Three classes of experiments are presented (with their `gym-environment-names`)
* **Low-dimensional environments**
  - Tiger: `Tiger-v0`
  - Voicemail: `Voicemail-v0`
  - Cheese Maze: `CheeseMaze-v0`
 * **Moderate-dimensional environments**
   - Rock Sample: `RockSampling-v0`
   - Drone Surveillance: `DroneSurveillance-v0`
 * **High-dimensional environments**

    Various grid-world models from [gym-minigrid](https://github.com/maximecb/gym-minigrid) (used in the BabyAI platform) including
    - Simple crossing: `MiniGrid-SimpleCrossingS9N1-v0`, `MiniGrid-SimpleCrossingS9N2-v0`, `MiniGrid-SimpleCrossingS9N3-v0`, `MiniGrid-SimpleCrossingS11N5-v0`
    - Lava crossing: `MiniGrid-LavaCrossingS9N1-v0`, `MiniGrid-LavaCrossingS9N2-v0`
    - Key corridor: `MiniGrid-KeyCorridorS3R1-v0`, `MiniGrid-KeyCorridorS3R2-v0`, `MiniGrid-KeyCorridorS3R3-v0`
    - Obstructed maze: `MiniGrid-ObstructedMaze-1Dl-v0`, `MiniGrid-ObstructedMaze-1Dlh-v0`
    - Misc `MiniGrid-Empty-8x8-v0`, `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-FourRooms-v0`

