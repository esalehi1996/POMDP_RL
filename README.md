# POMDP_RL

This repository contains code for solving Partially observable environments using off-policy model-free reinforcement learning algorithms such as Q-learning and Soft Actor-Critic. Our approach shows a significant boost in performance over traditional recurrent Q-learning on a wide variety of partially observable benchmarks.

In this work we used two approaches for learning state representations for partially-observable environments:
* **Generative model learning**
  - In this approach a generative model is trained to predict next step observations and current step rewards. We are using maximum mean discrepancy loss for observation prediction and Mean-squared error loss for reward prediction. In this case the recurrent component is trained in the model learning phase and it is frozen in the reinforcement learning phase. 
* **No seperate model learning**
  - In this approach the model is not trained individually and the recurrent component is trained alongside the Q and policy network during the reinforcement learning phase.

There are two options for the reinforcement learning algorithm:
* **Soft Actor-Critic**
  - This uses the Discrete environment case of the Soft Actor-Critic algorithm. (https://arxiv.org/abs/1910.07207)
* **Q-learning**
  - In this case we are using a Q-learning implementation using recurrent nets, double Q networks and polyak averaging for target Q updates.



Three classes of experiments are presented (followed by their `env_name` input arguments used.)
* **Low-dimensional environments**
  - Tiger: `Tiger` 
  - Voicemail: `Voicemail`
  - Cheese Maze: `Cheesemaze`
 * **Moderate-dimensional environments**
   - Rock Sample: `RockSampling`
   - Drone Surveillance: `DroneSurveillance`
 * **High-dimensional environments**
   - Compatible with environments from the MiniGrid family. (Please note that pretrained observation encoders are used for compressing the observations. The encoder weights for many MiniGrid environments are included in the autoencoder folder.


This program accepts the following command line arguments:

| Option          | Description |
| --------------- | ----------- |
| `--rl_alg` |  Reinforcement learning algorithm used for training the Policy/Q networks. It can be 'SAC' for Soft Actor-Critic or 'QL' for Q-learning. |
| `--model_alg` |  This specifies whether model-learning is done or not. 'AIS' is for model learning. 'None' is for no model learning. |
| `--seed` | Random seed used. |
| `--exp_name` | Experiment name which is used as a prefix in directory name (The prefix is followed by time and date of the experiment in the directory name used)|
| `--env_name` | The environment name.  |
| `--batch_size` | The batch size used for model updates and reinforcement learning updates. This specifies the number of trajectories that are to be sampled from the buffer. Each trajectory can have a random length (up to the real experience trajectry length) |
| `--hidden_size` | The number of neurons in the hidden layers of the Q network and the policy network. |
| `--gamma` | Discount Factor |
| `--SAC_alpha` |  The entropy regularizer coefficient used in Soft Actor-Critic. |
| `--tau` |  The coefficient used for updating the target Q network with Polyak averaging. |
| `--AIS_state_size` | The size of the recurrent network used as state representation for the POMDP. |
| `--rl_lr` | The learning rate used for updating Policy networks and Q networks. (Only Q-network in the case of Q-learning) |
| `--AIS_lr` | The learning rate used for updating the generative model. |
| `--automatic_entropy_tuning` | If set to True, the regularizer coefficient in Soft-Actor Critic is also learned during the procedure. Otherwise, it stays constant. |
| `--num_steps` | Total number of training steps taken in the environment.|
| `--p_q_updates_per_step` | The number of updates done to Policy and Q networks at each step (Only to Q-network with Q-learning) |
| `--model_updates_per_step` |  The number of updates done to model networks at each step |
| `--random_actions_until` |  This hyperparameters forces the agent to take randomly sampled actions until a specific number of steps in the environment has passed. |
| `--start_updates_to_model_after` |  After the agent has taken these number of environments steps, updating the model begins. |
| `--start_updates_to_p_q_after` |  After the agent has taken these number of environments steps, updating the policy and q network begins. |
| `--target_update_interval` |  This specifies the environment step intervals after which the target Q network is updated using polyak averaging. |
| `--replay_size` |  This spcecifies the number of episodes that are stored in the replay memory. After the replay buffer is filled, new experience episodes will overwrite the least recenet episodes in the buffer. |
| `--cuda` |  If set to True, GPU will be used for both the model learning and the Policy/Q learning. |
| `--AIS_lambda` |  The hyperparameter which specifies how we are averaging between reward learning loss and next observation predictions loss in the model learning phase. |
| `--max_env_steps` |  The maximum number of steps that the agent is allowed to take in each episode. |
| `--logging_freq` |  The frequency in terms of environment steps in which we evaluate the agent, log the results and save the neural network parameters on disk. |
| `--load_from_path` |  If a path is given, neural network parameters are loaded from the given path. |
| `--update_model_every_n_steps` |  It specifies the frequency in terms of environment steps at which we update the model. (Only applicable in the model-learning case) |
| `--rl_update_every_n_steps` |  It specifies the frequency in terms of environment steps at which we do reinforcement learning updates. |
| `--only_train_model` |  If set to true, only model learning is called. This is used when we want to pretrain the model on random agent experience. |
| `--EPS_start` |  This is only applicable in the Q-learning case. This specifies the start value for the epsilon hyperparameter used in Q-learning for exploration. |
| `--EPS_decay` |  This is only applicable in the Q-learning case. This specifies decay rate for the epsilon hyperparameter used in Q-learning for exploration |
| `--EPS_end` | This is only applicable in the Q-learning case. This specifies the end value for the epsilon hyperparameter used in Q-learning for exploration |
