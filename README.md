# POMDP_RL

This repository contains code for solving Partially observable environments using off-policy model-free reinforcement learning algorithms such as Q-learning and Soft Actor-Critic. Our approach shows a significant boost in performance over traditional recurrent Q-learning on a wide variety of partially observable benchmarks.

In this work we used two approaches for learning state representations for partially-observable environments:
* **Generative model learning (RQL-AIS) **
  - In this approach a generative model is trained to predict next step observations and current step rewards. We are using maximum mean discrepancy loss for observation prediction and Mean-squared error loss for reward prediction. In this case the recurrent component is trained in the model learning phase and it is frozen in the reinforcement learning phase. 
* **No seperate model learning (ND-R2D@2) **
  - In this approach the model is not trained individually and the recurrent component is trained alongside the Q and policy network during the reinforcement learning phase.

There are two options for the reinforcement learning algorithm:
* **Soft Actor-Critic**
  - This uses the Discrete environment case of the Soft Actor-Critic algorithm. (https://arxiv.org/abs/1910.07207)
* **Q-learning**
  - In this case we are using a Q-learning implementation using recurrent nets, double Q networks and polyak averaging for target Q updates.

The non-distributed variant of R2D2 is used as the RL algorithm. (https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning)

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
| `--rl_alg` |  Reinforcement learning algorithm used for training the Policy/Q networks. Only recurrent Q-learning is implemented at present. Use 'QL' for Q-learning. |
| `--model_alg` |  This specifies whether model-learning is done or not. 'AIS' is for model learning (RQL-AIS). 'None' is for no model learning (ND-R2D2). |
| `--num_seeds` | Total number of times the experiments is run. Experiment number i is run on seed number i |
| `--exp_name` | Experiment name which is used as a prefix in directory name (The prefix is followed by time and date of the experiment in the directory name used)|
| `--env_name` | The environment name.  |
| `--batch_size` | The batch size used for AIS updates and reinforcement learning updates. This specifies the number of samples drawn from the buffer. Each trajectory has a fixed length (learning_obs_len) |
| `--hidden_size` | The number of neurons in the hidden layers of the Q network. |
| `--gamma` | Discount Factor |
| `--AIS_state_size` | The size of the hidden vector and the output of the LSTM used as state representation for the POMDP. |
| `--rl_lr` | The learning rate used for updating Q networks and the LSTMs (for ND-R2D2) and only the Q-network (for RQL-AIS) |
| `--AIS_lr` | The learning rate used for updating the AIS components (for RQL-AIS). |
| `--num_steps` | Total number of training steps taken in the environment.|
| `--target_update_interval` |  This specifies the environment step intervals after which the target Q network (and target LSTM in case of ND-R2D2) is updated. |
| `--replay_size` |  This spcecifies the number of episodes that are stored in the replay memory. After the replay buffer is filled, new experience episodes will overwrite the least recenet episodes in the buffer. |
| `--cuda` |  If set to True, GPU will be used. |
| `--AIS_lambda` |  The hyperparameter which specifies how we are averaging between reward learning loss and next observation predictions loss in the AIS learning phase. |
| `--max_env_steps` |  The maximum number of steps that the agent is allowed to take in each episode. |
| `--logging_freq` |  The frequency in terms of environment steps in which we evaluate the agent, log the results and save the neural network parameters on disk. |
| `--rl_update_every_n_steps` |  It specifies the frequency in terms of environment steps at which we do reinforcement learning updates. |
| `--EPS_start` |  This specifies the start value for the epsilon hyperparameter used in Q-learning for exploration. |
| `--EPS_decay` |  This specifies decay rate for the epsilon hyperparameter used in Q-learning for exploration |
| `--EPS_end` | This specifies the end value for the epsilon hyperparameter used in Q-learning for exploration |
| `--burn_in_len` | Length of the preceding Burn-In Sequence saved with each sample in the R2D2 buffer. |
| `--learning_obs_len` | Sequence length of R2D2 samples. |
| `--forward_len` | The multi-step Q-learning length. |
| `--test_epsilon` | Epsilon value used at test time. Default is 0. |
| `--QL_VAE_disable` | Specifies whether the pretrained autoencoders should be disabled for ND-R2D2 or not. Default is false meaning the pretrained autoencoders are used for ND-R2D2 experiments.  |
| `--PER` | Specifies whether prioritized experience replay should be used or not. Default is False.  |



