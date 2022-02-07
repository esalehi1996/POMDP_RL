class Args():
    def __init__(self):
        self.seed = 233
        self.batch_size = 64
        self.hidden_size = 128
        self.gamma = 0.99
        self.tau = 0.01
        self.alpha = 0.1
        self.AIS_state_size = 128
        self.lr = 5e-4
        self.automatic_entropy_tuning = False
        self.num_steps = 100000
        self.p_q_updates_per_step = 1
        self.model_updates_per_step = 1
        self.random_actions_until = 0
        self.start_updates_to_model_after = 0
        self.start_updates_to_p_q_after = 0
        self.target_update_interval = 1
        self.replay_size = 50000
        self.cuda = False
        self.AIS_lr = 1e-3
        self.Lambda = 0.99
        self.alg = 'SAC+AIS'



args = Args()