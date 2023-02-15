import os
from khrylib.utils import load_yaml
from typing import Text, Dict


class Config:

    def __init__(self, cfg: Text, global_seed: int, tmp: bool, root_dir: Text,
                 agent: Text = 'rl-sgnn', cfg_dict: Dict = None):
        self.id = cfg
        self.seed = global_seed
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            file_path = 'urban_planning/cfg/**/{}.yaml'.format(self.id)
            cfg = load_yaml(file_path)
        # create dirs
        self.root_dir = '/tmp/urban_planning' if tmp else root_dir

        self.cfg_dir = os.path.join(self.root_dir, self.id, str(self.seed))
        self.model_dir = os.path.join(self.cfg_dir, 'models')
        self.log_dir = os.path.join(self.cfg_dir, 'log')
        self.tb_dir = os.path.join(self.cfg_dir, 'tb')
        self.plan_dir = os.path.join(self.cfg_dir, 'plan')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.plan_dir, exist_ok=True)

        self.agent = agent

        # env
        self.objectives_plan = cfg.get('objectives_plan', '')
        self.init_plan = cfg.get('init_plan', '')
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())

        # agent config
        self.agent_specs = cfg.get('agent_specs', dict())

        # training config
        self.skip_land_use = cfg.get('skip_land_use', False)
        self.skip_road = cfg.get('skip_road', False)
        self.road_ratio = cfg.get('road_ratio', 0.7)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.state_encoder_specs = cfg.get('state_encoder_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.value_specs = cfg.get('value_specs', dict())
        self.lr = cfg.get('lr', 4e-4)
        self.weightdecay = cfg.get('weightdecay', 0.0)
        self.eps = cfg.get('eps', 1e-5)
        self.value_pred_coef = cfg.get('value_pred_coef', 0.5)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.max_num_iterations = cfg.get('max_num_iterations', 1000)
        self.num_episodes_per_iteration = cfg.get('num_episodes_per_iteration', 1000)
        self.max_sequence_length = cfg.get('max_sequence_length', 100)
        self.original_max_sequence_length = cfg.get('max_sequence_length', 100)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 4)
        self.mini_batch_size = cfg.get('mini_batch_size', 1024)
        self.save_model_interval = cfg.get('save_model_interval', 10)

    def train(self) -> None:
        """Train land use only"""
        self.skip_land_use = False
        self.skip_road = True
        self.max_sequence_length = self.original_max_sequence_length // 2

    def finetune(self) -> None:
        """Change to road network only"""
        self.skip_land_use = True
        self.skip_road = False
        self.max_sequence_length = self.original_max_sequence_length // 2

    def log(self, logger, tb_logger):
        """Log cfg to logger and tensorboard."""
        logger.info(f'id: {self.id}')
        logger.info(f'seed: {self.seed}')
        logger.info(f'objectives_plan: {self.objectives_plan}')
        logger.info(f'init_plan: {self.init_plan}')
        logger.info(f'env_specs: {self.env_specs}')
        logger.info(f'reward_specs: {self.reward_specs}')
        logger.info(f'obs_specs: {self.obs_specs}')
        logger.info(f'agent_specs: {self.agent_specs}')
        logger.info(f'skip_land_use: {self.skip_land_use}')
        logger.info(f'skip_road: {self.skip_road}')
        logger.info(f'road_ratio: {self.road_ratio}')
        logger.info(f'gamma: {self.gamma}')
        logger.info(f'tau: {self.tau}')
        logger.info(f'state_encoder_specs: {self.state_encoder_specs}')
        logger.info(f'policy_specs: {self.policy_specs}')
        logger.info(f'value_specs: {self.value_specs}')
        logger.info(f'lr: {self.lr}')
        logger.info(f'weightdecay: {self.weightdecay}')
        logger.info(f'eps: {self.eps}')
        logger.info(f'value_pred_coef: {self.value_pred_coef}')
        logger.info(f'entropy_coef: {self.entropy_coef}')
        logger.info(f'clip_epsilon: {self.clip_epsilon}')
        logger.info(f'max_num_iterations: {self.max_num_iterations}')
        logger.info(f'num_episodes_per_iteration: {self.num_episodes_per_iteration}')
        logger.info(f'max_sequence_length: {self.max_sequence_length}')
        logger.info(f'num_optim_epoch: {self.num_optim_epoch}')
        logger.info(f'mini_batch_size: {self.mini_batch_size}')
        logger.info(f'save_model_interval: {self.save_model_interval}')

        if tb_logger is not None:
            tb_logger.add_hparams(
                hparam_dict={
                    'id': self.id,
                    'seed': self.seed,
                    'objectives_plan': self.objectives_plan,
                    'init_plan': self.init_plan,
                    'env_specs': str(self.env_specs),
                    'reward_specs': str(self.reward_specs),
                    'obs_specs': str(self.obs_specs),
                    'agent_specs': str(self.agent_specs),
                    'skip_land_use': self.skip_land_use,
                    'skip_road': self.skip_road,
                    'road_ratio': self.road_ratio,
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'state_encoder_specs': str(self.state_encoder_specs),
                    'policy_specs': str(self.policy_specs),
                    'value_specs': str(self.value_specs),
                    'lr': self.lr,
                    'weightdecay': self.weightdecay,
                    'eps': self.eps,
                    'value_pred_coef': self.value_pred_coef,
                    'entropy_coef': self.entropy_coef,
                    'clip_epsilon': self.clip_epsilon,
                    'max_num_iterations': self.max_num_iterations,
                    'num_episodes_per_iteration': self.num_episodes_per_iteration,
                    'max_sequence_length': self.max_sequence_length,
                    'num_optim_epoch': self.num_optim_epoch,
                    'mini_batch_size': self.mini_batch_size,
                    'save_model_interval': self.save_model_interval},
                metric_dict={'hparam/placeholder': 0.0})


