import math
import itertools
from khrylib.utils.stats_logger import StatsLogger


class LoggerRL:

    def __init__(self, init_stats_logger=True):
        self.num_steps = 0
        self.num_episodes = 0
        self.sample_time = 0
        self.stats_names = ['episode_len', 'reward', 'episode_reward', 'road_network', 'life_circle', 'greenness']
        if init_stats_logger:
            self.stats_loggers = {x: StatsLogger(is_nparray=False) for x in self.stats_names}
        self.plans = []

    def start_episode(self, env):
        self.episode_len = 0
        self.episode_reward = 0

    def step(self, env, reward, info):
        self.episode_len += 1
        self.episode_reward += reward
        self.stats_loggers['reward'].log(reward)

    def end_episode(self, info):
        self.num_steps += self.episode_len
        self.num_episodes += 1
        self.stats_loggers['episode_len'].log(self.episode_len)
        self.stats_loggers['episode_reward'].log(self.episode_reward)
        self.stats_loggers['road_network'].log(info['road_network'])
        self.stats_loggers['life_circle'].log(info['life_circle'])
        self.stats_loggers['greenness'].log(info['greenness'])

    def add_plan(self, info_plan):
        self.plans.append(info_plan)

    @classmethod
    def merge(cls, logger_list, **kwargs):
        logger = cls(init_stats_logger=False, **kwargs)
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.stats_loggers = {}
        for stats in logger.stats_names:
            logger.stats_loggers[stats] = StatsLogger.merge([x.stats_loggers[stats] for x in logger_list])

        logger.total_reward = logger.stats_loggers['reward'].total()
        logger.avg_episode_len = logger.stats_loggers['episode_len'].avg()
        logger.avg_episode_reward = logger.stats_loggers['episode_reward'].avg()
        logger.max_episode_reward = logger.stats_loggers['episode_reward'].max()
        logger.min_episode_reward = logger.stats_loggers['episode_reward'].min()
        logger.avg_episode_road_network_reward = logger.stats_loggers['road_network'].avg()
        logger.avg_episode_life_circle_reward = logger.stats_loggers['life_circle'].avg()
        logger.avg_episode_greenness_reward = logger.stats_loggers['greenness'].avg()
        logger.plans = list(itertools.chain(*[var.plans for var in logger_list]))
        return logger
