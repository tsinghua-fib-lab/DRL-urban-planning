import math
import numpy as np


class StatsLogger:

    def __init__(self, is_nparray=False):
        self.is_nparray = is_nparray
        self.total_val = 0
        self.min_val = math.inf
        self.max_val = -math.inf
        self.min_func = np.minimum if self.is_nparray else min
        self.max_func = np.maximum if self.is_nparray else max
        self.n = 0

    def log(self, val):
        self.total_val += val
        self.min_val = self.min_func(self.min_val, val)
        self.max_val = self.max_func(self.max_val, val)
        self.n += 1

    def avg(self):
        if self.n == 0:
            return 0
        return self.total_val / self.n

    def total(self):
        return self.total_val

    def min(self):
        return self.min_val

    def max(self):
        return self.max_val

    @classmethod
    def merge(cls, stats_loggers):
        logger = cls(is_nparray=stats_loggers[0].is_nparray)
        logger.total_val = sum([x.total_val for x in stats_loggers])
        logger.min_val = np.min(np.stack([x.min_val for x in stats_loggers]), axis=0)
        logger.max_val = np.max(np.stack([x.max_val for x in stats_loggers]), axis=0)
        logger.n = sum([x.n for x in stats_loggers])
        return logger