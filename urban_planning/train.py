import os
import setproctitle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from urban_planning.utils.config import Config
from urban_planning.agents.urban_planning_agent import UrbanPlanningAgent

flags.DEFINE_string('root_dir', '/data/zhengyu/drl_urban_planning/', 'Root directory for writing '
                                                                     'logs/summaries/checkpoints.')
flags.DEFINE_string('cfg', None, 'Configuration file of rl training.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_enum('agent', 'rl-sgnn', ['rl-sgnn', 'rl-mlp'], 'Agent type.')
flags.DEFINE_bool('separate_train', True, 'Whether to separate the training process of land use and road planning.')
flags.DEFINE_integer('num_threads', 20, 'The number of threads for sampling trajectories.')
flags.DEFINE_bool('use_nvidia_gpu', True, 'Whether to use Nvidia GPU for acceleration.')
flags.DEFINE_integer('gpu_index', 0, 'GPU ID.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', '0', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')
flags.DEFINE_bool('restore_best_rewards', True, 'Whether to restore the best rewards from a saved checkpoint. '
                                                'True for resume training. False for finetune with new reward.')

FLAGS = flags.FLAGS


def train_one_iteration(agent: UrbanPlanningAgent, iteration: int) -> None:
    """Train one iteration"""
    agent.optimize(iteration)
    agent.save_checkpoint(iteration)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


def main_loop(_):

    setproctitle.setproctitle(f'urban_planning_{FLAGS.cfg}_{FLAGS.global_seed}')

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if FLAGS.use_nvidia_gpu and torch.cuda.is_available():
        device = torch.device('cuda', index=FLAGS.gpu_index)
    else:
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = UrbanPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=FLAGS.num_threads,
                               training=True, checkpoint=checkpoint, restore_best_rewards=FLAGS.restore_best_rewards)

    if FLAGS.separate_train and not cfg.skip_land_use and not cfg.skip_road:
        agent.freeze_road()
        start_iteration = agent.start_iteration
        for iteration in range(start_iteration, cfg.max_num_iterations):
            train_one_iteration(agent, iteration)

        agent.freeze_land_use()
        for iteration in range(cfg.max_num_iterations, 2 * cfg.max_num_iterations):
            train_one_iteration(agent, iteration)
    else:
        start_iteration = agent.start_iteration
        if cfg.skip_land_use:
            agent.freeze_land_use()
        for iteration in range(start_iteration, cfg.max_num_iterations):
            train_one_iteration(agent, iteration)

    agent.logger.info('training done!')


if __name__ == '__main__':
    flags.mark_flags_as_required([
      'cfg',
      'global_seed'
    ])
    app.run(main_loop)
