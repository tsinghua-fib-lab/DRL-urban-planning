import os
from pprint import pprint

import pygad
import setproctitle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from urban_planning.utils.config import Config
from urban_planning.agents.urban_planning_agent import UrbanPlanningAgent


flags.DEFINE_string('root_dir', '/data1/mas/zhengyu/urban_planning/', 'Root directory for writing '
                                                                      'logs/summaries/checkpoints.')
flags.DEFINE_string('cfg', None, 'Configuration file of rl training.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_enum('agent', 'rl-sgnn', ['rl-sgnn', 'rl-mlp'], 'Agent type.')
flags.DEFINE_bool('mean_action', True, 'Whether to use greedy strategy.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the planning process.')
flags.DEFINE_bool('only_road', False, 'Whether to only visualize road planning.')
flags.DEFINE_bool('save_video', False, 'Whether to save a video of the planning process.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', '0', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')

FLAGS = flags.FLAGS


def main_loop(_):

    setproctitle.setproctitle('urban_planning@zhengyu')

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = UrbanPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=1,
                               training=False, checkpoint=checkpoint, restore_best_rewards=True)

    if FLAGS.only_road:
        agent.freeze_land_use()

    agent.infer(num_samples=1, mean_action=FLAGS.mean_action, visualize=FLAGS.visualize,
                save_video=FLAGS.save_video, only_road=FLAGS.only_road)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed'
    ])
    app.run(main_loop)
