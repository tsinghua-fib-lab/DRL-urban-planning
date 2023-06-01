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


flags.DEFINE_string('root_dir', '/data1/mas/zhengyu/drl_urban_planning/', 'Root directory for writing '
                                                                          'logs/summaries/checkpoints.')
flags.DEFINE_string('cfg', None, 'Configuration file of rl training.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('mean_action', True, 'Whether to use greedy strategy.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the planning process.')
flags.DEFINE_bool('only_road', False, 'Whether to only visualize road planning.')
flags.DEFINE_bool('save_video', False, 'Whether to save a video of the planning process.')
flags.DEFINE_integer('global_seed', None, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_integer('sol_per_pop', 20, 'The number of solutions per population.')
flags.DEFINE_integer('num_generations', 100, 'The number of generations.')
flags.DEFINE_integer('num_parents_mating', 2, 'The number of parents for mating.')
flags.DEFINE_integer('init_range_low', -5, 'Low range for gene initialization.')
flags.DEFINE_integer('init_range_high', 5, 'High range for gene initialization.')
flags.DEFINE_string('parent_selection_type', 'sss', 'Type of parent selection.')
flags.DEFINE_string('crossover_type', 'single_point', 'Type of crossover.')
flags.DEFINE_string('mutation_type', 'random', 'Type of mutation.')
flags.DEFINE_integer('mutation_percent_genes', 10, 'Percentage of genes for mutation.')

FLAGS = flags.FLAGS


def main_loop(_):

    setproctitle.setproctitle('urban_planning@zhengyu')

    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, 'ga')

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    """create agent"""
    agent = UrbanPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=1,
                               training=False, checkpoint=0, restore_best_rewards=True)

    if FLAGS.only_road:
        agent.freeze_land_use()

    def fitness_func(solution, solution_idx):
        fitness, _ = agent.fitness_ga(solution, num_samples=1, mean_action=False, visualize=FLAGS.visualize)
        return fitness

    def report_func(instance):
        print(f'Generation: {instance.generations_completed}')
        print(f'Best Fitness: {instance.best_solutions_fitness[-1]: .4f}')
        print(f'Last Generation Average Fitness: '
              f'{sum(instance.last_generation_fitness)/len(instance.last_generation_fitness): .4f}')
        print()

    ga_instance = pygad.GA(num_generations=FLAGS.num_generations,
                           num_parents_mating=FLAGS.num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=FLAGS.sol_per_pop,
                           num_genes=2*agent.node_dim + 1,
                           init_range_low=FLAGS.init_range_low,
                           init_range_high=FLAGS.init_range_high,
                           parent_selection_type=FLAGS.parent_selection_type,
                           keep_parents=1,
                           crossover_type=FLAGS.crossover_type,
                           mutation_type=FLAGS.mutation_type,
                           mutation_percent_genes=FLAGS.mutation_percent_genes,
                           on_generation=report_func,
                           stop_criteria="saturate_10",
                           random_seed=cfg.seed)

    ga_instance.run()

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=best_solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))

    agent.save_ga(best_solution, best_solution_fitness)

    _, plan = agent.fitness_ga(best_solution, num_samples=1, visualize=FLAGS.visualize)
    pprint(plan, indent=4, sort_dicts=False)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed'
    ])
    app.run(main_loop)
