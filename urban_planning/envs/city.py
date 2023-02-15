import logging
import math
import copy
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
from geopandas import GeoDataFrame
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from urban_planning.envs.plan_client import PlanClient
from urban_planning.envs.observation_extractor import ObservationExtractor
from urban_planning.envs import city_config
from urban_planning.utils.config import Config


class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(self.action, self.mask)


def reward_info_function(
    plc: PlanClient,
    name: Text,
    road_network_weight: float = 1.0,
    life_circle_weight: float = 1.0,
    greenness_weight: float = 1.0,
    concept_weight: float = 0.0,
    weight_by_area: bool = False) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greenness_weight: Weight of greenness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        weight_by_area: Whether to weight the life circle reward by the area of residential zones.

    Returns:
        The RL reward.
        Info dictionary.
    """
    proxy_reward = CityEnv.INTERMEDIATE_REWARD

    if name == 'intermediate':
        return proxy_reward, {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greenness': -1.0,
            'concept': -1.0,
        }
    elif name == 'road':
        proxy_reward = 0.0
        road_network = -1.0
        road_network_info = dict()
        if road_network_weight > 0.0:
            road_network, road_network_info = plc.get_road_network_reward()
            proxy_reward += road_network_weight * road_network
        return proxy_reward, {
            'road_network': road_network,
            'life_circle': -1.0,
            'greenness': -1.0,
            'concept': -1.0,
            'road_network_info': road_network_info
        }
    elif name == 'land_use':
        proxy_reward = 0.0
        life_circle = -1.0
        greenness = -1.0
        concept = -1.0

        life_circle_info = dict()
        if life_circle_weight > 0.0:
            life_circle, life_circle_info = plc.get_life_circle_reward(weight_by_area=weight_by_area)
            proxy_reward += life_circle_weight * life_circle

        if greenness_weight > 0.0:
            greenness = plc.get_greenness_reward()
            proxy_reward += greenness_weight * greenness

        concept_info = dict()
        if concept_weight > 0.0:
            concept, concept_info = plc.get_concept_reward()
            proxy_reward += concept_weight * concept

        return proxy_reward, {
            'road_network': -1.0,
            'life_circle': life_circle,
            'greenness': greenness,
            'concept': concept,
            'life_circle_info': life_circle_info,
            'concept_info': concept_info
        }
    else:
        raise ValueError('Invalid state.')


class CityEnv:
    """ Environment for urban planning."""
    FAILURE_REWARD = -1.0
    INTERMEDIATE_REWARD = 0.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn:
                 Callable[[PlanClient, Text, float, float, float, bool], Tuple[float, Dict]] = reward_info_function):
        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._plc = PlanClient(cfg.objectives_plan, cfg.init_plan)

        self._reward_info_fn = partial(reward_info_fn,
                                       road_network_weight=cfg.reward_specs.get('road_network_weight', 1.0),
                                       life_circle_weight=cfg.reward_specs.get('life_circle_weight', 1.0),
                                       greenness_weight=cfg.reward_specs.get('greenness_weight', 1.0),
                                       concept_weight=cfg.reward_specs.get('concept_weight', 0.0),
                                       weight_by_area=cfg.reward_specs.get('weight_by_area', False))

        self._all_stages = ['land_use', 'road', 'done']
        self._set_stage()
        self._done = False
        self._set_cached_reward_info()
        self._observation_extractor = ObservationExtractor(self._plc,
                                                           self.cfg.state_encoder_specs['max_num_nodes'],
                                                           self.cfg.state_encoder_specs['max_num_edges'],
                                                           len(self._all_stages))

    def _set_stage(self):
        """
        Set the stage.
        """
        self._land_use_steps = 0
        self._road_steps = 0
        if not self.cfg.skip_land_use:
            self._stage = 'land_use'
            self._land_use_done = False
            self._road_done = False
        elif not self.cfg.skip_road:
            self._stage = 'road'
            self._land_use_done = True
            self._road_done = False
        else:
            raise ValueError('Invalid stage. Land_use step and road step both reached max steps.')

    def _compute_total_road_steps(self) -> None:
        """
        Compute the total number of road steps.
        """
        if self._stage == 'road' and self._road_steps == 0:
            self._total_road_steps = math.floor(np.count_nonzero(self._current_road_mask)*self.cfg.road_ratio)
        else:
            raise ValueError('Invalid stage.')

    def _set_cached_reward_info(self):
        """
        Set the cached reward.
        """
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greenness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def freeze_land_use(self, info: Dict):
        """
        Freeze the land use.
        """
        land_use_gdf = info['land_use_gdf']
        self._plc.freeze_land_use(land_use_gdf)
        self._cached_land_use_gdf = land_use_gdf
        self._cached_land_use_reward = info['land_use_reward']
        self._cached_life_circle_reward = info['life_circle']
        self._cached_greenness_reward = info['greenness']
        self._cached_concept_reward = info['concept']
        self._cached_life_circle_info = info['life_circle_info']
        self._cached_concept_info = info['concept_info']
        self._frozen = True

    def get_reward_info(self) -> Tuple[float, Dict]:
        """
        Returns the RL reward and info.

        Returns:
            The RL reward.
            Info dictionary.
        """
        if self.cfg.skip_road:
            if self._stage == 'land_use':
                return self._reward_info_fn(self._plc, 'intermediate')
            elif self._stage == 'done':
                return self._reward_info_fn(self._plc, 'land_use')
            else:
                raise ValueError('Invalid stage.')
        elif self.cfg.skip_land_use:
            if self._stage == 'road':
                return self._reward_info_fn(self._plc, 'intermediate')
            elif self._stage == 'done':
                return self._reward_info_fn(self._plc, 'road')
            else:
                raise ValueError('Invalid stage.')
        else:
            if (self._stage == 'land_use') or (self._stage == 'road' and self._road_steps > 0):
                return self._reward_info_fn(self._plc, 'intermediate')
            elif self._stage == 'road' and self._road_steps == 0:
                return self._reward_info_fn(self._plc, 'land_use')
            elif self._stage == 'done':
                return self._reward_info_fn(self._plc, 'road')
            else:
                raise ValueError('Invalid stage.')

    def _get_all_reward_info(self) -> Tuple[float, Dict]:
        """
        Returns the entire reward and info. Used for loaded plans.
        """
        land_use_reward, land_use_info = self._reward_info_fn(self._plc, 'land_use')
        road_reward, road_info = self._reward_info_fn(self._plc, 'road')
        reward = land_use_reward + road_reward
        info = {
            'road_network': road_info['road_network'],
            'life_circle': land_use_info['life_circle'],
            'greenness': land_use_info['greenness'],
            'road_network_info': road_info['road_network_info'],
            'life_circle_info': land_use_info['life_circle_info']
        }
        return reward, info

    def eval(self):
        """
        Set the environment to eval mode.
        """
        self._is_eval = True

    def train(self):
        """
        Set the environment to training mode.
        """
        self._is_eval = False

    def get_numerical_feature_size(self):
        """
        Returns the numerical feature size.

        Returns:
            feature_size (int): the feature size.
        """
        return self._observation_extractor.get_numerical_feature_size()

    def get_node_dim(self):
        """
        Returns the node dimension.

        Returns:
            node_dim (int): the node dimension.
        """
        dummy_land_use = self._get_dummy_land_use()
        return self._observation_extractor.get_node_dim(dummy_land_use)

    def _get_dummy_land_use(self):
        """
        Returns the dummy land use.

        Returns:
            land_use (dictionary): the dummy land use.
        """
        dummy_land_use = dict()
        dummy_land_use['type'] = city_config.FEASIBLE
        dummy_land_use['x'] = 0.5
        dummy_land_use['y'] = 0.5
        dummy_land_use['area'] = 0.0
        dummy_land_use['length'] = 0.0
        dummy_land_use['width'] = 0.0
        dummy_land_use['height'] = 0.0
        dummy_land_use['rect'] = 0.5
        dummy_land_use['eqi'] = 0.5
        dummy_land_use['sc'] = 0.5
        return dummy_land_use

    def _get_land_use_and_mask(self):
        """
        Returns the current land use and mask.

        Returns:
            land_use (dictionary): the current land use.
            mask (np.ndarray): the current mask.
        """
        if self._stage != 'land_use':
            land_use = self._get_dummy_land_use()
            mask = np.zeros(self.cfg.state_encoder_specs['max_num_edges'], dtype=bool)
        else:
            land_use, mask = self._plc.get_current_land_use_and_mask()
        return land_use, mask

    def _get_road_mask(self):
        """
        Returns the current road mask.

        Returns:
            mask (np.ndarray): the current mask.
        """
        if self._stage == 'land_use':
            mask = np.zeros(self.cfg.state_encoder_specs['max_num_nodes'], dtype=bool)
        else:
            mask = self._plc.get_current_road_mask()
        return mask

    def _get_stage_obs(self) -> int:
        """
        Returns the current stage observation.

        Returns:
            obs (int): the current stage index.
        """
        return self._all_stages.index(self._stage)

    def _get_obs(self) -> List:
        """
        Returns the observation.

        Returns:
            observation (object): the observation
        """
        return self._observation_extractor.get_obs(self._current_land_use,
                                                   self._current_land_use_mask,
                                                   self._current_road_mask,
                                                   self._get_stage_obs())

    def place_land_use(self, land_use: Dict, action: int):
        """
        Places the land use.

        Args:
            land_use (dictionary): the land use.
            action (int): the action.
        """
        self._plc.place_land_use(land_use, action)

    def build_road(self, action: int):
        """
        Builds the road.

        Args:
            action (int): the action.
        """
        self._plc.build_road(action)

    def fill_leftover(self):
        """
        Fill the leftover space.
        """
        self._plc.fill_leftover()

    def snapshot_land_use(self):
        """
        Snapshot the land use.
        """
        return self._plc.snapshot()

    def build_all_road(self):
        """
        Build all the road.
        """
        self._plc.build_all_road()

    def transition_stage(self):
        """
        Transition to the next stage.
        """
        if self._stage == 'land_use':
            self._land_use_done = True
            if not self.cfg.skip_road:
                self._stage = 'road'
            else:
                self._road_done = True
                self._done = True
                self._stage = 'done'
        elif self._stage == 'road':
            self._road_done = True
            self._done = True
            self._stage = 'done'
        else:
            raise ValueError('Unknown stage: {}'.format(self._stage))

    def failure_step(self, logging_str, logger):
        """
        Logging and reset after a failure step.
        """
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greenness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    def step(self, action: np.ndarray, logger: logging.Logger) -> Tuple[List, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset
        the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (np.ndarray of size 2): The action to take.
                                           1 is the land_use placement action.
                                           1 is the building road action.
            logger (Logger): The logger.

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        if self._stage == 'land_use':
            land_use = self._current_land_use
            action = int(action[0])
            self._action_history.append((land_use, action))
            if self._current_land_use_mask[action] == 0:
                raise InfeasibleActionError(action, self._current_land_use_mask)

            try:
                self.place_land_use(land_use, action)
            except ValueError as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)
            except Exception as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)

            self._land_use_steps += 1
            land_use_done = self._plc.is_land_use_done()
            if land_use_done:
                self.fill_leftover()
                self._cached_land_use_gdf = self.snapshot_land_use()
                self.transition_stage()
            reward, info = self.get_reward_info()
            self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask()
            if not self._land_use_done and not np.any(self._current_land_use_mask):
                return self.failure_step('Actions took before becoming infeasible', logger)
            self._current_road_mask = self._get_road_mask()
            if self._stage != 'land_use':
                self._cached_land_use_reward = reward
                if self._stage == 'road':
                    if not np.any(self._current_road_mask):
                        return self.failure_step('Actions took before becoming infeasible', logger)
                    self._cached_life_circle_reward = info['life_circle']
                    self._cached_greenness_reward = info['greenness']
                    self._cached_concept_reward = info['concept']

                    self._cached_life_circle_info = info['life_circle_info']
                    self._cached_concept_info = info['concept_info']

                    self._compute_total_road_steps()
        elif self._stage == 'road':
            action = int(action[1])
            self._action_history.append(('road', action))
            if self._current_road_mask[action] == 0:
                raise InfeasibleActionError(action, self._current_road_mask)

            try:
                self.build_road(action)
            except ValueError as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)
            except Exception as err:
                logger.error(err)
                return self.failure_step('Actions took before failing to place land use', logger)

            self._road_steps += 1
            if self._road_steps >= self._total_road_steps:
                self.transition_stage()
            reward, info = self.get_reward_info()
            self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask()
            self._current_road_mask = self._get_road_mask()
        else:
            raise ValueError('Cannot step in stage: {}.'.format(self._stage))

        if self._done:
            info['land_use_reward'] = self._cached_land_use_reward
            if not self.cfg.skip_road:
                info['life_circle'] = self._cached_life_circle_reward
                info['greenness'] = self._cached_greenness_reward
                info['concept'] = self._cached_concept_reward

                info['life_circle_info'] = self._cached_life_circle_info
                info['concept_info'] = self._cached_concept_info
            else:
                self.build_all_road()
            if self._is_eval:
                info['gdf'] = self._plc.get_gdf()
                info['land_use_gdf'] = self._cached_land_use_gdf

        return self._get_obs(), reward, self._done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation from the reset
        """
        self._plc.unplan_all_land_use()
        self._action_history = []
        self._set_stage()
        self._done = False
        self._set_cached_reward_info()
        self._current_land_use, self._current_land_use_mask = self._get_land_use_and_mask()
        self._current_road_mask = self._get_road_mask()
        if self.cfg.skip_land_use:
            self._compute_total_road_steps()
        return self._get_obs()

    @staticmethod
    def filter_land_use_road(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Filter out the land use and road features.
        """
        land_use_road_gdf = copy.deepcopy(gdf[(gdf['existence'] == True) &
                                              (gdf['type'] != city_config.OUTSIDE) &
                                              (gdf['type'] != city_config.BOUNDARY) &
                                              (gdf['type'] != city_config.INTERSECTION)])
        return land_use_road_gdf

    @staticmethod
    def filter_road_boundary(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Filter out the road and boundary features.
        """
        road_boundary_gdf = copy.deepcopy(gdf[(gdf['existence'] == True) &
                                              ((gdf['type'] == city_config.ROAD) |
                                               (gdf['type'] == city_config.BOUNDARY))])
        return road_boundary_gdf

    @staticmethod
    def _add_legend_to_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Add legend to the gdf.
        """
        gdf['legend'] = gdf['type'].apply(lambda x: city_config.LAND_USE_ID_MAP_INV[x])
        return gdf

    @staticmethod
    def plot_and_save_gdf(gdf: GeoDataFrame, cmap: ListedColormap,
                          save_fig: bool = False, path: Text = None, legend: bool = False,
                          ticks: bool = True, bbox: bool = True) -> None:
        """
        Plot and save the gdf.
        """
        gdf = CityEnv._add_legend_to_gdf(gdf)
        gdf.plot(
            'legend',
            cmap=cmap,
            categorical=True,
            legend=legend,
            legend_kwds={'bbox_to_anchor': (1.8, 1)}
        )
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        if not bbox:
            plt.axis('off')
        if save_fig:
            assert path is not None
            plt.savefig(path, format='svg', transparent=True)
        plt.show()
        plt.close()

    def visualize(self, save_fig: bool = False, path: Text = None, legend: bool = True,
                  ticks: bool = True, bbox: bool = True) -> None:
        """
        Visualize the city plan.
        """
        gdf = self._plc.get_gdf()
        land_use_road_gdf = self.filter_land_use_road(gdf)
        existing_types = sorted([city_config.LAND_USE_ID_MAP_INV[var] for var in land_use_road_gdf['type'].unique()])
        cmap = ListedColormap(
            [city_config.TYPE_COLOR_MAP[var] for var in existing_types])
        self.plot_and_save_gdf(land_use_road_gdf, cmap, save_fig, path, legend, ticks, bbox)

    def visualize_road_and_boundary(self, save_fig: bool = False, path: Text = None, legend: bool = True,
                                    ticks: bool = True, bbox: bool = True) -> None:
        """
        Visualize the roads and boundaries.
        """
        gdf = self._plc.get_gdf()
        road_boundary_gdf = self.filter_road_boundary(gdf)
        existing_types = sorted([city_config.LAND_USE_ID_MAP_INV[var] for var in road_boundary_gdf['type'].unique()])
        cmap = ListedColormap(
            [city_config.TYPE_COLOR_MAP[var] for var in existing_types])
        self.plot_and_save_gdf(road_boundary_gdf, cmap, save_fig, path, legend, ticks, bbox)

    def load_plan(self, gdf: GeoDataFrame) -> None:
        """
        Load a city plan.
        """
        self._plc.load_plan(gdf)

    def score_plan(self, verbose=True) -> Tuple[float, Dict]:
        """
        Score the city plan.
        """
        reward, info = self._get_all_reward_info()
        if verbose:
            print(f'reward: {reward}')
            pprint(info, indent=4, sort_dicts=False)
        return reward, info

    def get_init_plan(self) -> Dict:
        """
        Get the gdf of the city plan.
        """
        return self._plc.get_init_plan()
