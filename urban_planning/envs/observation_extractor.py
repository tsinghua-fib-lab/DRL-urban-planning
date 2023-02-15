from typing import Dict, List, Tuple, Text

import numpy as np

from urban_planning.envs.plan_client import PlanClient
from urban_planning.envs import city_config


class ObservationExtractor:
    def __init__(self, plc: PlanClient, max_num_nodes: int, max_num_edges: int, max_num_stages: int) -> None:
        self._plc = plc
        self._max_num_nodes = max_num_nodes
        self._max_num_edges = max_num_edges
        self._max_num_stages = max_num_stages
        self._get_normalization_params()
        self._get_obs_static()

    def _get_normalization_params(self) -> None:
        """
        Returns the normalization parameters.

        Returns:
            normalization_params (np.ndarray): the normalization parameters.
        """
        self._max_area = self._plc.get_common_max_area()
        self._max_edge_length = self._plc.get_common_max_edge_length()

    def _get_obs_static(self) -> None:
        """
        Returns the numerical observation.

        Returns:
            obs_numerical (np.ndarray): the numerical observation.
        """
        required_plan_ratio, required_plan_count = self._plc.get_requirements()
        self.max_required_plan_count = required_plan_count.max()
        normalized_required_plan_count = required_plan_count / self.max_required_plan_count
        self._obs_static = np.concatenate([required_plan_ratio, normalized_required_plan_count])

    def _get_obs_numerical(self) -> np.ndarray:
        """
        Returns the numerical observation.

        Returns:
            obs_numerical (np.ndarray): the numerical observation.
        """
        plan_ratio, plan_count = self._plc.get_plan_ratio_and_count()
        normalized_plan_count = plan_count / self.max_required_plan_count
        obs_numerical = np.concatenate([self._obs_static, plan_ratio, normalized_plan_count], dtype=np.float32)
        return obs_numerical

    def _pad_mask(self, mask: np.ndarray, max_num: int, name: Text) -> np.ndarray:
        """
        Returns the mask observation.

        Args:
            mask (np.ndarray): the current mask.
            max_num (int): the maximum number of elements.
            name (str): the name of the mask.

        Returns:
            obs_mask (np.ndarray): the mask observation.
        """
        pad = (0, max_num - mask.size)
        if pad[1] < 0:
            raise ValueError('The number of {} exceeds the maximum limit.'.format(name))
        return np.pad(mask, pad, mode='constant', constant_values=False)

    def _pad_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """
        Returns the nodes observation.

        Args:
            nodes (np.ndarray): the current nodes.

        Returns:
            obs_nodes (np.ndarray): the nodes observation.
        """
        pad = ((0, self._max_num_nodes - nodes.shape[0]), (0, 0))
        if pad[0][1] < 0:
            raise ValueError('The number of nodes exceeds the maximum limit.')
        return np.pad(nodes, pad, mode='constant', constant_values=0)

    def _pad_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Returns the edges observation.

        Args:
            edges (np.ndarray): the current edges.

        Returns:
            obs_edges (np.ndarray): the edges observation.
        """
        pad = ((0, self._max_num_edges - edges.shape[0]), (0, 0))
        if pad[0][1] < 0:
            raise ValueError('The number of edges exceeds the maximum limit.')
        return np.pad(edges, pad, mode='constant', constant_values=self._max_num_nodes-1)

    def _get_obs_graph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the graph observation.

        Returns:
            obs_nodes (np.ndarray): the nodes observation.
            obs_edges (np.ndarray): the edges observation.
            obs_node_mask (np.ndarray): the node mask observation.
            obs_edge_mask (np.ndarray): the edge mask observation.
        """
        node_type, node_coordinates, node_area, node_length, node_width, node_height, node_domain, edges \
            = self._plc.get_graph_features()
        # transform node type to one-hot encoding
        node_type = np.eye(city_config.NUM_TYPES + 1)[node_type]
        node_coordinates = 2 * node_coordinates - 1
        node_area = 2 * np.expand_dims(node_area, axis=1)/self._max_area - 1
        node_length = 2 * np.expand_dims(node_length, axis=1)/self._max_edge_length - 1
        node_width = 2 * np.expand_dims(node_width, axis=1)/self._max_edge_length - 1
        node_height = 2 * np.expand_dims(node_height, axis=1)/self._max_edge_length - 1
        node_domain = 2 * node_domain - 1
        obs_nodes = np.concatenate(
            [node_type, node_coordinates, node_area, node_length, node_width, node_height, node_domain],
            axis=-1, dtype=np.float32)

        obs_node_mask = np.full(obs_nodes.shape[0], True)
        obs_node_mask = self._pad_mask(obs_node_mask, self._max_num_nodes, 'nodes')

        obs_edge_mask = np.full(edges.shape[0], True)
        obs_edge_mask = self._pad_mask(obs_edge_mask, self._max_num_edges, 'edges')

        obs_nodes = self._pad_nodes(obs_nodes)
        obs_edges = self._pad_edges(edges)

        return obs_nodes, obs_edges, obs_node_mask, obs_edge_mask

    def _get_obs_current_node(self, land_use: Dict) -> np.ndarray:
        """
        Returns the current node observation.

        Args:
            land_use (dictionary): the current land use.

        Returns:
            obs_current_node (np.ndarray): the current node observation.
        """
        node_type = np.eye(city_config.NUM_TYPES + 1)[land_use['type']]
        node_coordinates = 2*np.array([land_use['x'], land_use['y']]) - 1
        node_area_length_width_height = np.array(
            [2*land_use['area']/self._max_area - 1,
             2*land_use['length']/self._max_edge_length - 1,
             2*land_use['width']/self._max_edge_length - 1,
             2*land_use['height']/self._max_edge_length - 1])
        node_domain = np.array(
            [2*land_use['rect'] - 1,
             2*land_use['eqi'] - 1,
             2*land_use['sc'] - 1])
        obs_current_node = np.concatenate([node_type, node_coordinates, node_area_length_width_height, node_domain],
                                          dtype=np.float32)
        return obs_current_node

    def _get_obs_mask(self, mask: np.ndarray, max_num: int, name: Text) -> np.ndarray:
        """
        Returns the mask observation.

        Args:
            mask (np.ndarray): the current mask.
            max_num (int): the maximum number of elements.

        Returns:
            obs_mask (np.ndarray): the mask observation.
        """
        obs_mask = self._pad_mask(mask, max_num, name)
        return obs_mask

    def _get_obs_stage(self, stage: int) -> np.ndarray:
        """
        Returns the stage observation.

        Args:
            stage (int): the current stage.

        Returns:
            obs_stage (np.ndarray): the stage observation.
        """
        obs_stage = np.eye(self._max_num_stages, dtype=np.float32)[stage]
        return obs_stage

    def get_numerical_feature_size(self):
        """
        Returns the feature size.

        Returns:
            feature_size (int): the feature size.
        """
        return self._obs_static.size*2

    def get_node_dim(self, land_use: Dict) -> int:
        """
        Returns the node dimension.

        Args:
            land_use (dictionary): the current land use.

        Returns:
            node_dim (int): the node dimension.
        """
        return self._get_obs_current_node(land_use).size

    def get_obs(self, land_use: Dict, land_use_mask: np.ndarray, road_mask: np.ndarray, stage: int) -> List:
        """
        Returns the observation.

        Args:
            land_use (dictionary): the current land use.
            land_use_mask (np.ndarray): the current land_use mask.
            road_mask (np.ndarray): the current road mask.
            stage (int): the current stage.

        Returns:
            obs (list): the observation.
        """
        obs_numerical = self._get_obs_numerical()
        obs_nodes, obs_edges, obs_node_mask, obs_edge_mask = self._get_obs_graph()
        obs_current_node = self._get_obs_current_node(land_use)
        obs_land_use_mask = self._get_obs_mask(land_use_mask, self._max_num_edges, 'edges')
        obs_road_mask = self._get_obs_mask(road_mask, self._max_num_nodes, 'nodes')
        stage = self._get_obs_stage(stage)
        obs = [obs_numerical, obs_nodes, obs_edges, obs_current_node, obs_node_mask, obs_edge_mask,
               obs_land_use_mask, obs_road_mask, stage]
        return obs
