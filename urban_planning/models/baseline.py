import torch
from urban_planning.envs import city_config


class NullModel:
    def __init__(self):
        self.training = None
        self.device = None

    def train(self, mode=None):
        pass

    def to(self, device=None):
        pass

    @staticmethod
    def parameters() -> None:
        return None


class RuleCentralizedPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        node_features, edge_index, node_mask, edge_mask, land_use_mask, road_mask, stage = \
            x[0][1], x[0][2], x[0][4], x[0][5], x[0][-3], x[0][-2], x[0][-1]
        actions = torch.zeros(2)
        if stage.argmax() == 0:
            node_coordinates = node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            edge_coordinates = (node_coordinates[edge_index[:, 0]] + node_coordinates[edge_index[:, 1]]) / 2
            distance_to_center = torch.linalg.vector_norm(edge_coordinates, dim=1)
            land_use_logits = -torch.where(edge_mask,
                                           distance_to_center,
                                           torch.full_like(distance_to_center, torch.max(distance_to_center) + 1))
            land_use_paddings = torch.ones_like(land_use_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_land_use_logits = torch.where(land_use_mask, land_use_logits, land_use_paddings)
            land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)
            if mean_action:
                land_use_action = land_use_dist.probs.argmax().to(node_features.dtype)
            else:
                land_use_action = land_use_dist.sample().to(node_features.dtype)
            actions[0] = land_use_action
        else:
            node_length = node_features[:, city_config.NUM_TYPES+4]
            road_logits = torch.where(node_mask,
                                      node_length,
                                      torch.full_like(node_length, torch.min(node_length) - 1))
            road_paddings = torch.ones_like(road_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_road_logits = torch.where(road_mask, road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
            if mean_action:
                road_action = road_dist.probs.argmax().to(node_features.dtype)
            else:
                road_action = road_dist.sample().to(node_features.dtype)
            actions[1] = road_action

        actions = actions.unsqueeze(0)
        return actions


class RuleDecentralizedPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        node_features, edge_index, current_node, node_mask, edge_mask, land_use_mask, road_mask, stage = \
            x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8]
        actions = torch.zeros(2)
        if stage.argmax() == 0:
            node_coordinates = node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            edge_coordinates = (node_coordinates[edge_index[:, 0]] + node_coordinates[edge_index[:, 1]]) / 2
            current_land_use_type = torch.argmax(current_node[:city_config.NUM_TYPES+1])
            same_type_nodes = node_features[node_features[:, current_land_use_type] == 1]
            if same_type_nodes.shape[0] > 0:
                same_type_nodes_coordinates = same_type_nodes[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
                distance_to_same_type = torch.linalg.norm(
                    edge_coordinates.unsqueeze(1) - same_type_nodes_coordinates.unsqueeze(0), dim=2)
                distance_to_same_type = distance_to_same_type.mean(dim=1)
                land_use_logits = torch.where(edge_mask,
                                              distance_to_same_type,
                                              torch.full_like(distance_to_same_type,
                                                              torch.min(distance_to_same_type) - 1))
                land_use_paddings = torch.ones_like(land_use_mask, dtype=node_features.dtype)*(-2.**32+1)
                masked_land_use_logits = torch.where(land_use_mask, land_use_logits, land_use_paddings)
                land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)
                if mean_action:
                    land_use_action = land_use_dist.probs.argmax().to(node_features.dtype)
                else:
                    land_use_action = land_use_dist.sample().to(node_features.dtype)
                actions[0] = land_use_action
            else:
                valid_actions = torch.nonzero(land_use_mask.flatten()).flatten()
                if len(valid_actions) > 0:
                    index = torch.randint(0, len(valid_actions), (1,))
                    action = valid_actions[index]
                    actions[0] = action
        else:
            node_length = node_features[:, city_config.NUM_TYPES+4]
            road_logits = torch.where(node_mask,
                                      node_length,
                                      torch.full_like(node_length, torch.min(node_length) - 1))
            road_paddings = torch.ones_like(road_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_road_logits = torch.where(road_mask, road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
            if mean_action:
                road_action = road_dist.probs.argmax().to(node_features.dtype)
            else:
                road_action = road_dist.sample().to(node_features.dtype)
            actions[1] = road_action

        actions = actions.unsqueeze(0)
        return actions


class GSCAPolicy(NullModel):
    def __init__(self, grid_cols, grid_rows, cell_edge_length):
        super().__init__()
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_edge_length = cell_edge_length

    def rescale_coordinates(self, coordinates):
        coordinates[:, 0] = coordinates[:, 0] * self.grid_cols
        coordinates[:, 1] = coordinates[:, 1] * self.grid_rows
        return coordinates

    def select_action(self, x, mean_action=True):
        node_features, edge_index, current_node, node_mask, edge_mask, land_use_mask, road_mask, stage = \
            x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8]
        actions = torch.zeros(2)
        if stage.argmax() == 0:
            node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3] = \
                self.rescale_coordinates(node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3])
            node_coordinates = node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            edge_coordinates = (node_coordinates[edge_index[:, 0]] + node_coordinates[edge_index[:, 1]]) / 2
            current_land_use_type = torch.argmax(current_node[:city_config.NUM_TYPES+1])
            if current_land_use_type in (city_config.HOSPITAL_L, city_config.HOSPITAL_S):
                same_type_nodes = node_features[node_features[:, city_config.HOSPITAL_L] + node_features[:, city_config.HOSPITAL_S] >= 1]
            else:
                same_type_nodes = node_features[node_features[:, current_land_use_type] == 1]
            residential_nodes = node_features[node_features[:, city_config.RESIDENTIAL] == 1]
            residential_coordinates = residential_nodes[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            if same_type_nodes.shape[0] > 0:
                same_type_nodes_coordinates = same_type_nodes[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
                distance_residential_to_same_type = torch.linalg.norm(
                    residential_coordinates.unsqueeze(1) - same_type_nodes_coordinates.unsqueeze(0), dim=2)
                min_distance_residential_to_same_type = distance_residential_to_same_type.min(dim=1)[0]
                service_less = min_distance_residential_to_same_type * self.cell_edge_length > 500
                service_less_residential_nodes = residential_nodes[service_less]
                if service_less_residential_nodes.shape[0] == 0:
                    service_less_residential_nodes = residential_nodes
            else:
                service_less_residential_nodes = residential_nodes
            service_less_residential_coordinates = service_less_residential_nodes[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            distance_to_service_less_residential = torch.linalg.norm(
                edge_coordinates.unsqueeze(1) - service_less_residential_coordinates.unsqueeze(0), dim=2)
            num_served_residential = (distance_to_service_less_residential * self.cell_edge_length < 500).sum(dim=1)
            land_use_logits = torch.where(edge_mask,
                                          num_served_residential,
                                          torch.full_like(num_served_residential,
                                                          torch.min(num_served_residential) - 1)).float()
            land_use_paddings = torch.ones_like(land_use_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_land_use_logits = torch.where(land_use_mask, land_use_logits, land_use_paddings)
            land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)
            if mean_action:
                land_use_action = land_use_dist.probs.argmax().to(node_features.dtype)
            else:
                land_use_action = land_use_dist.sample().to(node_features.dtype)
            actions[0] = land_use_action
        else:
            node_length = node_features[:, city_config.NUM_TYPES+4]
            road_logits = torch.where(node_mask,
                                      node_length,
                                      torch.full_like(node_length, torch.min(node_length) - 1))
            road_paddings = torch.ones_like(road_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_road_logits = torch.where(road_mask, road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
            if mean_action:
                road_action = road_dist.probs.argmax().to(node_features.dtype)
            else:
                road_action = road_dist.sample().to(node_features.dtype)
            actions[1] = road_action

        actions = actions.unsqueeze(0)
        return actions


class GAPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, gene, mean_action=True):
        num_genes = len(gene)
        gene = torch.Tensor(gene)
        node_features, edge_index, current_node, node_mask, edge_mask, land_use_mask, road_mask, stage = \
            x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8]
        actions = torch.zeros(2)
        if stage.argmax() == 0:
            edge_features = (node_features[edge_index[:, 0]] + node_features[edge_index[:, 1]]) / 2

            node_coordinates = node_features[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
            edge_coordinates = (node_coordinates[edge_index[:, 0]] + node_coordinates[edge_index[:, 1]]) / 2
            current_land_use_type = torch.argmax(current_node[:city_config.NUM_TYPES+1])
            same_type_nodes = node_features[node_features[:, current_land_use_type] == 1]
            if same_type_nodes.shape[0] > 0:
                same_type_nodes_coordinates = same_type_nodes[:, city_config.NUM_TYPES+1:city_config.NUM_TYPES+3]
                distance_to_same_type = torch.linalg.norm(
                    edge_coordinates.unsqueeze(1) - same_type_nodes_coordinates.unsqueeze(0), dim=2)
                distance_to_same_type = distance_to_same_type.mean(dim=1)
            else:
                distance_to_same_type = torch.zeros_like(edge_coordinates[:, 0])
            edge_features = torch.cat([edge_features, distance_to_same_type.unsqueeze(1)], dim=1)

            land_use_logits = torch.sum(edge_features*torch.unsqueeze(torch.unsqueeze(gene[:num_genes//2 + 1], 0), 0),
                                        dim=2)
            land_use_logits = torch.where(edge_mask,
                                          land_use_logits,
                                          torch.full_like(land_use_logits, torch.min(land_use_logits) - 1))
            land_use_paddings = torch.ones_like(land_use_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_land_use_logits = torch.where(land_use_mask, land_use_logits, land_use_paddings)
            land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)
            if mean_action:
                land_use_action = land_use_dist.probs.argmax().to(node_features.dtype)
            else:
                land_use_action = land_use_dist.sample().to(node_features.dtype)
            actions[0] = land_use_action
        else:
            road_logits = torch.sum(node_features*torch.unsqueeze(torch.unsqueeze(gene[num_genes//2+1:], 0), 0), dim=2)
            road_logits = torch.where(node_mask,
                                      road_logits,
                                      torch.full_like(road_logits, torch.min(road_logits) - 1))
            road_paddings = torch.ones_like(road_mask, dtype=node_features.dtype)*(-2.**32+1)
            masked_road_logits = torch.where(road_mask, road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
            if mean_action:
                road_action = road_dist.probs.argmax().to(node_features.dtype)
            else:
                road_action = road_dist.sample().to(node_features.dtype)
            actions[1] = road_action

        actions = actions.unsqueeze(0)
        return actions
