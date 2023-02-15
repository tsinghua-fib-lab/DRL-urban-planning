import torch
import torch.nn as nn

from urban_planning.envs import city_config


class SGNNStateEncoder(nn.Module):
    """
    Single GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)

        self.node_encoder = nn.Linear(agent.node_dim, cfg['gcn_node_dim'])
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']
        self.edge_fc_layers = self.create_edge_fc_layers(cfg)
        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_key_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_value_layer = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])

        self.output_policy_land_use_size = cfg['gcn_node_dim']*4
        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim']*3 + cfg['state_encoder_hidden_size'][-1] + 3

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module(
                    'flatten_{}'.format(i),
                    nn.Flatten()
                )
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size, cfg['state_encoder_hidden_size'][i])
                )
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1], cfg['state_encoder_hidden_size'][i])
                )
            feature_encoder.add_module(
                'tanh_{}'.format(i),
                nn.Tanh()
            )
        return feature_encoder

    def create_edge_fc_layers(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                if i == 0:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim']*2, cfg['gcn_node_dim'])
                    )
                else:
                    seq.add_module(
                        'linear_{}'.format(i),
                        nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
                    )
                seq.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
            return seq
        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, edge_mask, max_num_nodes):
        """
        Aggregate edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            indices (torch.Tensor): Node indices. Shape: (batch, max_num_edges).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            count_edge (torch.Tensor): Edge counts per node. Shape: (batch, max_num_nodes, node_dim).
        """
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes, num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index, edge_mask, edge_fc_layer):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = torch.cat([h_edges1, h_edges2], -1)
        h_edges_21 = torch.cat([h_edges2, h_edges1], -1)
        h_edges = (edge_fc_layer(h_edges_12) + edge_fc_layer(h_edges_21)) / 2
        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def scatter_to_nodes(self, h_edges, edge_index, edge_mask, max_num_nodes):
        """
        Scatter edge embeddings to nodes.

        Args:
            h_edges (torch.Tensor): Edge embeddings. Shape: (batch, max_num_edges, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            max_num_nodes (int): Maximum number of nodes.

        Returns:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
        """
        h_nodes_1, count_1 = self.scatter_count(h_edges, edge_index[:, :, 0], edge_mask, max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges, edge_index[:, :, 1], edge_mask, max_num_nodes)
        h_nodes = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
        return h_nodes

    def self_attention(self, h_current_node, h_nodes, node_mask):
        """Self attention."""
        query = self.attention_query_layer(h_current_node).transpose(0, 1)
        keys = self.attention_key_layer(h_nodes).transpose(0, 1)
        values = self.attention_value_layer(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer(
            query,
            keys,
            values,
            key_padding_mask=~node_mask)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

    @staticmethod
    def batch_data(x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = zip(*x)
        numerical_features = torch.stack(numerical_features)
        node_features = torch.stack(node_features)
        edge_index = torch.stack(edge_index)
        current_node_features = torch.stack(current_node_features)
        node_mask = torch.stack(node_mask)
        edge_mask = torch.stack(edge_mask)
        land_use_mask = torch.stack(land_use_mask)
        road_mask = torch.stack(road_mask)
        stage = torch.stack(stage)
        return numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage

    @staticmethod
    def mean_features(h, mask):
        mean_h = (h*mask.unsqueeze(-1).float()).sum(dim=1)/mask.float().sum(dim=1, keepdim=True)
        return mean_h

    def forward(self, x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = self.batch_data(x)
        h_numerical_features = self.numerical_feature_encoder(numerical_features)

        h_nodes = self.node_encoder(node_features)
        current_node_features = torch.unsqueeze(current_node_features, 1)
        h_current_node = self.node_encoder(current_node_features)

        # GCN
        for edge_fc_layer in self.edge_fc_layers:
            h_edges = self.gather_to_edges(h_nodes, edge_index, edge_mask, edge_fc_layer)
            h_nodes_new = self.scatter_to_nodes(h_edges, edge_index, edge_mask, self.max_num_nodes)
            h_nodes = h_nodes + h_nodes_new

        h_edges_mean = self.mean_features(h_edges, edge_mask)
        h_nodes_mean = self.mean_features(h_nodes, node_mask)

        h_current_node_attended = self.self_attention(h_current_node, h_nodes, node_mask)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, h_edges_mean, h_current_node_attended, stage],
                                dim=1)

        h_current_node_repeated = h_current_node.repeat(1, self.max_num_edges, 1)
        state_policy_land_use = torch.cat(
            [h_edges, h_current_node_repeated, h_edges*h_current_node_repeated, h_edges - h_current_node_repeated],
            dim=-1)

        state_policy_road = torch.cat([h_nodes], dim=-1)

        return state_policy_land_use, state_policy_road, state_value, land_use_mask, road_mask, stage


class MLPStateEncoder(nn.Module):
    """
    State encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)

        self.node_encoder = nn.Linear(agent.node_dim, cfg['gcn_node_dim'])
        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.output_policy_land_use_size = cfg['gcn_node_dim']*4
        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim']*2 + cfg['state_encoder_hidden_size'][-1] + 3

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module(
                    'flatten_{}'.format(i),
                    nn.Flatten()
                )
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size, cfg['state_encoder_hidden_size'][i])
                )
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1], cfg['state_encoder_hidden_size'][i])
                )
            feature_encoder.add_module(
                'tanh_{}'.format(i),
                nn.Tanh()
            )
        return feature_encoder

    def compute_edge_features(self, h_nodes, edge_index, edge_mask):
        """
        Gather node embeddings to edges.

        Args:
            h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
            edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
            edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
            edge_fc_layer (torch.nn.Module): Edge fc layer.

        Returns:
            h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
        """
        h_edges1 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges2 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        edges2_type = torch.eq(torch.argmax(h_edges2[:, :, :city_config.NUM_TYPES+1], dim=-1), city_config.FEASIBLE)
        edges2_type_mask = torch.broadcast_to(edges2_type.unsqueeze(-1), h_edges2.size())
        h_edges = torch.where(edges2_type_mask, h_edges2, h_edges1)
        mask = torch.broadcast_to(edge_mask.unsqueeze(-1), h_edges.shape)
        h_edges = torch.where(mask, h_edges, torch.zeros_like(h_edges))
        return h_edges

    def forward(self, x):
        numerical_features, node_features, edge_index, current_node_features, node_mask, edge_mask, \
            land_use_mask, road_mask, stage = SGNNStateEncoder.batch_data(x)
        h_numerical_features = self.numerical_feature_encoder(numerical_features)

        edge_features = self.compute_edge_features(node_features, edge_index, edge_mask)

        h_nodes = self.node_encoder(node_features)
        h_edges = self.node_encoder(edge_features)
        current_node_features = torch.unsqueeze(current_node_features, 1)
        h_current_node = self.node_encoder(current_node_features)

        h_edges_mean = SGNNStateEncoder.mean_features(h_edges, edge_mask)
        h_nodes_mean = SGNNStateEncoder.mean_features(h_nodes, node_mask)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, h_edges_mean, stage],
                                dim=1)

        h_current_node_repeated = h_current_node.repeat(1, self.max_num_edges, 1)
        state_policy_land_use = torch.cat(
            [h_edges, h_current_node_repeated, h_edges*h_current_node_repeated, h_edges - h_current_node_repeated],
            dim=-1)

        state_policy_road = torch.cat([h_nodes], dim=-1)

        return state_policy_land_use, state_policy_road, state_value, land_use_mask, road_mask, stage
