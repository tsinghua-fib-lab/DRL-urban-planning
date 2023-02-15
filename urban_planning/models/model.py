import torch
import torch.nn as nn
from urban_planning.models.state_encoder import SGNNStateEncoder, MLPStateEncoder
from urban_planning.models.policy import UrbanPlanningPolicy
from urban_planning.models.value import UrbanPlanningValue


def create_sgnn_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = SGNNStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = UrbanPlanningPolicy(cfg.policy_specs, agent, shared_net)
    value_net = UrbanPlanningValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_mlp_model(cfg, agent):
    """Create a multi-layer perceptron model.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = MLPStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = UrbanPlanningPolicy(cfg.policy_specs, agent, shared_net)
    value_net = UrbanPlanningValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for parsing parameters.

    Args:
        actor_net (nn.Module): actor network.
        value_net (nn.Module): value network.
    """
    def __init__(self, actor_net, value_net):
        super().__init__()
        self.actor_net = actor_net
        self.value_net = value_net
