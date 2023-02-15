import torch.nn as nn


class UrbanPlanningValue(nn.Module):
    """
    Value network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        self.value_head = self.create_value_head(cfg)

    def create_value_head(self, cfg):
        """Create the value head."""
        value_head = nn.Sequential()
        for i in range(len(cfg['value_head_hidden_size'])):
            if i == 0:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.shared_net.output_value_size, cfg['value_head_hidden_size'][i])
                )
            else:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['value_head_hidden_size'][i - 1], cfg['value_head_hidden_size'][i])
                )
            if i < len(cfg['value_head_hidden_size']) - 1:
                value_head.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
        return value_head

    def forward(self, x):
        _, _, state_value, _, _, _ = self.shared_net(x)
        value = self.value_head(state_value)
        return value
