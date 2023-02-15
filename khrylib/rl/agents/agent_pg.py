from khrylib.rl.core import estimate_advantages
from khrylib.rl.agents.agent import Agent
from khrylib.utils.torch import *
import time


class AgentPG(Agent):

    def __init__(self, tau=0.95, optimizer=None,
                 value_pred_coef=0.5, entropy_coef=0.01,
                 opt_num_epochs=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer = optimizer
        self.value_pred_coef = value_pred_coef
        self.entropy_coef = entropy_coef
        self.opt_num_epochs = opt_num_epochs

    def value_loss(self, states, returns):
        """Get value loss"""
        values_pred = self.value_net(self.trans_value(states))
        value_loss = (values_pred - returns).pow(2).mean()
        return value_loss

    def update_policy(self, states, actions, returns, advantages, exps, iteration):
        """update policy"""
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        for _ in range(self.opt_num_epochs):
            value_loss = self.value_loss(states, returns)
            log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
            policy_loss = -(log_probs * advantages[ind]).mean()
            loss = policy_loss + self.value_pred_coef*value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_params(self, batch, iteration):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states))

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        self.update_policy(states, actions, returns, advantages, exps, iteration)

        return time.time() - t0
