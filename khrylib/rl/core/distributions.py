import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical


class DiagGaussian(Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value).sum(1, keepdim=True)

    def mean_sample(self):
        return self.loc


class Categorical(TorchCategorical):

    def __init__(self, probs=None, logits=None, uniform_prob=0.0):
        super().__init__(probs, logits)
        self.uniform_prob = uniform_prob
        if uniform_prob > 0.0:
            self.uniform = TorchCategorical(logits=torch.zeros_like(self.logits))

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        if self.uniform_prob == 0.0:
            return super().log_prob(value).unsqueeze(1)
        else:
            return super().log_prob(value).unsqueeze(1) * (1 - self.uniform_prob) + self.uniform.log_prob(value).unsqueeze(1) * self.uniform_prob

    def mean_sample(self):
        return self.probs.argmax(dim=1)

    def sample(self):
        if self.uniform_prob == 0.0:
            return super().sample()
        else:
            if torch.bernoulli(torch.tensor(self.uniform_prob)).bool():
                # print('unif')
                return self.uniform.sample()
            else:
                # print('original')
                return super().sample()


class GaussianCategorical:

    def __init__(self, logits, scale, gaussian_dim):
        self.gaussian_dim = gaussian_dim
        self.logits = logits
        self.loc = loc = logits[:, :gaussian_dim]
        self.scale = scale = scale[:, :gaussian_dim]
        self.gaussian = DiagGaussian(loc, scale)
        self.discrete = Categorical(logits=logits[:, gaussian_dim:])

    def log_prob(self, value):
        gaussian_log_prob = self.gaussian.log_prob(value[:, :self.gaussian_dim])
        discrete_log_prob = self.discrete.log_prob(value[:, -1])
        return gaussian_log_prob + discrete_log_prob

    def mean_sample(self):
        gaussian_samp = self.gaussian.mean_sample()
        discrete_samp = self.discrete.mean_sample().unsqueeze(1).float()
        return torch.cat([gaussian_samp, discrete_samp], dim=-1)

    def sample(self):
        gaussian_samp = self.gaussian.sample()
        discrete_samp = self.discrete.sample().unsqueeze(1).float()
        return torch.cat([gaussian_samp, discrete_samp], dim=-1)
