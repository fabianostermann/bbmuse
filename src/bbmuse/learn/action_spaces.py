import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MultiCategorical:
    def __init__(self, logits, nvec):
        self.dists = [torch.distributions.Categorical(logits=l)
                      for l in torch.split(logits, list(nvec), dim=-1)]

    def sample(self):
        return torch.stack([d.sample() for d in self.dists], dim=-1)

    def mode(self):
        return torch.stack([d.logits.argmax(-1) for d in self.dists], dim=-1)

    def log_prob(self, a):
        return torch.stack([d.log_prob(a[..., i]) for i, d in enumerate(self.dists)], -1).sum(-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], -1).sum(-1)

def make_ce_loss(nvec):
    """CE per segment. Targets are one-hot floats as emitted by _pack()."""
    nvec = list(nvec)
    def _loss(pred, target):
        return sum(
            F.cross_entropy(p, t)
            for p, t in zip(torch.split(pred, nvec, -1),
                            torch.split(target, nvec, -1))
        ) / len(nvec)
    return _loss