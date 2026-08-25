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
    """CE per segment (joint scale). Targets are one-hot floats as emitted by _pack()."""
    nvec = list(nvec)
    def _loss(pred, target):
        return sum(
            F.cross_entropy(p, t)
            for p, t in zip(torch.split(pred, nvec, -1),
                            torch.split(target, nvec, -1))
        )
    return _loss


def make_kl_losses(nvec):
    """
    Returns fn(ref_logits, pred_logits) -> (forward_kl, reverse_kl), both
    summed over segments -- the same joint scale as make_ce_loss, so every
    anchor metric lives on one axis.

    forward = KL(ref || policy): mass-covering. Matches the direction of
        kl_to_symbolic = CE(p_symbolic, q) - H(p_symbolic) = KL(p_symbolic || q),
        so the two anchors are directly comparable.
    reverse = KL(policy || ref): mode-seeking. The RLHF convention -- there it
        is used because it is estimable from on-policy samples; here both are
        exact, so the choice is purely about the objective.

    Both are computed from the same log-softmaxes, so they can never disagree
    on scale. Gradient flows only through `pred_logits`; `ref_logits` is
    expected to be detached by the caller.
    """
    nvec = list(nvec)

    def _kl(ref_logits, pred_logits):
        fwd = 0.0
        rev = 0.0
        for r, p in zip(torch.split(ref_logits, nvec, -1),
                        torch.split(pred_logits, nvec, -1)):
            log_pr = F.log_softmax(r, dim=-1)
            log_pq = F.log_softmax(p, dim=-1)
            diff = log_pr - log_pq
            fwd = fwd + (log_pr.exp() * diff).sum(-1).mean()   # sum_k p (log p - log q)
            rev = rev - (log_pq.exp() * diff).sum(-1).mean()   # sum_k q (log q - log p)
        return fwd, rev

    return _kl
