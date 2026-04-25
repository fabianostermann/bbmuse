import logging
import sys, os

from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from bbmuse.learn.module_clone import ModuleClone

class PolicyModel(nn.Module):
    def __init__(self, deterministic_model: ModuleClone):
        super().__init__()
        # TODO: ensure that all tensors are on the same device
        self.model = deterministic_model
        self.log_stds = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(*head.out_dims))
            for name, head in deterministic_model.output_heads.items()
        })

    def forward(self, inputs):
        return self.model(inputs)  # delegate

    def _build_dists(self, inputs):
        means = self.forward(inputs)
        return {
            name: torch.distributions.Normal(mean, self.log_stds[name].exp())
            for name, mean in means.items()
        }

    def sample_with_log_prob(self, inputs):
        dists = self._build_dists(inputs)
        actions, log_probs = {}, {}
        for name, dist in dists.items():
            action = dist.sample()
            actions[name] = action
            log_probs[name] = dist.log_prob(action).sum(-1)
        return actions, log_probs

    def log_prob(self, inputs, actions):
        dists = self._build_dists(inputs)
        return {
            name: dist.log_prob(actions[name]).sum(-1)
            for name, dist in dists.items()
        }