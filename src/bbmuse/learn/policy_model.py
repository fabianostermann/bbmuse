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
from bbmuse.learn.action_spaces import MultiCategorical

class PolicyModel(nn.Module):
    def __init__(self, deterministic_model: ModuleClone):
        super().__init__()
        # TODO: ensure that all tensors are on the same device
        self.model = deterministic_model
        self.nvecs = deterministic_model.config["action_spaces"]

    def forward(self, inputs):
        return self.model(inputs)  # delegate

    def _build_dists(self, inputs):
        logits = self.forward(inputs)
        return {n: MultiCategorical(logits[n], self.nvecs[n]) for n in logits}

    def sample_with_log_prob(self, inputs):
        dists = self._build_dists(inputs)
        actions, log_probs = {}, {}
        for name, dist in dists.items():
            action = dist.sample()
            actions[name] = action
            log_probs[name] = dist.log_prob(action)
        return actions, log_probs

    def log_prob_with_entropy(self, inputs, actions):
        dists = self._build_dists(inputs)
        log_probs, entropies = {}, {}
        for name, dist in dists.items():
            log_probs[name] = dist.log_prob(actions[name])
            entropies[name] = dist.entropy()
        return log_probs, entropies