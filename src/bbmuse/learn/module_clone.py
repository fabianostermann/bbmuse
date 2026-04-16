import logging
import sys, os

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

logger = logging.getLogger(__name__)

class ModuleClone(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dims: Dict[str, int],
    ):
        super().__init__()

        # init input encoders
        self.input_encoders = nn.ModuleDict({
            name: InputEncoder(in_dim)
            for name, in_dim in input_dims.items()
        })

        # infer sum of input encoder output dims
        with torch.no_grad():
            backbone_in_dim = sum(
                self.input_encoders[name](torch.zeros(1, in_dim)).shape[-1]
                for name, in_dim in input_dims.items()
            )

        # init backbone
        self.backbone = DefaultBackbone(backbone_in_dim)

        # infer backbone output dim
        with torch.no_grad():
            dummy = torch.zeros(1, backbone_in_dim)
            backbone_out = self.backbone(dummy)
            assert backbone_out.ndim == 2, f"Expected [B, D], got {tuple(backbone_out.shape)}"
            backbone_out_dim = backbone_out.shape[-1]

        # init output heads
        self.output_heads = nn.ModuleDict({
            name: OutputHead(backbone_out_dim, out_dim)
            for name, out_dim in output_dims.items()
        })

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = []
        for name, encoder in self.input_encoders.items():
            encoded.append(encoder(inputs[name]))

        fused = torch.cat(encoded, dim=-1)
        h = self.backbone(fused)

        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(h)

        return outputs

class InputEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Identity() 
        #self.net = nn.Sequential(
        #    nn.Linear(in_dim, out_dim),
        #    nn.ReLU(),
        #)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class OutputHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        # NOTICE: although no activations are ever applied here,
        #         you can use custom activations inside _loss() and _unpack()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DefaultBackbone(nn.Module):
    """
    This simple MLP backbone is used when no backbone is provided
    """

    def __init__(self, in_dim):
        super().__init__()
       
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
