import logging
import sys, os

from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class ModuleClone(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, Tuple[int, ...]],
        output_dims: Dict[str, Tuple[int, ...]],
        path_to_backbone: str = None,
    ):
        super().__init__()
        self.config = {
            "input_dims": input_dims,
            "output_dims": output_dims,
            "path_to_backbone": path_to_backbone,
        }

        # init input encoders
        self.input_encoders = nn.ModuleDict({
            name: InputEncoder(in_dims)
            for name, in_dims in input_dims.items()
        })
        logger.debug("Input encoders are: %s", self.input_encoders)

        # infer sum of input encoder output dims
        with torch.no_grad():
            fused_dim = sum(
                self.input_encoders[name](torch.zeros(1, *in_dims)).shape[-1]
                for name, in_dims in input_dims.items()
            )
        logger.debug("backbone input dimensions are: %s", fused_dim)

        # init backbone
        if path_to_backbone:
            self.backbone = BackboneWrapper(path_to_backbone, fused_dim)
        else:
            self.backbone = DefaultBackbone(fused_dim)


        # infer backbone output dim
        with torch.no_grad():
            backbone_out = self.backbone(torch.zeros(1, fused_dim))
            assert backbone_out.ndim == 2, f"Expected [B, D], got {tuple(backbone_out.shape)}"
            backbone_out_dim = backbone_out.shape[-1]

        logger.debug("Backbone is: %s", self.backbone)
        logger.debug("backbone output dimensions are: %s", backbone_out_dim)

        # init output heads
        self.output_heads = nn.ModuleDict({
            name: OutputHead(backbone_out_dim, out_dims)
            for name, out_dims in output_dims.items()
        })
        logger.debug("Output heads are: %s", self.output_heads)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logger.debug("Running forward on module clone.")
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
    def __init__(self, in_dims: Tuple[int, ...]):
        super().__init__()
        self.in_dims = in_dims
        # allow multi-dim inputs but flatten for now
        # TODO later: allow custom input encoders (like CNNs) 
        self.flat_dim = math.prod(in_dims)
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten but allow for leading batch dimension
        x = x.reshape(*x.shape[:-len(self.in_dims)], self.flat_dim)
        return self.net(x)

class OutputHead(nn.Module):
    def __init__(self, in_dim: int, out_dims: Tuple[int, ...]):
        super().__init__()
        self.out_dims = out_dims
        self.flat_out_dim = math.prod(out_dims)

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.flat_out_dim)
        )
        # NOTICE: no activations are applied here (logits are numerically safer),
        #         -> apply custom activations inside _loss() and _unpack()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # reshape from flattend to original shape 
        x = x.reshape(*x.shape[:-1], *self.out_dims)
        return x

class DefaultBackbone(nn.Module):
    """
    This simple MLP backbone is used when no backbone is provided
    """

    def __init__(self, in_dim):
        super().__init__()
       
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim) if in_dim else nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BackboneWrapper(nn.Module):

    def __init__(self, path_to_backbone: str | Path, in_dim: int):
        self.path = Path(path_to_backbone)
        self.name = self.path.stem

        self.backbone = None # TODO: dynamic load backbone from file, then instantiate and validate
        raise NotImplementedError("Not further implemented yet.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
