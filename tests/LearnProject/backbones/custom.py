import torch

class SomeCustomBackbone(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


# Then your framework knows:
# - fused encoder output size = in_dim
# - backbone feature size = out_dim
# - every output head consumes out_dim
#
# This is much cleaner than letting arbitrary shapes float around.
