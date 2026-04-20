valueA = 3
valueB = 4.567

# --- bblearn ---
import numpy as np
import torch
import torch.nn.functional as F

def _pack():
    return np.array([valueA, valueB])

def _unpack(arr: np.ndarray):
    global valueA, valueB
    valueA = float(arr[0])
    valueB = float(arr[1])

def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 10000 # F.mse_loss(pred, target)