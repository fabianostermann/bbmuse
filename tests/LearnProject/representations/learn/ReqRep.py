valueA = 2
valueB = 3.456

# --- bblearn ---
import numpy as np

def _pack():
    return np.array([valueA, valueB])

def _unpack(arr: np.ndarray):
    global valueA, valueB
    valueA = float(arr[0])
    valueB = float(arr[1])