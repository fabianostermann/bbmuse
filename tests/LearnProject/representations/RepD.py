# RepD — output of AgentD, the PARALLEL SIBLING of AgentB.
#
# 3 categories, deliberately matching RepB's noise segment: the parallel
# coordination reward (rewards/parallel_coordination.py) pays +1 when
# RepD.value == RepB.noise, and neither agent observes the other.
#
# Nothing reads RepD — it exists purely as one side of the coordination game.

value = 0

# --- bblearn ---
import numpy as np

N_VALUES = 3

# One segment of 3 categories. Sum (3) must equal len of _pack() output (3).
_action_space = [N_VALUES]


def _pack():
    one_hot = [0.0] * N_VALUES
    one_hot[value] = 1.0
    return np.array(one_hot)


def _unpack(action):
    # action: 1-D tensor of category indices, one per segment -> length 1
    global value
    value = int(action[0])
