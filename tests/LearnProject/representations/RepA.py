# RepA — output of AgentA, input to AgentB and AgentC.
#
# Single segment, 4 categories. The simplest possible MultiDiscrete rep:
# if anything breaks here, the bug is in the core pack/unpack/spec plumbing,
# not in multi-segment handling.

value = 0

# --- bblearn ---
import numpy as np

N_VALUES = 4

# One segment of 4 categories. Sum (4) must equal len of _pack() output (4).
_action_space = [N_VALUES]


def _pack():
    one_hot = [0.0] * N_VALUES
    one_hot[value] = 1.0
    return np.array(one_hot)


def _unpack(action):
    # action: 1-D tensor of category indices, one per segment -> length 1
    global value
    value = int(action[0])
