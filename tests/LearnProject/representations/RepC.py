# RepC — output of AgentC, end of the chain. Also the sculpting target
# (see rewards/prefer_c_zero.py).
#
# Single binary segment whose conditional entropy depends on BOTH upstream
# agents (RepA and RepB.noise). Expected floor is NOT ln(2)/2 = 0.3466 but
#
#     P(RepA.value + RepB.noise even) * ln2 = (35/72) * ln2 = 0.3369
#
# because AgentA's phase-dependent supports make P(RepA even) = 11/24, not 1/2.
# Verified by simulation (2M steps: 0.3370). If the measured floor lands at
# 0.3466 instead, the model/estimator is not conditioning on the full input.

value = 0

# --- bblearn ---
import numpy as np

N_VALUES = 2

# One binary segment. Sum (2) must equal len of _pack() output (2).
_action_space = [N_VALUES]


def _pack():
    one_hot = [0.0] * N_VALUES
    one_hot[value] = 1.0
    return np.array(one_hot)


def _unpack(action):
    # action: 1-D tensor of category indices, one per segment -> length 1
    global value
    value = int(action[0])
