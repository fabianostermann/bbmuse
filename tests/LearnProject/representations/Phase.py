# Phase — exogenous clock state, written by the Clock module (NOT armed).
#
# Input-only rep: it is REQUIRED by AgentA but never PROVIDED by an armed
# module, so it needs _pack() but no _unpack() and no _action_space.
#
# Cycles deterministically 0 -> 1 -> 2 -> 3 -> 0, so all four phases are
# visited equally often. That uniform visitation is assumed by the analytic
# entropy floor of AgentA: (ln4 + ln2 + 0 + ln3)/4 = 0.7945.

value = 0

# --- bblearn ---
import numpy as np

N_PHASES = 4


def _pack():
    one_hot = [0.0] * N_PHASES
    one_hot[value] = 1.0
    return np.array(one_hot)
