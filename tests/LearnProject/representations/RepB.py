# RepB — output of AgentB, input to AgentC.
#
# TWO segments with DIFFERENT entropies:
#   shift (4 categories): deterministic function of RepA  -> CE should go to ~0
#   noise (3 categories): uniform random                  -> CE floor ln(3)
#
# This is the rep that tests per-segment consistency: the averaged floor
# (0 + ln3)/2 = 0.5493 only comes out right if make_ce_loss's /len(nvec)
# normalization and the floor estimator use the same convention.
# A mismatch shows up as 1.0986 (no averaging) or 0.2747 (double averaging).
#
# Segment order in _pack() concatenation MUST match _action_space order.

shift = 0  # segment 0: (RepA.value + 1) % 4, deterministic
noise = 0  # segment 1: uniform over {0, 1, 2}

# --- bblearn ---
import numpy as np

N_SHIFT = 4
N_NOISE = 3

# Two segments: [4, 3]. Sum (7) must equal len of _pack() output (4 + 3 = 7).
_action_space = [N_SHIFT, N_NOISE]


def _pack():
    shift_one_hot = [0.0] * N_SHIFT
    shift_one_hot[shift] = 1.0

    noise_one_hot = [0.0] * N_NOISE
    noise_one_hot[noise] = 1.0

    return np.array(shift_one_hot + noise_one_hot)


def _unpack(action):
    # action: 1-D tensor of category indices, one per segment -> length 2
    global shift, noise
    shift = int(action[0])
    noise = int(action[1])
