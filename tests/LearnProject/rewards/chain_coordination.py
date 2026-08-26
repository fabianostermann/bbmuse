# chain_coordination — reward for the CHAIN cells of the 2x2.
#
# +1 when RepA.value == RepC.value. C observes RepA (directly), so C can
# best-respond to any A policy: sequential learning here is exact
# best-response on a chain and is predicted to match joint learning.
# This is the control cell against which the parallel cells are read.
#
# ANALYTIC LEVELS (RepC is binary, so agreement is only possible for a in {0,1}):
#   both symbolic:                      reward = 0.4201
#   C best-responds, A frozen symbolic: reward = 2/3
#     (C copies a whenever a in {0,1}; a=1 in phase 2 is guaranteed, etc.)
#   A also adapts (shifts mass into {0,1} where its support allows):
#     reward = 1.0, at mean KL cost for A of
#       [ln2, ln2, 0, ln(3/2)] / 4 = 0.4479
#     (phase 1 is the sharp case: support {0,2} -> A must go deterministic 0)
#
# IMPORTANT: keep only ONE reward file in the rewards dir per experiment --
# all *.py files there are loaded and averaged (see parallel_coordination.py).

_weight = 0.0

def _reward(bb):
    return 1.0 if bb.RepA.value == bb.RepC.value else 0.0
