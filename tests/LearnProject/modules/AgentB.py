# AgentB — middle of the chain: RepA -> RepB.
#
# WHAT THIS TESTS:
#   1. Input wiring. RepB.shift = (RepA.value + 1) % 4 is a pure function of
#      the input. Its CE going to ~0 proves the model actually reads RepA.
#      If it plateaus at the marginal entropy of shift instead, inputs are
#      not reaching the model (encoder order, key mismatch, ...).
#   2. Per-segment normalization. shift is deterministic (H=0), noise is
#      uniform over 3 (H=ln3). Averaged floor: (0 + ln3)/2 = 0.5493.
#      loss__RepB converging to 0.5493 confirms make_ce_loss and the floor
#      estimator share the /len(nvec) convention.
#   3. Edge lesioning. Zeroing RepA before AgentB reads it must destroy the
#      shift segment completely (it has no other information source) —
#      a binary pass/fail check that information flows along this edge.

USES = []
REQUIRES = ["RepA"]
PROVIDES = ["RepB"]

import random


def _update(bb):
    bb.RepB.shift = (bb.RepA.value + 1) % 4
    bb.RepB.noise = random.choice([0, 1, 2])
