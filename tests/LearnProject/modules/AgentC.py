# AgentC — end of the chain: (RepA, RepB) -> RepC. Multi-input agent and
# the sculpting target.
#
# Rule: if (RepA.value + RepB.noise) is even, flip a fair coin;
#       otherwise deterministically output 1.
#
# WHAT THIS TESTS:
#   1. Multi-input conditioning. C's entropy depends on both upstream reps.
#      Expected floor: P(even) * ln2 = (35/72) * ln2 = 0.3369  (not 0.3466!
#      AgentA's phase-dependent supports make P(RepA even) = 11/24).
#      Simulation-verified: 0.3370 over 2M steps.
#   2. Sculpting, in two distinct regimes at once (reward prefers RepC == 0):
#        even states: symbolic is 50/50   -> reward should SHIFT the
#                     distribution toward 0; with an UNNORMALIZED advantage
#                     scale the KL-regularized optimum is closed-form:
#                     pi*(0) = e^(r/beta) / (e^(r/beta) + 1),  beta = bc_coef
#        odd states:  symbolic is certain (always 1) -> reward pushes
#                     AGAINST a deterministic anchor; how far it bends
#                     measures anchor strength where the expert is confident.
#      NOTE: compute_advantages currently standardizes returns, which
#      destroys the absolute reward scale — the closed-form check needs an
#      unnormalized mode. The qualitative signature (even states shift,
#      odd states resist) is testable either way.

USES = []
REQUIRES = ["RepA", "RepB"]
PROVIDES = ["RepC"]

import random


def _update(bb):
    if (bb.RepA.value + bb.RepB.noise) % 2 == 0:
        bb.RepC.value = random.choice([0, 1])
    else:
        bb.RepC.value = 1
