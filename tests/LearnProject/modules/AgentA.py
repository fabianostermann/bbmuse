# AgentA — first armed agent in the chain: Phase -> RepA.
#
# WHAT THIS TESTS: compute_entropy_floor's grouping-by-state logic.
# The support (and therefore the entropy) of the choice differs per phase:
#
#     phase 0: {0,1,2,3}  -> H = ln4 = 1.3863
#     phase 1: {0,2}      -> H = ln2 = 0.6931
#     phase 2: {1}        -> H = 0        (deterministic)
#     phase 3: {0,1,3}    -> H = ln3 = 1.0986
#
# With uniform phase visitation the mean floor is 0.7945. If the estimator
# reports per-group entropies [1.3863, 0.6931, 0, 1.0986], grouping works.
# A state-blind estimator would instead report the marginal entropy of RepA.
#
# Cloning check: loss__RepA should converge to ~0.7945 and kl__RepA to ~0.

USES = []
REQUIRES = ["Phase"]
PROVIDES = ["RepA"]

import random

SUPPORTS = {
    0: [0, 1, 2, 3],
    1: [0, 2],
    2: [1],
    3: [0, 1, 3],
}


def _update(bb):
    bb.RepA.value = random.choice(SUPPORTS[bb.Phase.value])
