# AgentD — PARALLEL SIBLING of AgentB: RepA -> RepD.
#
# B and D both read RepA; NEITHER reads the other. This is the structure the
# chain scenario lacks: a genuinely simultaneous pair with no observation
# path between them. Under the parallel coordination reward
# (+1 if RepB.noise == RepD.value), neither agent can best-respond to the
# other's current action — they must converge on a shared convention f(a)
# through the common reward signal alone. That is the case where sequential
# (round-robin) and joint learning are predicted to differ.
#
# Symbolic rule:
#     RepA.value even -> uniform over {0, 1, 2}   (H = ln3)
#     RepA.value odd  -> deterministic RepA.value % 3   (H = 0)
#                        (a=1 -> 1, a=3 -> 0)
#
# ANALYTIC VALUES (simulation-verified, 2M steps):
#   Visitation under symbolic A: P(a) = [13, 19, 9, 7] / 48, P(a even) = 11/24.
#   Cloning floor for RepD:  (11/24) * ln3 = 0.5035
#   Baseline coordination reward (both symbolic): exactly 1/3.
#   Full coordination = 1.0 via any shared f(a) in {0,1,2}^4 -> but the odd-a
#   anchor makes f(1)=1 and f(3)=0 the only finite-KL choices, leaving exactly
#   3 x 3 = 9 minimal-cost conventions (free choice at a=0 and a=2).
#   Minimal mean KL at full coordination:
#       AgentB (noise):  ln3          = 1.0986   (uniform -> f(a), every state)
#       AgentD:          (11/24)*ln3  = 0.5035   (even states only)
#   Which of the 9 conventions the learners lock onto -- and whether
#   sequential and joint lock onto one at all -- is the experiment.
#
# NOTE: the chain floors (AgentA 0.7945, AgentB 1.0986, AgentC 0.3368) are
# unchanged by adding this module: nothing upstream of them is modified and
# nothing reads RepD.

USES = []
REQUIRES = ["RepA"]
PROVIDES = ["RepD"]

import random


def _update(bb):
    a = bb.RepA.value
    if a % 2 == 0:
        bb.RepD.value = random.choice([0, 1, 2])
    else:
        bb.RepD.value = a % 3
