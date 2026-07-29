# Clock — exogenous driver of the chain. Plays the "leader/soloist" role.
#
# DO NOT ARM THIS MODULE. It has no inputs, and cloning a module with empty
# input_dims would break ModuleClone (torch.cat over an empty encoder list).
# It exists only so that AgentA has a moving, fully-observable input.
#
# Deterministic cycle 0..3 -> reproducible runs, uniform phase visitation
# (assumed by AgentA's analytic floor). For a stochastic leader later
# (perturbation battery / scrambled-input null), replace the increment with:
#     bb.Phase.value = random.randrange(4)

USES = []
REQUIRES = []
PROVIDES = ["Phase"]


def _update(bb):
    bb.Phase.value = (bb.Phase.value + 1) % 4
