# prefer_c_zero — the sculpting reward for the chain_debug scenario.
#
# +1 when RepC == 0. The symbolic AgentC outputs 0 only on the fair coin in
# "even" states (P(RepC=0) = 35/144 ~ 0.24 overall), so there is real room
# to improve — and the improvement must come from shifting a distribution,
# not from copying a label. See AgentC.py for the two-regime analysis and
# the closed-form optimum (valid only with unnormalized advantages).
#
# Caveat for compute_advantages: if the sculpted policy saturates (always 0
# or always 1 within a rollout), this reward becomes constant and its
# per-episode std ~ 0 — exercising exactly the divide-by-noise guard
# discussed for compute_advantages. That is intentional: this toy should
# trigger that edge case on purpose rather than leaving it for the music
# domain to find.

_weight = 0.5

def _reward(bb):
    return 1.0 if bb.RepC.value == 0 else 0.0
