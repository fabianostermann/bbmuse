# parallel_coordination — reward for the PARALLEL cells of the 2x2.
#
# +1 when RepB.noise == RepD.value. Neither agent observes the other, so
# this can only be maximized by converging on a shared convention f(a)
# (both read RepA). See AgentD.py for the analytic values: baseline exactly
# 1/3, optimum 1.0, exactly 9 minimal-KL conventions, per-agent KL costs
# ln3 (B) and (11/24)*ln3 (D).
#
# What to watch beyond mean reward: WHICH convention gets locked (log the
# learned f(a) per agent), whether sequential oscillates between phases,
# and whether joint-without-critic chases (both moving targets).
#
# IMPORTANT: the rewards directory is loaded globally -- every *.py in it is
# active during any sculpt run, and multiple rewards get averaged into one
# advantage signal. Keep ONLY the reward file for the current experiment in
# the rewards dir (move the others out, e.g. to rewards/disabled/).


def _reward(bb):
    return 1.0 if bb.RepB.noise == bb.RepD.value else 0.0
