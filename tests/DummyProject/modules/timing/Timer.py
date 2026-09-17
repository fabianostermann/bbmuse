from time import time

# RepA comes from another control group and RepB would close a cycle through
# ModB, so both are read as of the previous cycle
DELAYED = [ "RepA", "RepB" ]
PROVIDES = [ "Clock" ]
RATE = 1   # the group schedules the tick; no busy-wait needed

def _update(bb):
    bb.Clock.delta = time() - bb.Clock.now
    bb.Clock.now += bb.Clock.delta
