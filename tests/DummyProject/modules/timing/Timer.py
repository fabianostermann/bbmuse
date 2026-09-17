from time import time

# RepA comes from another control group and RepB would close a cycle through
# ModB, so both are read as of the previous cycle
DELAYED = [ "RepA", "RepB" ]
PROVIDES = [ "Clock" ]

def _update(bb):
    now = bb.Clock.now
    while time() - now < 1:
        continue

    bb.Clock.delta = time() - bb.Clock.now
    bb.Clock.now += bb.Clock.delta
