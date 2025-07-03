from time import time

USES = [ "RepB" ]
REQUIRES = [ "RepA" ]
PROVIDES = [ "Clock" ]

def _update(bb):
    now = bb.Clock.now
    while time() - now < 1:
        continue

    bb.Clock.delta = time() - bb.Clock.now
    bb.Clock.now += bb.Clock.delta
