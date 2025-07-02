from time import time

USES = [ "RepB" ]
REQUIRES = [ "RepA" ]
PROVIDES = [ "Clock" ]

def _update():

    while time() - Clock.now < 1:
        continue

    Clock.delta = time() - Clock.now
    Clock.now += Clock.delta
