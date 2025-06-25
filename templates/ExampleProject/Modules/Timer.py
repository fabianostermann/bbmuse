from time import time

REQUIRES = []
PROVIDES = [ "Clock" ]

def _update(bb):
    theClock = bb["Clock"]

    while time() - theClock.now < 1:
        continue

    theClock.delta = time() - theClock.now
    theClock.now += theClock.delta
