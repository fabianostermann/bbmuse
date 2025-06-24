from time import time

class Timer():
    requires = []
    provides = [ "Clock" ]

    def update(bb):
        theClock = bb["Clock"]

        while time() - theClock.now < 1:
            continue

        theClock.delta = time() - theClock.now
        theClock.now += theClock.delta
