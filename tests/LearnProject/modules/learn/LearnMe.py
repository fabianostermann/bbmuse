USES = [ "UsedRep" ]
REQUIRES = [ "ReqRep" ]
PROVIDES = [ "ProvRep", "UsedRep" ]

import time
import random

def _update(bb):
    print("Updating everything..")

    bb.ProvRep.valueA = random.uniform(-2, 2)
    bb.ProvRep.valueB = random.randint(-10, 10)

    bb.UsedRep.valueA = random.uniform(-2, 2)
    bb.UsedRep.valueB = random.randint(-10, 10)

    time.sleep(0.1)