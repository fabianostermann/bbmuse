# UsedRep is both read and written here, so it has to be read as of the
# previous cycle -- otherwise the module would depend on its own output
DELAYED  = [ "UsedRep" ]
REQUIRES = [ "ReqRep" ]
PROVIDES = [ "ProvRep", "UsedRep" ]
RATE = 10

import random

def _update(bb):
    print("Updating everything..")

    bb.ProvRep.valueA = random.uniform(-2, 2)
    bb.ProvRep.valueB = random.randint(-10, 10)

    bb.UsedRep.valueA = random.uniform(-2, 2)
    bb.UsedRep.valueB = random.randint(-10, 10)