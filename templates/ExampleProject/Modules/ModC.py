REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def _update(bb):
    print(bb["RepA"].ValueA)
    print(bb["RepB"].StringB)
