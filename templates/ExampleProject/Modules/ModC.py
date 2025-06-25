REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def update(bb):
    print(bb["RepA"].ValueA)
    print(bb["RepB"].StringB)
