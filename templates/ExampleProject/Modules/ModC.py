REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def _update(bb):
    print(bb["RepA"].ValueA)
    print(bb["RepB"].StringB)

    bb["RepA"].ValueNew = "This is new (and forbidden..)"
    print(bb["RepA"].ValueNew)
