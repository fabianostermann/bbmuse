REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def _update(bb):
    print("RepA.ValueA contains:", bb["RepA"].ValueA)
    print("RepB.StringB contains:", bb["RepB"].StringB)

    #bb["RepA"].ValueNew = "This is new (and forbidden..)"
    #print(bb["RepA"].ValueNew)
