REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def _update():
    print("RepA.ValueA contains:", RepA.ValueA)
    print("RepB.StringB contains:", RepB.StringB)

    #RepA.ValueNew = "This is new (and forbidden..)"
    #print(RepA.ValueNew)

    print("REQUIRES:", REQUIRES)
