USES = [ "Clock" ]
PROVIDES = [ "RepA" ]

internal_var = 0

def _update(bb):
    #bb.RepA = "Overwriting blackboard entries should be forbidden."

    bb.RepA.ValueA += 2
    print("Modified RepA.ValueA (+=2):", bb.RepA.ValueA)

    global internal_var
    internal_var += 5
    print("internal_var =", internal_var)