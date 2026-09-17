GROUP = "special"
REQUIRES = [ "RepConfig" ]
DELAYED = [ "Clock" ]   # provided by Timer in the 'default' group
PROVIDES = [ "RepA" ]

import time

internal_var = 0

def _update(bb):
    time.sleep(0.2)
    #bb.RepA = "Overwriting blackboard entries should be forbidden."

    bb.RepA.ValueA += 2
    print("Modified RepA.ValueA (+=2):", bb.RepA.ValueA)

    global internal_var
    internal_var += 5
    print("internal_var =", internal_var)
