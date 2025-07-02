USES = [ "Clock" ]
PROVIDES = [ "RepA" ]

internal_var = 0

def _update():
    RepA.ValueA += 2

    global internal_var
    internal_var += 5
    print("internal_var =", internal_var)

