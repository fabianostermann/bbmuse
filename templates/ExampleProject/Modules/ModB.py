REQUIRES = [ "Clock" ]
PROVIDES = [ "RepB" ]
    
def _update(bb):
    bb["RepB"].StringB += "!"