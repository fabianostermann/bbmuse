REQUIRES = [ "Clock" ]
PROVIDES = [ "RepA" ]
    
def _update(bb):
    bb["RepA"].ValueA += 2