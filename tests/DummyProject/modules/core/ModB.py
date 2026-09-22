REQUIRES = [ "Clock" ]
PROVIDES = [ "RepB" ]
RATE = 2
    
def _update(bb):
    bb.RepB.StringB += "!"
    print("Modified RepB.StringB (+='!'):", bb.RepB.StringB)