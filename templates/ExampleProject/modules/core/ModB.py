REQUIRES = [ "Clock" ]
PROVIDES = [ "RepB" ]
    
def _update(bb):
    bb.RepB.StringB += "!"
    print("Modified RepB.StringB (+='!'):", bb.RepB.StringB)