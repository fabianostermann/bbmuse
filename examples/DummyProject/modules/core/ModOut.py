REQUIRES = [ "BibRep" ]
PROVIDES = [ "RepOut1", "RepOut2", ]    

def _update(bb):
    print("I am providing outside of the main loop without requirements!")

    #print("I should not be allowed to do this: clock.now =", bb["Clock"].now)