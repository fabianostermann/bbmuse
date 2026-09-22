GROUP = "output-group"
DELAYED = [ "BibRep" ]  # constant, provided by Init in the 'default' group
PROVIDES = [ "RepOut1", "RepOut2", ]    
RATE = 10

def _update(bb):
    print("I am providing from a special group thread.")
    #print("I should not be allowed to do this: clock.now =", bb["Clock"].now)