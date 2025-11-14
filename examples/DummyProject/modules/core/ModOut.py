GROUP = "output-group"
REQUIRES = [ "BibRep" ]
PROVIDES = [ "RepOut1", "RepOut2", ]    

from time import sleep

def _update(bb):
    print("I am providing from a special group thread.")
    sleep(0.1)
    #print("I should not be allowed to do this: clock.now =", bb["Clock"].now)