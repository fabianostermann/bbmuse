REQUIRES = []
PROVIDES = [ "RepOut1", "RepOut2", ]    

ASDF = ["test"]

def _update(bb):
    print("I am providing out of the loop!")

    #print("I should not be allowed to do this: clock.now =", bb["Clock"].now)