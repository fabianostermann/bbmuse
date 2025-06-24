class ModA():
    requires = [ "Clock" ]
    provides = [ "RepA" ]
        
    def update(bb):
        bb["RepA"].ValueA += 2