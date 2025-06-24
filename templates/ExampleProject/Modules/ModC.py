class ModC():
    requires = [ "RepA", "RepB" ]
    provides = []
        
    def update(bb):
        print(bb["RepA"].ValueA)
        print(bb["RepB"].StringB)