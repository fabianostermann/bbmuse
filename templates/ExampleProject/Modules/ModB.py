class ModB():
    requires = [ "Clock" ]
    provides = [ "RepB" ]
        
    def update(bb):
        bb["RepB"].StringB += "!"