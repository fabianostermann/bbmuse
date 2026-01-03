REQUIRES = [ "RepA", "RepB" ]
PROVIDES = []

def _update(bb):
    print("RepA.ValueA contains:", bb.RepA.ValueA)
    print("RepB.StringB contains:", bb.RepB.StringB)

    print("RepA.square_me(RepA.ValueA):", bb.RepA.square_me(bb.RepA.ValueA))

    bb.RepA.AList.append(99)
    print("This is a problem. RepA.AList was altered:", bb.RepA.AList)

    #bb.RepA.ValueA = "Settings values on required representations is forbidden."
    #bb.RepA.ValueNew = "Creating values on representations is always forbidden."
    #bb.RepA = "Overwriting blackboard entries is forbidden."
    bb = "Overwriting the local variable 'bb' is possible but makes no sense."
    print(bb)
