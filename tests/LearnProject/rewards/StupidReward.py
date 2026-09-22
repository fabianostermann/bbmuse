# A reward judges the whole system, so it can read any representation on the
# blackboard -- not only the ones the module under test declared.
def _reward(bb):
    return -abs(bb.ProvRep.valueA - bb.ReqRep.valueA)
