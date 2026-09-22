# Works at the top level, counting bars and asking for a cadence. Bottom-up
# focus means it is scheduled after the levels it reads from.
REQUIRES = [ "Pulse" ]
PROVIDES = [ "Phrase" ]

def _trigger(bb):
    return bb.Pulse.is_downbeat and bb.Pulse.beat >= 0

def _update(bb):
    bb.Phrase.bars = bb.Pulse.beat // 4 + 1
    bb.Phrase.cadence_due = (bb.Phrase.bars % 4 == 0)
