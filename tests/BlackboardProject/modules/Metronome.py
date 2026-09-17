# The only module driven by the clock: it turns transport position into a
# beat number that everything else reacts to.
PROVIDES = [ "Pulse" ]
RATE = 100

def _update(bb):
    beat = int(bb.transport.beat)
    bb.Pulse.beat = beat
    bb.Pulse.is_downbeat = (beat % 4 == 0)
