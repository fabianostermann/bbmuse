# A knowledge source: it only has something to say when the beat changes, so
# it says so rather than running every cycle and checking.
REQUIRES = [ "Pulse" ]
PROVIDES = [ "Chord" ]

PROGRESSION = [ [60, 64, 67], [57, 60, 64], [65, 69, 72], [55, 59, 62] ]

last_beat = None

def _trigger(bb):
    return bb.Pulse.beat != last_beat

def _update(bb):
    global last_beat
    last_beat = bb.Pulse.beat
    bb.Chord.pitches = PROGRESSION[bb.Pulse.beat % len(PROGRESSION)]
