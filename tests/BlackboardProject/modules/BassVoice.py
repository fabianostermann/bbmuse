# A second contributor to Chord. It never sees what the Harmoniser wrote:
# both write into private copies and Chord._merge() decides.
REQUIRES = [ "Pulse" ]
PROVIDES = [ "Chord" ]

last_beat = None

def _trigger(bb):
    return bb.Pulse.beat != last_beat

def _update(bb):
    global last_beat
    last_beat = bb.Pulse.beat
    bb.Chord.pitches = [ 36 + (bb.Pulse.beat % 4) * 2 ]
