# Schedules its own note-offs rather than tracking them by hand, and reads the
# queue back with a lookahead the way a real device driver would.
REQUIRES = [ "Chord" ]
PROVIDES = [ "Played" ]
RATE = 100

LOOKAHEAD_TICKS = 24
last_pitches = None

def _trigger(bb):
    return bb.Chord.pitches != last_pitches or bb.transport.pending() > 0

def _update(bb):
    global last_pitches
    if bb.Chord.pitches != last_pitches:
        last_pitches = list(bb.Chord.pitches)
        for pitch in last_pitches:
            bb.transport.after(0, ("on", pitch))
            bb.transport.after_beats(0.5, ("off", pitch))

    for tick, event in bb.transport.due(LOOKAHEAD_TICKS):
        bb.Played.log.append((round(tick), event))
