LEVEL = "harmony"

# Two modules contribute to this one, so it arbitrates between them rather
# than letting whichever ran last win.
pitches = []
votes = {}

def _merge(contributions):
    global pitches, votes
    votes = {name: sorted(c.pitches) for name, c in contributions.items()}
    pitches = sorted({p for voted in votes.values() for p in voted})

def _validate():
    assert all(isinstance(p, int) for p in pitches), "pitches must be midi note numbers"
    assert pitches == sorted(set(pitches)), "pitches must be sorted and unique"
