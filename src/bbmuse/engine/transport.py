import logging

from heapq import heappush, heappop
from itertools import count
from time import monotonic

logger = logging.getLogger(__name__)

DEFAULT_TEMPO = 120.0
DEFAULT_PPQ = 960

class Transport:
    """
    The project's musical clock.

    Position is kept in ticks, at PPQ pulses per quarter note, and derived from
    the wall clock rather than accumulated per cycle, so it does not drift with
    the update rate and is correct whenever it is read. A tempo change
    re-anchors: everything before it keeps the tempo it was played at.

    In deterministic step mode the clock is virtual and advanced by a fixed
    amount per cycle instead, so a run is reproducible.
    """

    def __init__(self, tempo=DEFAULT_TEMPO, ppq=DEFAULT_PPQ, virtual=False):
        if tempo <= 0:
            raise ValueError(f"tempo must be positive, got {tempo}")
        if ppq <= 0:
            raise ValueError(f"ppq must be positive, got {ppq}")
        self._tempo = float(tempo)
        self._ppq = int(ppq)
        self._virtual = virtual

        self._anchor_tick = 0.0
        self._anchor_time = 0.0 if virtual else monotonic()
        self._virtual_seconds = 0.0
        self._running = True

    # --- position -----------------------------------------------------------
    @property
    def now(self):
        """ Current position in ticks. """
        if not self._running:
            return self._anchor_tick
        return self._anchor_tick + self.elapsed_seconds() * self.ticks_per_second()

    @property
    def seconds(self):
        """ Current position in seconds since the transport started. """
        return self.ticks_to_seconds(self.now)

    @property
    def beat(self):
        """ Current position in quarter notes. """
        return self.now / self._ppq

    def elapsed_seconds(self):
        clock = self._virtual_seconds if self._virtual else monotonic()
        return clock - self._anchor_time

    # --- tempo --------------------------------------------------------------
    @property
    def tempo(self):
        return self._tempo

    @property
    def ppq(self):
        return self._ppq

    def ticks_per_second(self):
        return self._tempo / 60.0 * self._ppq

    def ticks_to_seconds(self, ticks):
        return ticks / self.ticks_per_second()

    def seconds_to_ticks(self, seconds):
        return seconds * self.ticks_per_second()

    def set_tempo(self, bpm):
        if bpm <= 0:
            raise ValueError(f"tempo must be positive, got {bpm}")
        # re-anchor first, so the ticks already played keep the old tempo
        self._anchor_tick = self.now
        self._anchor_time = self._virtual_seconds if self._virtual else monotonic()
        self._tempo = float(bpm)
        logger.debug("Tempo set to %s BPM at tick %.1f", bpm, self._anchor_tick)

    # --- running ------------------------------------------------------------
    @property
    def is_running(self):
        return self._running

    def start(self):
        if self._running:
            return
        self._anchor_time = self._virtual_seconds if self._virtual else monotonic()
        self._running = True

    def stop(self):
        if not self._running:
            return
        self._anchor_tick = self.now
        self._running = False

    def advance_virtual(self, seconds):
        """ Step-mode only: move the virtual clock forward. """
        if not self._virtual:
            raise RuntimeError("advance_virtual() is only valid for a virtual transport.")
        self._virtual_seconds += seconds

class ModuleSchedule:
    """
    A module's own queue of things to do at a future tick.

    Each module gets its own, so two modules cannot consume each other's
    events. Sending something to another module is what the blackboard is for;
    this is for a module scheduling its own future work -- the note-off that
    belongs with a note-on, a fill that lands four bars from now.
    """

    def __init__(self, transport, owner_name=""):
        self._transport = transport
        self._owner_name = owner_name
        self._queue = []
        self._sequence = count() # keeps equal ticks in insertion order

    def at(self, tick, payload):
        """ Queue `payload` for the given absolute tick. """
        heappush(self._queue, (float(tick), next(self._sequence), payload))
        return self

    def after(self, ticks, payload):
        """ Queue `payload` that many ticks from now. """
        return self.at(self._transport.now + ticks, payload)

    def after_beats(self, beats, payload):
        return self.after(beats * self._transport.ppq, payload)

    def after_seconds(self, seconds, payload):
        return self.after(self._transport.seconds_to_ticks(seconds), payload)

    def due(self, lookahead_ticks=0):
        """
        Everything scheduled up to now (plus an optional lookahead), removed
        from the queue and returned in tick order as (tick, payload) pairs.

        The lookahead is how an output module hands events to a device early
        enough for the device to play them on time.
        """
        horizon = self._transport.now + lookahead_ticks
        ready = []
        while self._queue and self._queue[0][0] <= horizon:
            tick, _, payload = heappop(self._queue)
            ready.append((tick, payload))
        return ready

    def due_within_seconds(self, seconds):
        return self.due(self._transport.seconds_to_ticks(seconds))

    def pending(self):
        """ How many events are still queued. """
        return len(self._queue)

    def peek(self):
        """ The next (tick, payload) without removing it, or None. """
        if not self._queue:
            return None
        tick, _, payload = self._queue[0]
        return (tick, payload)

    def clear(self):
        self._queue.clear()

class _TransportView:
    """
    What a module sees as `bb.transport`: the shared clock, read-only, plus
    that module's own schedule.
    """

    def __init__(self, transport: Transport, owner_name=""):
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_schedule", ModuleSchedule(transport, owner_name))

    # the module's own queue
    def at(self, tick, payload):
        return self._schedule.at(tick, payload)

    def after(self, ticks, payload):
        return self._schedule.after(ticks, payload)

    def after_beats(self, beats, payload):
        return self._schedule.after_beats(beats, payload)

    def after_seconds(self, seconds, payload):
        return self._schedule.after_seconds(seconds, payload)

    def due(self, lookahead_ticks=0):
        return self._schedule.due(lookahead_ticks)

    def due_within_seconds(self, seconds):
        return self._schedule.due_within_seconds(seconds)

    def pending(self):
        return self._schedule.pending()

    def peek(self):
        return self._schedule.peek()

    def clear(self):
        return self._schedule.clear()

    # the shared clock
    @property
    def now(self):
        return self._transport.now

    @property
    def seconds(self):
        return self._transport.seconds

    @property
    def beat(self):
        return self._transport.beat

    @property
    def tempo(self):
        return self._transport.tempo

    @property
    def ppq(self):
        return self._transport.ppq

    @property
    def is_running(self):
        return self._transport.is_running

    def set_tempo(self, bpm):
        """ Tempo is a fact about the whole piece, so any module may set it. """
        self._transport.set_tempo(bpm)

    def ticks_to_seconds(self, ticks):
        return self._transport.ticks_to_seconds(ticks)

    def seconds_to_ticks(self, seconds):
        return self._transport.seconds_to_ticks(seconds)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"'{name}' cannot be set on the transport. Use set_tempo() to change tempo.")
