# BlackBoard MUSic Engine

BbMuse is an open-source platform-independent Python framework and backend for interactive AI systems based on the good old concept of blackboard systems.
It aims to fill the gap between ad hoc multi-agent systems that require complex manual "wiring" and monolithic neural models that act as opaque "black boxes".
By adapting a modern blackboard architecture from real-time robotics, we create a system where real-time musical composition is treated as a distributed hierarchical decision-making process.

## Links

- [<img src="https://upload.wikimedia.org/wikipedia/commons/0/04/PyPI-Logo-notext.svg" alt="" height="24"/> PyPI release](https://pypi.org/project/bbmuse)

- [<img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Github_Topicon.svg" alt="" height="24"/> GitHub repo](https://github.com/fabianostermann/bbmuse)

- [Collection of templates and examples](https://github.com/fabianostermann/bbmuse-templates)

- [<img src="https://icmc2026.ligeti-zentrum.de/wp-content/uploads/2026/04/icmc2026_logo_w-scaled.png" alt="ICMC 2026" height="24"/> conference paper](https://icmc2026.ligeti-zentrum.de/proceedings) introducing BbMuse and its main concepts.
 
- [<img src="https://aimusiccreativity.org/2025/images/logo_white_short.svg" alt="AIMC 2026" height="24"/> conference paper](https://aimc2026.org/home) presenting BbLearn: an AI extension for BbMuse living in the same repo.

## Installation and first-run experience

To comfortably install via PyPI, use: `pip install bbmuse`

To install locally (e.g., if cloned from GitHub), use: `pip install -e .` (from inside the cloned directory)

To quickly test your installation, run: `bbmuse tests/DummyProject/ --quit-after 5`

For additional usage, run: `bbmuse --help`

## The execution model

A **module** is a Python file with an `_update(bb)` function and a handful of
optional declarations. A **representation** is a Python file holding state.
The engine reads the declarations, works out the order in which modules must
run, and calls them in a loop.

```python
PROVIDES = [ "Harmony" ]   # written by this module; exactly one module may provide each
REQUIRES = [ "Meter" ]     # read, and this module runs after whoever provides it
DELAYED  = [ "Melody" ]    # read as of the previous cycle, via bb.prev.Melody
GROUP    = "composition"   # which control group (thread) this module belongs to
RATE     = 20              # updates per second; omit to run as often as possible
```

**Ordering.** `REQUIRES` is what orders modules, and it only orders them
*within a control group*. Groups are separate threads, so an edge that crosses
a group boundary gives no ordering and no one-to-one pairing -- the consumer
sees whichever value happens to be there. The build warns about every such
edge, and `--mode DEBUG` refuses to start.

**Cycles.** A cycle built from `REQUIRES` is rejected. Break it with `DELAYED`,
which reads a snapshot taken before the cycle began: the same value for every
module in the group, unaffected by anything written during the cycle. That is
also the right way to read across a group boundary. `USES` is the older,
unordered live read; it still works but what it returns is not well defined,
so prefer `DELAYED`.

**Rates.** Declare `RATE` rather than sleeping inside `_update()`. The group
schedules due modules and sleeps holding nothing, so an infrequent module costs
nothing and does not hold anything up. Timing statistics report the budget and
any overruns against it.

**Concurrency.** Each representation carries its own lock, taken for the
duration of an update in a fixed order. Modules whose representations do not
overlap run in parallel; modules that share one serialise.

**Determinism.** `bbmuse <dir> --steps N --seed S` runs the project
single-threaded for exactly N cycles with the random sources seeded, so runs
are reproducible and a project can be tested. `project.step(n_cycles, seed=...)`
is the same thing from Python.

**Checking.** `--mode DEBUG` additionally calls `_validate()` on every
representation after it is written, and reports any module that mutates a
representation it only declared as read-only.

