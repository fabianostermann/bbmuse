# BlackBoard MUSic Engine

BbMuse is an open-source platform-independent Python framework and backend for interactive AI systems based on the good old concept of blackboard systems.
It aims to fill the gap between ad hoc multi-agent systems that require complex manual "wiring" and monolithic neural models that act as opaque "black boxes".
By adapting a modern blackboard architecture from real-time robotics, we create a system where real-time musical composition is treated as a distributed hierarchical decision-making process.

## Links

- [<img src="https://upload.wikimedia.org/wikipedia/commons/0/04/PyPI-Logo-notext.svg" alt="" height="24"/> PyPI release](https://pypi.org/project/bbmuse)

- [<img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Github_Topicon.svg" alt="" height="24"/> GitHub repo](https://github.com/fabianostermann/bbmuse)

- Templates and example projects will be released in another online repository soon. Link will be provided here.

- A corresponding paper was recently accepted for the [International Computer Music Conference (ICMC)](https://ligeti-zentrum.de/icmc-2026/). Link will be provided soon.

## Installation and first-run experience

To comfortably install via PyPI, use: `pip install bbmuse`

To install locally (e.g., if cloned from GitHub), use: `pip install -e .` (from inside the cloned directory)

To quickly test your installation, run: `bbmuse tests/DummyProject/ --quit-after 5`

For additional usage, run: `bbmuse --help`

