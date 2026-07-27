# BlackBoard MUSic Engine

BbMuse is an open-source platform-independent Python framework and backend for interactive AI systems based on the good old concept of blackboard systems.
It aims to fill the gap between ad hoc multi-agent systems that require complex manual "wiring" and monolithic neural models that act as opaque "black boxes".
By adapting a modern blackboard architecture from real-time robotics, we create a system where real-time musical composition is treated as a distributed hierarchical decision-making process.

## Links

- [<img src="https://upload.wikimedia.org/wikipedia/commons/0/04/PyPI-Logo-notext.svg" alt="" height="24"/> PyPI release](https://pypi.org/project/bbmuse)

- [<img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Github_Topicon.svg" alt="" height="24"/> GitHub repo](https://github.com/fabianostermann/bbmuse)

- [Collection of templates and examples](https://github.com/fabianostermann/bbmuse-templates)

- [<img src="https://icmc2026.ligeti-zentrum.de/wp-content/uploads/2026/04/icmc2026_logo_w-scaled.png" alt="ICMC 2026" height="24"/> conference paper](https://icmc2026.ligeti-zentrum.de/proceedings) presenting the main concept of BbMuse.
 
- [<img src="https://aimusiccreativity.org/2025/images/logo_white_short.svg" alt="AIMC 2026" height="24"/> conference paper](https://aimc2026.org/home) presenting an AI extension called BbLearn.

## Installation and first-run experience

To comfortably install via PyPI, use: `pip install bbmuse`

To install locally (e.g., if cloned from GitHub), use: `pip install -e .` (from inside the cloned directory)

To quickly test your installation, run: `bbmuse tests/DummyProject/ --quit-after 5`

For additional usage, run: `bbmuse --help`

