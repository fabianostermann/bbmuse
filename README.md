# BlackBoard MUSic Engine

A blackboard system for realtime AI music generation.

This is ongoing research. The Github repository is currently set to private, but will be available soon.

## Installation and first-run experience

To install via PyPI, use: `pip install bbmuse`

To install locally (e.g., if cloned from github), use: `pip install -e .` (from inside this directory)

To quickly test the installation, run: `bbmuse tests/DummyProject/ --quit-after 5`

For additional usage, use: `bbmuse --help`

More example projects can be found online soon.

## TODO list

- [ ] Create and link example collection
- [ ] separate TODO list from README (maybe use github wiki?)

- [ ] think about the atomic update lock in module_handler. Maybe optional per module?

- [ ] real multiprocessing (main problem: share representations between processes -> better: threading + native libs w/o GIL like numpy)

- [ ] profiling: time for system calls vs module calculations? And startup time
- [ ] Make timing stats a command-line option?

- [x] Introduce run modes: DEBUG, NORMAL, PERFORM (NaoDevil eq.: DEBUG, DEVELOP, RELEASE)
- [x] DEBUG mode: sanity check representation (introduce an optional hook for that: `_validate()`).
- [ ] DEBUG mode: Maybe also: debug symbols/prints/draws.
- [ ] DEBUG mode: check Representations for undesired changes (using pickle)
- [x] NORMAL mode: run normally, halt on errors
- [x] PERFORM mode: no halt when exception are raised. Restart when crashed
- [x] PERFORM mode: disable gc


