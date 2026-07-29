TODO: Warning -> Not yet finished nor tested!

class RewardCollector:
    """Evaluates all rewards once per timestep, after every module has written."""

    def __init__(self, project, rewards):
        self.project = project
        self.rewards = rewards
        self.buffer = {}          # reward_name -> list of scalars
        self._active = False

    def activate(self, last_handler):
        self._original = last_handler.call_update
        original, collector = self._original, self

        def wrapped(bb):
            result = original(bb)
            collector._collect()   # after the final module of the cycle
            return result

        last_handler.call_update = wrapped
        self._handler = last_handler
        self._active = True

    def _collect(self):
        bb = self.project.get_blackboard()      # full scope, all reps
        for reward in self.rewards:
            name = reward.get_name()
            self.buffer.setdefault(name, []).append(reward.call_reward(bb))

    def flush(self):
        out = {f"rewards__{k}": list(v) for k, v in self.buffer.items()}
        self.buffer.clear()
        return out