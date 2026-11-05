from collections.abc import Iterable


class ScriptedLLMClient:
    """Returns or raises the given outcomes in order, and records each prompt."""

    def __init__(self, outcomes: Iterable[str | Exception]):
        self._outcomes = iter(outcomes)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
