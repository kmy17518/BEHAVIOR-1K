"""ARAT session protocol: item order, short-circuit rules, subscale aggregation.

Per the scoring guide's test order:
- If the first item of a subscale scores 3, the whole subscale is credited its maximum
  and no further items are administered (remaining items imputed 3).
- Otherwise the second item is administered; if it scores 0, the subscale scores 0 and
  no further items are administered.
- The gross-movement subscale additionally ends with 0 if its first item scores 0.

The raw sum of administered items is also reported alongside the guide-literal subscale
score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ITEM_MAX_SCORE = 3
SUBSCALE_MAX_SCORES = {"grasp": 18, "grip": 12, "pinch": 18, "gross_movement": 9}
SESSION_MAX_SCORE = sum(SUBSCALE_MAX_SCORES.values())


@dataclass
class SubscaleState:
    subscale: str
    activities: tuple[str, ...]
    scores: dict[str, int] = field(default_factory=dict)
    stopped_reason: str | None = None

    @property
    def administered(self) -> int:
        return len(self.scores)

    @property
    def finished(self) -> bool:
        return self.stopped_reason is not None or self.administered == len(self.activities)


class AratSessionScorer:
    """Tracks one or more subscales through the ARAT administration protocol."""

    def __init__(self, subscales: dict[str, tuple[str, ...]]):
        for name in subscales:
            if name not in SUBSCALE_MAX_SCORES:
                raise ValueError(f"Unknown ARAT subscale {name!r}")
        self._states = {
            name: SubscaleState(subscale=name, activities=tuple(activities))
            for name, activities in subscales.items()
        }

    def record(self, activity: str, score: int) -> None:
        state = self._state_for(activity)
        expected = state.activities[state.administered]
        if activity != expected:
            raise ValueError(f"Out-of-order item {activity!r}; expected {expected!r}")
        if not 0 <= score <= ITEM_MAX_SCORE:
            raise ValueError(f"Score {score} out of range for {activity!r}")
        state.scores[activity] = score

        index = state.administered - 1
        if index == 0 and score == ITEM_MAX_SCORE:
            state.stopped_reason = "first_item_max"
        elif index == 0 and score == 0 and state.subscale == "gross_movement":
            state.stopped_reason = "first_item_zero"
        elif index == 1 and score == 0:
            state.stopped_reason = "second_item_zero"

    def should_administer(self, activity: str) -> bool:
        state = self._state_for(activity)
        if state.finished:
            return False
        return state.activities[state.administered] == activity

    def subscale_score(self, subscale: str) -> int:
        state = self._states[subscale]
        if state.stopped_reason == "first_item_max":
            return SUBSCALE_MAX_SCORES[subscale]
        if state.stopped_reason in ("first_item_zero", "second_item_zero"):
            return 0
        return sum(state.scores.values())

    def imputed_scores(self, subscale: str) -> dict[str, int | None]:
        """Per-activity scores including protocol-imputed values for skipped items."""
        state = self._states[subscale]
        imputed_value: int | None
        if state.stopped_reason == "first_item_max":
            imputed_value = ITEM_MAX_SCORE
        elif state.stopped_reason in ("first_item_zero", "second_item_zero"):
            imputed_value = 0
        else:
            imputed_value = None
        return {
            activity: state.scores.get(activity, imputed_value)
            for activity in state.activities
        }

    def total_score(self) -> int:
        return sum(self.subscale_score(name) for name in self._states)

    def to_dict(self) -> dict:
        subscales = {}
        for name, state in self._states.items():
            subscales[name] = {
                "score": self.subscale_score(name),
                "max_score": SUBSCALE_MAX_SCORES[name],
                "administered": dict(state.scores),
                "imputed": self.imputed_scores(name),
                "stopped_reason": state.stopped_reason,
                "raw_sum_of_administered": sum(state.scores.values()),
            }
        return {
            "subscales": subscales,
            "total": self.total_score(),
            "max_total": sum(SUBSCALE_MAX_SCORES[name] for name in self._states),
        }

    def _state_for(self, activity: str) -> SubscaleState:
        for state in self._states.values():
            if activity in state.activities:
                return state
        raise ValueError(f"Activity {activity!r} is not part of this session")
