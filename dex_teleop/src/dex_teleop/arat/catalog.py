"""ARAT BDDL activity catalog and subscale expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


CATALOG_PATH = Path(__file__).with_name("tasks.yaml")


@dataclass(frozen=True)
class AratTask:
    activity: str
    subscale: str
    label: str
    layout: str
    instances: dict[str, str]


class AratTaskCatalog:
    def __init__(self, path: Path = CATALOG_PATH) -> None:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        placeholder_goals = raw.get("placeholder_goals")
        if not isinstance(placeholder_goals, bool):
            raise ValueError("ARAT task catalog must explicitly state whether its goals are placeholders")
        self.scene = str(raw["scene"])
        self.placeholder_goals = placeholder_goals
        self.tasks = {
            activity: AratTask(
                activity=activity,
                subscale=str(data["subscale"]),
                label=str(data["label"]),
                layout=str(data["layout"]),
                instances=dict(data["instances"]),
            )
            for activity, data in raw["tasks"].items()
        }
        self.subscales = {name: tuple(activities) for name, activities in raw["subscales"].items()}
        self._validate()

    def resolve(self, task: str | None, subscale: str | None) -> tuple[AratTask, ...]:
        if (task is None) == (subscale is None):
            raise ValueError("Specify exactly one of task or subscale")
        if task is not None:
            try:
                return (self.tasks[task],)
            except KeyError as error:
                raise ValueError(f"Unknown ARAT task {task!r}; choose from {tuple(self.tasks)}") from error
        assert subscale is not None
        try:
            return tuple(self.tasks[activity] for activity in self.subscales[subscale.lower()])
        except KeyError as error:
            raise ValueError(f"Unknown ARAT subscale {subscale!r}; choose from {tuple(self.subscales)}") from error

    def _validate(self) -> None:
        listed = [activity for activities in self.subscales.values() for activity in activities]
        if len(listed) != len(set(listed)):
            raise ValueError("An ARAT activity appears in more than one subscale")
        if set(listed) != set(self.tasks):
            raise ValueError("ARAT subscale membership does not exactly cover the task catalog")
        expected_counts = {"grasp": 6, "grip": 4, "pinch": 6, "gross_movement": 3}
        actual_counts = {name: len(activities) for name, activities in self.subscales.items()}
        if actual_counts != expected_counts:
            raise ValueError(f"Expected ARAT task counts {expected_counts}, got {actual_counts}")
        layouts = {task.layout for task in self.tasks.values()}
        if len(layouts) != 13:
            raise ValueError(f"Expected 13 physical layouts, got {len(layouts)}")
