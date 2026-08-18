"""Static hand-model contracts shared by retargeting and execution adapters."""

from __future__ import annotations

from dataclasses import dataclass

from dex_teleop.types import RetargetedHandCommand


SHARPA_ACTION_JOINTS = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)


@dataclass(frozen=True)
class HandProfile:
    """Expected retargeting output and current execution support for one hand."""

    name: str
    degrees_of_freedom: int
    omnigibson_execution: bool

    def validate(self, command: RetargetedHandCommand) -> None:
        if command.hand_model != self.name:
            raise ValueError(f"Expected a {self.name} command, got {command.hand_model}")
        if len(command.joint_names) != self.degrees_of_freedom:
            raise ValueError(
                f"{self.name} must produce {self.degrees_of_freedom} joints, "
                f"got {len(command.joint_names)}"
            )
        if len(set(command.joint_names)) != len(command.joint_names):
            raise ValueError(f"{self.name} command contains duplicate joint names")


HAND_PROFILES = {
    "shadow": HandProfile("shadow", 22, False),
    "sharpa": HandProfile("sharpa", 22, True),
    "wuji": HandProfile("wuji", 20, False),
}


def get_hand_profile(hand_model: str) -> HandProfile:
    try:
        return HAND_PROFILES[hand_model.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported hand model {hand_model!r}; choose from {tuple(HAND_PROFILES)}") from error
