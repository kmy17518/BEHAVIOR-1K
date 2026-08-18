"""Result artifacts: per-item and per-session JSON files plus console summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dex_teleop.arat.eval.scorer import ItemResult
from dex_teleop.arat.eval.session import AratSessionScorer


def make_results_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base).expanduser() / stamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def write_item_result(result: ItemResult, results_dir: Path) -> Path:
    path = Path(results_dir) / f"{result.activity}.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)
        stream.write("\n")
    return path


def write_session_result(session: AratSessionScorer, results_dir: Path) -> Path:
    path = Path(results_dir) / "session.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(session.to_dict(), stream, indent=2)
        stream.write("\n")
    return path


def format_item_summary(result: ItemResult) -> str:
    parts = [f"ARAT score {result.score}/3 for {result.activity}"]
    if result.completed and result.t_complete is not None:
        parts.append(f"completed at t={result.t_complete:.1f}s")
    elif result.completed:
        parts.append("completed (with flaw)")
    else:
        parts.append(f"not completed within {result.elapsed:.1f}s")
    if result.reasons:
        parts.append("reasons: " + ", ".join(result.reasons))
    return "; ".join(parts)


def format_session_summary(session: AratSessionScorer) -> str:
    data = session.to_dict()
    lines = ["ARAT session summary:"]
    for name, sub in data["subscales"].items():
        stopped = f" (protocol stop: {sub['stopped_reason']})" if sub["stopped_reason"] else ""
        lines.append(f"  {name}: {sub['score']}/{sub['max_score']}{stopped}")
    lines.append(f"  total: {data['total']}/{data['max_total']}")
    return "\n".join(lines)
