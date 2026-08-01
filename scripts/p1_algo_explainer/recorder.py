"""In-memory Gantt snapshot recorder (plan §2.2).

The Gantt plotter consumes tuple-keyed ``(job, stage, machine) -> int`` maps
directly, so disk serialization is unnecessary here. We keep an ordered list of
small snapshot records that the render helper can replay panel-by-panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

OpKey = tuple[str, str, str]  # (job, stage, machine)
HighlightKey = tuple[str, str]  # (job, stage)


@dataclass
class GanttSnapshot:
    """A single captured Gantt state."""

    label: str
    start_map: Mapping[OpKey, int]
    end_map: Mapping[OpKey, int]
    highlight: set[HighlightKey] | None = None
    axis: tuple[int, int] | None = None  # (force_start, force_end) hint
    note: str | None = None


@dataclass
class GanttSnapshotRecorder:
    """Holds an ordered list of Gantt snapshots in memory."""

    snapshots: list[GanttSnapshot] = field(default_factory=list)

    def record(
        self,
        label: str,
        start_map: Mapping[OpKey, int],
        end_map: Mapping[OpKey, int],
        *,
        highlight: set[HighlightKey] | None = None,
        axis: tuple[int, int] | None = None,
        note: str | None = None,
    ) -> GanttSnapshot:
        """Append a snapshot built from explicit start/end maps."""
        snap = GanttSnapshot(
            label=label,
            start_map=dict(start_map),
            end_map=dict(end_map),
            highlight=set(highlight) if highlight is not None else None,
            axis=axis,
            note=note,
        )
        self.snapshots.append(snap)
        return snap

    def record_schedule(self, label: str, schedule, **kw) -> GanttSnapshot:
        """Append a snapshot pulled from a HybridFlowshopLiteSchedule.

        Convenience wrapper that extracts the tuple-keyed start/end maps via
        ``schedule.get_jik_2_start_time_map()`` / ``get_jik_2_end_time_map()``.
        """
        return self.record(
            label,
            schedule.get_jik_2_start_time_map(),
            schedule.get_jik_2_end_time_map(),
            **kw,
        )
