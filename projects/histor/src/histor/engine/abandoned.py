"""Abandoned objects: a bag or suitcase that stays put with nobody near it.

For each tracked object of `classes` (COCO: backpack, handbag, suitcase by default):

    still       its centre stays within `still_ratio` of its own size from where it stopped
    unattended  no person's centre within `owner_distance` person-heights of it (a person
                sitting next to their bag attends it; one walking past for a moment resets
                the clock too)
    event       `abandoned` (rule `abandoned_object`) once the object has been both still and
                unattended for `seconds`; once per object

A bag carried by someone moves with them, so it is never still; set down and left, it fires
after `seconds`. Optionally only inside the named areas (`zones`, from the `zones` section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from histor.engine.tracker import Track
    from histor.engine.zones import Zone

EVENT_TYPE = "abandoned"
RULE = "abandoned_object"
CLASSES = ("backpack", "handbag", "suitcase")


@dataclass
class _State:
    anchor: tuple[float, float]
    still_since: float
    alone_since: float | None
    fired: bool = False


@dataclass
class AbandonedEvent:
    time: float
    track_id: int
    label: str
    seconds: float  # how long it had been still and alone


def _centre(box: Any) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


class AbandonedWatch:
    def __init__(self, classes: list[str] | tuple[str, ...] = CLASSES, seconds: float = 60.0,
                 still_ratio: float = 0.25, owner_distance: float = 1.5,
                 zones: list[Zone] | None = None) -> None:
        self.classes = set(classes)
        self.seconds = seconds
        self.still_ratio = still_ratio
        self.owner_distance = owner_distance
        self.zones = zones or []
        self._state: dict[int, _State] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], zones: list[Zone] | None = None
                    ) -> AbandonedWatch:
        """config: the `abandoned` section; its `zones` names areas of the `zones` section."""
        names = list(config.get("zones") or [])
        known = {z.name: z for z in zones or []}
        missing = [n for n in names if n not in known]
        if missing:
            raise ValueError(f"abandoned.zones: {missing} are not areas in the `zones` section")
        options = {k: v for k, v in config.items() if k not in {"enabled", "cameras", "zones"}}
        return cls(**options, zones=[known[n] for n in names])

    def update(self, tracks: list[Track], removed: list[Track], t: float) -> list[AbandonedEvent]:
        for track in removed:
            self._state.pop(track.id, None)
        people = [np.asarray(tr.box, float) for tr in tracks
                  if not tr.missed and tr.label == "person"]
        events = []
        for track in tracks:
            if track.missed or track.label not in self.classes:
                continue  # a coasting track keeps its state
            box = np.asarray(track.box, float)
            centre = _centre(box)
            if self.zones and not any(z.contains(z.anchor(box)) for z in self.zones):
                self._state.pop(track.id, None)
                continue
            size = max(box[2] - box[0], box[3] - box[1], 1.0)
            state = self._state.get(track.id)
            if state is None or np.hypot(centre[0] - state.anchor[0],
                                         centre[1] - state.anchor[1]) > self.still_ratio * size:
                state = _State(centre, t, None)  # (re)started: it moved
                self._state[track.id] = state
            attended = any(
                np.hypot(*(np.array(_centre(p)) - centre)) <= self.owner_distance * (p[3] - p[1])
                for p in people)
            if attended:
                state.alone_since = None
            elif state.alone_since is None:
                state.alone_since = t
            if state.fired or state.alone_since is None:
                continue
            since = max(state.still_since, state.alone_since)
            if t - since >= self.seconds:
                state.fired = True
                events.append(AbandonedEvent(t, track.id, str(track.label), round(t - since, 1)))
        return events


def abandoned_enabled(config: dict[str, Any] | None, camera: str) -> bool:
    if not config or not config.get("enabled"):
        return False
    cameras = config.get("cameras") or []
    return not cameras or camera in cameras
