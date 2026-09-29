"""Split a search sentence into *what* to look for, *when*, and *which cameras*. No LLM needed.

    "red shirt last friday"                  -> "red shirt", Friday 00:00-24:00
    "white van on gate yesterday 6pm to 9pm" -> "white van", yesterday 18:00-21:00, cameras [gate]
    "person with a backpack in the last 2 hours"

Times are the server's local time. Unknown words are kept as the visual description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PARTS_OF_DAY = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21),
                "night": (21, 30), "tonight": (21, 30)}  # 30 = 06:00 next day
FILLER = {
    "search", "find", "show", "me", "look", "for", "any", "all", "the", "a", "an", "in", "on",
    "at", "during", "from", "of", "past", "last", "this", "was", "there", "is", "who", "did",
    "anyone", "somebody", "someone", "please", "camera", "cameras", "cam", "footage", "video",
    "between", "and", "to", "until", "till", "after", "before", "since", "around", "within",
    "wearing", "with",
}
# Words that describe *what* even though they look like filler: keep "with"/"wearing" if
# they sit between visual words (handled below by only stripping filler at the edges).


@dataclass
class ParsedQuery:
    text: str
    since: datetime | None = None
    until: datetime | None = None
    cameras: list[str] = field(default_factory=list)
    when: str = ""  # human-readable time range

    def range(self) -> tuple[float | None, float | None]:
        return (self.since.timestamp() if self.since else None,
                self.until.timestamp() if self.until else None)


_TIME = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"


def _hour(h: str, m: str | None, ampm: str | None) -> float:
    hour = int(h) % 24
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return hour + (int(m) / 60 if m else 0)


def _at(day: datetime, hour: float) -> datetime:
    return day + timedelta(hours=hour)


def parse_query(text: str, now: datetime | None = None,
                camera_names: list[str] | None = None) -> ParsedQuery:
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    s = " " + text.lower().strip().rstrip("?.!") + " "
    day: datetime | None = None
    since = until = None
    labels: list[str] = []

    def cut(pattern: str) -> re.Match[str] | None:
        nonlocal s
        m = re.search(pattern, s)
        if m:
            s = s[: m.start()] + " " + s[m.end():]
        return m

    # --- cameras: "on gate", "camera gate", or a bare camera name
    cameras = []
    for name in sorted(camera_names or [], key=len, reverse=True):
        if cut(rf"\b(?:on |at |from |camera |cam )?{re.escape(name.lower())}\b(?: camera| cam)?"):
            cameras.append(name)

    # --- relative spans: "last 2 hours", "past 30 minutes", "last 3 days"
    m = cut(r"\b(?:in )?(?:the )?(?:last|past) (\d+) (minute|min|hour|hr|day)s?\b")
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                 "hour": timedelta(hours=n), "hr": timedelta(hours=n),
                 "day": timedelta(days=n)}[unit]
        since, until = now - delta, now
        labels.append(m.group(0).strip())

    # --- explicit dates: 2026-09-25 or 25/09(/2026)
    m = cut(r"\b(\d{4})-(\d{2})-(\d{2})\b")
    if m:
        day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = cut(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
        if m:
            year = int(m.group(3)) if m.group(3) else now.year
            year += 2000 if year < 100 else 0
            day = datetime(year, int(m.group(2)), int(m.group(1)))

    # --- named days
    if day is None and since is None:
        if cut(r"\b(?:the )?day before yesterday\b"):
            day = today - timedelta(days=2)
        elif cut(r"\byesterday\b"):
            day = today - timedelta(days=1)
        elif cut(r"\btoday\b"):
            day = today
        elif m := cut(r"\b(?:(last|past|previous|this|on) )?(" + "|".join(WEEKDAYS) + r")s?\b"):
            target = WEEKDAYS.index(m.group(2))
            back = (now.weekday() - target) % 7
            if back == 0 and m.group(1) in {"last", "past", "previous"}:
                back = 7  # "last friday" said on a Friday = a week ago
            day = today - timedelta(days=back)
        elif cut(r"\b(?:last|this|past) week\b"):
            since, until = today - timedelta(days=7), now
            labels.append("last 7 days")

    # --- time of day on top of the day
    base = day or today
    if m := cut(r"\b(?:between|from) " + _TIME + r" (?:and|to|-|until) " + _TIME + r"\b"):
        start, end = _hour(*m.group(1, 2, 3)), _hour(*m.group(4, 5, 6))
        if m.group(3) is None and m.group(6) == "pm":  # "2 to 4pm" -> both pm
            start = _hour(m.group(1), m.group(2), "pm")
        since, until = _at(base, start), _at(base, end if end > start else end + 24)
    elif m := cut(r"\b" + _TIME + r" ?(?:to|-|until) ?" + _TIME + r"\b"):
        start, end = _hour(*m.group(1, 2, 3)), _hour(*m.group(4, 5, 6))
        if m.group(3) is None and m.group(6) == "pm":
            start = _hour(m.group(1), m.group(2), "pm")
        since, until = _at(base, start), _at(base, end if end > start else end + 24)
    elif m := cut(r"\bafter " + _TIME + r"\b"):
        since, until = _at(base, _hour(*m.group(1, 2, 3))), _at(base, 24)
    elif m := cut(r"\bbefore " + _TIME + r"\b"):
        since, until = base, _at(base, _hour(*m.group(1, 2, 3)))
    elif m := cut(r"\b(?:at|around) " + _TIME + r"\b"):
        centre = _at(base, _hour(*m.group(1, 2, 3)))
        since, until = centre - timedelta(minutes=30), centre + timedelta(minutes=30)
    elif m := cut(r"\b(?:in the |this )?(" + "|".join(PARTS_OF_DAY) + r")\b"):
        a, b = PARTS_OF_DAY[m.group(1)]
        since, until = _at(base, a), _at(base, b)
    elif day is not None:
        since, until = day, day + timedelta(days=1)

    if until is not None and until > now and since is not None and since <= now:
        until = now

    words = s.split()
    while words and words[0] in FILLER:  # strip filler at the edges only, so
        words.pop(0)                     # "man with a backpack" keeps its "with a"
    while words and words[-1] in FILLER:
        words.pop()
    description = " ".join(words) or "a person"

    when = ""
    if since is not None and until is not None:
        fmt = "%a %d %b %H:%M"
        when = f"{since.strftime(fmt)} - {until.strftime(fmt if until.date() != since.date() else '%H:%M')}"
    return ParsedQuery(description, since, until, cameras, when)
