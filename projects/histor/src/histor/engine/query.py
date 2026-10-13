"""Split a search sentence into *what* to look for, *when*, and *which cameras*. No LLM needed.

    "red shirt last friday"                  -> "red shirt", Friday 00:00-24:00
    "white van on gate yesterday 6pm to 9pm" -> "white van", yesterday 18:00-21:00, cameras [gate]
    "person with a backpack in the last 2 hours"
    "anyone at reception after 6pm"          -> "reception" is an alias of camera lobby

Times are the server's local time. Unknown words are kept as the visual description.
"""

from __future__ import annotations

import re
from collections.abc import Callable
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


ISO = r"(\d{4}-\d{2}-\d{2})(?:[ t](\d{1,2}:\d{2}(?::\d{2})?))?"  # a date, maybe a time
# (queries are lower-cased first: ISO's "T" arrives as "t")


def _exact(date: str, clock: str | None, end: bool = False) -> datetime:
    """An ISO date and time; a date alone that ends a range runs to the end of its day."""
    if clock:
        hour, rest = clock.split(":", 1)
        return datetime.fromisoformat(f"{date} {int(hour):02d}:{rest}")
    day = datetime.fromisoformat(date)
    return day + timedelta(days=1) if end else day


def _year(text: str | None, now: datetime) -> int:
    year = int(text) if text else now.year
    return year + 2000 if year < 100 else year


def parse_query(text: str, now: datetime | None = None,
                camera_names: list[str] | None = None,
                camera_aliases: dict[str, str] | None = None) -> ParsedQuery:
    """camera_aliases: other names for a camera, e.g. {"reception": "lobby"} (the cameras'
    `about.aliases` in the config)."""
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

    # --- cameras: "on gate", "camera gate", a bare camera name or one of its aliases
    names = {name: name for name in camera_names or []} | dict(camera_aliases or {})
    cameras = []
    for name in sorted(names, key=len, reverse=True):  # longest first: "back gate" before "gate"
        if cut(rf"\b(?:on |at |in |from |camera |cam )?(?:the )?{re.escape(name.lower())}\b"
               r"(?: camera| cam)?") and names[name] not in cameras:
            cameras.append(names[name])

    # --- relative spans: "last 2 hours", "past 30 minutes", "last 3 days"
    m = cut(r"\b(?:in )?(?:the )?(?:last|past) (\d+) (minute|min|hour|hr|day)s?\b")
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                 "hour": timedelta(hours=n), "hr": timedelta(hours=n),
                 "day": timedelta(days=n)}[unit]
        since, until = now - delta, now
        labels.append(m.group(0).strip())

    def cut_date(pattern: str, ymd: Callable[[re.Match[str]], tuple[int, int, int]]
                 ) -> datetime | None:
        nonlocal s
        for m in re.finditer(pattern, s):
            try:
                found = datetime(*ymd(m))
            except ValueError:  # "30/2", "50/50", "9/25": not a date, so it stays in the text
                continue
            s = s[: m.start()] + " " + s[m.end():]
            return found
        return None

    def cut_exact() -> tuple[datetime, datetime] | None:
        """A range of ISO dates / times, or one ISO time (around it, like "at 10pm")."""
        nonlocal s
        for m in re.finditer(rf"\b(?:from |between )?{ISO} ?(?:to|until|till|and|-) ?{ISO}\b", s):
            try:
                span = _exact(m[1], m[2]), _exact(m[3], m[4], end=True)
            except ValueError:
                continue
            s = s[: m.start()] + " " + s[m.end():]
            return span
        for m in re.finditer(rf"\b(?:at |around )?{ISO}\b", s):
            try:
                at = _exact(m[1], m[2]) if m[2] else None
            except ValueError:
                at = None
            if at is not None:  # a date alone is for the explicit dates below
                s = s[: m.start()] + " " + s[m.end():]
                return at - timedelta(minutes=30), at + timedelta(minutes=30)
        return None

    # --- exact times, as a model passes them: "2026-10-12 22:10 to 2026-10-12 22:30",
    #     "2026-10-01 to 2026-10-03" (to the end of that day), or one "2026-10-12 22:10"
    if since is None and (span := cut_exact()):
        since, until = span

    # --- explicit dates: 2026-09-25 or 25/09(/2026)
    day = cut_date(r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: (int(m[1]), int(m[2]), int(m[3])))
    if day is None:
        day = cut_date(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b",
                       lambda m: (_year(m[3], now), int(m[2]), int(m[1])))

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
    # (not inside a date: the "02-30" of "2026-02-30" is no time range)
    elif m := cut(r"(?<![-/\d])\b" + _TIME + r" ?(?:to|-|until) ?" + _TIME + r"\b(?![-/\d])"):
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
    elif m := cut(r"\b(?:in the |(this|last) )?(" + "|".join(PARTS_OF_DAY) + r")\b"):
        a, b = PARTS_OF_DAY[m.group(2)]
        since, until = _at(base, a), _at(base, b)
        if day is None and since > now and m.group(1) != "this" and m.group(2) != "tonight":
            # "last night" asked in the morning: the night that just ended, not tonight
            since, until = since - timedelta(days=1), until - timedelta(days=1)
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
