"""
Web search + page-reading tools for the Researcher.

Three tools, all returning a plain ``dict`` with a ``success`` flag (never
raising, so a tool failure is something the LLM can read and react to):

    web_search(query, recency, read) -> ranked results; recency= -> news vertical,
                                        read=True -> also fetch the top pages' text
    fetch_page(url)                  -> clean extracted main text of one known URL
    get_weather(location, date)      -> current + forecast (Open-Meteo, keyless API)

Responsible-use guarantees (see shared/config.py for the knobs):

* **Public web only.** GET requests, no cookies, no auth headers, no form
  submission. Requests to localhost / private / link-local addresses and the
  cloud metadata IP are refused (SSRF guard). Redirects are followed manually,
  one hop at a time, with that check re-run before each hop is connected to.
* **robots.txt is honoured**, including ``Crawl-delay``. A politeness gap is
  kept between consecutive hits on the same host.
* **Honest identity.** An identifiable bot User-Agent with a contact URL - not
  a spoofed browser string.
* **Bounded.** Response bodies are size-capped; redirects are limited and must
  stay http(s); extracted text is length-capped.
* **No circumvention.** Nothing here logs in, pays, solves CAPTCHAs, or works
  around a paywall / rate limit. If a site says no, the tool returns
  ``success: false`` and the agent moves on.

Heavy imports (ddgs, trafilatura, httpx) are done lazily inside the functions.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse, urlunparse

from google.adk.agents.llm_agent import ToolUnion

from shared.config import (
    NEWS_DEFAULT_TIMELIMIT,
    SCRAPE_MAX_BYTES,
    SCRAPE_MAX_CHARS,
    SCRAPE_MAX_REDIRECTS,
    SCRAPE_POLITENESS_DELAY_SECONDS,
    SCRAPE_RESPECT_ROBOTS,
    SCRAPE_TIMEOUT_SECONDS,
    SEARCH_MAX_RESULTS,
    SEARCH_PROVIDER,
    SEARCH_REGION,
    WEATHER_FORECAST_DAYS,
    WEATHER_FORECAST_URL,
    WEATHER_GEOCODE_URL,
    WEB_INTEL_USER_AGENT,
)

logger = logging.getLogger(__name__)

if not SCRAPE_RESPECT_ROBOTS:
    logger.warning(
        "SCRAPE_RESPECT_ROBOTS is disabled - robots.txt will be ignored. "
        "This is the operator's legal/ethical responsibility; re-enable it "
        "unless you have a specific reason not to."
    )

# Per-host timestamp of the last fetch, for the politeness delay.
_last_fetch_at: dict[str, float] = {}

_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}
_METADATA_ADDRESSES = {"169.254.169.254", "fd00:ec2::254"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _normalise_url(url: str) -> str:
    """Add a scheme if missing and strip the fragment."""
    url = (url or "").strip()
    if not url:
        return url
    if "://" not in url:
        url = "https://" + url
    parts = urlparse(url)
    return urlunparse(parts._replace(fragment=""))


def _address_is_private(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or str(ip) in _METADATA_ADDRESSES
    )


def _is_public_url(url: str) -> bool:
    """Best-effort SSRF guard: reject internal / non-public destinations.

    Blocks obvious internal hostnames, private/loopback/link-local IP literals,
    and hostnames that resolve to such addresses. Fails open only when a name
    cannot be resolved at all (the request itself will then just fail).
    """
    host = (urlparse(url).hostname or "").lower().strip(".")
    if not host:
        return False
    if host in _BLOCKED_HOSTNAMES:
        return False
    if host.endswith((".local", ".internal", ".localdomain", ".lan")):
        return False
    if _address_is_private(host):  # IP literal
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # unresolvable now; let the real request fail later
    return not any(_address_is_private(info[4][0]) for info in infos)


def _fetch_robots_txt(robots_url: str) -> str | None:
    """GET robots.txt with our real User-Agent.

    Using ``RobotFileParser.read()`` directly is a trap: it fetches with
    urllib's default UA, which many sites (Wikipedia, Google, ...) answer with
    403 - and the parser then treats a 403 as "disallow everything", so the
    the agent would refuse pages it is actually allowed to read. We fetch it
    ourselves instead. A non-200 (no robots.txt, or a server error) is treated
    as "no stated restriction".
    """
    import httpx

    try:
        resp = httpx.get(
            robots_url,
            follow_redirects=True,
            timeout=min(SCRAPE_TIMEOUT_SECONDS, 10.0),
            headers={"User-Agent": WEB_INTEL_USER_AGENT},
        )
    except Exception:
        return None
    if resp.status_code == 200 and len(resp.content) <= 512 * 1024:
        return resp.text
    return None


def _robots(url: str) -> tuple[bool, str | None, float]:
    """Return (allowed, reason, crawl_delay_seconds).

    Fails open on a missing/unreadable robots.txt (that is 'not disallowed',
    not 'forbidden'). crawl_delay is 0.0 when none is advertised.
    """
    if not SCRAPE_RESPECT_ROBOTS:
        return True, None, 0.0
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"

    text = _fetch_robots_txt(robots_url)
    if text is None:
        return True, None, 0.0

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(text.splitlines())

    # NOT `can_fetch(ours) or can_fetch("*")`: RobotFileParser already picks the
    # most specific matching group and falls back to `*` itself, so OR-ing the
    # wildcard in would let us ignore a rule written specifically to block us.
    allowed = parser.can_fetch(WEB_INTEL_USER_AGENT, url)
    delay = 0.0
    try:
        raw = parser.crawl_delay(WEB_INTEL_USER_AGENT)
        if raw is None:
            raw = parser.crawl_delay("*")
        delay = float(raw) if raw is not None else 0.0
    except (AttributeError, TypeError, ValueError):
        delay = 0.0
    if not allowed:
        return False, f"Disallowed by {robots_url}", delay
    return True, None, delay


def _be_polite(host: str, crawl_delay: float) -> None:
    """Sleep so consecutive hits on one host are spaced out."""
    gap = max(SCRAPE_POLITENESS_DELAY_SECONDS, crawl_delay)
    if gap <= 0:
        return
    last = _last_fetch_at.get(host)
    now = time.monotonic()
    if last is not None and (wait := gap - (now - last)) > 0:
        time.sleep(wait)
    _last_fetch_at[host] = time.monotonic()


def _robots_allows(url: str) -> tuple[bool, str | None]:
    """Back-compat wrapper: (allowed, reason) only."""
    allowed, reason, _ = _robots(url)
    return allowed, reason


# --------------------------------------------------------------------------- #
# Low-level fetch (GET only, no cookies, size + redirect capped, SSRF-guarded)
# --------------------------------------------------------------------------- #
def _http_fetch(url: str) -> dict:
    """GET ``url``, following redirects MANUALLY so every hop is SSRF-checked
    *before* the connection is made.

    httpx's own ``follow_redirects=True`` would have already contacted an
    internal host (e.g. a ``302`` to ``http://169.254.169.254/...``) before we
    could look at the ``Location`` - by which point the SSRF request has
    happened. So redirects are disabled on the client and walked here, running
    the scheme + ``_is_public_url`` checks on each target first.
    """
    import httpx

    try:
        client = httpx.Client(
            follow_redirects=False,
            timeout=SCRAPE_TIMEOUT_SECONDS,
            headers={"User-Agent": WEB_INTEL_USER_AGENT, "Accept": "text/html,*/*;q=0.5"},
            cookies=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"Client init failed: {exc}"}

    current = url
    try:
        with client:
            for _ in range(SCRAPE_MAX_REDIRECTS + 1):
                if urlparse(current).scheme not in ("http", "https"):
                    return {"ok": False, "error": f"Refusing a non-http(s) URL: {current}"}
                if not _is_public_url(current):
                    return {"ok": False, "error": "Refusing a non-public / internal address."}

                with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        loc = resp.headers.get("location")
                        if not loc:
                            return {"ok": False, "error": "Redirect with no Location header."}
                        current = urljoin(str(resp.url), loc)
                        continue

                    if resp.status_code >= 400:
                        return {
                            "ok": False,
                            "status_code": resp.status_code,
                            "error": f"HTTP {resp.status_code}",
                        }
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body.extend(chunk)
                        if len(body) > SCRAPE_MAX_BYTES:
                            break
                    return {
                        "ok": True,
                        "final_url": str(resp.url),
                        "status_code": resp.status_code,
                        "content_type": resp.headers.get("content-type", ""),
                        "text": bytes(body).decode(resp.encoding or "utf-8", errors="replace"),
                    }
            return {"ok": False, "error": f"Too many redirects (>{SCRAPE_MAX_REDIRECTS})."}
    except Exception as exc:
        return {"ok": False, "error": f"Request failed: {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# 1. web_search
# --------------------------------------------------------------------------- #
def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    raw = DDGS().text(query, max_results=max_results, region=SEARCH_REGION)
    results = []
    for item in raw or []:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("href") or item.get("url", ""),
                "snippet": item.get("body") or item.get("snippet", ""),
            }
        )
    return results


def _news_duckduckgo(query: str, max_results: int, timelimit: str) -> list[dict]:
    from ddgs import DDGS

    raw = DDGS().news(
        query, max_results=max_results, region=SEARCH_REGION, timelimit=timelimit
    )
    results = []
    for item in raw or []:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url") or item.get("href", ""),
            "snippet": item.get("body") or item.get("excerpt", ""),
            "date": item.get("date", ""),
            "source": item.get("source", ""),
        })
    return results


def _web_search_blocking(
    query: str,
    recency: str | None = None,
    read: bool = False,
    max_results: int | None = None,
) -> dict:
    """
    Search the public web and return ranked results (title, URL, snippet).

    Args:
        query: natural-language query or keywords.
        recency: set to "d" / "w" / "m" / "y" to search the NEWS vertical within
            that window (past day / week / month / year) instead of the general
            web index - use it for questions about recent events (product
            launches, funding, announcements, rule changes). News results also carry
            `date` and `source`.
        read: when True, also fetch and extract the main text of the top results
            (one page fetch each, with a robots.txt lookup + politeness delay -
            so keep max_results small). Use it when you want page content, not
            just snippets. When False, follow up with fetch_page(url) on the
            result you want.
        max_results: how many results (default 5 for a plain search, 2 when
            read=True; capped at 20, or 5 when read=True).

    Returns:
        dict: success, query, mode ("web" | "news"), provider, count, and
            results - a list of {title, url, snippet, date?, source?, content?,
            truncated?, fetch_error?}. On failure success=False + error.
    """
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "Empty search query."}
    if SEARCH_PROVIDER not in ("duckduckgo", "ddg", "ddgs"):
        return {"success": False, "error": f"Unknown SEARCH_PROVIDER "
                f"{SEARCH_PROVIDER!r} (expected 'duckduckgo')."}

    news_mode = recency is not None
    timelimit = (recency or "").strip().lower()
    if news_mode and timelimit not in ("d", "w", "m", "y"):
        timelimit = (NEWS_DEFAULT_TIMELIMIT or "w").strip().lower()

    default_n = 2 if read else SEARCH_MAX_RESULTS
    limit = _clamp(max_results if max_results is not None else default_n,
                   1, 5 if read else 20)

    try:
        results = (
            _news_duckduckgo(query, limit, timelimit) if news_mode
            else _search_duckduckgo(query, limit)
        )
    except Exception as exc:  # ddgs raises on rate limits / transient upstream errors
        return {
            "success": False,
            "provider": SEARCH_PROVIDER,
            "query": query,
            "mode": "news" if news_mode else "web",
            "error": f"Search failed: {type(exc).__name__}: {exc}",
        }

    if read:
        for hit in results:
            page = _fetch_page_blocking(hit.get("url", ""))
            if page["success"]:
                hit["content"] = page["content"]
                hit["truncated"] = page["truncated"]
            else:
                hit["content"] = None
                hit["fetch_error"] = page.get("error")

    out = {
        "success": True,
        "provider": SEARCH_PROVIDER,
        "query": query,
        "mode": "news" if news_mode else "web",
        "count": len(results),
        "results": results,
    }
    if news_mode:
        out["timelimit"] = timelimit
    return out


# --------------------------------------------------------------------------- #
# 2. fetch_page
# --------------------------------------------------------------------------- #
def _fetch_page_blocking(url: str, max_chars: int | None = None) -> dict:
    """
    Fetch one public web page and return its main text, cleaned of navigation,
    ads, and boilerplate.

    Only plain GET requests to http(s) pages are made - no logins, no cookies,
    no paywall workarounds. robots.txt is respected. If a page can't be fetched
    for any of those reasons, this returns success=false; that is a normal
    outcome, not an error to work around.

    Args:
        url: the page URL (a missing scheme is assumed to be https)
        max_chars: truncate the extracted text to this many characters
            (default 8000, capped at 40000)

    Returns:
        dict: success, url, final_url, title, content, char_count, truncated;
            on failure, success=False and an error.
    """
    import trafilatura

    url = _normalise_url(url)
    if not url:
        return {"success": False, "error": "Empty URL."}
    if urlparse(url).scheme not in ("http", "https"):
        return {"success": False, "error": f"Unsupported URL scheme: {url!r}"}
    if not _is_public_url(url):
        return {
            "success": False,
            "url": url,
            "error": "Refusing to fetch a non-public / internal address.",
        }

    allowed, reason, crawl_delay = _robots(url)
    if not allowed:
        return {"success": False, "url": url, "error": reason}

    limit = _clamp(max_chars if max_chars is not None else SCRAPE_MAX_CHARS, 200, 40000)

    _be_polite(urlparse(url).netloc, crawl_delay)
    fetched = _http_fetch(url)
    if not fetched["ok"]:
        out = {"success": False, "url": url, "error": fetched["error"]}
        if "status_code" in fetched:
            out["status_code"] = fetched["status_code"]
        return out

    content_type = fetched["content_type"]
    if content_type and "html" not in content_type and "xml" not in content_type:
        return {
            "success": False,
            "url": url,
            "content_type": content_type,
            "error": f"Not an HTML page (content-type: {content_type}).",
        }

    extracted = trafilatura.extract(
        fetched["text"], url=url, include_comments=False, include_tables=True
    )
    if not extracted:
        return {
            "success": False,
            "url": url,
            "error": "Could not extract readable content from the page.",
        }

    metadata = trafilatura.extract_metadata(fetched["text"])
    title = getattr(metadata, "title", None) if metadata else None

    truncated = len(extracted) > limit
    return {
        "success": True,
        "url": url,
        "final_url": fetched["final_url"],
        "title": title or "",
        "content": extracted[:limit],
        "char_count": min(len(extracted), limit),
        "truncated": truncated,
    }


# --------------------------------------------------------------------------- #
# 3. get_weather  (Open-Meteo — free, keyless JSON API, not a scrape)
# --------------------------------------------------------------------------- #
def _http_get_json(url: str, params: dict) -> dict:
    """Small bounded GET for a JSON API. SSRF-guarded, size-capped, honest UA,
    no redirects, no cookies. Used for documented programmatic endpoints
    (Open-Meteo) - not for crawling pages, so no robots.txt lookup."""
    import httpx

    if not _is_public_url(url):
        return {"ok": False, "error": "Refusing to call a non-public address."}
    try:
        resp = httpx.get(
            url,
            params=params,
            follow_redirects=False,
            timeout=SCRAPE_TIMEOUT_SECONDS,
            headers={"User-Agent": WEB_INTEL_USER_AGENT, "Accept": "application/json"},
        )
    except Exception as exc:
        return {"ok": False, "error": f"Request failed: {type(exc).__name__}: {exc}"}

    if resp.status_code >= 400:
        return {"ok": False, "status_code": resp.status_code, "error": f"HTTP {resp.status_code}"}
    if resp.is_redirect:
        return {"ok": False, "error": f"Unexpected redirect from {url}."}
    if len(resp.content) > SCRAPE_MAX_BYTES:
        return {"ok": False, "error": "API response too large."}
    try:
        return {"ok": True, "data": resp.json()}
    except Exception as exc:
        return {"ok": False, "error": f"Bad JSON from API: {exc}"}


# WMO weather interpretation codes (the subset Open-Meteo returns), condensed.
_WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _describe_weather_code(code) -> str:
    try:
        return _WMO_WEATHER_CODES.get(int(code), f"Code {code}")
    except (TypeError, ValueError):
        return "Unknown"


# Route / corridor joiners: "Dubai-Abu Dhabi", "Dubai to Abu Dhabi", "Dubai / Al
# Ain". Open-Meteo's name index has no entry for a compound like this, so we split
# on the first one and keep the leading leg.
_WEATHER_ROUTE_SEPARATORS = (" — ", " – ", " - ", " to ", " and ", " & ", "/", "-", "–", "—", "&")


def _weather_location_candidates(location: str) -> list[str]:
    """Progressively narrower forms of a place string to try against Open-Meteo's
    geocoder, most specific first.

    Open-Meteo geocoding matches a single toponym. It copes with "City, Country"
    but not a hyphenated multi-city corridor ("Dubai-Abu Dhabi, UAE"), a trailing
    descriptor word ("Jebel Ali Port, UAE"), or a route string ("Dubai to Al
    Ain"). We hand back the original plus these fallbacks and stop at the first
    that resolves.
    """
    out: list[str] = []

    def _add(value: str) -> None:
        value = " ".join(value.split()).strip(" ,-–—/&")
        if value and value.casefold() not in {s.casefold() for s in out}:
            out.append(value)

    _add(location)

    # First leg of a route / corridor.
    head = location
    for sep in _WEATHER_ROUTE_SEPARATORS:
        if sep in head:
            head = head.split(sep, 1)[0]
    _add(head)

    # Drop trailing comma-separated qualifiers one at a time, then the bare first
    # segment ("Jebel Ali Port, Dubai, UAE" -> "Jebel Ali Port, Dubai" -> "Jebel
    # Ali Port").
    parts = [p.strip() for p in location.split(",") if p.strip()]
    for i in range(len(parts) - 1, 0, -1):
        _add(", ".join(parts[:i]))
    if parts:
        first = parts[0]
        for sep in _WEATHER_ROUTE_SEPARATORS:
            if sep in first:
                first = first.split(sep, 1)[0]
        _add(first)

    return out


def _resolve_weather_location(location: str) -> tuple[dict | None, str | None]:
    """Resolve a place string to a match dict with at least latitude/longitude
    (and name/admin1/country when known).

    Tries Open-Meteo's geocoder over `_weather_location_candidates`.
    Returns (None, error) if nothing resolves.
    """
    last_error = f"Could not find a place called {location!r}."
    for name in _weather_location_candidates(location):
        geo = _http_get_json(WEATHER_GEOCODE_URL, {"name": name, "count": 1})
        if not geo["ok"]:
            last_error = geo["error"]
            continue
        matches = (geo["data"] or {}).get("results") or []
        if matches:
            return matches[0], None

    if any(sep in location.casefold() for sep in _WEATHER_ROUTE_SEPARATORS):
        last_error += (
            " If this is a route or several places, call get_weather once per "
            "city (e.g. 'Dubai', then 'Abu Dhabi')."
        )
    return None, last_error


def _get_weather_blocking(location: str, date: str | None = None) -> dict:
    """
    Current conditions and a multi-day forecast for a place, via the Open-Meteo
    public API (free, no key). Use this instead of scraping a weather website -
    it is faster and returns structured data.

    Args:
        location: a SINGLE place name, e.g. "Dubai", "Abu Dhabi", "Jebel Ali".
            "City, Country" is fine. NOT a route or a list - for "Dubai to Abu
            Dhabi" call this once per city; a compound like "Dubai-Abu Dhabi" is
            not a place.
        date: optional ISO date (YYYY-MM-DD) to focus on. Must fall inside the
            forecast window (about a week ahead). Omit for the whole forecast.

    Returns:
        dict: success, location (resolved name/country/lat/lon/timezone),
            current (temp, humidity, wind, conditions), and daily - a list of
            {date, conditions, temp_min_c, temp_max_c, precipitation_mm,
            precipitation_probability_pct, wind_max_kmh}. On failure
            success=False with an error.
    """
    location = (location or "").strip()
    if not location:
        return {"success": False, "error": "Empty location."}

    place, geo_error = _resolve_weather_location(location)
    if place is None:
        return {"success": False, "location": location, "error": geo_error}
    lat, lon = place.get("latitude"), place.get("longitude")

    forecast = _http_get_json(
        WEATHER_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "forecast_days": _clamp(WEATHER_FORECAST_DAYS, 1, 16),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
            ),
        },
    )
    if not forecast["ok"]:
        return {"success": False, "location": location, "error": forecast["error"]}
    data = forecast["data"] or {}

    cur = data.get("current") or {}
    current = {
        "temp_c": cur.get("temperature_2m"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "conditions": _describe_weather_code(cur.get("weather_code")),
    }

    d = data.get("daily") or {}
    days = d.get("time") or []
    daily = []
    for i, day in enumerate(days):
        daily.append({
            "date": day,
            "conditions": _describe_weather_code(_at(d.get("weather_code"), i)),
            "temp_min_c": _at(d.get("temperature_2m_min"), i),
            "temp_max_c": _at(d.get("temperature_2m_max"), i),
            "precipitation_mm": _at(d.get("precipitation_sum"), i),
            "precipitation_probability_pct": _at(d.get("precipitation_probability_max"), i),
            "wind_max_kmh": _at(d.get("wind_speed_10m_max"), i),
        })

    resolved = {
        "name": place.get("name"),
        "admin1": place.get("admin1"),
        "country": place.get("country"),
        "latitude": lat,
        "longitude": lon,
        "timezone": data.get("timezone"),
    }

    if date:
        match = [row for row in daily if row["date"] == date]
        if not match:
            return {
                "success": False,
                "location": resolved,
                "error": f"{date} is outside the forecast window "
                         f"({days[0]}..{days[-1]})." if days else "No forecast data.",
                "current": current,
            }
        daily = match

    return {
        "success": True,
        "location": resolved,
        "current": current,
        "daily": daily,
        "units": {"temp": "C", "wind": "km/h", "precipitation": "mm"},
    }


def _at(seq, i):
    """seq[i] or None - Open-Meteo returns parallel arrays."""
    try:
        return seq[i]
    except (TypeError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Tool entry points
#
# ADK calls a synchronous tool function DIRECTLY on the agent server's event
# loop (google/adk/flows/llm_flows/functions.py: the non-live path goes straight
# to FunctionTool.run_async, and the thread-pool runner is only bound on the
# live path). Every one of these does blocking I/O - robots.txt lookups, a
# time.sleep politeness gap, page downloads - so as plain `def` they froze the
# whole worker: no health-check replies, no status streaming, nothing else
# served, for the entire research turn.
#
# So the blocking bodies live in the `_*_blocking` functions above and the
# registered tools are coroutines that hand them to a worker thread. The
# signatures and docstrings are what ADK turns into the tool schema, so they
# must match the implementations exactly.
# --------------------------------------------------------------------------- #
async def web_search(
    query: str,
    recency: str | None = None,
    read: bool = False,
    max_results: int | None = None,
) -> dict:
    """
    Search the public web and return ranked results (title, URL, snippet).

    Args:
        query: natural-language query or keywords.
        recency: set to "d" / "w" / "m" / "y" to search the NEWS vertical within
            that window (past day / week / month / year) instead of the general
            web index - use it for questions about recent events (product
            launches, funding, announcements, rule changes). News results also carry
            `date` and `source`.
        read: when True, also fetch and extract the main text of the top results
            (one page fetch each, with a robots.txt lookup + politeness delay -
            so keep max_results small). Use it when you want page content, not
            just snippets. When False, follow up with fetch_page(url) on the
            result you want.
        max_results: how many results (default 5 for a plain search, 2 when
            read=True; capped at 20, or 5 when read=True).

    Returns:
        dict: success, query, mode ("web" | "news"), provider, count, and
            results - a list of {title, url, snippet, date?, source?, content?,
            truncated?, fetch_error?}. On failure success=False + error.
    """
    return await asyncio.to_thread(
        _web_search_blocking, query, recency, read, max_results
    )


async def fetch_page(url: str, max_chars: int | None = None) -> dict:
    """
    Fetch one public web page and return its main text, cleaned of navigation,
    ads, and boilerplate.

    Only plain GET requests to http(s) pages are made - no logins, no cookies,
    no paywall workarounds. robots.txt is respected. If a page can't be fetched
    for any of those reasons, this returns success=false; that is a normal
    outcome, not an error to work around.

    Args:
        url: the page URL (a missing scheme is assumed to be https)
        max_chars: truncate the extracted text to this many characters
            (default 8000, capped at 40000)

    Returns:
        dict: success, url, final_url, title, content, char_count, truncated;
            on failure, success=False and an error.
    """
    return await asyncio.to_thread(_fetch_page_blocking, url, max_chars)


async def get_weather(location: str, date: str | None = None) -> dict:
    """
    Current conditions and a multi-day forecast for a place, via the Open-Meteo
    public API (free, no key). Use this instead of scraping a weather website -
    it is faster and returns structured data.

    Args:
        location: a SINGLE place name, e.g. "Dubai", "Abu Dhabi", "Jebel Ali".
            "City, Country" is fine. NOT a route or a list - for "Dubai to Abu
            Dhabi" call this once per city; a compound like "Dubai-Abu Dhabi" is
            not a place.
        date: optional ISO date (YYYY-MM-DD) to focus on. Must fall inside the
            forecast window (about a week ahead). Omit for the whole forecast.

    Returns:
        dict: success, location (resolved name/country/lat/lon/timezone),
            current (temp, humidity, wind, conditions), and daily - a list of
            {date, conditions, temp_min_c, temp_max_c, precipitation_mm,
            precipitation_probability_pct, wind_max_kmh}. On failure
            success=False with an error.
    """
    return await asyncio.to_thread(_get_weather_blocking, location, date)



# Tool collection for the Researcher's tool list.
WEB_INTEL_TOOLS: list[ToolUnion] = [
    web_search,
    fetch_page,
    get_weather,
]
