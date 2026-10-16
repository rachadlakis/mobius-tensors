"""
Tests for shared/web_tools.py — the Researcher's search + page-reading tools.

Every network boundary is faked: `ddgs.DDGS`, the low-level `_http_fetch` (or the
`httpx.Client` under it), `trafilatura`, DNS resolution, robots.txt, and sleeps.
The focus is the responsible-use guarantees: robots.txt, the SSRF guard,
politeness delays, size/redirect caps, and "failure is a return value, not a
raise".
"""

import pytest

from shared import web_tools as wt


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    @pytest.mark.parametrize("val, out", [(-1, 1), (0, 1), (5, 5), (999, 20)])
    def test_clamp(self, val, out):
        assert wt._clamp(val, 1, 20) == out

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("example.com", "https://example.com"),
            ("example.com/a/b#frag", "https://example.com/a/b"),
            ("http://x.io/p?q=1", "http://x.io/p?q=1"),
            ("  https://y.io  ", "https://y.io"),
            ("", ""),
        ],
    )
    def test_normalise_url(self, raw, expected):
        assert wt._normalise_url(raw) == expected


# --------------------------------------------------------------------------- #
# SSRF guard
# --------------------------------------------------------------------------- #
class TestIsPublicUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/x",
            "http://127.0.0.1/x",
            "http://127.99.99.99/x",
            "http://10.0.0.5/x",
            "http://192.168.1.1/x",
            "http://172.16.4.4/x",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://[::1]/x",
            "http://0.0.0.0/x",
            "http://box.internal/x",
            "http://printer.local/x",
        ],
    )
    def test_blocks_internal_targets(self, url):
        assert wt._is_public_url(url) is False

    @pytest.mark.parametrize("url", ["http://93.184.216.34/x", "http://[2606:2800:220:1::]/x"])
    def test_allows_public_ip_literals(self, url):
        assert wt._is_public_url(url) is True

    def test_blocks_hostname_that_resolves_private(self, monkeypatch):
        monkeypatch.setattr(
            wt.socket, "getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 0))],
        )
        assert wt._is_public_url("https://sneaky.example/x") is False

    def test_allows_hostname_that_resolves_public(self, monkeypatch):
        monkeypatch.setattr(
            wt.socket, "getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
        )
        assert wt._is_public_url("https://example.com/x") is True

    def test_unresolvable_host_fails_open(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("nxdomain")

        monkeypatch.setattr(wt.socket, "getaddrinfo", _boom)
        assert wt._is_public_url("https://nope.invalid/x") is True

    def test_empty_host(self):
        assert wt._is_public_url("not-a-url") is False


# --------------------------------------------------------------------------- #
# robots.txt
# --------------------------------------------------------------------------- #
@pytest.fixture
def robots(monkeypatch):
    """Control the robots.txt body _robots() sees (None = unreachable)."""
    box = {"text": None}
    monkeypatch.setattr(wt, "SCRAPE_RESPECT_ROBOTS", True)
    monkeypatch.setattr(wt, "_fetch_robots_txt", lambda robots_url: box["text"])
    return box


class TestRobots:
    def test_disabled_always_allows(self, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_RESPECT_ROBOTS", False)
        assert wt._robots("https://any.example/x") == (True, None, 0.0)

    def test_unreachable_robots_fails_open(self, robots):
        robots["text"] = None
        assert wt._robots("https://x.example/p") == (True, None, 0.0)

    def test_allowed_path(self, robots):
        robots["text"] = "User-agent: *\nDisallow: /private/\n"
        assert wt._robots("https://x.example/public/page") == (True, None, 0.0)

    def test_disallowed_path_blocked(self, robots):
        robots["text"] = "User-agent: *\nDisallow: /secret/\n"
        allowed, reason, _ = wt._robots("https://x.example/secret/doc")
        assert allowed is False
        assert "robots.txt" in reason

    def test_disallow_all_blocks(self, robots):
        robots["text"] = "User-agent: *\nDisallow: /\n"
        assert wt._robots("https://x.example/anything")[0] is False

    def test_wildcard_ua_applies_when_no_specific_rule(self, robots):
        # Our UA has no section; the "*" section governs.
        robots["text"] = "User-agent: BadBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
        assert wt._robots("https://x.example/p")[0] is True

    def test_rule_naming_our_agent_overrides_a_permissive_wildcard(self, robots):
        # A group written specifically for us wins over `*` - we must not fall
        # back to the wildcard to get ourselves un-blocked.
        # robotparser matches on the product token (the bit before the "/").
        token = wt.WEB_INTEL_USER_AGENT.split("/")[0]
        robots["text"] = f"User-agent: {token}\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
        allowed, reason, _ = wt._robots("https://x.example/p")
        assert allowed is False
        assert "robots.txt" in reason

    def test_crawl_delay_is_reported(self, robots):
        robots["text"] = "User-agent: *\nCrawl-delay: 3\nDisallow: /x/\n"
        allowed, reason, delay = wt._robots("https://x.example/ok")
        assert (allowed, reason) == (True, None)
        assert delay == 3.0

    def test_fetch_robots_txt_uses_our_user_agent(self, monkeypatch):
        seen = {}

        class _Resp:
            status_code = 200
            content = b"User-agent: *\nDisallow:\n"
            text = "User-agent: *\nDisallow:\n"

        def _get(url, **kwargs):
            seen.update(kwargs)
            return _Resp()

        monkeypatch.setattr("httpx.get", _get)
        wt._fetch_robots_txt("https://x.example/robots.txt")
        assert "ErgonResearcher" in seen["headers"]["User-Agent"]

    def test_fetch_robots_txt_non_200_is_none(self, monkeypatch):
        class _Resp:
            status_code = 403
            content = b"nope"
            text = "nope"

        monkeypatch.setattr("httpx.get", lambda url, **k: _Resp())
        assert wt._fetch_robots_txt("https://x.example/robots.txt") is None


# --------------------------------------------------------------------------- #
# Politeness delay
# --------------------------------------------------------------------------- #
class FakeClock:
    """Deterministic stand-in for the `time` module."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(wt, "time", c)
    wt._last_fetch_at.clear()
    yield c
    wt._last_fetch_at.clear()


class TestBePolite:
    def test_first_hit_does_not_sleep(self, clock, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_POLITENESS_DELAY_SECONDS", 1.0)
        wt._be_polite("host.example", 0.0)
        assert clock.slept == []

    def test_immediate_second_hit_sleeps_the_gap(self, clock, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_POLITENESS_DELAY_SECONDS", 1.0)
        wt._be_polite("host.example", 0.0)
        wt._be_polite("host.example", 0.0)
        assert clock.slept == [1.0]

    def test_no_sleep_once_enough_time_has_passed(self, clock, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_POLITENESS_DELAY_SECONDS", 1.0)
        wt._be_polite("h.example", 0.0)
        clock.advance(5.0)
        wt._be_polite("h.example", 0.0)
        assert clock.slept == []

    def test_crawl_delay_overrides_default(self, clock, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_POLITENESS_DELAY_SECONDS", 0.0)
        wt._be_polite("h.example", 5.0)
        wt._be_polite("h.example", 5.0)
        assert clock.slept == [5.0]

    def test_per_host_tracking(self, clock, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_POLITENESS_DELAY_SECONDS", 1.0)
        wt._be_polite("a.example", 0.0)
        wt._be_polite("b.example", 0.0)  # different host, no wait
        assert clock.slept == []


# --------------------------------------------------------------------------- #
# _http_fetch — manual per-hop redirect walk, SSRF re-check, size cap
# --------------------------------------------------------------------------- #
class _FakeResp:
    """One response in a (possibly redirecting) chain. `location` set => a
    redirect the walker must validate before following."""

    def __init__(self, *, url="https://example.io/p", status=200, headers=None,
                 chunks=(b"<html>ok</html>",), encoding="utf-8", location=None):
        self.url = url
        self.status_code = status
        self.is_redirect = location is not None
        self.headers = dict(headers or {"content-type": "text/html"})
        if location is not None:
            self.headers["location"] = location
        self._chunks = chunks
        self.encoding = encoding

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_bytes(self):
        yield from self._chunks


class FakeClient:
    """Serves a queue of _FakeResp, one per .stream() call."""

    queue: list = []

    def __init__(self, *a, **k): ...
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def stream(self, method, url):
        if not FakeClient.queue:
            raise AssertionError(f"unexpected extra request to {url}")
        return FakeClient.queue.pop(0)


@pytest.fixture
def fake_httpx(monkeypatch):
    FakeClient.queue = []
    monkeypatch.setattr("httpx.Client", FakeClient)
    monkeypatch.setattr(wt, "_is_public_url", lambda url: True)
    return FakeClient


class TestHttpFetch:
    def test_success(self, fake_httpx):
        fake_httpx.queue = [_FakeResp(url="https://example.io/final")]
        out = wt._http_fetch("https://example.io/p")
        assert out["ok"] is True
        assert out["final_url"] == "https://example.io/final"
        assert "ok" in out["text"]

    def test_size_cap_stops_download(self, fake_httpx, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_MAX_BYTES", 10)
        fake_httpx.queue = [_FakeResp(url="https://x.io/big",
                                      chunks=[b"a" * 8, b"b" * 8, b"c" * 8])]
        out = wt._http_fetch("https://x.io/big")
        assert out["ok"] is True
        assert len(out["text"]) <= 16  # stopped after exceeding the 10-byte cap

    def test_http_error_status(self, fake_httpx):
        fake_httpx.queue = [_FakeResp(url="https://x.io/missing", status=404)]
        out = wt._http_fetch("https://x.io/missing")
        assert out["ok"] is False
        assert out["status_code"] == 404

    def test_follows_a_redirect_to_a_public_target(self, fake_httpx):
        fake_httpx.queue = [
            _FakeResp(url="https://x.io/a", status=302, location="https://x.io/b"),
            _FakeResp(url="https://x.io/b", chunks=[b"<html>dest</html>"]),
        ]
        out = wt._http_fetch("https://x.io/a")
        assert out["ok"] is True
        assert out["final_url"] == "https://x.io/b"
        assert "dest" in out["text"]

    def test_relative_redirect_is_resolved(self, fake_httpx):
        fake_httpx.queue = [
            _FakeResp(url="https://x.io/a/b", status=301, location="/c"),
            _FakeResp(url="https://x.io/c"),
        ]
        out = wt._http_fetch("https://x.io/a/b")
        assert out["ok"] is True
        assert out["final_url"] == "https://x.io/c"

    def test_redirect_to_private_address_is_blocked_before_the_hop(self, monkeypatch):
        monkeypatch.setattr("httpx.Client", FakeClient)
        FakeClient.queue = [
            _FakeResp(url="https://x.io/p", status=302,
                      location="http://169.254.169.254/latest/meta-data/"),
            _FakeResp(url="http://169.254.169.254/latest/meta-data/",
                      chunks=[b"CLOUD-SECRETS"]),  # must never be reached
        ]
        monkeypatch.setattr(wt, "_is_public_url", lambda url: "169.254" not in url)
        out = wt._http_fetch("https://x.io/p")
        assert out["ok"] is False
        assert "non-public" in out["error"]
        assert len(FakeClient.queue) == 1  # the metadata response was never consumed

    def test_redirect_off_the_web_is_blocked(self, fake_httpx):
        fake_httpx.queue = [
            _FakeResp(url="https://x.io/p", status=302, location="file:///etc/passwd"),
        ]
        out = wt._http_fetch("https://x.io/p")
        assert out["ok"] is False
        assert "non-http" in out["error"]

    def test_redirect_chain_is_bounded(self, fake_httpx, monkeypatch):
        monkeypatch.setattr(wt, "SCRAPE_MAX_REDIRECTS", 3)
        fake_httpx.queue = [
            _FakeResp(url=f"https://x.io/{i}", status=302, location=f"https://x.io/{i + 1}")
            for i in range(10)
        ]
        out = wt._http_fetch("https://x.io/0")
        assert out["ok"] is False
        assert "redirect" in out["error"].lower()

    def test_transport_exception_becomes_error_dict(self, monkeypatch):
        class _Raises(FakeClient):
            def stream(self, *a):
                raise ConnectionError("boom")

        monkeypatch.setattr("httpx.Client", _Raises)
        monkeypatch.setattr(wt, "_is_public_url", lambda url: True)
        out = wt._http_fetch("https://x.io/p")
        assert out["ok"] is False
        assert "Request failed" in out["error"]


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
class FakeDDGS:
    payload: list = []
    raises: Exception | None = None
    news_payload: list = []
    news_raises: Exception | None = None

    def text(self, query, max_results=None, region=None):
        if FakeDDGS.raises:
            raise FakeDDGS.raises
        FakeDDGS.last_call = {"query": query, "max_results": max_results, "region": region}
        return FakeDDGS.payload

    def news(self, query, max_results=None, region=None, timelimit=None):
        if FakeDDGS.news_raises:
            raise FakeDDGS.news_raises
        FakeDDGS.last_news_call = {
            "query": query, "max_results": max_results,
            "region": region, "timelimit": timelimit,
        }
        return FakeDDGS.news_payload


@pytest.fixture
def fake_ddgs(monkeypatch):
    FakeDDGS.payload = []
    FakeDDGS.raises = None
    FakeDDGS.news_payload = []
    FakeDDGS.news_raises = None
    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    return FakeDDGS


class TestWebSearch:
    def test_empty_query(self):
        assert wt._web_search_blocking("   ")["success"] is False

    def test_maps_ddgs_fields_to_stable_shape(self, fake_ddgs):
        fake_ddgs.payload = [
            {"title": "T1", "href": "https://a.io", "body": "snippet one"},
            {"title": "T2", "href": "https://b.io", "body": "snippet two"},
        ]
        result = wt._web_search_blocking("ai act rules")
        assert result["success"] is True
        assert result["mode"] == "web"
        assert result["count"] == 2
        assert result["results"][0] == {
            "title": "T1", "url": "https://a.io", "snippet": "snippet one",
        }

    def test_max_results_clamped(self, fake_ddgs):
        wt._web_search_blocking("x", max_results=999)
        assert fake_ddgs.last_call["max_results"] == 20
        wt._web_search_blocking("x", max_results=0)
        assert fake_ddgs.last_call["max_results"] == 1

    def test_upstream_error_is_returned_not_raised(self, fake_ddgs):
        fake_ddgs.raises = RuntimeError("rate limited")
        result = wt._web_search_blocking("x")
        assert result["success"] is False
        assert "rate limited" in result["error"]

    def test_unknown_provider(self, monkeypatch):
        monkeypatch.setattr(wt, "SEARCH_PROVIDER", "googol")
        assert wt._web_search_blocking("x")["success"] is False

    # --- recency= -> news vertical ---
    def test_recency_uses_news_vertical_and_maps_extra_fields(self, fake_ddgs):
        fake_ddgs.news_payload = [
            {"title": "Road closed", "url": "https://n.io/1", "body": "the E11 is shut",
             "date": "2026-08-26", "source": "Gulf News"},
        ]
        out = wt._web_search_blocking("dubai road closure", recency="w")
        assert out["success"] is True
        assert out["mode"] == "news"
        assert out["timelimit"] == "w"
        assert out["results"][0]["date"] == "2026-08-26"
        assert out["results"][0]["source"] == "Gulf News"

    def test_bad_recency_falls_back_to_default_window(self, fake_ddgs):
        fake_ddgs.news_payload = []
        wt._web_search_blocking("x", recency="bogus")
        assert fake_ddgs.last_news_call["timelimit"] == "w"

    def test_recency_day_passthrough(self, fake_ddgs):
        fake_ddgs.news_payload = []
        wt._web_search_blocking("x", recency="d")
        assert fake_ddgs.last_news_call["timelimit"] == "d"

    # --- read=True -> also fetch top pages ---
    def test_read_true_scrapes_each_result(self, fake_ddgs, monkeypatch):
        fake_ddgs.payload = [
            {"title": "A", "href": "https://a.io", "body": "sa"},
            {"title": "B", "href": "https://b.io", "body": "sb"},
        ]
        monkeypatch.setattr(
            wt, "_fetch_page_blocking",
            lambda url, *a, **k: {"success": True, "content": f"body {url}", "truncated": False},
        )
        out = wt._web_search_blocking("q", read=True)
        assert [r["content"] for r in out["results"]] == ["body https://a.io", "body https://b.io"]

    def test_read_true_captures_per_page_fetch_error(self, fake_ddgs, monkeypatch):
        fake_ddgs.payload = [{"title": "A", "href": "https://a.io", "body": "s"}]
        monkeypatch.setattr(wt, "_fetch_page_blocking",
                            lambda url, *a, **k: {"success": False, "error": "HTTP 500"})
        out = wt._web_search_blocking("q", read=True)
        assert out["results"][0]["content"] is None
        assert out["results"][0]["fetch_error"] == "HTTP 500"

    def test_read_true_default_and_cap_on_max_results(self, fake_ddgs, monkeypatch):
        fake_ddgs.payload = []
        monkeypatch.setattr(wt, "_fetch_page_blocking", lambda *a, **k: {"success": True, "content": "", "truncated": False})
        wt._web_search_blocking("q", read=True)
        assert fake_ddgs.last_call["max_results"] == 2      # read default
        wt._web_search_blocking("q", read=True, max_results=99)
        assert fake_ddgs.last_call["max_results"] == 5      # read cap


# --------------------------------------------------------------------------- #
# fetch_page
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_fetch(monkeypatch):
    state = {
        "http": {"ok": True, "final_url": "https://example.io/final", "status_code": 200,
                 "content_type": "text/html; charset=utf-8", "text": "<html>...</html>"},
        "extract": "Clean body text.",
    }

    class _Meta:
        title = "Page Title"

    monkeypatch.setattr(wt, "_http_fetch", lambda url: state["http"])
    monkeypatch.setattr(wt, "_is_public_url", lambda url: True)
    monkeypatch.setattr(wt, "_be_polite", lambda host, delay: None)
    monkeypatch.setattr(wt, "_robots", lambda url: (True, None, 0.0))
    monkeypatch.setattr("trafilatura.extract", lambda *a, **k: state["extract"])
    monkeypatch.setattr("trafilatura.extract_metadata", lambda *a, **k: _Meta())
    return state


class TestFetchPage:
    def test_success_shape(self, fake_fetch):
        result = wt._fetch_page_blocking("example.io/article")
        assert result["success"] is True
        assert result["url"] == "https://example.io/article"
        assert result["final_url"] == "https://example.io/final"
        assert result["title"] == "Page Title"
        assert result["content"] == "Clean body text."
        assert result["truncated"] is False

    def test_empty_url(self, fake_fetch):
        assert wt._fetch_page_blocking("")["success"] is False

    def test_rejects_non_http_scheme(self, fake_fetch):
        assert wt._fetch_page_blocking("ftp://x.io/f")["success"] is False

    def test_rejects_non_public_address_without_fetching(self, monkeypatch):
        called = []
        monkeypatch.setattr(wt, "_http_fetch", lambda url: called.append(url) or {})
        monkeypatch.setattr(wt, "_is_public_url", lambda url: False)
        result = wt._fetch_page_blocking("http://10.0.0.1/admin")
        assert result["success"] is False
        assert "non-public" in result["error"]
        assert called == []  # never hit the network

    def test_robots_block(self, monkeypatch, fake_fetch):
        monkeypatch.setattr(wt, "_robots", lambda url: (False, "Disallowed by robots", 0.0))
        result = wt._fetch_page_blocking("https://x.io/p")
        assert result["success"] is False
        assert "Disallowed" in result["error"]

    def test_http_error_propagates(self, fake_fetch):
        fake_fetch["http"] = {"ok": False, "error": "HTTP 403", "status_code": 403}
        result = wt._fetch_page_blocking("https://x.io/forbidden")
        assert result["success"] is False
        assert result["status_code"] == 403

    def test_non_html_content_type(self, fake_fetch):
        fake_fetch["http"] = {**fake_fetch["http"], "content_type": "application/pdf"}
        result = wt._fetch_page_blocking("https://x.io/file.pdf")
        assert result["success"] is False
        assert "content-type" in result["error"]

    def test_no_extractable_content(self, fake_fetch):
        fake_fetch["extract"] = None
        result = wt._fetch_page_blocking("https://x.io/empty")
        assert result["success"] is False

    def test_truncation(self, fake_fetch):
        fake_fetch["extract"] = "x" * 5000
        result = wt._fetch_page_blocking("https://x.io/long", max_chars=1000)
        assert result["truncated"] is True
        assert result["char_count"] == 1000
        assert len(result["content"]) == 1000

    def test_max_chars_clamped_low(self, fake_fetch):
        fake_fetch["extract"] = "y" * 5000
        result = wt._fetch_page_blocking("https://x.io/p", max_chars=1)
        assert result["char_count"] == 200

    def test_politeness_is_applied(self, monkeypatch, fake_fetch):
        seen = {}
        monkeypatch.setattr(wt, "_robots", lambda url: (True, None, 2.5))
        monkeypatch.setattr(wt, "_be_polite", lambda host, delay: seen.update(host=host, delay=delay))
        wt._fetch_page_blocking("https://news.example/article")
        assert seen == {"host": "news.example", "delay": 2.5}


def test_web_intel_tools_collection():
    assert [t.__name__ for t in wt.WEB_INTEL_TOOLS] == [
        "web_search",
        "fetch_page",
        "get_weather",
    ]


# --------------------------------------------------------------------------- #
# get_weather  (Open-Meteo, keyless JSON API)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_open_meteo(monkeypatch):
    """Stub _http_get_json: first call -> geocode payload, second -> forecast."""
    box = {
        "geocode": {"ok": True, "data": {"results": [{
            "name": "Dubai", "admin1": "Dubai", "country": "United Arab Emirates",
            "latitude": 25.07, "longitude": 55.31,
        }]}},
        "forecast": {"ok": True, "data": {
            "timezone": "Asia/Dubai",
            "current": {"temperature_2m": 38.4, "relative_humidity_2m": 55,
                        "wind_speed_10m": 14.0, "weather_code": 1},
            "daily": {
                "time": ["2026-08-27", "2026-08-28"],
                "weather_code": [0, 95],
                "temperature_2m_max": [41.0, 40.0],
                "temperature_2m_min": [31.0, 30.0],
                "precipitation_sum": [0.0, 3.2],
                "precipitation_probability_max": [0, 40],
                "wind_speed_10m_max": [18.0, 22.0],
            },
        }},
    }
    calls = {"n": 0}

    def _fake(url, params):
        calls["n"] += 1
        return box["geocode"] if calls["n"] == 1 else box["forecast"]

    monkeypatch.setattr(wt, "_http_get_json", _fake)
    box["calls"] = calls
    return box


class TestGetWeather:
    def test_empty_location(self):
        assert wt._get_weather_blocking("")["success"] is False

    def test_happy_path_shape(self, fake_open_meteo):
        out = wt._get_weather_blocking("Dubai")
        assert out["success"] is True
        assert out["location"]["country"] == "United Arab Emirates"
        assert out["current"]["conditions"] == "Mainly clear"
        assert out["current"]["temp_c"] == 38.4
        assert len(out["daily"]) == 2
        assert out["daily"][0]["conditions"] == "Clear sky"
        assert out["daily"][1]["conditions"] == "Thunderstorm"
        assert out["daily"][1]["precipitation_mm"] == 3.2

    def test_date_filter_selects_one_day(self, fake_open_meteo):
        out = wt._get_weather_blocking("Dubai", date="2026-08-28")
        assert out["success"] is True
        assert [d["date"] for d in out["daily"]] == ["2026-08-28"]

    def test_date_outside_window_is_error(self, fake_open_meteo):
        out = wt._get_weather_blocking("Dubai", date="2030-01-01")
        assert out["success"] is False
        assert "forecast window" in out["error"]

    def test_unknown_place(self, monkeypatch):
        monkeypatch.setattr(wt, "_http_get_json", lambda u, p: {"ok": True, "data": {"results": []}})
        out = wt._get_weather_blocking("Nowheresville")
        assert out["success"] is False
        assert "Could not find" in out["error"]

    def test_geocode_api_error_is_returned(self, monkeypatch):
        monkeypatch.setattr(wt, "_http_get_json", lambda u, p: {"ok": False, "error": "HTTP 503"})
        out = wt._get_weather_blocking("Dubai")
        assert out["success"] is False
        assert out["error"] == "HTTP 503"

    def test_compound_location_retries_a_narrower_form(self, monkeypatch):
        """'Dubai-Abu Dhabi, UAE' - the string the agent actually sent - should
        fall back to the first leg ('Dubai') rather than failing outright."""
        seen = []

        def _fake(url, params):
            if "geocoding" in url:
                seen.append(params["name"])
                if params["name"] == "Dubai":
                    return {"ok": True, "data": {"results": [
                        {"name": "Dubai", "latitude": 25.07, "longitude": 55.31}]}}
                return {"ok": True, "data": {"results": []}}
            return {"ok": True, "data": {
                "timezone": "Asia/Dubai", "current": {}, "daily": {"time": []}}}

        monkeypatch.setattr(wt, "_http_get_json", _fake)
        out = wt._get_weather_blocking("Dubai-Abu Dhabi, UAE")
        assert out["success"] is True
        assert out["location"]["latitude"] == 25.07
        assert "Dubai-Abu Dhabi, UAE" in seen and "Dubai" in seen

    def test_multi_place_failure_tells_the_agent_to_split(self, monkeypatch):
        monkeypatch.setattr(wt, "_http_get_json",
                            lambda u, p: {"ok": True, "data": {"results": []}})
        out = wt._get_weather_blocking("Dubai and Abu Dhabi")
        assert out["success"] is False
        assert "once per city" in out["error"]

    def test_forecast_api_error_is_returned(self, monkeypatch):
        seq = [
            {"ok": True, "data": {"results": [{"latitude": 1, "longitude": 2, "name": "X"}]}},
            {"ok": False, "error": "HTTP 500"},
        ]
        monkeypatch.setattr(wt, "_http_get_json", lambda u, p: seq.pop(0))
        out = wt._get_weather_blocking("X")
        assert out["success"] is False
        assert out["error"] == "HTTP 500"


class TestWeatherLocationCandidates:
    def test_plain_name_is_the_only_candidate(self):
        assert wt._weather_location_candidates("Dubai") == ["Dubai"]

    def test_original_comes_first(self):
        assert wt._weather_location_candidates("Jebel Ali Port, UAE")[0] == "Jebel Ali Port, UAE"

    def test_corridor_string_yields_the_first_leg(self):
        cands = wt._weather_location_candidates("Dubai-Abu Dhabi, UAE")
        assert "Dubai" in cands

    def test_route_word_is_split(self):
        assert "Dubai" in wt._weather_location_candidates("Dubai to Al Ain")

    def test_trailing_qualifiers_are_dropped_progressively(self):
        cands = wt._weather_location_candidates("Marina, Jebel Ali, Dubai, UAE")
        assert cands == [
            "Marina, Jebel Ali, Dubai, UAE",
            "Marina, Jebel Ali, Dubai",
            "Marina, Jebel Ali",
            "Marina",
        ]

    def test_no_duplicates(self):
        cands = wt._weather_location_candidates("Dubai, Dubai")
        assert len(cands) == len(set(c.casefold() for c in cands))


class TestHttpGetJson:
    def test_refuses_non_public(self, monkeypatch):
        monkeypatch.setattr(wt, "_is_public_url", lambda url: False)
        out = wt._http_get_json("http://169.254.169.254/x", {})
        assert out["ok"] is False
        assert "non-public" in out["error"]

    def test_http_error(self, monkeypatch):
        class _Resp:
            status_code = 500
            is_redirect = False
            content = b""

            def json(self):
                return {}

        monkeypatch.setattr(wt, "_is_public_url", lambda url: True)
        monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
        out = wt._http_get_json("https://api.example/x", {})
        assert out == {"ok": False, "status_code": 500, "error": "HTTP 500"}

    def test_ok_parses_json(self, monkeypatch):
        class _Resp:
            status_code = 200
            is_redirect = False
            content = b'{"a": 1}'

            def json(self):
                return {"a": 1}

        monkeypatch.setattr(wt, "_is_public_url", lambda url: True)
        monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
        out = wt._http_get_json("https://api.example/x", {})
        assert out == {"ok": True, "data": {"a": 1}}


# --------------------------------------------------------------------------- #
# Tool entry points must not block the event loop
#
# ADK calls a synchronous tool function directly on the agent server's loop, so
# a `def` tool that does HTTP + time.sleep freezes the whole worker for the turn
# (no health-check replies, no status streaming). The registered tools must be
# coroutines that offload their blocking body to a thread.
# --------------------------------------------------------------------------- #
class TestToolsAreNonBlocking:
    def test_every_io_bound_web_tool_is_a_coroutine(self):
        import inspect

        blocking_io = {"web_search", "fetch_page", "get_weather"}
        for tool in wt.WEB_INTEL_TOOLS:
            if tool.__name__ in blocking_io:
                assert inspect.iscoroutinefunction(tool), (
                    f"{tool.__name__} does I/O but is sync - it will block the "
                    f"agent server's event loop"
                )

    async def test_fetch_page_delegates_to_the_blocking_impl(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            wt, "_fetch_page_blocking",
            lambda url, max_chars: seen.update(url=url, max_chars=max_chars) or {"success": True},
        )
        assert await wt.fetch_page("https://x.io/p", max_chars=1234) == {"success": True}
        assert seen == {"url": "https://x.io/p", "max_chars": 1234}

    async def test_web_search_delegates_to_the_blocking_impl(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            wt, "_web_search_blocking",
            lambda q, rec, read, mx: seen.update(q=q, rec=rec, read=read, mx=mx) or {"success": True},
        )
        assert await wt.web_search("fuel", recency="w", read=True, max_results=3)["success"] is True \
            if False else True  # call below does the real assert
        out = await wt.web_search("fuel", recency="w", read=True, max_results=3)
        assert out == {"success": True}
        assert seen == {"q": "fuel", "rec": "w", "read": True, "mx": 3}

    async def test_get_weather_delegates(self, monkeypatch):
        monkeypatch.setattr(wt, "_get_weather_blocking", lambda loc, d: {"ok": loc, "d": d})
        assert await wt.get_weather("Dubai", "2026-09-02") == {"ok": "Dubai", "d": "2026-09-02"}

    async def test_the_blocking_body_really_runs_off_the_loop(self, monkeypatch):
        """A slow tool must not stall other coroutines on the same loop."""
        import asyncio
        import time as real_time

        monkeypatch.setattr(
            wt, "_fetch_page_blocking",
            lambda url, max_chars: (real_time.sleep(0.3), {"success": True})[1],
        )
        ticks = 0
        stop = False

        async def heartbeat():
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.01)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        assert (await wt.fetch_page("https://x.io/p"))["success"] is True
        stop = True
        await hb
        assert ticks >= 5  # a blocked loop would show ~0
