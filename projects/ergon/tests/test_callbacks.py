"""
Tests for shared/callbacks.py - the observability + Researcher fetch-budget
callbacks wired into every employee.

All offline: no ADK runner, just the callback functions called directly with
minimal fake tool / context objects.
"""

import logging

import pytest

from shared import callbacks as cb


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeCtx:
    def __init__(self, fcid="fc-1", invocation_id="inv-1", agent_name="Test_Agent"):
        self.function_call_id = fcid
        self.invocation_id = invocation_id
        self.agent_name = agent_name


class FakeUsage:
    prompt_token_count = 100
    candidates_token_count = 20
    total_token_count = 120


class FakeLlmResponse:
    def __init__(self, usage=None):
        self.usage_metadata = usage


@pytest.fixture(autouse=True)
def _clean_state():
    cb._tool_starts.clear()
    cb._fetch_counts.clear()
    yield
    cb._tool_starts.clear()
    cb._fetch_counts.clear()


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #
class TestToolLogging:
    def test_before_records_start_and_logs(self, caplog):
        ctx = FakeCtx()
        with caplog.at_level(logging.INFO, logger="ergon.employees"):
            out = cb.log_before_tool(FakeTool("list_files"), {"limit": 5}, ctx)
        assert out is None
        assert cb._call_key(ctx) in cb._tool_starts
        assert "list_files" in caplog.text

    def test_after_pops_start_and_reports_ok(self, caplog):
        ctx = FakeCtx()
        cb.log_before_tool(FakeTool("get_file"), {}, ctx)
        with caplog.at_level(logging.INFO, logger="ergon.employees"):
            out = cb.log_after_tool(FakeTool("get_file"), {}, ctx, {"success": True})
        assert out is None
        assert cb._call_key(ctx) not in cb._tool_starts
        assert "ok" in caplog.text

    def test_after_flags_tool_failure_as_warning(self, caplog):
        ctx = FakeCtx()
        with caplog.at_level(logging.WARNING, logger="ergon.employees"):
            cb.log_after_tool(
                FakeTool("fetch_page"),
                {"url": "x"},
                ctx,
                {"success": False, "error": "blocked by robots.txt"},
            )
        assert "FAILED" in caplog.text
        assert "robots.txt" in caplog.text

    def test_non_dict_response_is_treated_as_ok(self, caplog):
        ctx = FakeCtx()
        with caplog.at_level(logging.INFO, logger="ergon.employees"):
            cb.log_after_tool(FakeTool("t"), {}, ctx, "plain string result")
        assert "ok" in caplog.text

    def test_logging_never_raises_on_bad_context(self):
        # No attributes at all - must not blow up the request.
        assert cb.log_before_tool(FakeTool("t"), {}, object()) is None
        assert cb.log_after_tool(FakeTool("t"), {}, object(), {}) is None

    def test_short_args_truncates(self):
        long = {"q": "x" * 500}
        assert len(cb._short_args(long)) <= 161
        assert cb._short_args({"a": 1}) == "a=1"


class TestModelLogging:
    def test_logs_token_usage(self, caplog):
        with caplog.at_level(logging.INFO, logger="ergon.employees"):
            out = cb.log_after_model(FakeCtx(), FakeLlmResponse(FakeUsage()))
        assert out is None
        assert "total=120" in caplog.text

    def test_no_usage_metadata_is_silent(self, caplog):
        with caplog.at_level(logging.INFO, logger="ergon.employees"):
            cb.log_after_model(FakeCtx(), FakeLlmResponse(None))
        assert "tokens" not in caplog.text


# --------------------------------------------------------------------------- #
# Researcher fetch budget
# --------------------------------------------------------------------------- #
class TestScrapeBudget:
    def test_non_fetch_tool_is_ignored(self):
        assert cb.enforce_scrape_budget(FakeTool("get_weather"), {}, FakeCtx()) is None
        assert cb._fetch_counts == {}

    def test_plain_web_search_is_not_charged(self):
        # web_search without read=True fetches no pages
        assert cb.enforce_scrape_budget(FakeTool("web_search"), {}, FakeCtx()) is None
        assert cb._fetch_counts == {}

    def test_web_search_read_charges_max_results(self):
        ctx = FakeCtx(invocation_id="inv-r")
        cb.enforce_scrape_budget(FakeTool("web_search"), {"read": True, "max_results": 3}, ctx)
        assert cb._fetch_counts["inv-r"] == 3

    def test_web_search_read_default_cost_is_two(self):
        ctx = FakeCtx(invocation_id="inv-r2")
        cb.enforce_scrape_budget(FakeTool("web_search"), {"read": True}, ctx)
        assert cb._fetch_counts["inv-r2"] == 2

    def test_web_search_read_cost_is_capped_like_the_tool(self):
        # web_search clamps read=True to 5 pages; the budget must not bill 99.
        ctx = FakeCtx(invocation_id="inv-cap")
        cb.enforce_scrape_budget(
            FakeTool("web_search"), {"read": True, "max_results": 99}, ctx
        )
        assert cb._fetch_counts["inv-cap"] == cb._WEB_SEARCH_READ_CAP

    def test_web_search_read_cost_survives_a_junk_max_results(self):
        ctx = FakeCtx(invocation_id="inv-junk")
        assert cb.enforce_scrape_budget(
            FakeTool("web_search"), {"read": True, "max_results": "lots"}, ctx
        ) is None
        assert cb._fetch_counts["inv-junk"] == 2  # falls back to the default

    def test_fetch_page_counts_one_each(self):
        ctx = FakeCtx(invocation_id="inv-a")
        for _ in range(cb.WEB_INTEL_MAX_FETCHES_PER_TASK):
            assert cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, ctx) is None
        assert cb._fetch_counts["inv-a"] == cb.WEB_INTEL_MAX_FETCHES_PER_TASK

    def test_over_budget_fetch_is_skipped(self):
        ctx = FakeCtx(invocation_id="inv-b")
        for _ in range(cb.WEB_INTEL_MAX_FETCHES_PER_TASK):
            cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, ctx)
        blocked = cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, ctx)
        assert blocked is not None
        assert blocked["success"] is False
        assert "budget" in blocked["error"].lower()


    def test_budget_is_per_invocation(self):
        a = FakeCtx(invocation_id="inv-1")
        b = FakeCtx(invocation_id="inv-2")
        for _ in range(cb.WEB_INTEL_MAX_FETCHES_PER_TASK):
            cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, a)
        # b is a different request - still has full budget
        assert cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, b) is None

    def test_clear_scrape_budget_resets_counter(self):
        ctx = FakeCtx(invocation_id="inv-x")
        cb.enforce_scrape_budget(FakeTool("fetch_page"), {}, ctx)
        cb.clear_scrape_budget(ctx)
        assert "inv-x" not in cb._fetch_counts


# --------------------------------------------------------------------------- #
# strip_unreadable_file_parts - keep un-ingestible blobs out of the LLM request
# --------------------------------------------------------------------------- #
class _Req:
    def __init__(self, contents):
        self.contents = contents


def _content(parts, role="user"):
    from google.genai import types
    return types.Content(role=role, parts=parts)


def _blob_part(data: bytes, mime, name=None):
    from google.genai import types
    blob = types.Blob(data=data, mime_type=mime)
    if name is not None:
        blob.display_name = name
    return types.Part(inline_data=blob)


class TestStripUnreadableFileParts:
    def test_xlsx_blob_becomes_a_note(self):
        from google.genai import types
        xlsx = _blob_part(
            b"PK\x03\x04binary",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sales.xlsx",
        )
        req = _Req([_content([types.Part(text="analyse this"), xlsx])])

        assert cb.strip_unreadable_file_parts(None, req) is None

        parts = req.contents[0].parts
        assert parts[0].text == "analyse this"
        assert parts[1].inline_data is None
        assert "sales.xlsx" in parts[1].text
        assert "file-aware tool" in parts[1].text

    def test_text_and_image_and_pdf_blobs_are_kept(self):
        from google.genai import types
        keep = [
            _blob_part(b"col1,col2\n", "text/csv", "d.csv"),
            _blob_part(b"\x89PNG", "image/png", "c.png"),
            _blob_part(b"%PDF-1.4", "application/pdf", "r.pdf"),
        ]
        req = _Req([_content([types.Part(text="hi"), *keep])])
        cb.strip_unreadable_file_parts(None, req)
        assert all(p.inline_data is not None for p in req.contents[0].parts[1:])

    def test_does_not_mutate_the_original_content_object(self):
        # the orchestrator's user_content references the same object; it must
        # keep its blob so _remember_uploads can still stash the upload.
        original = _content([_blob_part(b"PK\x03\x04", "application/zip", "x.xlsx")])
        req = _Req([original])
        cb.strip_unreadable_file_parts(None, req)
        assert original.parts[0].inline_data.data == b"PK\x03\x04"   # untouched
        assert req.contents[0] is not original                       # replaced
        assert req.contents[0].parts[0].inline_data is None          # noted

    def test_no_file_parts_is_a_no_op(self):
        from google.genai import types
        req = _Req([_content([types.Part(text="just text")])])
        cb.strip_unreadable_file_parts(None, req)
        assert req.contents[0].parts[0].text == "just text"

    def test_never_raises_on_a_weird_request(self):
        assert cb.strip_unreadable_file_parts(None, object()) is None
        assert cb.strip_unreadable_file_parts(None, _Req(None)) is None


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
class TestCallbackBundles:
    def test_worker_bundle_shape(self):
        assert cb.WORKER_CALLBACKS["before_tool_callback"] == [cb.log_before_tool]
        assert cb.log_after_tool in cb.WORKER_CALLBACKS["after_tool_callback"]
        assert cb.log_after_model in cb.WORKER_CALLBACKS["after_model_callback"]
        assert cb.WORKER_CALLBACKS["before_model_callback"] == [cb.strip_unreadable_file_parts]

    def test_web_intel_bundle_runs_budget_before_logging(self):
        before = cb.WEB_INTEL_CALLBACKS["before_tool_callback"]
        assert before == [cb.enforce_scrape_budget, cb.log_before_tool]
        assert cb.WEB_INTEL_CALLBACKS["after_agent_callback"] is cb.clear_scrape_budget
        assert cb.strip_unreadable_file_parts in cb.WEB_INTEL_CALLBACKS["before_model_callback"]

    def test_agents_import_with_callbacks_wired(self):
        # Import-time smoke: the agent modules spread the bundles into LlmAgent.
        import data_agent_adk.agent as data_analyst
        import web_intel_agent_adk.agent as web_intel

        assert data_analyst.create_agent is not None
        assert web_intel.create_agent is not None
